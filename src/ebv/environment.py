"""Environmental predictors for every grid cell, from four open raster datasets.

Each raster is read on a pixel grid that nests exactly inside the 0.1 degree cells
(k x k pixels per cell), so per-cell statistics are plain reshapes, not resampling.
Only land pixels count, so coastal cells are not diluted by the sea.

Dataset              Read at            Pixels per cell   Variables
WorldClim 2.1        1/120 deg (~1 km)  12 x 12           19 bioclimatic variables
Copernicus GLO-90    1/600 deg (~180 m) 60 x 60           elevation mean / SD / range, slope
ESA WorldCover 2021  1/750 deg (~150 m) 75 x 75           land-cover fractions, Shannon
                                                          diversity, distance to inland water
WorldCover S2 NDVI   1/750 deg (~150 m) 75 x 75           median NDVI, NDVI seasonal range
"""
import warnings
from contextlib import ExitStack

import numpy as np
import pandas as pd
import rasterio
from rasterio.features import rasterize
from rasterio.merge import merge
from rasterio.transform import from_bounds
from rasterio.windows import from_bounds as window_from_bounds
from scipy.ndimage import distance_transform_edt

from . import config
from .region import GridSpec

WORLDCLIM = config.RAW / "wc2.1_30s_bio_tile33.tif"
WORLDCLIM_URL = "https://geodata.ucdavis.edu/climate/worldclim/2_1/tiles/tile/tile_33_wc2.1_30s_bio.tif"
DEM_URL = ("https://copernicus-dem-90m.s3.amazonaws.com/Copernicus_DSM_COG_30_{ns}{lat:02d}_00_E{lon:03d}_00_DEM/"
           "Copernicus_DSM_COG_30_{ns}{lat:02d}_00_E{lon:03d}_00_DEM.tif")
WORLDCOVER_URL = ("https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/map/"
                  "ESA_WorldCover_10m_2021_v200_N{lat:02d}E{lon:03d}_Map.tif")
NDVI_URL = ("https://esa-worldcover-s2.s3.eu-central-1.amazonaws.com/ndvi/2021/N{lat:02d}/"
            "ESA_WorldCover_10m_2021_v200_N{lat:02d}E{lon:03d}_NDVI.tif")

LAND_COVER = {10: "tree", 20: "shrub", 30: "grass", 40: "crop", 50: "built",
              60: "bare", 80: "water", 90: "wetland", 95: "mangrove"}
WATER_CLASS = 80
REMOTE_READ_OPTIONS = dict(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
                           CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif", GDAL_HTTP_MULTIRANGE="YES",
                           GDAL_HTTP_MERGE_CONSECUTIVE_RANGES="YES", GDAL_HTTP_MAX_RETRY="5",
                           GDAL_HTTP_RETRY_DELAY="3")


# ----------------------------------------------------------------------------- helpers
def _shape(spec: GridSpec, k: int) -> tuple[int, int]:
    return spec.n_rows * k, spec.n_cols * k


def land_mask(spec: GridSpec, k: int, land_polygons) -> np.ndarray:
    """True for pixels whose centre falls on land (any Indian state)."""
    height, width = _shape(spec, k)
    transform = from_bounds(*spec.bounds, width, height)
    return rasterize(((g, 1) for g in land_polygons), out_shape=(height, width),
                     transform=transform, fill=0, dtype="uint8").astype(bool)


def per_cell(pixels: np.ndarray, spec: GridSpec, k: int) -> np.ndarray:
    """Group a pixel array into (rows, cols, k*k): one vector of pixels per cell."""
    height, width = _shape(spec, k)
    if pixels.shape != (height, width):
        raise ValueError(f"raster shape {pixels.shape} does not match grid ({height}, {width})")
    return (pixels.reshape(spec.n_rows, k, spec.n_cols, k)
                  .transpose(0, 2, 1, 3)
                  .reshape(spec.n_rows, spec.n_cols, k * k))


def to_cells(values: np.ndarray, spec: GridSpec, cells: pd.DataFrame) -> np.ndarray:
    """Pick each cell's value out of a (rows, cols) array."""
    rows, cols = spec.row_col(cells)
    return values[rows, cols]


def nan_stat(func, blocks: np.ndarray) -> np.ndarray:
    """Apply a nan-aware statistic over the pixels of each cell, quietly giving NaN for empty cells."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return func(blocks, axis=-1)


def read_mosaic(urls: list[str], spec: GridSpec, k: int, nodata, band=1) -> np.ndarray:
    """Mosaic remote Cloud-Optimized GeoTIFF tiles onto the grid's pixel lattice.

    Tiles that do not exist (open-ocean tiles) are skipped. GDAL picks the internal
    overview closest to the requested pixel size, so only the needed bytes are fetched.
    """
    with rasterio.Env(**REMOTE_READ_OPTIONS), ExitStack() as stack:
        sources = []
        for url in urls:
            try:
                sources.append(stack.enter_context(rasterio.open(f"/vsicurl/{url}")))
            except rasterio.errors.RasterioIOError:
                continue
        mosaic, _ = merge(sources, bounds=spec.bounds, res=spec.cell / k, nodata=nodata,
                          indexes=[band] if np.isscalar(band) else list(band))
    return mosaic[0] if np.isscalar(band) else mosaic


def _tile_range(spec: GridSpec, step: int) -> tuple[range, range]:
    """Latitudes and longitudes (south-west corners) of `step`-degree tiles covering the grid."""
    lats = range(int(np.floor(spec.south / step) * step), int(np.ceil(spec.north)), step)
    lons = range(int(np.floor(spec.west / step) * step), int(np.ceil(spec.east)), step)
    return lats, lons


# ----------------------------------------------------------------------------- datasets
def climate(spec: GridSpec, cells: pd.DataFrame, land_polygons) -> pd.DataFrame:
    """Mean of each WorldClim bioclimatic variable over the cell's land pixels."""
    k = 12
    with rasterio.open(WORLDCLIM) as src:
        window = window_from_bounds(*spec.bounds, src.transform).round_offsets().round_lengths()
        bio = src.read(window=window).astype("float32")
        bio[bio <= src.nodata] = np.nan
    land = land_mask(spec, k, land_polygons)
    out = {}
    for i, layer in enumerate(bio, start=1):
        layer[~land] = np.nan
        out[f"bio{i}"] = to_cells(nan_stat(np.nanmean, per_cell(layer, spec, k)), spec, cells)
    return pd.DataFrame(out, index=cells.index)


def terrain(spec: GridSpec, cells: pd.DataFrame, land_polygons) -> pd.DataFrame:
    """Elevation mean, standard deviation (ruggedness), 5-95% range and mean slope."""
    k = 60
    lats, lons = _tile_range(spec, 1)
    urls = [DEM_URL.format(ns="N", lat=la, lon=lo) for la in lats for lo in lons]
    dem = read_mosaic(urls, spec, k, nodata=np.nan).astype("float32")
    dem[~land_mask(spec, k, land_polygons)] = np.nan

    pixel_deg = spec.cell / k
    pixel_lat = spec.north - (np.arange(dem.shape[0]) + 0.5) * pixel_deg
    dy = pixel_deg * 110_574.0                                        # metres per pixel north-south
    dx = pixel_deg * 111_320.0 * np.cos(np.radians(pixel_lat))[:, None]  # and east-west
    grad_y, grad_x = np.gradient(dem)                                 # NaN-aware: edges become NaN
    slope = np.degrees(np.arctan(np.hypot(grad_x / dx, grad_y / dy)))

    blocks = per_cell(dem, spec, k)
    pct = lambda q: nan_stat(lambda a, axis: np.nanpercentile(a, q, axis=axis), blocks)
    out = {"elev_mean": nan_stat(np.nanmean, blocks),
           "elev_sd": nan_stat(np.nanstd, blocks),
           "elev_range": pct(95) - pct(5),
           "slope": nan_stat(np.nanmean, per_cell(slope, spec, k))}
    return pd.DataFrame({name: to_cells(v, spec, cells) for name, v in out.items()}, index=cells.index)


def land_cover(spec: GridSpec, cells: pd.DataFrame, land_polygons) -> pd.DataFrame:
    """Share of each land-cover class, Shannon diversity of the class mix, and the mean
    distance (km) from the cell's land to the nearest inland water body."""
    k = 75
    lats, lons = _tile_range(spec, 3)
    urls = [WORLDCOVER_URL.format(lat=la, lon=lo) for la in lats for lo in lons]
    classes = read_mosaic(urls, spec, k, nodata=0)
    land = land_mask(spec, k, land_polygons)
    valid = land & (classes > 0)

    blocks = per_cell(classes, spec, k)
    valid_blocks = per_cell(valid, spec, k)
    n_valid = valid_blocks.sum(axis=-1)
    out = {}
    with np.errstate(invalid="ignore", divide="ignore"):
        for code, name in LAND_COVER.items():
            out[f"lc_{name}"] = ((blocks == code) & valid_blocks).sum(axis=-1) / n_valid
        shares = np.stack(list(out.values()), axis=-1)
        out["lc_shannon"] = -np.nansum(np.where(shares > 0, shares * np.log(shares), 0.0), axis=-1)
    out["lc_shannon"][n_valid == 0] = np.nan

    # Distance to inland water (sea is excluded by the land mask). Pixel size in km at the
    # region's mid-latitude; the east-west error from varying latitude stays below 3%.
    pixel_deg = spec.cell / k
    mid_lat = np.radians((spec.north + spec.south) / 2)
    sampling = (pixel_deg * 110.574, pixel_deg * 111.320 * np.cos(mid_lat))
    inland_water = land & (classes == WATER_CLASS)
    distance = distance_transform_edt(~inland_water, sampling=sampling).astype("float32")
    distance[~land] = np.nan
    out["dist_water_km"] = nan_stat(np.nanmean, per_cell(distance, spec, k))
    return pd.DataFrame({name: to_cells(v, spec, cells) for name, v in out.items()}, index=cells.index)


def vegetation(spec: GridSpec, cells: pd.DataFrame, land_polygons) -> pd.DataFrame:
    """Median NDVI (greenness) and NDVI seasonal range (90th minus 10th percentile), 2021."""
    k = 75
    lats, lons = _tile_range(spec, 1)
    urls = [NDVI_URL.format(lat=la, lon=lo) for la in lats for lo in lons]
    nodata = 255
    p90, p50, p10 = read_mosaic(urls, spec, k, nodata=nodata, band=(1, 2, 3)).astype("float32")
    land = land_mask(spec, k, land_polygons)
    median = np.where(land & (p50 != nodata), p50 * 0.008 - 1.0, np.nan)          # DN -> NDVI
    seasonal = np.where(land & (p90 != nodata) & (p10 != nodata), (p90 - p10) * 0.008, np.nan)
    out = {"ndvi_median": to_cells(nan_stat(np.nanmean, per_cell(median, spec, k)), spec, cells),
           "ndvi_range": to_cells(nan_stat(np.nanmean, per_cell(seasonal, spec, k)), spec, cells)}
    return pd.DataFrame(out, index=cells.index)
