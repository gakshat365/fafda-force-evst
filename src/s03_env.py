"""Step 3: environmental predictors per grid cell.

- WorldClim 2.1 bioclim (30 arc-sec)      -> 19 climate variables
- Copernicus DEM GLO-90 (read remotely)   -> mean / sd elevation, mean slope
- ESA WorldCover 2021 10 m (read remotely,
  ~75 m overview level)                   -> land-cover fractions + Shannon diversity
"""
import numpy as np
import pandas as pd
import rasterio
from rasterio.merge import merge
from config import RAW, PROC, CELL

W, E, S, N = 74.85, 77.45, 8.25, 12.85
grid = pd.read_csv(PROC / "grid.csv")


def cell_blocks(arr, res):
    """Yield (cell_id, pixel block) for every grid cell from a mosaic aligned to W/N."""
    k = int(round(CELL / res))
    for r in grid.itertuples():
        c0 = int(round((r.lon0 - W) / res))
        r0 = int(round((N - (r.lat0 + CELL)) / res))
        yield r.cell_id, arr[..., r0:r0 + k, c0:c0 + k]


def mosaic(urls, res, nodata):
    srcs = []
    for u in urls:
        try:
            srcs.append(rasterio.open(u))
        except Exception:
            pass  # ocean tiles do not exist
    arr, _ = merge(srcs, bounds=(W, S, E, N), res=res, nodata=nodata)
    for s in srcs:
        s.close()
    return arr[0]


out = {}

# ---- WorldClim -------------------------------------------------------------
print("WorldClim")
with rasterio.open(RAW / "wc2.1_30s_bio_tile33.tif") as src:
    win = rasterio.windows.from_bounds(W, S, E, N, src.transform)
    bio = src.read(window=win).astype("float32")
    bio[bio < -1e30] = np.nan
    res = src.res[0]
for cid, b in cell_blocks(bio, res):
    out.setdefault(cid, {}).update({f"bio{i+1}": np.nanmean(b[i]) for i in range(19)})

# ---- Copernicus DEM ----------------------------------------------------------
print("Copernicus DEM")
tiles = [f"/vsicurl/https://copernicus-dem-90m.s3.amazonaws.com/"
         f"Copernicus_DSM_COG_30_N{la:02d}_00_E{lo:03d}_00_DEM/"
         f"Copernicus_DSM_COG_30_N{la:02d}_00_E{lo:03d}_00_DEM.tif"
         for la in range(8, 13) for lo in range(74, 78)]
res = CELL / 60
dem = mosaic(tiles, res, nodata=np.nan).astype("float32")
dem[dem <= -100] = np.nan
lat = N - (np.arange(dem.shape[0]) + 0.5) * res
dy = res * 110574.0
dx = res * 111320.0 * np.cos(np.deg2rad(lat))[:, None]
gy, gx = np.gradient(np.nan_to_num(dem))
slope = np.degrees(np.arctan(np.hypot(gx / dx, gy / dy)))
slope[np.isnan(dem)] = np.nan
for (cid, d), (_, s) in zip(cell_blocks(dem, res), cell_blocks(slope, res)):
    out[cid].update(elev_mean=np.nanmean(d), elev_sd=np.nanstd(d),
                    elev_range=np.nanpercentile(d, 95) - np.nanpercentile(d, 5),
                    slope_mean=np.nanmean(s))

# ---- ESA WorldCover ---------------------------------------------------------
print("ESA WorldCover")
classes = {10: "tree", 20: "shrub", 30: "grass", 40: "crop", 50: "built",
           60: "bare", 80: "water", 90: "wetland", 95: "mangrove"}
tiles = [f"/vsicurl/https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/map/"
         f"ESA_WorldCover_10m_2021_v200_N{la:02d}E{lo:03d}_Map.tif"
         for la in (6, 9, 12) for lo in (72, 75)]
res = CELL / 75
lc = mosaic(tiles, res, nodata=0)
for cid, b in cell_blocks(lc, res):
    v = b[b > 0]
    fr = {f"lc_{n}": (v == k).mean() if v.size else np.nan for k, n in classes.items()}
    p = np.array([x for x in fr.values() if x > 0])
    fr["lc_shannon"] = float(-(p * np.log(p)).sum()) if p.size else np.nan
    out[cid].update(fr)

env = pd.DataFrame.from_dict(out, orient="index").rename_axis("cell_id").reset_index()
env.to_csv(PROC / "env.csv", index=False)
print(env.describe().T.round(2).to_string())
