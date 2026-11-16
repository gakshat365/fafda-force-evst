"""Study region (five South Indian states) and the regular latitude-longitude grid."""
import json
import unicodedata
from dataclasses import dataclass

import numpy as np
import pandas as pd
import requests
from shapely.geometry import box, shape
from shapely.ops import unary_union
from shapely.prepared import prep

from . import config


def _ascii(name: str) -> str:
    """'Tamil Nādu' -> 'Tamil Nadu' so state names can be matched reliably."""
    return unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()


def load_admin_boundaries() -> dict:
    """India's state polygons from geoBoundaries (downloaded once), keyed by ASCII name."""
    path = config.RAW / "india_adm1.geojson"
    if not path.exists():
        path.write_bytes(requests.get(config.ADMIN_URL, timeout=600).content)
    features = json.loads(path.read_text())["features"]
    return {_ascii(f["properties"]["shapeName"]): shape(f["geometry"]) for f in features}


def study_states(admin: dict) -> dict:
    """The five study states; Puducherry is clipped to drop its distant Yanam exclave."""
    states = {name: admin[name] for name in config.STATES}
    states["Puducherry"] = states["Puducherry"].intersection(box(60, 0, config.MAX_LONGITUDE, 40))
    return states


@dataclass(frozen=True)
class GridSpec:
    """A regular grid of square cells, anchored at its north-west corner."""
    west: float
    south: float
    east: float
    north: float
    cell: float = config.CELL_DEG

    @property
    def n_cols(self) -> int:
        return round((self.east - self.west) / self.cell)

    @property
    def n_rows(self) -> int:
        return round((self.north - self.south) / self.cell)

    @property
    def bounds(self) -> tuple:
        return self.west, self.south, self.east, self.north

    @classmethod
    def around(cls, geometry, cell: float = config.CELL_DEG) -> "GridSpec":
        """Smallest grid with edges on multiples of `cell` that covers `geometry`."""
        w, s, e, n = geometry.bounds
        snap_down = lambda v: round(np.floor(v / cell) * cell, 6)
        snap_up = lambda v: round(np.ceil(v / cell) * cell, 6)
        return cls(snap_down(w), snap_down(s), snap_up(e), snap_up(n), cell)

    @classmethod
    def from_cells(cls, cells: pd.DataFrame, cell: float = config.CELL_DEG) -> "GridSpec":
        return cls(cells.lon0.min(), cells.lat0.min(),
                   round(cells.lon0.max() + cell, 6), round(cells.lat0.max() + cell, 6), cell)

    def row_col(self, cells: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Row (counted from the north edge) and column of each cell."""
        rows = np.round((self.north - cells.lat0.values - self.cell) / self.cell).astype(int)
        cols = np.round((cells.lon0.values - self.west) / self.cell).astype(int)
        return rows, cols


def build_grid(states: dict) -> pd.DataFrame:
    """One row per grid cell with at least MIN_REGION_FRACTION of its area in the region.

    Each cell is labelled with the state that covers most of it.
    """
    region = unary_union(list(states.values()))
    spec = GridSpec.around(region)
    region_prepared = prep(region)
    rows = []
    for i in range(spec.n_cols):
        for j in range(spec.n_rows):
            lon0 = round(spec.west + i * spec.cell, 6)
            lat0 = round(spec.south + j * spec.cell, 6)
            square = box(lon0, lat0, lon0 + spec.cell, lat0 + spec.cell)
            if not region_prepared.intersects(square):
                continue
            share = {name: geom.intersection(square).area / square.area for name, geom in states.items()}
            fraction = sum(share.values())
            if fraction >= config.MIN_REGION_FRACTION:
                rows.append(dict(cell_id=f"{lon0:.2f}_{lat0:.2f}", lon0=lon0, lat0=lat0,
                                 lon=lon0 + spec.cell / 2, lat=lat0 + spec.cell / 2,
                                 region_fraction=round(fraction, 4),
                                 state=max(share, key=share.get)))
    return pd.DataFrame(rows).sort_values(["lat0", "lon0"], ignore_index=True)
