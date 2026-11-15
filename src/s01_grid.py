"""Step 1: build a 0.05 degree grid over Kerala."""
import json
import numpy as np
import pandas as pd
from shapely.geometry import shape, box
from shapely.prepared import prep
from config import KERALA_GEOJSON, CELL, MIN_LAND_FRAC, PROC

if not KERALA_GEOJSON.exists():  # Kerala polygon from geoBoundaries India ADM1
    import requests
    url = ("https://github.com/wmgeolab/geoBoundaries/raw/9469f09/releaseData/"
           "gbOpen/IND/ADM1/geoBoundaries-IND-ADM1.geojson")
    feats = requests.get(url, timeout=300).json()["features"]
    k = [f for f in feats if f["properties"]["shapeName"].lower() == "kerala"]
    json.dump({"type": "FeatureCollection", "features": k}, open(KERALA_GEOJSON, "w"))

geom = shape(json.load(open(KERALA_GEOJSON))["features"][0]["geometry"])
pg = prep(geom)
minx, miny, maxx, maxy = geom.bounds
x0, y0 = np.floor(minx / CELL) * CELL, np.floor(miny / CELL) * CELL

rows = []
for x in np.arange(x0, maxx, CELL):
    for y in np.arange(y0, maxy, CELL):
        x, y = round(x, 4), round(y, 4)
        c = box(x, y, x + CELL, y + CELL)
        if not pg.intersects(c):
            continue
        frac = geom.intersection(c).area / c.area
        if frac >= MIN_LAND_FRAC:
            rows.append(dict(cell_id=f"{x:.2f}_{y:.2f}", lon0=x, lat0=y,
                             lon=x + CELL / 2, lat=y + CELL / 2, land_frac=frac))

grid = pd.DataFrame(rows)
grid.to_csv(PROC / "grid.csv", index=False)
print(f"{len(grid)} grid cells")
