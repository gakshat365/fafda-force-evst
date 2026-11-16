"""Step 3: climate, terrain, land cover, water and vegetation predictors for every grid cell."""
import time

import pandas as pd
import requests
from shapely.geometry import box

from ebv import config
from ebv.environment import WORLDCLIM, WORLDCLIM_URL, climate, land_cover, terrain, vegetation
from ebv.region import GridSpec, load_admin_boundaries

grid = pd.read_csv(config.PROCESSED / "grid.csv")
spec = GridSpec.from_cells(grid)

if not WORLDCLIM.exists():
    WORLDCLIM.write_bytes(requests.get(WORLDCLIM_URL, timeout=1800).content)

# Land = every Indian state touching the grid, so neighbours' rivers and coasts are handled right
frame = box(*spec.bounds)
land = [geom.simplify(0.0005) for geom in load_admin_boundaries().values() if geom.intersects(frame)]

parts = [grid[["cell_id"]]]
for name, extract in [("climate", climate), ("terrain", terrain),
                      ("land cover", land_cover), ("vegetation", vegetation)]:
    start = time.time()
    parts.append(extract(spec, grid, land))
    print(f"{name:<11} {time.time() - start:6.0f} s", flush=True)

environment = pd.concat(parts, axis=1)
environment.to_csv(config.PROCESSED / "environment.csv", index=False)
print(f"{environment.shape[1] - 1} variables for {len(environment):,} cells")
print("cells with missing values:", int(environment.isna().any(axis=1).sum()))
print(environment.describe().T[["mean", "min", "max"]].round(2).to_string())
