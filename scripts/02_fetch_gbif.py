"""Step 2: bird species and record counts for every grid cell, plus species names and IUCN status.

Resumable: each cell is appended to a JSON-lines file as soon as it is fetched, so the
script can be stopped and restarted without repeating finished cells.
"""
import json

import pandas as pd

from ebv import config
from ebv.gbif import GbifClient

cells_path = config.RAW / "gbif_cells.jsonl"
species_path = config.PROCESSED / "species.csv"

grid = pd.read_csv(config.PROCESSED / "grid.csv")
client = GbifClient()

done = set()
if cells_path.exists():
    done = {json.loads(line)["cell_id"] for line in cells_path.read_text().splitlines()}
todo = grid[~grid.cell_id.isin(done)]
print(f"{len(done):,} cells cached, {len(todo):,} to fetch", flush=True)

with cells_path.open("a") as out:
    for i, cell in enumerate(todo.itertuples(), 1):
        result = client.species_counts(cell.lon0, cell.lat0, cell.lon0 + config.CELL_DEG,
                                       cell.lat0 + config.CELL_DEG)
        out.write(json.dumps({"cell_id": cell.cell_id, **result}) + "\n")
        out.flush()
        if i % 250 == 0:
            print(f"  {i:,}/{len(todo):,} cells", flush=True)

# Species names and IUCN Red List categories (two small requests per species)
records = [json.loads(line) for line in cells_path.read_text().splitlines()]
keys = sorted({k for r in records for k in r["species"]}, key=int)
species = pd.DataFrame([client.species_info(k) for k in keys])
species.to_csv(species_path, index=False)
print(f"{len(keys)} species; IUCN categories:\n{species.iucn.value_counts().to_string()}")
