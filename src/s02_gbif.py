"""Step 2: per-cell bird species lists from the GBIF occurrence API.

For every grid cell we ask GBIF for a facet (count of records) on speciesKey.
This gives, per cell: total records (sampling effort) and records per species.
"""
import json
import time
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
import requests
from config import GBIF_API, AVES_CLASS_KEY, YEARS, CELL, PROC, RAW

OUT = RAW / "gbif_cell_species.json"
EPS = 1e-6


def query(row):
    params = dict(
        classKey=AVES_CLASS_KEY,
        decimalLatitude=f"{row.lat0},{row.lat0 + CELL - EPS}",
        decimalLongitude=f"{row.lon0},{row.lon0 + CELL - EPS}",
        hasCoordinate="true", hasGeospatialIssue="false",
        occurrenceStatus="PRESENT", basisOfRecord="HUMAN_OBSERVATION",
        year=YEARS, limit=0, facet="speciesKey", facetLimit=3000,
    )
    for attempt in range(6):
        try:
            r = requests.get(f"{GBIF_API}/occurrence/search", params=params, timeout=60)
            r.raise_for_status()
            d = r.json()
            counts = d["facets"][0]["counts"] if d["facets"] else []
            return row.cell_id, {"n_records": d["count"],
                                 "species": {c["name"]: c["count"] for c in counts}}
        except Exception as e:  # retry with back-off
            time.sleep(2 ** attempt)
    raise RuntimeError(f"failed {row.cell_id}")


def species_info(key):
    for attempt in range(5):
        try:
            s = requests.get(f"{GBIF_API}/species/{key}", timeout=30).json()
            r = requests.get(f"{GBIF_API}/species/{key}/iucnRedListCategory", timeout=30)
            iucn = r.json().get("category") if r.status_code == 200 and r.text else None
            return dict(speciesKey=key, scientificName=s.get("canonicalName", s.get("scientificName")),
                        vernacularName=s.get("vernacularName"), family=s.get("family"),
                        order=s.get("order"), iucn=iucn)
        except Exception:
            time.sleep(2 ** attempt)
    return dict(speciesKey=key)


if __name__ == "__main__":
    grid = pd.read_csv(PROC / "grid.csv")
    done = json.load(open(OUT)) if OUT.exists() else {}
    todo = [r for r in grid.itertuples() if r.cell_id not in done]
    print(f"{len(done)} cached, {len(todo)} to query")
    with ThreadPoolExecutor(8) as ex:
        for i, (cid, res) in enumerate(ex.map(query, todo)):
            done[cid] = res
            if i % 100 == 0:
                print(i, cid, res["n_records"], len(res["species"]), flush=True)
                json.dump(done, open(OUT, "w"))
    json.dump(done, open(OUT, "w"))

    keys = sorted({k for v in done.values() for k in v["species"]}, key=int)
    print(f"{len(keys)} species; fetching names + IUCN categories")
    with ThreadPoolExecutor(8) as ex:
        info = list(ex.map(species_info, keys))
    pd.DataFrame(info).to_csv(PROC / "species.csv", index=False)
    print("done")
