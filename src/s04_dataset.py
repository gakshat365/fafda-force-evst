"""Step 4: build the modelling table (one row per grid cell).

EBV targets
- S_obs      : observed number of bird species (biased by sampling effort)
- S_rare     : rarefied richness = expected species in a random sample of
               MIN_RECORDS records (Hurlbert 1971) -> effort-standardised
- n_threat   : species in IUCN NT/VU/EN/CR categories
Per-species presence/absence matrix is also saved for occurrence models.
"""
import json
import numpy as np
import pandas as pd
from scipy.special import gammaln
from config import RAW, PROC, MIN_RECORDS

grid = pd.read_csv(PROC / "grid.csv")
env = pd.read_csv(PROC / "env.csv")
sp = pd.read_csv(PROC / "species.csv")
cells = json.load(open(RAW / "gbif_cell_species.json"))

threat = set(sp.loc[sp.iucn.isin(["NEAR_THREATENED", "VULNERABLE", "ENDANGERED",
                                  "CRITICALLY_ENDANGERED"]), "speciesKey"].astype(str))


def lchoose(n, k):
    return gammaln(n + 1) - gammaln(k + 1) - gammaln(n - k + 1)


def rarefy(counts, n):
    counts = np.asarray(counts, float)
    N = counts.sum()
    if N < n:
        return np.nan
    return float(np.sum(1 - np.exp(lchoose(N - counts, n) - lchoose(N, n))))


rows, pa = [], {}
for cid, d in cells.items():
    c = d["species"]
    rows.append(dict(cell_id=cid, n_records=d["n_records"], S_obs=len(c),
                     S_rare=rarefy(list(c.values()), MIN_RECORDS),
                     n_threat=sum(k in threat for k in c)))
    pa[cid] = {k: 1 for k in c}

tab = grid.merge(pd.DataFrame(rows), on="cell_id").merge(env, on="cell_id")
tab["log_records"] = np.log10(tab.n_records.clip(lower=1))
tab["well_sampled"] = tab.n_records >= MIN_RECORDS
tab.to_csv(PROC / "model_table.csv", index=False)

pa = pd.DataFrame.from_dict(pa, orient="index").fillna(0).astype("int8")
pa.index.name = "cell_id"
pa.to_csv(PROC / "presence.csv.gz")

print(f"cells: {len(tab)}, well sampled (>= {MIN_RECORDS} records): {tab.well_sampled.sum()}")
print(f"records: {tab.n_records.sum():,}, species: {pa.shape[1]}, threatened spp: {len(threat)}")
print(tab[["n_records", "S_obs", "S_rare", "n_threat"]].describe().round(1).to_string())
