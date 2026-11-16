"""Step 4: biodiversity targets for every cell, the presence matrix, and the modelling table.

Targets (EBVs), defined for cells with at least RAREFACTION_SIZE records
- richness:            expected species in a random sample of RAREFACTION_SIZE records
- threatened:          expected Near-Threatened-or-worse species in that same sample
- richness_coverage:   expected species at TARGET_COVERAGE sample coverage; the alternative
                       standardisation we tested (compared in effort_bias.csv, not modelled)
"""
import json

import numpy as np
import pandas as pd

from ebv import config
from ebv.diversity import estimated_coverage, rarefied_richness, sample_size_for_coverage

grid = pd.read_csv(config.PROCESSED / "grid.csv")
environment = pd.read_csv(config.PROCESSED / "environment.csv")
species = pd.read_csv(config.PROCESSED / "species.csv", dtype={"species_key": str})
threatened = set(species.loc[species.iucn.isin(config.THREATENED_OR_NT), "species_key"])
cells = {r["cell_id"]: r for r in map(json.loads, (config.RAW / "gbif_cells.jsonl").read_text().splitlines())}

rows, species_by_cell = [], {}
for cell_id in grid.cell_id:
    record = cells[cell_id]
    keys = list(record["species"])
    counts = np.array([record["species"][k] for k in keys], float)
    is_threatened = np.array([k in threatened for k in keys], bool)
    n = int(counts.sum())
    enough = n >= config.RAREFACTION_SIZE
    m = sample_size_for_coverage(counts, config.TARGET_COVERAGE) if enough else None
    rows.append({
        "cell_id": cell_id,
        "n_records": record["n_records"],                  # all bird records = sampling effort
        "n_species_records": n,                            # records identified to species
        "observed_species": len(keys),
        "observed_threatened": int(is_threatened.sum()),
        "coverage": estimated_coverage(counts) if n else np.nan,
        "richness": rarefied_richness(counts, config.RAREFACTION_SIZE) if enough else np.nan,
        "threatened": rarefied_richness(counts, config.RAREFACTION_SIZE, is_threatened) if enough else np.nan,
        "richness_coverage": rarefied_richness(counts, m) if m else np.nan,
    })
    species_by_cell[cell_id] = dict.fromkeys(keys, 1)

config.TABLES.mkdir(parents=True, exist_ok=True)
targets = pd.DataFrame(rows)
table = grid.merge(environment, on="cell_id").merge(targets, on="cell_id")
table["log_records"] = np.log10(table.n_records.clip(lower=1))
table["has_richness"] = table.richness.notna()
table.to_csv(config.PROCESSED / "model_table.csv", index=False)

presence = (pd.DataFrame.from_dict(species_by_cell, orient="index")
            .reindex(grid.cell_id).fillna(0).astype("int8"))      # every cell, even those with no records
presence.index.name = "cell_id"
presence.to_csv(config.PROCESSED / "presence.csv.gz")

# How strongly does each richness measure still follow sampling effort? (same cells for all)
measures = {"observed_species": "Observed species",
            "richness_coverage": f"Species at {config.TARGET_COVERAGE:.0%} coverage (tested)",
            "richness": f"Species in {config.RAREFACTION_SIZE} records (used)"}
common = table.dropna(subset=list(measures))
bias = pd.DataFrame([{"measure": label, "column": col, "cells_available": int(table[col].notna().sum()),
                      "r_with_log_effort": common[col].corr(common.log_records),
                      "r_on_own_cells": table[col].corr(table.log_records)}
                     for col, label in measures.items()])
bias.to_csv(config.TABLES / "effort_bias.csv", index=False)

# Region summary: the whole study area and each state (used to compare Kerala with the region)
summary = []
for area, part in [("All five states", table)] + [(name, table[table.state == name]) for name in config.STATES]:
    seen = presence.loc[part.cell_id].sum() > 0
    summary.append({"area": area, "cells": len(part), "records": int(part.n_records.sum()),
                    "species": int(seen.sum()), "threatened_species": int(seen[seen.index.isin(threatened)].sum()),
                    "cells_with_richness": int(part.has_richness.sum()),
                    "elevation_min": part.elev_mean.min(), "elevation_max": part.elev_mean.max(),
                    "rainfall_min": part.bio12.min(), "rainfall_max": part.bio12.max(),
                    "temperature_min": part.bio1.min(), "temperature_max": part.bio1.max()})
pd.DataFrame(summary).to_csv(config.TABLES / "region_summary.csv", index=False)

print(f"cells: {len(table):,} | records: {table.n_records.sum():,} | species: {presence.shape[1]}")
print(f"cells with at least {config.RAREFACTION_SIZE} records (modelled): {table.has_richness.sum():,}")
print(f"Near-Threatened-or-worse species: {len(threatened & set(presence.columns))}")
print(table[["n_records", "coverage", "observed_species", "richness", "threatened",
             "richness_coverage"]].describe().round(2).to_string())
print(bias.round(3).to_string(index=False))
