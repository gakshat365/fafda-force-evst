"""Step 7: how well do protected areas cover the predicted bird hotspots?

Hotspots are the top HOTSPOT_SHARE of cells by predicted richness or predicted
threatened-species richness. We compare their protected share with the whole region
and count the unprotected hotspot cells ("gaps") in each state.
"""
import json

import numpy as np
import pandas as pd
from shapely.geometry import mapping
from shapely.ops import unary_union

from ebv import config
from ebv.protected import fetch_osm, protected_areas, protected_fraction
from ebv.region import GridSpec, load_admin_boundaries, study_states

HOTSPOT_SHARE = 0.10
WELL_PROTECTED = 0.25          # a cell counts as "protected" if at least a quarter of it is

grid = pd.read_csv(config.PROCESSED / "grid.csv")
osm = fetch_osm(GridSpec.from_cells(grid).bounds, config.RAW / "osm_protected_areas.json")
areas = protected_areas(osm)
grid["protected"] = protected_fraction(grid, areas.geometry)

region = unary_union(list(study_states(load_admin_boundaries()).values()))
features = [{"type": "Feature", "properties": {"name": name}, "geometry": mapping(geometry.simplify(0.003))}
            for name, geometry in zip(areas.name, areas.geometry)
            if geometry.intersects(region)]                                  # study-region parks only, simplified
(config.PROCESSED / "protected_areas.geojson").write_text(json.dumps({"type": "FeatureCollection",
                                                                       "features": features}))

rows, cells = [], grid[["cell_id", "state", "lon", "lat", "protected"]].copy()
for target, label in (("richness", "Species richness"), ("threatened", "Threatened-species richness")):
    predicted = pd.read_csv(config.TABLES / f"{target}_predictions.csv")[["cell_id", "predicted"]]
    merged = grid.merge(predicted, on="cell_id")
    hotspot = merged.predicted >= merged.predicted.quantile(1 - HOTSPOT_SHARE)
    cells[f"{target}_hotspot"] = hotspot.values
    rows.append({"layer": label, "hotspot_cells": int(hotspot.sum()),
                 "mean_protected_share_hotspots": merged.protected[hotspot].mean(),
                 "mean_protected_share_all": merged.protected.mean(),
                 "hotspots_well_protected": (merged.protected[hotspot] >= WELL_PROTECTED).mean(),
                 "all_cells_well_protected": (merged.protected >= WELL_PROTECTED).mean()})

summary = pd.DataFrame(rows)
summary.to_csv(config.TABLES / "protected_coverage.csv", index=False)
cells.to_csv(config.TABLES / "protected_cells.csv", index=False)

# Unprotected hotspot cells per state (both layers): where new protection would help most
gaps = cells[(cells.richness_hotspot | cells.threatened_hotspot) & (cells.protected < WELL_PROTECTED)]
gap_states = gaps.groupby("state").size().rename("unprotected_hotspot_cells").reset_index()
gap_states.to_csv(config.TABLES / "protection_gaps_by_state.csv", index=False)

print(f"{len(areas)} protected areas ({len(features)} in the study region); "
      f"{np.mean(grid.protected >= WELL_PROTECTED):.1%} of cells well protected")
print(summary.round(3).to_string(index=False))
print(gap_states.to_string(index=False))
