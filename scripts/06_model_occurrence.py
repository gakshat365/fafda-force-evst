"""Step 6: occurrence probability (species-distribution EBV) for the focal threatened birds.

Presence:          the species was recorded in the cell.
Inferred absence:  the cell has >= ABSENCE_MIN_RECORDS bird records but none of this species.
Four classifiers (linear GLM, quadratic GLM, Random Forest, LightGBM) are compared with
nested, spatially blocked and class-stratified cross-validation.
"""
import json

import pandas as pd

from ebv import config
from ebv.modelling import cross_validate, fit_final, predict
from ebv.models import classification_models

table = pd.read_csv(config.PROCESSED / "model_table.csv")
presence = pd.read_csv(config.PROCESSED / "presence.csv.gz", index_col="cell_id")
species = pd.read_csv(config.PROCESSED / "species.csv", dtype={"species_key": str}).set_index("scientific_name")
features = json.loads((config.TABLES / "predictors.json").read_text())
specs = classification_models()

rows, maps = [], table[["cell_id", "lon", "lat", "lon0", "lat0", "state"]].copy()
for scientific, (common, habitat) in config.FOCAL_SPECIES.items():
    key = species.loc[scientific, "species_key"]
    present = table.cell_id.map(presence[key]).fillna(0).astype(int).values
    usable = (present == 1) | (table.n_records >= config.ABSENCE_MIN_RECORDS).values
    data = table[usable].reset_index(drop=True)
    y = present[usable]
    if y.sum() < config.MIN_PRESENCE_CELLS:
        raise ValueError(f"{common} has only {y.sum()} presence cells (< {config.MIN_PRESENCE_CELLS})")
    X = data[features]

    scores = {}
    for spec in specs:
        result = cross_validate(spec, X, y, data, "spatial", classification=True)
        scores[spec.name] = result.scores()
        rows.append({"species": common, "scientific_name": scientific, "habitat": habitat,
                     "iucn": species.loc[scientific, "iucn"], "presences": int(y.sum()),
                     "absences": int((1 - y).sum()), "model": spec.name, **scores[spec.name]})
    best = max(scores, key=lambda name: scores[name]["AUC"])
    model, _ = fit_final(next(s for s in specs if s.name == best), X, y, data, classification=True)
    maps[f"p_{common}"] = predict(model, table[features], True)
    maps[f"present_{common}"] = present
    print(f"{common:<28} {int(y.sum()):>4} pres | best {best:<16} AUC {scores[best]['AUC']:.3f}",
          flush=True)

pd.DataFrame(rows).to_csv(config.TABLES / "occurrence_cv.csv", index=False)
maps.to_csv(config.TABLES / "occurrence_predictions.csv", index=False)
