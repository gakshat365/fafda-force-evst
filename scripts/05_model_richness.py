"""Step 5: model the two community EBVs (species richness, threatened-species richness).

For each target:
1. drop near-duplicate predictors (|r| >= 0.8)
2. nested cross-validation of 4 models under 3 schemes (random, spatial blocks, whole states)
3. explain the best model (permutation importance, SHAP, response curves)
4. map predictions and uncertainty for every cell, including under-sampled ones
"""
import json

import numpy as np
import pandas as pd

from ebv import config
from ebv.features import CANDIDATES, LABELS, select_uncorrelated
from ebv.modelling import (cross_validate, cv_permutation_importance, fit_final, outside_training_range,
                           predict, response_curves, shap_values, tree_spread)
from ebv.models import regression_models
from ebv.validation import SCHEMES, regression_scores, state_groups

TARGETS = {"richness": "Bird species richness", "threatened": "Threatened-species richness"}

table = pd.read_csv(config.PROCESSED / "model_table.csv")
train = table[table.has_richness].reset_index(drop=True)
features, dropped = select_uncorrelated(train)
dropped.to_csv(config.TABLES / "dropped_predictors.csv", index=False)
(config.TABLES / "predictors.json").write_text(json.dumps(features, indent=1))
(config.TABLES / "predictor_labels.json").write_text(json.dumps({n: LABELS[n] for n in CANDIDATES}, indent=1))
print(f"{len(train):,} training cells; {len(features)} predictors kept, dropped:\n{dropped.to_string(index=False)}")

X, X_all = train[features], table[features]
specs = {spec.name: spec for spec in regression_models()}

for target, title in TARGETS.items():
    y = train[target].values
    print(f"\n=== {title} ===", flush=True)
    rows, by_state, results = [], [], {}
    for scheme in SCHEMES:
        for spec in specs.values():
            result = cross_validate(spec, X, y, train, scheme, keep_models=scheme == "spatial")
            results[(scheme, spec.name)] = result
            fold_r2 = [f["R2"] for f in result.folds]
            rows.append({"scheme": scheme, "model": spec.name, **result.scores(),
                         "fold_R2_sd": np.std(fold_r2)})
            print(f"  {scheme:<8} {spec.name:<17} R2={rows[-1]['R2']:.3f}", flush=True)
            if scheme == "state":
                groups = state_groups(train)
                for (_, test), fold in zip(result.splits, result.folds):
                    by_state.append({"model": spec.name, "held_out_state": groups[test][0],
                                     "n_cells": len(test), "R2": fold["R2"], "MAE": fold["MAE"]})
    cv = pd.DataFrame(rows)
    cv.to_csv(config.TABLES / f"{target}_cv.csv", index=False)
    pd.DataFrame(by_state).to_csv(config.TABLES / f"{target}_state_transfer.csv", index=False)

    spatial = cv[cv.scheme == "spatial"].sort_values("R2", ascending=False)
    best_name = spatial.model.iloc[0]
    best = results[("spatial", best_name)]

    # Spatial-CV accuracy within each state (the mid-evaluation only covered Kerala)
    within = pd.DataFrame([{"state": state, "n_cells": len(idx),
                            **regression_scores(y[idx], best.predictions[idx])}
                           for state, idx in train.groupby("state").indices.items() if len(idx) >= 20])
    within.to_csv(config.TABLES / f"{target}_spatial_by_state.csv", index=False)
    pd.DataFrame({"cell_id": train.cell_id, "observed": y, "predicted": best.predictions}).to_csv(
        config.TABLES / f"{target}_oof.csv", index=False)

    importance = cv_permutation_importance(best, X, LABELS)
    importance.to_csv(config.TABLES / f"{target}_importance.csv", index=False)

    # Final models on all training cells: the best for maps, a Random Forest for uncertainty
    model, params = fit_final(specs[best_name], X, y, train)
    forest, _ = fit_final(specs["Random Forest"], X, y, train)
    explainer_model = model if specs[best_name].tree_based else forest
    values, sample = shap_values(explainer_model, X)
    np.savez_compressed(config.TABLES / f"{target}_shap.npz", values=values, X=sample.values,
                        columns=np.array(features))
    curves = response_curves(explainer_model, X, importance.feature.head(4).tolist())
    (config.TABLES / f"{target}_curves.json").write_text(json.dumps(curves))

    predictions = table[["cell_id", "lon", "lat", "lon0", "lat0", "state", "has_richness"]].copy()
    predictions["observed"] = table[target]
    predictions["predicted"] = predict(model, X_all, False)
    predictions["uncertainty"] = tree_spread(forest, X_all)
    predictions["extrapolated"] = outside_training_range(X, X_all)
    predictions.to_csv(config.TABLES / f"{target}_predictions.csv", index=False)

    summary = {"target": target, "title": title, "best_model": best_name, "best_params": params,
               "n_train": len(train), "n_cells": len(table), "features": features,
               "extrapolated_cells": int(predictions.extrapolated.sum()),
               "train_R2_best": regression_scores(y, predict(model, X, False))["R2"]}
    (config.TABLES / f"{target}_summary.json").write_text(json.dumps(summary, indent=1, default=str))
    print(f"  best: {best_name} | spatial R2 {spatial.R2.iloc[0]:.3f}", flush=True)
