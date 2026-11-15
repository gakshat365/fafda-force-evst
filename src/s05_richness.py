"""Step 5: EBV 1 - bird species richness from environment.

Target: rarefied richness (species expected in 200 records) in well-sampled cells.
Compares 4 models under random and spatial-block cross-validation, explains the
best model (permutation importance, SHAP) and predicts richness for all cells.
"""
import json
import numpy as np
import pandas as pd
import shap
from sklearn.base import clone
from sklearn.inspection import permutation_importance, partial_dependence
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from config import PROC, RES, SEED
from common import select_features, splitters, regressors, LABELS

tab = pd.read_csv(PROC / "model_table.csv")
train = tab[tab.well_sampled].reset_index(drop=True)
feats = select_features(train)
X, y = train[feats], train.S_rare
print(f"{len(train)} training cells, {len(feats)} predictors: {feats}")

# --- Effort bias check -------------------------------------------------------
bias = {
    "r_Sobs_vs_logeffort_all": tab[["S_obs", "log_records"]].corr().iloc[0, 1],
    "r_Srare_vs_logeffort_train": train[["S_rare", "log_records"]].corr().iloc[0, 1],
    "r_Sobs_vs_logeffort_train": train[["S_obs", "log_records"]].corr().iloc[0, 1],
}

# --- Cross-validated model comparison ---------------------------------------
rows, oof = [], {}
for cv_name, folds in splitters(train).items():
    for m_name, model in regressors().items():
        pred = np.zeros(len(y))
        fold_r2 = []
        for tr, te in folds:
            m = clone(model).fit(X.iloc[tr], y.iloc[tr])
            pred[te] = m.predict(X.iloc[te])
            fold_r2.append(r2_score(y.iloc[te], pred[te]))
        rows.append(dict(cv=cv_name, model=m_name, R2=r2_score(y, pred),
                         R2_fold_sd=np.std(fold_r2),
                         RMSE=mean_squared_error(y, pred) ** 0.5,
                         MAE=mean_absolute_error(y, pred)))
        oof[(cv_name, m_name)] = pred
        print(rows[-1])
cvres = pd.DataFrame(rows)
cvres.to_csv(RES / "richness_cv.csv", index=False)

spatial = cvres[cvres.cv == "Spatial block CV"]
best = spatial.sort_values("R2", ascending=False).model.iloc[0]
print("best model (spatial CV):", best)
train["pred_oof"] = oof[("Spatial block CV", best)]
train[["cell_id", "lon", "lat", "S_rare", "pred_oof"]].to_csv(RES / "richness_oof.csv", index=False)

# --- Permutation importance on spatial held-out folds (best model) ----------
imp = []
for tr, te in splitters(train)["Spatial block CV"]:
    m = clone(regressors()[best]).fit(X.iloc[tr], y.iloc[tr])
    pi = permutation_importance(m, X.iloc[te], y.iloc[te], n_repeats=20,
                                random_state=SEED, scoring="r2")
    imp.append(pi.importances_mean)
imp = pd.DataFrame({"feature": feats, "label": [LABELS[f] for f in feats],
                    "importance": np.mean(imp, 0), "sd": np.std(imp, 0)})
imp = imp.sort_values("importance", ascending=False)
imp.to_csv(RES / "richness_importance.csv", index=False)
print(imp.to_string())

# --- Final fit, SHAP and partial dependence ---------------------------------
final = clone(regressors()[best]).fit(X, y)
tree_model = final if best in ("Random Forest", "LightGBM") else \
    clone(regressors()["LightGBM"]).fit(X, y)
sv = shap.TreeExplainer(tree_model).shap_values(X)
np.save(RES / "richness_shap.npy", sv)
X.to_csv(RES / "richness_X.csv", index=False)

pdp = {}
for f in imp.feature.head(6):
    r = partial_dependence(final, X, [f], grid_resolution=40)
    pdp[f] = dict(x=r["grid_values"][0].tolist(), y=r["average"][0].tolist())
json.dump(pdp, open(RES / "richness_pdp.json", "w"))

# --- Predict richness for every cell (fills under-sampled gaps) -------------
tab["S_pred"] = final.predict(tab[feats])
# Uncertainty: spread across random-forest trees
rf = clone(regressors()["Random Forest"]).fit(X, y)
tree_preds = np.stack([t.predict(tab[feats].values) for t in rf.estimators_])
tab["S_pred_sd"] = tree_preds.std(0)
# Extrapolation flag: any predictor outside the training range
lo, hi = X.min(), X.max()
tab["extrapolated"] = ((tab[feats] < lo) | (tab[feats] > hi)).any(axis=1)
tab[["cell_id", "lon", "lat", "lon0", "lat0", "n_records", "well_sampled",
     "S_obs", "S_rare", "S_pred", "S_pred_sd", "extrapolated"]].to_csv(
    RES / "richness_predictions.csv", index=False)

summary = dict(n_train=len(train), n_cells=len(tab), features=feats, best_model=best,
               under_sampled=int((~tab.well_sampled).sum()),
               extrapolated=int(tab.extrapolated.sum()), **bias)
json.dump(summary, open(RES / "richness_summary.json", "w"), indent=2)
print(json.dumps(summary, indent=2))
