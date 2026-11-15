"""Step 6: EBV 2 - occurrence probability of focal (threatened / endemic) birds.

Presence = species recorded in a cell (any effort).
Absence  = well-surveyed cell (>= 1000 records) where the species was never recorded
           ("inferred absence" given high observation effort).
Models: Logistic regression (classic SDM baseline), Random Forest, LightGBM.
Evaluation: spatial-block CV, AUC and TSS (max over thresholds).
"""
import json
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import roc_auc_score, roc_curve
from config import PROC, RES
from common import select_features, splitters, classifiers

tab = pd.read_csv(PROC / "model_table.csv")
pa = pd.read_csv(PROC / "presence.csv.gz", index_col="cell_id")
sp = pd.read_csv(PROC / "species.csv")
sp["speciesKey"] = sp.speciesKey.astype(str)

FOCAL = {  # scientific name -> (common name, habitat note); IUCN status from GBIF
    "Ocyceros griseus": ("Malabar Grey Hornbill", "Forest, W. Ghats endemic"),
    "Buceros bicornis": ("Great Hornbill", "Large evergreen forest"),
    "Trochalopteron fairbanki": ("Kerala Laughingthrush", "Montane shola, endemic"),
    "Sterna aurantia": ("River Tern", "Rivers and wetlands"),
}
ABSENCE_MIN_RECORDS = 1000  # a cell counts as "absence" only if this well surveyed

feats = select_features(tab[tab.well_sampled])


def tss(y, p):
    fpr, tpr, thr = roc_curve(y, p)
    i = np.argmax(tpr - fpr)
    return tpr[i] - fpr[i], thr[i]


rows, maps = [], {}
for sci, (common, habitat) in FOCAL.items():
    key = sp.loc[sp.scientificName == sci, "speciesKey"]
    if key.empty or key.iloc[0] not in pa.columns:
        print("missing", sci)
        continue
    key = key.iloc[0]
    iucn = sp.loc[sp.speciesKey == key, "iucn"].iloc[0]
    pres = tab.cell_id.map(pa[key]).fillna(0).astype(int)
    train = tab[(pres == 1) | (tab.n_records >= ABSENCE_MIN_RECORDS)].reset_index(drop=True)
    X = train[feats]
    folds = splitters(train)["Spatial block CV"]
    y = train.cell_id.map(pa[key]).fillna(0).astype(int).values
    best = (None, -1)
    for m_name, model in classifiers().items():
        p = np.zeros(len(y))
        for tr, te in folds:
            p[te] = clone(model).fit(X.iloc[tr], y[tr]).predict_proba(X.iloc[te])[:, 1]
        auc = roc_auc_score(y, p)
        t, _ = tss(y, p)
        rows.append(dict(species=common, scientific=sci, iucn=iucn, habitat=habitat,
                         presences=int(y.sum()), absences=int((1 - y).sum()),
                         model=m_name, AUC=auc, TSS=t))
        print(rows[-1])
        if auc > best[1]:
            best = (m_name, auc)
    final = clone(classifiers()[best[0]]).fit(X, y)
    maps[common] = dict(model=best[0], prob=final.predict_proba(tab[feats])[:, 1].tolist(),
                        presence=tab.cell_id.map(pa[key]).fillna(0).astype(int).tolist())

pd.DataFrame(rows).to_csv(RES / "occurrence_cv.csv", index=False)
out = tab[["cell_id", "lon", "lat", "lon0", "lat0", "well_sampled"]].copy()
for k, v in maps.items():
    out[f"p_{k}"] = v["prob"]
    out[f"obs_{k}"] = v["presence"]
out.to_csv(RES / "occurrence_predictions.csv", index=False)
json.dump({k: v["model"] for k, v in maps.items()}, open(RES / "occurrence_best.json", "w"))
