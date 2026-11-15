"""Step 8: extra explanatory figures for the study guide / detailed deck."""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.base import clone
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.tree import DecisionTreeRegressor, plot_tree
from config import PROC, RES, FIG, RAW
from common import select_features, splitters, classifiers, LABELS, CANDIDATES
from scipy.special import gammaln


def rarefy(counts, n):
    """Hurlbert (1971) expected species in n records (same formula as s04_dataset.py)."""
    counts = np.asarray(counts, float)
    N = counts.sum()
    lc = lambda a, b: gammaln(a + 1) - gammaln(b + 1) - gammaln(a - b + 1)
    return float(np.sum(1 - np.exp(lc(N - counts, n) - lc(N, n)))) if N >= n else np.nan

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 14, "axes.titlesize": 16, "axes.titleweight": "bold",
    "axes.labelsize": 15, "axes.spines.top": False, "axes.spines.right": False,
    "savefig.dpi": 200, "savefig.bbox": "tight",
})
INK, GREEN, AMBER, GREY = "#1B2E25", "#2D6A4F", "#E07A1F", "#9AA59F"
tab = pd.read_csv(PROC / "model_table.csv")
cells = json.load(open(RAW / "gbif_cell_species.json"))

# 1. Rarefaction curves: Kochi city vs Thattekad bird sanctuary --------------------
fig, ax = plt.subplots(figsize=(8.5, 5.2))
for cid, name, col in [("76.70_10.10", "Thattekad (forest, bird sanctuary)", GREEN),
                       ("76.25_9.95", "Kochi city (mostly built-up)", AMBER)]:
    c = list(cells[cid]["species"].values())
    N = sum(c)
    ns = np.unique(np.geomspace(1, N, 80).astype(int))
    ax.plot(ns, [rarefy(c, n) for n in ns], color=col, lw=3,
            label=f"{name}: {N:,} records, {len(c)} species")
    ax.plot(200, rarefy(c, 200), "o", color=col, ms=10)
    ax.annotate(f"{rarefy(c, 200):.0f}", (200, rarefy(c, 200)), xytext=(8, -18 if col == AMBER else 6),
                textcoords="offset points", color=col, fontsize=15, fontweight="bold")
ax.axvline(200, color=GREY, ls="--", lw=1.5)
ax.set_xscale("log")
ax.set_xlabel("Number of records sampled (log scale)")
ax.set_ylabel("Expected number of species")
ax.set_title("Rarefaction: compare cells at the same effort (200 records)")
ax.legend(frameon=False, fontsize=12, loc="upper left")
fig.savefig(FIG / "teach_rarefaction.png"); plt.close(fig)

# 2. Random vs spatial-block CV folds -------------------------------------------------
train = tab[tab.well_sampled].reset_index(drop=True)
sp = splitters(train)
fig, axs = plt.subplots(1, 2, figsize=(9, 7))
cols = ["#2D6A4F", "#E07A1F", "#3B6E8F", "#B7472A", "#95D5B2"]
for ax, (name, folds) in zip(axs, sp.items()):
    fold = np.zeros(len(train), int)
    for k, (_, te) in enumerate(folds):
        fold[te] = k
    ax.scatter(train.lon, train.lat, c=[cols[f] for f in fold], s=9, marker="s", lw=0)
    ax.set_title(name); ax.set_aspect("equal"); ax.axis("off")
fig.suptitle("Colour = test fold. Spatial CV hides whole 28 km blocks.", fontsize=14, y=0.05)
fig.savefig(FIG / "teach_cv_folds.png"); plt.close(fig)

# 3. ROC curve: Great Hornbill, GLM vs LightGBM -------------------------------------
pa = pd.read_csv(PROC / "presence.csv.gz", index_col="cell_id")
spp = pd.read_csv(PROC / "species.csv"); spp["speciesKey"] = spp.speciesKey.astype(str)
key = spp.loc[spp.scientificName == "Buceros bicornis", "speciesKey"].iloc[0]
feats = select_features(tab[tab.well_sampled])
pres = tab.cell_id.map(pa[key]).fillna(0).astype(int)
tr = tab[(pres == 1) | (tab.n_records >= 1000)].reset_index(drop=True)
y = tr.cell_id.map(pa[key]).fillna(0).astype(int).values
fig, ax = plt.subplots(figsize=(6.5, 6))
for m, col in [("Logistic (GLM)", GREY), ("LightGBM", GREEN)]:
    p = np.zeros(len(y))
    for a, b in splitters(tr)["Spatial block CV"]:
        p[b] = clone(classifiers()[m]).fit(tr[feats].iloc[a], y[a]).predict_proba(tr[feats].iloc[b])[:, 1]
    fpr, tpr, _ = roc_curve(y, p)
    i = np.argmax(tpr - fpr)
    ax.plot(fpr, tpr, color=col, lw=3, label=f"{m}  (AUC = {roc_auc_score(y, p):.2f})")
    ax.plot([fpr[i], fpr[i]], [fpr[i], tpr[i]], color=col, ls=":", lw=2)
ax.plot([0, 1], [0, 1], color="black", lw=1, ls="--", label="Random guess (AUC = 0.50)")
ax.set_xlabel("False positive rate (1 − specificity)")
ax.set_ylabel("True positive rate (sensitivity)")
ax.set_title("ROC curve – Great Hornbill")
ax.legend(frameon=False, fontsize=12, loc="lower right")
ax.text(0.33, 0.52, "dotted = TSS\n(max gap above diagonal)", fontsize=11, color=INK)
fig.savefig(FIG / "teach_roc.png"); plt.close(fig)

# 4. A small decision tree ---------------------------------------------------------
X = train[feats].rename(columns=LABELS)
dt = DecisionTreeRegressor(max_depth=2, min_samples_leaf=50, random_state=0).fit(X, train.S_rare)
fig, ax = plt.subplots(figsize=(13, 5.5))
plot_tree(dt, feature_names=list(X.columns), filled=True, rounded=True, impurity=False,
          precision=1, fontsize=13, ax=ax)
ax.set_title("A depth-2 decision tree for richness (Random Forest averages 500 deeper trees)")
fig.savefig(FIG / "teach_tree.png"); plt.close(fig)

# 5. Permutation importance bar ------------------------------------------------------
imp = pd.read_csv(RES / "richness_importance.csv").iloc[::-1]
fig, ax = plt.subplots(figsize=(8, 6))
ax.barh(imp.label, imp.importance, xerr=imp.sd, color=GREEN, ecolor=GREY, capsize=3)
ax.set_xlabel("Drop in R² when the predictor is shuffled")
ax.set_title("Permutation importance (Random Forest, spatial CV)")
fig.savefig(FIG / "teach_importance.png"); plt.close(fig)

# 6. Correlation matrix of candidate predictors ----------------------------------------
c = tab[tab.well_sampled][CANDIDATES].corr()
fig, ax = plt.subplots(figsize=(9.5, 8))
im = ax.imshow(c, cmap="RdBu_r", vmin=-1, vmax=1)
lab = [LABELS[x] for x in CANDIDATES]
ax.set_xticks(range(len(lab)), lab, rotation=90, fontsize=11)
ax.set_yticks(range(len(lab)), lab, fontsize=11)
for i in range(len(lab)):
    for j in range(len(lab)):
        if abs(c.iloc[i, j]) >= 0.8 and i != j:
            ax.text(j, i, f"{c.iloc[i, j]:.2f}", ha="center", va="center", fontsize=8, color="white")
plt.colorbar(im, ax=ax, shrink=0.7, label="Pearson r")
ax.set_title("Predictor correlations (|r| ≥ 0.8 labelled)")
for s in ax.spines.values():
    s.set_visible(False)
fig.savefig(FIG / "teach_correlation.png"); plt.close(fig)
print("teaching figures written")
