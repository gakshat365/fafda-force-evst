"""Step 7: figures for the presentation (maps, scatter, SHAP, partial dependence)."""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import shap
from config import PROC, RES, FIG, KERALA_GEOJSON, CELL, MIN_RECORDS
from common import LABELS

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 15, "axes.titlesize": 18,
    "axes.titleweight": "bold", "axes.labelsize": 16, "xtick.labelsize": 13,
    "ytick.labelsize": 13, "axes.spines.top": False, "axes.spines.right": False,
    "savefig.dpi": 200, "savefig.bbox": "tight", "savefig.transparent": False,
})
INK, GREEN, AMBER, GREY = "#1B2E25", "#2D6A4F", "#E07A1F", "#B9C2BC"
GREENS = LinearSegmentedColormap.from_list("g", ["#F1F7EE", "#95D5B2", "#2D6A4F", "#081C15"])
W, E, S, N = 74.85, 77.45, 8.25, 12.85

outline = json.load(open(KERALA_GEOJSON))["features"][0]["geometry"]
polys = outline["coordinates"] if outline["type"] == "MultiPolygon" else [outline["coordinates"]]


def raster(df, col):
    nx, ny = round((E - W) / CELL), round((N - S) / CELL)
    a = np.full((ny, nx), np.nan)
    ix = np.round((df.lon0 - W) / CELL).astype(int)
    iy = np.round((N - df.lat0 - CELL) / CELL).astype(int)
    a[iy, ix] = df[col].values
    return a


def kmap(ax, df, col, title, cmap=GREENS, cbar=None, vmin=None, vmax=None, gaps=None):
    if gaps is not None:  # draw under-sampled cells in grey underneath
        g = df.assign(_g=1.0)
        ax.imshow(raster(g, "_g"), extent=(W, E, S, N), cmap=LinearSegmentedColormap.from_list(
            "grey", [GREY, GREY]), interpolation="nearest")
    im = ax.imshow(raster(df, col), extent=(W, E, S, N), cmap=cmap, vmin=vmin, vmax=vmax,
                   interpolation="nearest")
    for p in polys:
        r = np.array(p[0])
        ax.plot(r[:, 0], r[:, 1], color=INK, lw=0.7)
    ax.set_xlim(74.8, 77.5); ax.set_ylim(8.2, 12.9)
    ax.set_title(title, pad=10)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    if cbar:
        cb = plt.colorbar(im, ax=ax, shrink=0.55, pad=0.0, aspect=18)
        cb.set_label(cbar, fontsize=14); cb.outline.set_visible(False)
    return im


tab = pd.read_csv(PROC / "model_table.csv")
rp = pd.read_csv(RES / "richness_predictions.csv")

# 1. Study area: elevation -----------------------------------------------------
fig, ax = plt.subplots(figsize=(5.2, 7.5))
ELEV = LinearSegmentedColormap.from_list("elev", ["#CFE8C4", "#7FB069", "#2D6A4F", "#8C6D46", "#EDE6DA"])
kmap(ax, tab, "elev_mean", "Kerala: 1,282 grid cells", cmap=ELEV, vmin=0, vmax=2000,
     cbar="Elevation (m)")
ax.annotate("Western\nGhats", xy=(76.15, 11.7), xytext=(76.35, 12.25), fontsize=15, color=INK, fontweight="bold", arrowprops=dict(arrowstyle="->", color=INK, lw=1.5))
ax.annotate("Arabian\nSea", (75.0, 9.3), fontsize=15, color="#3B6E8F", style="italic")
fig.savefig(FIG / "study_area.png")

# 2. Sampling effort ------------------------------------------------------------
fig, axs = plt.subplots(1, 2, figsize=(12, 6.8), gridspec_kw=dict(width_ratios=[1, 1.25]))
kmap(axs[0], tab, "log_records", "GBIF records per cell", cmap="magma_r",
     cbar="log10(records)")
ax = axs[1]
ax.scatter(tab.n_records.clip(lower=1), tab.S_obs, s=14, c=GREEN, alpha=0.55, lw=0)
ax.set_xscale("log")
ax.axvline(MIN_RECORDS, color=AMBER, lw=2, ls="--")
ax.text(MIN_RECORDS * 0.8, 280, f"{MIN_RECORDS} records\n(well-sampled)", color=AMBER, fontsize=14, ha="right")
ax.set_xlabel("Records in cell (log scale)"); ax.set_ylabel("Observed species")
r = np.corrcoef(tab.log_records, tab.S_obs)[0, 1]
ax.set_title(f"More effort → more species  (r = {r:.2f})")
fig.tight_layout(); fig.savefig(FIG / "effort.png")

# 3. Predictor maps ---------------------------------------------------------------
fig, axs = plt.subplots(1, 4, figsize=(16, 6.5))
for ax, (c, t, cm, lab) in zip(axs, [("bio1", "Temperature", "RdYlBu_r", "°C"),
                                    ("bio12", "Rainfall", "Blues", "mm / yr"),
                                    ("lc_tree", "Tree cover", GREENS, "fraction"),
                                    ("lc_shannon", "Land-cover diversity", "YlOrBr", "Shannon H")]):
    kmap(ax, tab, c, t, cmap=cm, cbar=lab)
fig.tight_layout(); fig.savefig(FIG / "predictors.png")

# 4. Observed vs predicted (spatial CV) -------------------------------------------------
oof = pd.read_csv(RES / "richness_oof.csv")
summ = json.load(open(RES / "richness_summary.json"))
fig, ax = plt.subplots(figsize=(6.4, 6.0))
ax.scatter(oof.S_rare, oof.pred_oof, s=18, c=GREEN, alpha=0.6, lw=0)
lim = [oof[["S_rare", "pred_oof"]].min().min() - 3, oof[["S_rare", "pred_oof"]].max().max() + 3]
ax.plot(lim, lim, color=AMBER, lw=2)
ax.set_xlim(lim); ax.set_ylim(lim)
ss = 1 - ((oof.S_rare - oof.pred_oof) ** 2).sum() / ((oof.S_rare - oof.S_rare.mean()) ** 2).sum()
ax.set_xlabel("Observed richness"); ax.set_ylabel("Predicted richness")
ax.set_title(f"{summ['best_model']} – spatial CV (R² = {ss:.2f})")
fig.tight_layout(); fig.savefig(FIG / "obs_vs_pred.png")

# 5. Richness maps -------------------------------------------------------------------
fig, axs = plt.subplots(1, 3, figsize=(15, 7.2))
vmin, vmax = np.nanpercentile(rp.S_pred, 1), np.nanpercentile(rp.S_rare, 99)
kmap(axs[0], rp[rp.well_sampled], "S_rare", "Observed (sampled cells)", vmin=vmin, vmax=vmax,
     gaps=rp, cbar="species / 200 records")
kmap(axs[1], rp, "S_pred", "ML prediction (all cells)", vmin=vmin, vmax=vmax,
     cbar="species / 200 records")
kmap(axs[2], rp, "S_pred_sd", "Uncertainty", cmap="Oranges", cbar="SD across trees")
fig.tight_layout(); fig.savefig(FIG / "richness_maps.png")

# 6. SHAP summary ----------------------------------------------------------------------
sv = np.load(RES / "richness_shap.npy")
X = pd.read_csv(RES / "richness_X.csv").rename(columns=LABELS)
plt.figure()
shap.summary_plot(sv, X, show=False, max_display=8, plot_size=(9, 6.2), cmap="viridis")
f = plt.gcf()
for a in f.axes:
    a.tick_params(labelsize=15)
    a.xaxis.label.set_size(15)
f.axes[0].set_xlabel("SHAP value (effect on predicted richness)")
f.savefig(FIG / "shap_summary.png"); plt.close("all")

# 7. Partial dependence -------------------------------------------------------------------
pdp = json.load(open(RES / "richness_pdp.json"))
units = {"bio1": "°C", "bio12": "mm", "bio4": "SD×100", "bio15": "CV %", "elev_mean": "m",
         "elev_sd": "m", "slope_mean": "°", "bio2": "°C", "bio14": "mm", "bio18": "mm"}
keys = list(pdp)[:4]
fig, axs = plt.subplots(1, 4, figsize=(17, 4.6), sharey=True)
for ax, k in zip(axs, keys):
    ax.plot(pdp[k]["x"], pdp[k]["y"], color=GREEN, lw=3.5)
    ax.set_title(LABELS[k]); ax.set_xlabel(units.get(k, "fraction"))
axs[0].set_ylabel("Predicted richness")
fig.tight_layout(); fig.savefig(FIG / "pdp.png")

# 8. Occurrence probability maps ---------------------------------------------------------
occ = pd.read_csv(RES / "occurrence_predictions.csv")
names = [c[2:] for c in occ.columns if c.startswith("p_")]
fig, axs = plt.subplots(1, len(names), figsize=(4.4 * len(names), 7.4))
for ax, n in zip(axs, names):
    kmap(ax, occ, f"p_{n}", n, cmap="YlGn", vmin=0, vmax=1)
    pres = occ[(occ[f"obs_{n}"] == 1)]
    ax.scatter(pres.lon, pres.lat, s=1.6, c=AMBER, lw=0, alpha=0.9)
sm = plt.cm.ScalarMappable(cmap="YlGn", norm=plt.Normalize(0, 1))
cb = fig.colorbar(sm, ax=axs, orientation="horizontal", fraction=0.035, pad=0.03, aspect=50)
cb.set_label("Predicted occurrence probability   (orange dots = GBIF records)")
cb.outline.set_visible(False)
fig.savefig(FIG / "occurrence_maps.png")
print("figures written to", FIG)
