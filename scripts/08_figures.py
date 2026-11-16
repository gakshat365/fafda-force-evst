"""Step 8: all result figures (written to outputs/figures)."""
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from matplotlib.patches import Patch
from shapely.geometry import shape

from ebv import config
from ebv.diversity import expected_coverage, rarefied_richness, sample_size_for_coverage
from ebv.features import SHORT, unit
from ebv.plotting import (AMBER, BLUE, GREEN, GREENS, GREY, INK, MINT, ORANGES, RED, TERRAIN, MapCanvas,
                          use_style)
from ebv.validation import outer_splits

use_style()
FIG, TAB = config.FIGURES, config.TABLES
FIG.mkdir(parents=True, exist_ok=True)
table = pd.read_csv(config.PROCESSED / "model_table.csv")
canvas = MapCanvas(table)
cells_json = {r["cell_id"]: r for r in map(json.loads, (config.RAW / "gbif_cells.jsonl").read_text().splitlines())}
STATE_LABELS = {"Karnataka": (75.6, 16.2), "Tamil Nadu": (78.4, 10.9), "Kerala": (75.75, 10.25)}


def save(fig, name):
    fig.savefig(FIG / name)
    plt.close(fig)


def study_region():
    fig, ax = plt.subplots(figsize=(7.2, 9))
    canvas.draw(ax, table.elev_mean, cmap=TERRAIN, vmin=0, vmax=2000, colorbar="Elevation (m)")
    for name, (x, y) in STATE_LABELS.items():
        ax.text(x, y, name, fontsize=15, fontweight="bold", color=INK, ha="center")
    ax.annotate("Western\nGhats", xy=(75.0, 13.3), xytext=(73.6, 12.3), fontsize=13, color=INK,
                arrowprops=dict(arrowstyle="->", color=INK, lw=1.3))
    ax.annotate("Goa", xy=(74.1, 15.3), xytext=(73.75, 16.45), fontsize=15, fontweight="bold", color=INK,
                arrowprops=dict(arrowstyle="->", color=INK, lw=1.3))
    ax.text(76.0, 14.35, "Deccan\nPlateau", fontsize=13, color=INK, ha="center", style="italic")
    save(fig, "study_region.png")


def predictors():
    panels = [("bio1", "Mean temperature", "RdYlBu_r", "°C"), ("bio12", "Annual rainfall", "Blues", "mm"),
              ("ndvi_median", "Greenness (NDVI)", GREENS, "NDVI"), ("lc_tree", "Tree cover", GREENS, "share"),
              ("lc_crop", "Cropland", ORANGES, "share"), ("dist_water_km", "Distance to water", "YlGnBu", "km")]
    fig, axs = plt.subplots(2, 3, figsize=(15, 13.5))
    for ax, (col, title, cmap, label) in zip(axs.flat, panels):
        vmax = 12 if col == "dist_water_km" else None          # most cells lie within a few km of water
        canvas.draw(ax, table[col], title, cmap=cmap, vmax=vmax, colorbar=label)
    fig.tight_layout()
    save(fig, "predictors.png")


def effort():
    fig, ax = plt.subplots(figsize=(7.2, 9))
    canvas.draw(ax, table.log_records, cmap="magma_r", vmin=0, vmax=5.5, colorbar="Bird records per cell (log10)")
    save(fig, "effort_map.png")

    fig, axs = plt.subplots(1, 2, figsize=(13, 5.4), sharex=True)
    panels = [(table, "observed_species", "Observed species", RED),       # every cell
              (table[table.has_richness], "richness", f"Species in {config.RAREFACTION_SIZE} records", GREEN)]
    for ax, (data, col, title, color) in zip(axs, panels):
        r = data[col].corr(data.log_records)
        ax.scatter(data.n_records.clip(lower=1), data[col], s=9, alpha=0.45, color=color, lw=0)
        ax.set_xscale("log")
        ax.set_xlabel("Bird records in cell (log scale)")
        ax.set_title(f"{title}   r = {r:.2f}")
    axs[0].set_ylabel("Species")
    fig.tight_layout()
    save(fig, "bias_scatter.png")


def standardisation_curves():
    """Left: species expected in n records (we compare cells at n = 200).
    Right: sample coverage; reaching 95% needs several hundred records, which many cells lack."""
    examples = {"76.60_10.10": ("Thattekad forest (Kerala)", GREEN), "80.20_13.00": ("Chennai city (Tamil Nadu)", AMBER)}
    fig, axs = plt.subplots(1, 2, figsize=(13.5, 5.2))
    for cell_id, (label, color) in examples.items():
        counts = np.array(list(cells_json[cell_id]["species"].values()), float)
        sizes = np.unique(np.geomspace(1, counts.sum(), 70).astype(int))
        n, m = config.RAREFACTION_SIZE, sample_size_for_coverage(counts, config.TARGET_COVERAGE)
        axs[0].plot(sizes, [rarefied_richness(counts, k) for k in sizes], color=color, lw=3, label=label)
        axs[0].plot(n, rarefied_richness(counts, n), "o", color=color, ms=10)
        axs[0].annotate(f"{rarefied_richness(counts, n):.0f} species", (n, rarefied_richness(counts, n)),
                        xytext=(12, -26) if color == AMBER else (-112, 6), textcoords="offset points", color=color,
                        fontsize=14, fontweight="bold")
        axs[1].plot(sizes, [expected_coverage(counts, k) for k in sizes], color=color, lw=3, label=label)
        axs[1].plot(m, config.TARGET_COVERAGE, "o", color=color, ms=10)
        axs[1].annotate(f"{m:,} records", (m, config.TARGET_COVERAGE), xytext=(14, -30 if color == AMBER else -56),
                        textcoords="offset points", color=color, fontsize=14, fontweight="bold")
    axs[0].axvline(config.RAREFACTION_SIZE, color=GREY, ls="--", lw=1.4)
    axs[0].set_ylabel("Expected species")
    axs[0].set_title(f"Used: species in {config.RAREFACTION_SIZE} records")
    axs[1].axhline(config.TARGET_COVERAGE, color=GREY, ls="--", lw=1.4)
    axs[1].set_ylabel("Sample coverage")
    axs[1].set_title(f"Tested: records needed for {config.TARGET_COVERAGE:.0%} coverage")
    for ax in axs:
        ax.set_xscale("log")
        ax.set_xlabel("Records sampled (log scale)")
    axs[0].legend(loc="upper left", fontsize=12)
    fig.tight_layout()
    save(fig, "standardisation_curves.png")


def validation_schemes():
    train = table[table.has_richness].reset_index(drop=True)
    colours = np.array([GREEN, AMBER, BLUE, RED, MINT])
    fig, axs = plt.subplots(1, 3, figsize=(16, 7.5))
    for ax, (scheme, title) in zip(axs, [("random", "Random CV"), ("spatial", "Spatial block CV"),
                                         ("state", "Leave-one-state-out")]):
        fold = np.zeros(len(train), int)
        for k, (_, test) in enumerate(outer_splits(train, scheme)):
            fold[test] = k
        ax.scatter(train.lon, train.lat, c=colours[fold], s=6, marker="s", lw=0)
        for geometry in canvas.states.values():
            canvas.outline(ax, geometry, color=INK, lw=0.6)
        ax.set_aspect("equal"); ax.axis("off"); ax.set_title(title)
    save(fig, "validation_schemes.png")


def model_maps(target, label):
    pred = pd.read_csv(TAB / f"{target}_predictions.csv")
    vmin, vmax = np.nanpercentile(pred.predicted, 1), np.nanpercentile(pred.observed, 99)
    fig, axs = plt.subplots(1, 3, figsize=(16.5, 7.6))
    observed = pred.observed.where(pred.has_richness)
    canvas.draw(axs[0], observed, "Observed", GREENS, vmin, vmax, colorbar=label, background=pred.predicted)
    canvas.draw(axs[1], pred.predicted, "Predicted (all cells)", GREENS, vmin, vmax, colorbar=label)
    canvas.draw(axs[2], pred.uncertainty, "Uncertainty", ORANGES, colorbar="SD across trees")
    fig.tight_layout()
    save(fig, f"{target}_maps.png")

    oof = pd.read_csv(TAB / f"{target}_oof.csv")
    r2 = 1 - ((oof.observed - oof.predicted) ** 2).sum() / ((oof.observed - oof.observed.mean()) ** 2).sum()
    fig, ax = plt.subplots(figsize=(6.2, 5.8))
    ax.scatter(oof.observed, oof.predicted, s=10, alpha=0.45, color=GREEN, lw=0)
    lim = [min(oof.observed.min(), oof.predicted.min()), max(oof.observed.max(), oof.predicted.max())]
    ax.plot(lim, lim, color=AMBER, lw=2.2)
    ax.set_xlabel(f"Observed {label.lower()}")
    ax.set_ylabel(f"Predicted {label.lower()}")
    ax.set_title(f"Spatial cross-validation   R² = {r2:.2f}")
    save(fig, f"{target}_obs_vs_pred.png")

    shap_data = np.load(TAB / f"{target}_shap.npz", allow_pickle=True)
    sample = pd.DataFrame(shap_data["X"], columns=[SHORT[c] for c in shap_data["columns"]])
    plt.figure()
    shap.summary_plot(shap_data["values"], sample, show=False, max_display=8, plot_size=(9, 6), cmap="viridis")
    fig = plt.gcf()
    fig.axes[0].set_xlabel(f"Effect on predicted {label.lower()} (SHAP value)", fontsize=14)
    fig.axes[0].tick_params(labelsize=14)
    save(fig, f"{target}_shap.png")

    curves = json.loads((TAB / f"{target}_curves.json").read_text())
    fig, axs = plt.subplots(1, len(curves), figsize=(4.3 * len(curves), 4.3), sharey=True)
    for ax, (feature, curve) in zip(np.atleast_1d(axs), curves.items()):
        ax.plot(curve["x"], curve["y"], color=GREEN, lw=3.2)
        ax.set_title(SHORT[feature])
        ax.set_xlabel(unit(feature))
    np.atleast_1d(axs)[0].set_ylabel(f"Predicted {label.lower()}")
    fig.tight_layout()
    save(fig, f"{target}_curves.png")


def occurrence():
    occ = pd.read_csv(TAB / "occurrence_predictions.csv")
    names = [c[2:] for c in occ.columns if c.startswith("p_")]
    n_cols = 6                                                   # 12 species in two rows
    n_rows = int(np.ceil(len(names) / n_cols))
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(4.2 * n_cols, 5.6 * n_rows))
    for ax, name in zip(axs.flat, names):
        canvas.draw(ax, occ[f"p_{name}"], name, cmap="YlGn", vmin=0, vmax=1)
        present = occ[occ[f"present_{name}"] == 1]
        ax.scatter(present.lon, present.lat, s=1.2, color=AMBER, lw=0)
    for ax in axs.flat[len(names):]:
        ax.axis("off")
    bar = fig.colorbar(plt.cm.ScalarMappable(cmap="YlGn", norm=plt.Normalize(0, 1)), ax=axs,
                       orientation="horizontal", fraction=0.025, pad=0.02, aspect=60)
    bar.set_label("Predicted occurrence probability (orange dots: cells with GBIF records)", fontsize=15)
    bar.outline.set_visible(False)
    save(fig, "occurrence_maps.png")


def protected():
    cells = pd.read_csv(TAB / "protected_cells.csv")
    areas = json.loads((config.PROCESSED / "protected_areas.geojson").read_text())["features"]
    category = np.select([cells.richness_hotspot & cells.threatened_hotspot, cells.threatened_hotspot,
                          cells.richness_hotspot], [3, 2, 1], 0).astype(float)
    category[category == 0] = np.nan
    fig, ax = plt.subplots(figsize=(8, 9.6))
    canvas.draw(ax, np.zeros(len(cells)), cmap=plt.matplotlib.colors.ListedColormap(["#EEF1EE"]))
    canvas.draw(ax, category, cmap=plt.matplotlib.colors.ListedColormap([MINT, AMBER, GREEN]), vmin=0.5, vmax=3.5)
    for feature in areas:
        geometry = shape(feature["geometry"])
        for part in getattr(geometry, "geoms", [geometry]):
            x, y = part.exterior.xy
            ax.fill(x, y, facecolor="none", edgecolor=BLUE, lw=1.0, hatch="////", alpha=0.9)
    ax.legend(handles=[Patch(color=MINT, label="Species-richness hotspot"),
                       Patch(color=AMBER, label="Threatened-species hotspot"),
                       Patch(color=GREEN, label="Both"),
                       Patch(facecolor="none", edgecolor=BLUE, hatch="////", label="Protected area")],
              loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=2, fontsize=13)
    save(fig, "protected_hotspots.png")


study_region()
predictors()
effort()
standardisation_curves()
validation_schemes()
model_maps("richness", "Species richness")
model_maps("threatened", "Threatened species")
occurrence()
protected()
print("figures written to", FIG)
