# AI for Essential Biodiversity Variables — Birds of Kerala

EVST course project (Project 10), mid-evaluation, October 2026.
Team: Divya Padariya (2024113010), Ayush Kanani (2024101049), Akshat Gupta (2024111005)

**Slides:** [presentation/EVST_MidEval_Kerala_Bird_EBVs.pptx](presentation/EVST_MidEval_Kerala_Bird_EBVs.pptx) (18 slides, main deck) and
[presentation/EVST_MidEval_Kerala_Bird_EBVs_v2.pptx](presentation/EVST_MidEval_Kerala_Bird_EBVs_v2.pptx) (31 slides, detailed 2.0 version). PDF copies alongside; speaker notes included.

**Team study guide:** [docs/Study_Guide_Kerala_Bird_EBVs.pdf](docs/Study_Guide_Kerala_Bird_EBVs.pdf): terms, data sources and preprocessing, models and settings, metrics, all results, likely panel questions with answers, and a slide-by-slide speaking guide.

## Problem statement
Can machine learning estimate Essential Biodiversity Variables (EBVs) — **species richness** and
**occurrence probability** — from environmental conditions?
Scope: one taxon (**birds, class Aves**) in one region (**Kerala, India**). This targets the gap found in
our literature review: no comparison of ML methods for one Indian region and taxon using only open data.

## Data used
| Dataset | Used for | Access |
|---|---|---|
| GBIF occurrences | Bird species list + record count per cell (human observations, 2000–2025) | Public occurrence API (facet on `speciesKey` per cell) |
| WorldClim 2.1 | 19 bioclimatic variables, 30 arc-sec | geodata.ucdavis.edu tile 33 |
| Copernicus DEM GLO-90 | Mean/SD elevation, elevation range, slope | AWS open data COGs |
| ESA WorldCover 2021 | Fraction of 9 land-cover classes + Shannon diversity | AWS open data COGs (~75 m overview) |
| IUCN Red List | Threat category of each species | GBIF species API (`/iucnRedListCategory`) |
| geoBoundaries (IND ADM1) | Kerala boundary | geoboundaries.org |

CHELSA was unreachable during download, so WorldClim was used for climate.

## Methodology
1. **Grid** – Kerala split into 0.05° cells (~5.5 km); cells with ≥50% land in Kerala kept → 1,282 cells.
2. **Bird data** – GBIF species list and record counts per cell → 7.96 M records, 573 species.
3. **Environment** – 34 predictors per cell; collinear ones removed (|r| ≥ 0.8) → 13 predictors.
4. **Sampling-bias correction** – raw richness correlates with effort (r = 0.84). Target EBV 1 is
   **rarefied richness** (expected species in 200 records, Hurlbert 1971) for the 1,254 cells with ≥200 records (r with effort drops to 0.40).
5. **EBV 2 – occurrence** for 4 IUCN-listed birds: presence = recorded in a cell; inferred absence =
   cell with ≥1,000 records but no record of the species.
6. **Models** – richness: Ridge, MLP neural net, Random Forest, LightGBM; occurrence: logistic GLM, Random Forest, LightGBM.
7. **Validation** – random 5-fold CV vs **spatial block 5-fold CV** (0.25° ≈ 28 km blocks).
8. **Explainability & maps** – permutation importance, SHAP, partial dependence; prediction maps with RF uncertainty.

## Results (spatial block CV)
| Richness model | R² random CV | R² spatial CV |
|---|---|---|
| Ridge | 0.27 | 0.22 |
| Neural net (MLP) | 0.32 | 0.11 |
| **Random Forest** | 0.52 | **0.42** |
| LightGBM | 0.53 | 0.42 |

| Species (IUCN) | Presences | GLM AUC | RF AUC | LightGBM AUC |
|---|---|---|---|---|
| Malabar Grey Hornbill (VU) | 656 | 0.88 | 0.93 | 0.93 |
| Great Hornbill (VU) | 63 | 0.71 | 0.86 | 0.89 |
| Kerala Laughingthrush (NT) | 56 | 0.95 | 0.96 | 0.96 |
| River Tern (VU) | 338 | 0.71 | 0.82 | 0.82 |

Top richness drivers (permutation importance / SHAP): temperature seasonality, mean annual temperature,
driest-month rainfall, built-up fraction (negative), grassland and open water (positive).

## Observations
- ML can estimate bird EBVs from open data: moderate for richness, strong for occurrence.
- Sampling effort must be corrected first; raw counts mostly measure birding effort.
- Random CV overstates accuracy by ~0.1 R²; spatial CV is the honest test for gap-filling.
- Non-linear tree models beat linear/GLM baselines, most clearly for the rare Great Hornbill.
- Predicted maps match known ecology (hornbills in Western Ghats forest, laughingthrush on high peaks, tern along rivers/backwaters).

## Next steps (remaining ~40%)
Add NDVI and distance-to-water; coverage-based richness and effort covariates; threatened-species
richness map; more species; final report.

## Reproduce
```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
# WorldClim tile: download https://geodata.ucdavis.edu/climate/worldclim/2_1/tiles/tile/tile_33_wc2.1_30s_bio.tif
#   to data/raw/wc2.1_30s_bio_tile33.tif
cd src
../.venv/bin/python s01_grid.py        # Kerala boundary + 0.05° grid
../.venv/bin/python s02_gbif.py        # GBIF species per cell + IUCN status (~15 min, cached)
../.venv/bin/python s03_env.py         # WorldClim / DEM / WorldCover predictors
../.venv/bin/python s04_dataset.py     # modelling table, rarefied richness
../.venv/bin/python s05_richness.py    # EBV 1 models
../.venv/bin/python s06_occurrence.py  # EBV 2 models
../.venv/bin/python s07_figures.py     # figures
../.venv/bin/python s08_teaching_figures.py   # extra explanatory figures (guide + 2.0 deck)
cd ../presentation && npm install && node build_deck.js       # main deck
python3 make_v2.py && node build_deck_v2.js                   # detailed 2.0 deck (re-uses build_deck.js + v2_new_slides.js)
cd ../docs && ../.venv/bin/python build_study_guide.py         # study guide PDF (needs reportlab)
```
Settings live in `src/config.py`; shared model definitions in `src/common.py`.
Outputs: `outputs/results/*.csv|json`, `outputs/figures/*.png`.
