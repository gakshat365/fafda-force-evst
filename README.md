# AI for Essential Biodiversity Variables: Birds of South India

EVST course project 10 (final submission).
Team: Divya Padariya (2024113010), Ayush Kanani (2024101049), Akshat Gupta (2024111005)

## Problem statement
Can machine learning estimate Essential Biodiversity Variables (EBVs) for one taxon (birds, class Aves)
from open environmental data? And how well do the models work in places they have never seen?

We estimate three EBVs:
1. **Species richness**: the expected number of species in 200 records. This is rarefied richness, so it does not depend on birding effort.
2. **Threatened-species richness**: the same measure for birds listed by the IUCN as Near Threatened to Critically Endangered.
3. **Occurrence probability** of 12 threatened bird species.

**Study region:** Kerala, Tamil Nadu, Karnataka, Goa and mainland Puducherry (~365,000 km²), split into
3,043 grid cells of 0.1° (≈ 11 km). The data cover 20.9 million bird records of 686 species.

**Gaps addressed (from our literature review):**
- few machine-learning studies of birds come from India;
- Indian studies usually cover one city or one small area;
- ML methods are rarely compared on the same region, taxon and open data;
- the main drivers change between regions, and transfer to new regions is rarely tested;
- random cross-validation overstates accuracy;
- raw citizen-science counts follow where birders go.

## From mid-evaluation to final
The mid-evaluation version (Kerala only, 0.05° cells, 4 species) is kept in the commit history (`48111af`).
Since then we have made these changes:
- **Region:** expanded from Kerala to five states (about 9× the area and 2.6× the records). This lets us test transfer between states.
- **Model selection:** nested cross-validation now keeps test cells out of every tuning and model choice.
- **Baseline:** added a quadratic GLM, which can capture hump-shaped responses.
- **Predictors:** added NDVI greenness and distance to water.
- **New outputs:** added threatened-species richness and a leave-one-state-out transfer test, and increased the species models from 4 to 12.
- **Richness correction:** tested coverage-based richness (Chao & Jost 2012). It removed less bias than rarefaction, so rarefaction stays.
- **IUCN status:** now fetched per species, which fills in species that the earlier shortcut missed.
- **Protected areas:** compared the threatened-species hotspots with protected areas.

## Data (all open)
| Dataset | What we use | Access |
|---|---|---|
| GBIF occurrences | bird species and record counts per cell (human observations, 2000–2025) | GBIF occurrence API, one facet query per cell |
| WorldClim 2.1 | 19 bioclimatic variables (1970–2000 normals, ~1 km) | geodata.ucdavis.edu tile 33 |
| Copernicus DEM GLO-90 | elevation, ruggedness, slope | AWS open data (COG) |
| ESA WorldCover 2021 | land-cover shares, habitat mix, distance to inland water | AWS open data (COG) |
| ESA WorldCover Sentinel-2 NDVI 2021 | median greenness and greenness seasonality | AWS open data (COG) |
| IUCN Red List | threat category of each species | GBIF species API (`/iucnRedListCategory`) |
| geoBoundaries | state boundaries | geoboundaries.org |
| OpenStreetMap | protected areas (national parks, sanctuaries, tiger reserves) | Overpass API |

## Methodology
1. **Grid**: 0.1° cells over the five states (`01_build_grid.py`).
2. **Birds**: GBIF species lists and record counts per cell, plus IUCN status (`02_fetch_gbif.py`).
3. **Environment**: climate, terrain, land cover and greenness, aggregated onto the grid. Predictors that
   correlate at |r| ≥ 0.8 are removed, which leaves 18 (`03_extract_environment.py`, `04_build_dataset.py`).
4. **Sampling bias**: raw species counts follow birding effort (r = 0.91). Richness in 200 records removes most
   of this (r = 0.06), so we use it as the target for cells with at least 200 records.
5. **Models**:
   - Richness: Ridge regression, a neural network, Random Forest and XGBoost (`05_model_richness.py`).
   - Occurrence: linear GLM, quadratic GLM, Random Forest and XGBoost (`06_model_occurrence.py`).
   - Hyperparameters are tuned inside each training fold (nested CV).
6. **Validation**: three ways to hide data.
   - Random 5-fold CV, shown for contrast only.
   - Spatial-block CV with 0.5° (≈ 55 km) blocks.
   - Leave-one-state-out CV.
7. **Explanation and maps**: permutation importance, SHAP, response curves and prediction maps (`08_figures.py`).
8. **Conservation check**: threatened-species hotspots compared with OpenStreetMap protected areas (`07_protected_areas.py`).

## Results (spatial-block CV unless stated)
| Model | Species richness R² | Threatened-species richness R² |
|---|---|---|
| Ridge regression | 0.34 | 0.26 |
| Neural network | 0.49 | 0.40 |
| **Random Forest** | **0.53** | **0.41** |
| XGBoost | 0.51 | 0.40 |

For species richness, random CV gives a higher R² (0.60) and holding out a whole state gives a much lower one (0.18).

| Species (IUCN) | Presences | Best AUC | Best model |
|---|---|---|---|
| White-bellied Blue Robin (NT) | 46 | 0.98 | XGBoost |
| Malabar Grey Hornbill (VU) | 660 | 0.97 | Random Forest |
| Kerala Laughingthrush (NT) | 57 | 0.96 | Random Forest |
| Malabar Pied Hornbill (NT) | 426 | 0.93 | XGBoost |
| Rufous-tailed Lark (NT) | 691 | 0.93 | Random Forest |
| Indian Courser (NT) | 115 | 0.92 | Random Forest |
| Spot-billed Pelican (NT) | 593 | 0.89 | GLM (quadratic) |
| River Tern (VU) | 1,245 | 0.88 | Random Forest |
| White-rumped Vulture (CR) | 95 | 0.88 | Random Forest |
| Woolly-necked Stork (NT) | 1,355 | 0.86 | Random Forest |
| Great Hornbill (VU) | 94 | 0.80 | GLM (quadratic) |
| Black-bellied Tern (EN) | 42 | 0.72 | Random Forest |

## Conclusions
- **Machine learning can estimate bird EBVs from open data.** Richness is predicted moderately well (R² 0.53)
  and occurrence well (mean AUC 0.89). Tree models are best for 10 of the 12 species.
- **Effort must be corrected first.** Raw counts mostly show where birders go.
- **Climate leads and water fine-tunes.** Mean temperature and driest-month rainfall matter most. Open water adds
  about 5 species and built-up land removes about 3.
- **Transfer is limited.** A model trained on two states predicts Kerala reasonably well (R² 0.34). It fails for
  Karnataka and Tamil Nadu, because their environments have no match in the training states.
- **Threatened species cluster in wetlands, not forests.** They concentrate around Karnataka's reservoirs and
  wetlands and in coastal wetlands. Only 13% of threatened-species hotspot cells are at least 25% protected.
  402 hotspot cells are unprotected, 356 of them in Karnataka.

## Limitations and future scope
**Limitations:**
- Sampling is uneven across cells.
- The climate normals (1970–2000) are older than the bird records (2000–2025).
- 11 km cells blur narrow habitats.
- Absences are inferred, not confirmed.
- OpenStreetMap protected areas may be incomplete.
- IUCN status is global, not specific to India.

**Next steps:**
- Track change over time (2005–15 vs 2015–25).
- Project ranges under future climate (CMIP6).
- Build detection-aware models from eBird checklists.
- Cover more taxa and all of India.
- Validate against bird-atlas surveys.

## Repository layout
```
src/ebv/            the analysis package
  config.py         every setting (region, grid, years, species, CV design)
  region.py         study region and 0.1° grid
  gbif.py           rate-limited GBIF client
  environment.py    raster predictors aggregated onto the grid
  diversity.py      rarefaction and coverage-based richness
  features.py       predictor names and the collinearity filter
  validation.py     random / spatial-block / leave-one-state-out CV and metrics
  models.py         models and their tuning search spaces
  neural.py         small PyTorch network with a scikit-learn interface
  modelling.py      nested cross-validation, final fits, explanations
  protected.py      OpenStreetMap protected areas
  plotting.py       shared figure style and map drawing
scripts/            01–08: the pipeline, run in order (run_pipeline.sh runs everything)
tests/              unit tests (pytest)
data/raw/           cached GBIF and OpenStreetMap downloads (large rasters and boundaries are re-downloaded)
data/processed/     grid, environment, species and modelling tables
outputs/tables/     cross-validation scores, predictions, importance, transfer and protection results
outputs/figures/    maps and plots
Final_presentation.pdf, Mid_presentation.pdf, literature_review_final.pdf
```

## Reproduce
```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt && .venv/bin/pip install -e .
```
```bash
scripts/run_pipeline.sh       # runs the tests, then steps 01–08
```
The first run downloads the WorldClim tile (214 MB) and the state boundaries automatically. The GBIF
download is cached in `data/raw/gbif_cells.jsonl`, and the fetch resumes where it stopped if interrupted.
