"""Candidate predictors, their display names, and the collinearity filter."""
import pandas as pd

# Candidates in priority order: when two are near-duplicates, the earlier one is kept.
CANDIDATES = [
    "bio1", "bio12", "bio4", "bio15", "bio2", "bio14", "bio18",          # climate
    "ndvi_median", "ndvi_range",                                         # vegetation
    "elev_sd", "slope", "elev_mean",                                     # terrain
    "lc_tree", "lc_shrub", "lc_grass", "lc_crop", "lc_built",            # land cover
    "lc_water", "lc_wetland", "dist_water_km", "lc_shannon",             # water, habitat mix
]

LABELS = {
    "bio1": "Mean annual temperature", "bio2": "Day-night temperature range",
    "bio4": "Temperature seasonality", "bio12": "Annual rainfall",
    "bio14": "Driest-month rainfall", "bio15": "Rainfall seasonality",
    "bio18": "Warmest-quarter rainfall", "ndvi_median": "Greenness (median NDVI)",
    "ndvi_range": "Greenness seasonality (NDVI range)", "elev_mean": "Elevation",
    "elev_sd": "Terrain ruggedness", "slope": "Slope", "lc_tree": "Tree cover",
    "lc_shrub": "Shrubland", "lc_grass": "Grassland", "lc_crop": "Cropland",
    "lc_built": "Built-up", "lc_water": "Open water", "lc_wetland": "Wetland",
    "dist_water_km": "Distance to inland water", "lc_shannon": "Land-cover diversity",
}

UNITS = {
    "bio1": "°C", "bio2": "°C", "bio4": "SD × 100", "bio12": "mm", "bio14": "mm", "bio15": "CV %",
    "bio18": "mm", "ndvi_median": "NDVI", "ndvi_range": "NDVI", "elev_mean": "m", "elev_sd": "m",
    "slope": "degrees", "dist_water_km": "km", "lc_shannon": "Shannon H",
}

SHORT = {  # compact labels for crowded charts
    "bio1": "Mean temp.", "bio2": "Day-night range", "bio4": "Temp. seasonality",
    "bio12": "Annual rain", "bio14": "Driest-month rain", "bio15": "Rain seasonality",
    "bio18": "Warm-quarter rain", "ndvi_median": "Greenness", "ndvi_range": "Green seasonality",
    "elev_mean": "Elevation", "elev_sd": "Ruggedness", "slope": "Slope", "lc_tree": "Tree cover",
    "lc_shrub": "Shrubland", "lc_grass": "Grassland", "lc_crop": "Cropland", "lc_built": "Built-up",
    "lc_water": "Open water", "lc_wetland": "Wetland", "dist_water_km": "Dist. to water",
    "lc_shannon": "Habitat mix",
}


def unit(name: str) -> str:
    return UNITS.get(name, "fraction of cell")


def select_uncorrelated(frame: pd.DataFrame, candidates=CANDIDATES, threshold: float = 0.8):
    """Keep candidates in priority order, dropping any whose |Pearson r| with an already
    kept predictor reaches `threshold`.

    Returns the kept names and a table of dropped predictors with the partner that
    replaced them and their correlation.
    """
    corr = frame[candidates].corr()
    kept, dropped = [], []
    for name in candidates:
        if frame[name].std() == 0:
            dropped.append((name, None, float("nan")))
            continue
        partner = corr.loc[name, kept].abs().idxmax() if kept else None
        if partner is not None and abs(corr.loc[name, partner]) >= threshold:
            dropped.append((name, partner, round(float(corr.loc[name, partner]), 2)))
        else:
            kept.append(name)
    return kept, pd.DataFrame(dropped, columns=["dropped", "kept_instead", "r"])
