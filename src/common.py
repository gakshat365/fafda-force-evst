"""Shared helpers: predictor selection, spatial blocks, model zoo."""
import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import KFold, GroupKFold
from lightgbm import LGBMRegressor, LGBMClassifier
from config import SEED

# Candidate predictors in priority order (ecologically interpretable first).
CANDIDATES = [
    "bio1", "bio12", "bio4", "bio15", "bio2", "bio14", "bio18",
    "elev_sd", "slope_mean", "elev_mean",
    "lc_tree", "lc_crop", "lc_built", "lc_water", "lc_grass", "lc_wetland",
    "lc_shannon",
]
LABELS = {
    "bio1": "Mean annual temp.", "bio2": "Diurnal temp. range",
    "bio4": "Temp. seasonality", "bio12": "Annual rainfall",
    "bio14": "Driest-month rain", "bio15": "Rainfall seasonality",
    "bio18": "Warmest-qtr rain", "elev_mean": "Elevation",
    "elev_sd": "Terrain ruggedness", "slope_mean": "Slope",
    "lc_tree": "Tree cover", "lc_crop": "Cropland", "lc_built": "Built-up",
    "lc_water": "Open water", "lc_grass": "Grassland", "lc_wetland": "Wetland",
    "lc_shannon": "Land-cover diversity",
}
BLOCK = 0.25  # spatial block size in degrees (~28 km)


def select_features(df, thr=0.8):
    """Greedy removal of collinear predictors (|Pearson r| >= thr)."""
    keep = []
    corr = df[CANDIDATES].corr().abs()
    for c in CANDIDATES:
        if df[c].std() > 0 and all(corr.loc[c, k] < thr for k in keep):
            keep.append(c)
    return keep


def blocks(df):
    return (np.floor(df.lon / BLOCK).astype(int).astype(str) + "_" +
            np.floor(df.lat / BLOCK).astype(int).astype(str))


def splitters(df):
    return {
        "Random CV": list(KFold(5, shuffle=True, random_state=SEED).split(df)),
        "Spatial block CV": list(GroupKFold(5, shuffle=True, random_state=SEED)
                                 .split(df, groups=blocks(df))),
    }


def regressors():
    return {
        "Linear (Ridge)": make_pipeline(StandardScaler(), RidgeCV(np.logspace(-3, 3, 13))),
        "Neural net (MLP)": make_pipeline(StandardScaler(), MLPRegressor(hidden_layer_sizes=
            (32, 16), alpha=1e-2, max_iter=3000, early_stopping=True, random_state=SEED)),
        "Random Forest": RandomForestRegressor(500, min_samples_leaf=3, max_features=0.5,
                                               n_jobs=-1, random_state=SEED),
        "LightGBM": LGBMRegressor(n_estimators=400, learning_rate=0.03, num_leaves=15,
                                  min_child_samples=10, subsample=0.8, subsample_freq=1,
                                  colsample_bytree=0.8, random_state=SEED, verbose=-1),
    }


def classifiers():
    return {
        "Logistic (GLM)": make_pipeline(StandardScaler(), LogisticRegression(
            C=1.0, max_iter=2000, class_weight="balanced")),
        "Random Forest": RandomForestClassifier(500, min_samples_leaf=3, max_features=0.5,
                                                class_weight="balanced_subsample",
                                                n_jobs=-1, random_state=SEED),
        "LightGBM": LGBMClassifier(n_estimators=300, learning_rate=0.03, num_leaves=15,
                                   min_child_samples=10, subsample=0.8, subsample_freq=1,
                                   colsample_bytree=0.8, is_unbalance=True,
                                   random_state=SEED, verbose=-1),
    }
