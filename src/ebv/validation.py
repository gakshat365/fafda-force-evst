"""Cross-validation schemes and evaluation metrics.

Three outer schemes, from most to least optimistic:
- random:  cells shuffled into folds (neighbours leak information between folds)
- spatial: whole 0.5 degree blocks held out together (~55 km)
- state:   one state held out at a time (Goa joins Karnataka, Puducherry joins Tamil Nadu)
"""
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, roc_auc_score, roc_curve
from sklearn.model_selection import (GroupKFold, KFold, LeaveOneGroupOut, StratifiedGroupKFold,
                                     StratifiedKFold)

from . import config

SCHEMES = ("random", "spatial", "state")
STATE_GROUP = {"Goa": "Karnataka", "Puducherry": "Tamil Nadu"}


def spatial_blocks(cells: pd.DataFrame, size: float = config.BLOCK_DEG) -> np.ndarray:
    """Integer id of the size x size degree block containing each cell centre."""
    col = np.floor(cells.lon.values / size).astype(int)
    row = np.floor(cells.lat.values / size).astype(int)
    return row * 10_000 + col


def state_groups(cells: pd.DataFrame) -> np.ndarray:
    return cells.state.replace(STATE_GROUP).values


def outer_splits(cells: pd.DataFrame, scheme: str, y=None, classification: bool = False) -> list:
    """(train, test) index pairs for one outer cross-validation scheme."""
    n_folds, seed = config.N_OUTER_FOLDS, config.SEED
    if scheme == "random":
        splitter = (StratifiedKFold if classification else KFold)(n_folds, shuffle=True, random_state=seed)
        return list(splitter.split(cells, y))
    if scheme == "spatial":
        splitter = (StratifiedGroupKFold if classification else GroupKFold)(n_folds, shuffle=True,
                                                                            random_state=seed)
        return list(splitter.split(cells, y, groups=spatial_blocks(cells)))
    if scheme == "state":
        return list(LeaveOneGroupOut().split(cells, y, groups=state_groups(cells)))
    raise ValueError(f"unknown scheme {scheme!r}; choose from {SCHEMES}")


def inner_splitter(scheme: str, classification: bool):
    """Splitter used for hyper-parameter tuning inside each outer training set.

    It mirrors the outer scheme, so tuning is judged on the same kind of hold-out the
    model is finally scored on (blocks for spatial/state, random cells for random).
    """
    n, seed = config.N_INNER_FOLDS, config.SEED
    if scheme == "random":
        return (StratifiedKFold if classification else KFold)(n, shuffle=True, random_state=seed)
    return (StratifiedGroupKFold if classification else GroupKFold)(n, shuffle=True, random_state=seed)


# ----------------------------------------------------------------------------- metrics
def regression_scores(y, pred) -> dict:
    return {"R2": r2_score(y, pred), "RMSE": mean_squared_error(y, pred) ** 0.5,
            "MAE": mean_absolute_error(y, pred)}


def true_skill_statistic(y, prob) -> float:
    """Maximum over thresholds of sensitivity + specificity - 1 (Allouche et al. 2006)."""
    fpr, tpr, _ = roc_curve(y, prob)
    return float(np.max(tpr - fpr))


def classification_scores(y, prob) -> dict:
    return {"AUC": roc_auc_score(y, prob), "TSS": true_skill_statistic(y, prob)}
