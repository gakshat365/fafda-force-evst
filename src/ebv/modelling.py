"""Nested cross-validation, final model fitting, and model explanations.

Nested CV: for every outer fold, hyper-parameters are tuned by an *inner* CV that only
sees that fold's training cells. The outer test cells never influence any modelling
choice, so the outer score is an honest estimate of performance on new places.
"""
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import shap
from sklearn.base import clone
from sklearn.inspection import partial_dependence, permutation_importance
from sklearn.model_selection import RandomizedSearchCV

from . import config
from .models import ModelSpec
from .validation import (classification_scores, inner_splitter, outer_splits, regression_scores,
                         spatial_blocks)


def fit_tuned(spec: ModelSpec, X: pd.DataFrame, y: np.ndarray, groups: np.ndarray,
              scheme: str = "spatial", classification: bool = False):
    """Fit `spec` on (X, y), tuning hyper-parameters by inner CV if it has a search space."""
    if spec.search_space is None:
        return clone(spec.estimator).fit(X, y), {}
    search = RandomizedSearchCV(clone(spec.estimator), spec.search_space,
                                n_iter=config.N_TUNING_ITER, random_state=config.SEED,
                                scoring="roc_auc" if classification else "r2",
                                cv=inner_splitter(scheme, classification), n_jobs=spec.search_jobs)
    search.fit(X, y, groups=None if scheme == "random" else groups)    # random folds ignore groups
    return search.best_estimator_, search.best_params_


def predict(model, X: pd.DataFrame, classification: bool) -> np.ndarray:
    return model.predict_proba(X)[:, 1] if classification else model.predict(X)


@dataclass
class CVResult:
    model: str
    scheme: str
    y: np.ndarray
    predictions: np.ndarray                 # out-of-fold: each cell predicted by a model that never saw it
    splits: list
    folds: list = field(default_factory=list)
    fitted: list = field(default_factory=list)
    classification: bool = False

    def scores(self) -> dict:
        score = classification_scores if self.classification else regression_scores
        return score(self.y, self.predictions)


def cross_validate(spec: ModelSpec, X: pd.DataFrame, y, cells: pd.DataFrame, scheme: str,
                   classification: bool = False, keep_models: bool = False) -> CVResult:
    y = np.asarray(y)
    blocks = spatial_blocks(cells)
    splits = outer_splits(cells, scheme, y, classification)
    result = CVResult(spec.name, scheme, y, np.full(len(y), np.nan), splits,
                      classification=classification)
    score = classification_scores if classification else regression_scores
    for fold, (train, test) in enumerate(splits):
        model, params = fit_tuned(spec, X.iloc[train], y[train], blocks[train], scheme, classification)
        result.predictions[test] = predict(model, X.iloc[test], classification)
        result.folds.append({"fold": fold, "n_test": len(test), "params": params,
                             **score(y[test], result.predictions[test])})
        if keep_models:
            result.fitted.append(model)
    return result


def fit_final(spec: ModelSpec, X: pd.DataFrame, y, cells: pd.DataFrame, classification: bool = False):
    """Tune (inner spatial CV) and fit on all training cells; used only for maps and explanations."""
    return fit_tuned(spec, X, np.asarray(y), spatial_blocks(cells), "spatial", classification)


# ----------------------------------------------------------------------------- explanations
def cv_permutation_importance(result: CVResult, X: pd.DataFrame, labels: dict) -> pd.DataFrame:
    """Drop in held-out score when one predictor is shuffled, averaged over the outer folds."""
    if not result.fitted:
        raise ValueError("cross_validate(..., keep_models=True) is needed for permutation importance")
    scoring = "roc_auc" if result.classification else "r2"
    drops = []
    for (_, test), model in zip(result.splits, result.fitted):
        pi = permutation_importance(model, X.iloc[test], result.y[test], scoring=scoring,
                                    n_repeats=10, random_state=config.SEED)
        drops.append(pi.importances_mean)
    drops = np.array(drops)
    table = pd.DataFrame({"feature": X.columns, "label": [labels[c] for c in X.columns],
                          "importance": drops.mean(0), "sd": drops.std(0)})
    return table.sort_values("importance", ascending=False, ignore_index=True)


def shap_values(model, X: pd.DataFrame, n: int = 1500) -> tuple[np.ndarray, pd.DataFrame]:
    """SHAP values of a tree model for a random sample of n cells."""
    sample = X.sample(min(n, len(X)), random_state=config.SEED)
    return np.asarray(shap.TreeExplainer(model).shap_values(sample)), sample


def response_curves(model, X: pd.DataFrame, features: list[str]) -> dict:
    """Partial dependence: average prediction as one predictor varies (5th-95th percentile)."""
    curves = {}
    for f in features:
        pd_result = partial_dependence(model, X, [f], grid_resolution=40, percentiles=(0.05, 0.95))
        curves[f] = {"x": pd_result["grid_values"][0].tolist(), "y": pd_result["average"][0].tolist()}
    return curves


def tree_spread(forest, X: pd.DataFrame) -> np.ndarray:
    """Standard deviation of the individual trees' predictions: a simple uncertainty map."""
    return np.stack([tree.predict(X.values) for tree in forest.estimators_]).std(axis=0)


def outside_training_range(X_train: pd.DataFrame, X_all: pd.DataFrame) -> np.ndarray:
    """True where any predictor lies outside the range seen in training (extrapolation)."""
    return ((X_all < X_train.min()) | (X_all > X_train.max())).any(axis=1).values
