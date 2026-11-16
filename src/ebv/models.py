"""The model zoo: every model with its fixed settings and its tuning search space.

Gradient boosting (XGBoost) and the neural network (PyTorch) train on the GPU when one
is available; the other models are small or CPU-parallel and run on the processor.
"""
from dataclasses import dataclass

import numpy as np
from scipy.stats import loguniform
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from xgboost import XGBClassifier, XGBRegressor

from . import config
from .neural import DEVICE, TorchMLPRegressor


@dataclass
class ModelSpec:
    """A model and how to tune it. `search_space=None` means no outer tuning is needed."""
    name: str
    estimator: BaseEstimator
    search_space: dict | None = None
    search_jobs: int = 1          # -1 (all cores) only for single-threaded CPU models
    tree_based: bool = False


def add_squares(x):
    """[x, x^2] for each (already standardised) predictor: lets a GLM draw hump-shaped responses."""
    return np.hstack([x, x ** 2])


class BoostedTreesRegressor(XGBRegressor):
    """XGBoost trained on the GPU; predictions run on the CPU inputs without device copies."""

    def fit(self, X, y, **kwargs):
        super().fit(X, y, **kwargs)
        self.get_booster().set_param({"device": "cpu"})
        return self


class BoostedTreesClassifier(XGBClassifier):
    """XGBoost classifier that weights presences and absences equally (rare species)."""

    def fit(self, X, y, **kwargs):
        y = np.asarray(y)
        self.set_params(scale_pos_weight=(len(y) - y.sum()) / max(y.sum(), 1))
        super().fit(X, y, **kwargs)
        self.get_booster().set_param({"device": "cpu"})
        return self


BOOSTING_SPACE = {"n_estimators": [200, 400, 800], "learning_rate": [0.02, 0.05, 0.1],
                  "max_depth": [3, 4, 6, 8], "min_child_weight": [1, 3, 5, 10],
                  "subsample": [0.7, 0.9], "colsample_bytree": [0.6, 0.8, 1.0],
                  "reg_lambda": [0.0, 1.0, 5.0]}


def regression_models() -> list[ModelSpec]:
    seed = config.SEED
    return [
        ModelSpec("Ridge regression",
                  make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-3, 3, 25)))),
        ModelSpec("Neural network",
                  make_pipeline(StandardScaler(), TorchMLPRegressor(random_state=seed)),
                  {"torchmlpregressor__hidden_layers": [(32,), (64, 32), (128, 64), (64, 64, 32)],
                   "torchmlpregressor__weight_decay": loguniform(1e-5, 1e-1),
                   "torchmlpregressor__learning_rate": [1e-3, 3e-3, 1e-2],
                   "torchmlpregressor__dropout": [0.0, 0.1, 0.2]}),
        ModelSpec("Random Forest",
                  RandomForestRegressor(n_estimators=400, n_jobs=-1, random_state=seed),
                  {"max_features": [0.3, 0.5, 0.7, 1.0], "min_samples_leaf": [1, 3, 5, 10],
                   "max_depth": [None, 12, 24]},
                  tree_based=True),
        ModelSpec("XGBoost",
                  BoostedTreesRegressor(tree_method="hist", device=DEVICE.type, random_state=seed),
                  BOOSTING_SPACE, tree_based=True),
    ]


def classification_models() -> list[ModelSpec]:
    seed = config.SEED
    glm = lambda: LogisticRegression(class_weight="balanced", max_iter=5000)
    c_space = {"logisticregression__C": loguniform(1e-3, 1e2)}
    return [
        ModelSpec("GLM (linear)", make_pipeline(StandardScaler(), glm()), c_space, search_jobs=-1),
        ModelSpec("GLM (quadratic)",
                  make_pipeline(StandardScaler(), FunctionTransformer(add_squares), StandardScaler(), glm()),
                  c_space, search_jobs=-1),
        ModelSpec("Random Forest",
                  RandomForestClassifier(n_estimators=400, class_weight="balanced_subsample",
                                         n_jobs=-1, random_state=seed),
                  {"max_features": [0.3, 0.5, 0.7], "min_samples_leaf": [1, 3, 5, 10]},
                  tree_based=True),
        ModelSpec("XGBoost",
                  BoostedTreesClassifier(tree_method="hist", device=DEVICE.type, random_state=seed),
                  {k: v for k, v in BOOSTING_SPACE.items() if k != "n_estimators"} | {"n_estimators": [200, 400]},
                  tree_based=True),
    ]
