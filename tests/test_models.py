import numpy as np
from sklearn.base import clone
from sklearn.metrics import r2_score

from ebv.models import BoostedTreesClassifier, add_squares, classification_models, regression_models
from ebv.neural import TorchMLPRegressor

RNG = np.random.default_rng(3)


def test_add_squares_appends_squared_columns():
    x = np.array([[1.0, -2.0], [3.0, 0.5]])
    assert np.allclose(add_squares(x), [[1, -2, 1, 4], [3, 0.5, 9, 0.25]])


def test_neural_network_learns_a_smooth_function_and_clones():
    X = RNG.normal(size=(400, 3)).astype("float32")
    y = 2 * X[:, 0] - X[:, 1] ** 2 + RNG.normal(scale=0.1, size=400)
    model = TorchMLPRegressor(hidden_layers=(32, 16), max_epochs=800, random_state=0).fit(X, y)
    assert r2_score(y, model.predict(X)) > 0.9
    assert clone(model).get_params() == model.get_params()


def test_boosted_classifier_balances_rare_presences():
    X = RNG.normal(size=(300, 4))
    y = (X[:, 0] > 1.2).astype(int)                      # ~12% presences
    model = BoostedTreesClassifier(n_estimators=20, tree_method="hist", device="cpu").fit(X, y)
    assert model.get_params()["scale_pos_weight"] == (len(y) - y.sum()) / y.sum()
    assert model.predict_proba(X).shape == (300, 2)


def test_model_zoo_names_are_unique():
    for models in (regression_models(), classification_models()):
        names = [m.name for m in models]
        assert len(names) == len(set(names)) == 4
