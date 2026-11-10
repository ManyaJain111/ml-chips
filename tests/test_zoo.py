"""Tests for leakml.ml.zoo (P2)."""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from leakml.ml import zoo


def _data(n: int = 200, seed: int = 0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 6))
    Z = np.column_stack([X[:, 0] + X[:, 1] ** 2, np.sin(X[:, 2]), X[:, 3] * X[:, 4], -X[:, 5]])
    return X, Z


FAST = {
    "poly_ridge": {"degree": 2},
    "random_forest": {"n_estimators": 20},
    "gbdt": {"n_estimators": 30},
    "mlp": {"hidden_layer_sizes": (16,), "max_iter": 200},
}


@pytest.mark.filterwarnings("ignore::sklearn.exceptions.ConvergenceWarning")  # tiny test MLP
@pytest.mark.parametrize("name", zoo.MODEL_TYPES)
def test_every_model_fits_multioutput_and_is_seeded(name):
    X, Z = _data()
    p1 = zoo.make_model(name, seed=3, **FAST[name]).fit(X, Z).predict(X)
    p2 = zoo.make_model(name, seed=3, **FAST[name]).fit(X, Z).predict(X)
    assert p1.shape == (len(X), 4) and np.isfinite(p1).all()
    np.testing.assert_allclose(p1, p2, rtol=1e-12, atol=1e-12)  # same seed -> same model
    # (not bit-equal: parallel tree averaging sums in a different order, ~1e-16 noise)


@pytest.mark.parametrize("name", zoo.MODEL_TYPES)
def test_search_space_keys_are_valid_overrides(name):
    space = zoo.search_space(name)
    assert space and set(space) <= set(zoo.default_hyperparams(name))
    for combo in itertools.product(*space.values()):
        zoo.make_model(name, seed=0, **dict(zip(space, combo, strict=True)))


def test_overrides_and_errors():
    m = zoo.make_model("random_forest", seed=1, n_estimators=7)
    assert m.named_steps["model"].n_estimators == 7
    assert m.named_steps["model"].random_state == 1
    with pytest.raises(ValueError, match="unknown hyperparameters"):
        zoo.make_model("gbdt", seed=0, n_estimator=5)
    with pytest.raises(ValueError, match="unknown model type"):
        zoo.make_model("svr", seed=0)
    d = zoo.default_hyperparams("mlp")
    d["alpha"] = 99.0
    assert zoo.default_hyperparams("mlp")["alpha"] != 99.0  # defaults are copied
