"""The 4 model types compared in Deliverable 4.  Owner: P2.

Line-up (record the hyperparameters in the report):
    poly_ridge     StandardScaler + PolynomialFeatures(degree) + StandardScaler + Ridge
                   (interpretable baseline; Ridge is natively multi-output)
    random_forest  RandomForestRegressor (natively multi-output)
    gbdt           LightGBM, one LGBMRegressor per target via MultiOutputRegressor
    mlp            MLPRegressor on standardised features AND standardised targets
                   (TransformedTargetRegressor), early stopping on an internal 10 % hold-out
                   (patience 150 epochs)

Every model is a single sklearn object: fit(X, Z) with X the build_features() matrix and Z the
(n, 4) target-space array from TargetTransform; predict(X) -> (n, 4). Each is seeded with
random_state=seed (hard rule 3). Hyperparameters are flat dicts whose keys are the ones in
default_hyperparams(name); search_space(name) uses the same keys.
"""

from __future__ import annotations

from typing import Any

from lightgbm import LGBMRegressor
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

MODEL_TYPES: tuple[str, ...] = ("poly_ridge", "random_forest", "gbdt", "mlp")

_DEFAULTS: dict[str, dict[str, Any]] = {
    "poly_ridge": {"degree": 3, "alpha": 1e-4},
    "random_forest": {
        "n_estimators": 300,
        "max_features": 1.0,
        "min_samples_leaf": 1,
        "n_jobs": -1,
    },
    "gbdt": {
        "n_estimators": 1000,
        "learning_rate": 0.05,
        "num_leaves": 31,
        "min_child_samples": 10,
        "subsample": 0.8,
        "subsample_freq": 1,
        "colsample_bytree": 0.8,
        "n_jobs": -1,
    },
    "mlp": {
        "hidden_layer_sizes": (128, 128),
        "alpha": 1e-4,
        "learning_rate_init": 1e-3,
        "batch_size": 128,
        "max_iter": 2000,
        "early_stopping": True,
        "validation_fraction": 0.1,
        "n_iter_no_change": 150,  # 30 stopped at ~150-200 epochs; 150 cut val p95 by ~1/3
    },
}

_SEARCH: dict[str, dict[str, list[Any]]] = {
    "poly_ridge": {"degree": [2, 3], "alpha": [1e-6, 1e-4, 1e-2]},
    "random_forest": {"max_features": [0.33, 0.66, 1.0], "min_samples_leaf": [1, 3]},
    "gbdt": {"num_leaves": [15, 31, 63], "learning_rate": [0.03, 0.1]},
    "mlp": {
        "hidden_layer_sizes": [(128, 128), (128, 128, 128), (256, 256, 256)],
        "alpha": [1e-5, 1e-3],
    },
}


def _check_name(name: str) -> None:
    if name not in MODEL_TYPES:
        raise ValueError(f"unknown model type {name!r}, expected one of {MODEL_TYPES}")


def make_model(name: str, seed: int, **overrides: Any):
    """Return an UNFITTED sklearn-compatible multi-output regressor (fit(X, Z) with Z shape (n, 4)).

    `overrides` are applied on top of default_hyperparams(name); unknown keys raise, so a typo in
    a search space cannot be silently ignored.
    """
    p = default_hyperparams(name)
    unknown = sorted(set(overrides) - set(p))
    if unknown:
        raise ValueError(f"{name}: unknown hyperparameters {unknown}, expected keys of {sorted(p)}")
    p.update(overrides)

    if name == "poly_ridge":
        return Pipeline(
            [
                ("scale_in", StandardScaler()),
                ("poly", PolynomialFeatures(degree=p["degree"], include_bias=False)),
                ("scale_poly", StandardScaler()),
                ("model", Ridge(alpha=p["alpha"], random_state=seed)),
            ]
        )
    if name == "random_forest":
        return Pipeline(
            [
                ("scale_in", StandardScaler()),
                ("model", RandomForestRegressor(random_state=seed, **p)),
            ]
        )
    if name == "gbdt":
        lgbm = LGBMRegressor(random_state=seed, verbose=-1, deterministic=True, **p)
        return Pipeline([("scale_in", StandardScaler()), ("model", MultiOutputRegressor(lgbm))])
    # mlp
    net = Pipeline(
        [("scale_in", StandardScaler()), ("model", MLPRegressor(random_state=seed, **p))]
    )
    return TransformedTargetRegressor(regressor=net, transformer=StandardScaler())


def default_hyperparams(name: str) -> dict[str, Any]:
    """Defaults per model type (a fresh copy). Record them in the report's Models section."""
    _check_name(name)
    return dict(_DEFAULTS[name])


def search_space(name: str) -> dict[str, list[Any]]:
    """Small grid tuned on the VALIDATION split only (never on test). Keys match the defaults.

    Grid sizes: poly_ridge 6, random_forest 6, gbdt 6, mlp 6 (24 fits per case).
    """
    _check_name(name)
    return {k: list(v) for k, v in _SEARCH[name].items()}
