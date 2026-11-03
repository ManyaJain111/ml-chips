"""The 4 model types compared in Deliverable 4.  Owner: P2.

Default line-up (can change, but keep it to 4 and record the hyperparameters in the report):
    poly_ridge     PolynomialFeatures(deg 2-4) + StandardScaler + Ridge  (interpretable baseline)
    random_forest  RandomForestRegressor / ExtraTreesRegressor
    gbdt           LightGBM (one regressor per target, or MultiOutputRegressor). For XGBoost,
                   `uv add xgboost-cpu`, not `xgboost`, which pulls ~200 MB of CUDA libs
    mlp            sklearn MLPRegressor on standardised features (or torch via `uv sync --group nn`)
Alternatives worth trying if one under-performs: GaussianProcessRegressor (great for smooth
targets at this sample size), KNN, SVR.
"""

from __future__ import annotations

from typing import Any

MODEL_TYPES: tuple[str, ...] = ("poly_ridge", "random_forest", "gbdt", "mlp")


def make_model(name: str, seed: int, **overrides: Any):
    """Return an UNFITTED sklearn-compatible multi-output regressor (fit(X, Z) with Z shape (n, 4)).

    TODO(P2):
      - wrap in a sklearn Pipeline (scaler + estimator) so that saving and loading is one object
      - pass random_state=seed wherever the estimator takes one
      - apply **overrides on top of the defaults from default_hyperparams(name)
    """
    raise NotImplementedError


def default_hyperparams(name: str) -> dict[str, Any]:
    """Sensible defaults per model type. Record them in the report's Models section.

    TODO(P2)
    """
    raise NotImplementedError


def search_space(name: str) -> dict[str, list[Any]]:
    """Small grid / random-search space, tuned on the VALIDATION split only (never on test).

    TODO(P2): keep it small. 10k rows x 4 cases x 4 models has to stay fast enough to re-run.
    """
    raise NotImplementedError
