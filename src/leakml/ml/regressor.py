"""The saved transistor-model object (the bundle's "regressor", SPEC §4.6).  Owner: P2.

TransistorRegressor owns ALL preprocessing, so P3/P4 only ever call predict() on raw inputs:

    raw X (INPUT_COLUMNS) -> features.build_features -> fitted estimator -> target space Z
                          -> TargetTransform.inverse_transform -> currents in A (id, ig, is, ib)

It lives inside the leakml package so joblib can unpickle it from any script.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from leakml.constants import INPUT_COLUMNS, TARGET_COLUMNS, Case
from leakml.ml.features import TargetTransform, build_features

_I = {t: TARGET_COLUMNS.index(t) for t in TARGET_COLUMNS}


def is_from_kcl(Y: np.ndarray) -> np.ndarray:
    """Copy of Y (n, 4, TARGET_COLUMNS order) with is replaced by -(id + ig + ib).

    Kirchhoff: the four terminal currents of a FET sum to zero. id and ib are predicted much more
    accurately than is (in the OFF cases is is tiny and changes sign), so deriving is from them
    can beat predicting it directly. fit_one decides per case on validation.
    """
    Y = np.array(Y, dtype=float, copy=True)
    Y[:, _I["is"]] = -(Y[:, _I["id"]] + Y[:, _I["ig"]] + Y[:, _I["ib"]])
    return Y


class TransistorRegressor:
    """features + fitted estimator + target transform, as one picklable object."""

    def __init__(
        self,
        case: Case | str,
        estimator: Any,
        target_transform: TargetTransform,
        feature_names: list[str],
        nominal: dict[str, float] | None = None,
        is_kcl: bool = False,
    ):
        self.case = Case(case)
        self.is_kcl = bool(is_kcl)
        self.estimator = estimator
        self.target_transform = target_transform
        self.feature_names = list(feature_names)
        self.nominal = dict(nominal) if nominal is not None else None

    def features(self, X: pd.DataFrame) -> pd.DataFrame:
        F = build_features(X, self.case, self.nominal)
        if list(F.columns) != self.feature_names:
            raise ValueError(
                f"{self.case.value}: build_features now returns {list(F.columns)}, but this model "
                f"was trained on {self.feature_names}. Retrain the model."
            )
        return F

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """X: DataFrame[INPUT_COLUMNS], raw values -> (n, 4) currents in A, TARGET_COLUMNS order."""
        X = X.loc[:, list(INPUT_COLUMNS)]
        Z = np.asarray(self.estimator.predict(self.features(X)), dtype=float)
        if Z.ndim == 1:
            Z = Z.reshape(-1, 1)
        Y = self.target_transform.inverse_transform(Z, X).to_numpy()
        # getattr: models pickled before is_kcl existed predict is directly
        return is_from_kcl(Y) if getattr(self, "is_kcl", False) else Y

    def __repr__(self) -> str:
        return f"TransistorRegressor({self.case.value}, {type(self.estimator).__name__})"
