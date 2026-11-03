"""STAND-IN transistor model. NOT a deliverable and NOT one of the 4 compared model types.

It exists only so P3/P4 can run the full Part B/C pipeline before P2's real models are ready:

    uv run python scripts/dev_train_baseline.py --config configs/dev.yaml --quick

It conforms to the bundle contract in predictor.py, so swapping in P2's models later changes
nothing downstream. Deliberately simple: ExtraTrees on asinh(I / I0) of a few raw/derived features.
Expect errors of several % or worse. P2 may delete this file once the real models exist.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor

from leakml.constants import INPUT_COLUMNS, TARGET_COLUMNS


def _features(X: pd.DataFrame) -> np.ndarray:
    F = X.loc[:, list(INPUT_COLUMNS)].astype(float).copy()
    F["vds"] = F["vd"] - F["vs"]
    F["vgs"] = F["vg"] - F["vs"]
    F["vbs"] = F["vb"] - F["vs"]
    F["vgd"] = F["vg"] - F["vd"]
    return F.to_numpy()


class AsinhExtraTrees:
    """ExtraTrees multi-output regressor in asinh(I / i0) space (sign-safe, log-like for |I| >> i0)."""

    def __init__(self, i0: float = 1e-15, n_estimators: int = 200, seed: int = 0):
        self.i0 = i0
        self.n_estimators = n_estimators
        self.seed = seed

    def fit(self, X: pd.DataFrame, Y: pd.DataFrame) -> AsinhExtraTrees:
        Z = np.arcsinh(Y.loc[:, list(TARGET_COLUMNS)].to_numpy(dtype=float) / self.i0)
        self.model_ = ExtraTreesRegressor(
            n_estimators=self.n_estimators, random_state=self.seed, n_jobs=-1
        ).fit(_features(X), Z)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.sinh(self.model_.predict(_features(X))) * self.i0
