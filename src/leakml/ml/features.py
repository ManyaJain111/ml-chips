"""Feature engineering and target transforms.  Owner: P2.

Why this matters: leakage currents span ~10 decades and change sign with Vd vs Vs. Regressing
raw amperes with MSE fits only the largest currents. Predict in a transformed space (SPEC §4.5).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from leakml.constants import Case


def build_features(
    df: pd.DataFrame, case: Case, nominal: dict[str, float] | None = None
) -> pd.DataFrame:
    """Map raw input columns (constants.INPUT_COLUMNS) to the model's feature matrix.

    TODO(P2). Ideas, to be judged on the validation set:
      - raw: vdd, temp, w, vd, vs, vth0, toxe, ndep, u0, xj
      - vg and vb are fixed per case (vdd or 0), so they are redundant. Use vgs/vbs instead
      - physics features: vds = vd - vs, |vds|, vgs = vg - vs, vbs = vb - vs, vgd, vbd,
        T_K = temp + 273.15, thermal voltage kT/q, vgs/vT, (|vgs| - |vth0|)/(n*vT),
        w/lmin, log(w), 1 - exp(-|vds|/vT) (subthreshold drain factor)
      - normalise process params by their nominal (p / p_nom), mostly in [0.9, 1.1]
      - for PMOS, flip signs so features are "NMOS-like", e.g. vsg instead of vgs
    The SAME function must be used in training and in predictor.TransistorModel.predict, so the
    fitted feature list is stored inside the saved model file.
    """
    raise NotImplementedError


@dataclass
class TargetTransform:
    """Invertible map between currents (A) and the regression target space.

    TODO(P2): choose and justify one in the report (the brief explicitly mentions log|I|).
      (a) z = log10(|I|), plus sign handled separately, either from physics (e.g. sign(id) =
          sign(vd - vs) for OFF devices, ib is junction-dominated, ...) or by a small classifier
      (b) z = asinh(I / I0) with I0 ~ 1e-15 A: smooth, handles sign and zero, log-like for |I| >> I0
      (c) predict id, ig, ib and set is = -(id + ig + ib) to enforce KCL, but check the accuracy
          of `is`, which is often the smallest current
    """

    kind: str = "log10_abs"  # "log10_abs" | "asinh"
    i0: float = 1e-15

    def fit(self, y: pd.DataFrame) -> TargetTransform:
        """TODO(P2): learn anything needed (for example a per-target i0 from data percentiles)."""
        raise NotImplementedError

    def transform(self, y: pd.DataFrame) -> np.ndarray:
        """TODO(P2): currents (A) to target space, shape (n, 4) in TARGET_COLUMNS order."""
        raise NotImplementedError

    def inverse_transform(self, z: np.ndarray, X_raw: pd.DataFrame | None = None) -> pd.DataFrame:
        """TODO(P2): target space back to currents (A). X_raw is available if sign rules need vd/vs."""
        raise NotImplementedError
