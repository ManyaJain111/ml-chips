"""Feature engineering and target transforms.  Owner: P2.

Why this matters: leakage currents span ~10 decades and change sign with Vd vs Vs. Regressing
raw amperes with MSE fits only the largest currents. Predict in a transformed space (SPEC §4.5).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from leakml.constants import CASE_DEVICE, INPUT_COLUMNS, PROCESS_PARAMS, TARGET_COLUMNS, Case

K_OVER_Q = 8.617333262e-5  # Boltzmann constant / electron charge, V/K
T_ZERO_C = 273.15


def build_features(
    df: pd.DataFrame, case: Case, nominal: dict[str, float] | None = None
) -> pd.DataFrame:
    """Map raw input columns (constants.INPUT_COLUMNS) to the model's feature matrix.

    Deterministic and stateless: the same function is used in training and inside the saved
    regressor, and the column order is fixed, so the feature list stored in the model file is
    simply list(result.columns).

    Features (all "NMOS-like": for PMOS every terminal voltage difference is negated, so e.g.
    `vgs` is really v_sg and is positive when the device is ON; `vth` is |vth0|):
      - supply / temperature: vdd, temp_k, inv_temp_k, v_t = kT/q
      - geometry: w, log_w
      - bias: vgs, vds, vbs, vgd, vbd, abs_vds (vg/vb themselves are redundant: 0 or vdd)
      - subthreshold physics: vgs_over_vt, vds_over_vt, overdrive_over_vt = (vgs - vth)/v_t,
        drain_factor = 1 - exp(-|vds|/v_t)
      - process: vth, toxe, ndep, u0, xj, either raw or divided by the device's nominal value
        (mostly in [0.9, 1.1]) when `nominal` is given, plus inv_toxe (gate tunnelling grows
        steeply as the oxide gets thinner)
    """
    case = Case(case)
    missing = [c for c in INPUT_COLUMNS if c not in df.columns]
    if missing:
        raise KeyError(f"missing input columns {missing}")
    X = df.loc[:, list(INPUT_COLUMNS)].astype(float)
    s = 1.0 if CASE_DEVICE[case] == "nmos" else -1.0

    temp_k = X["temp"] + T_ZERO_C
    v_t = K_OVER_Q * temp_k
    vgs = s * (X["vg"] - X["vs"])
    vds = s * (X["vd"] - X["vs"])
    vth = X["vth0"].abs()

    proc = {p: X[p] for p in PROCESS_PARAMS}
    proc["vth0"] = vth
    if nominal is not None:
        proc = {p: proc[p] / abs(float(nominal[p])) for p in PROCESS_PARAMS}

    F = pd.DataFrame(
        {
            "vdd": X["vdd"],
            "temp_k": temp_k,
            "inv_temp_k": 1.0 / temp_k,
            "v_t": v_t,
            "w": X["w"],
            "log_w": np.log(X["w"]),
            "vgs": vgs,
            "vds": vds,
            "vbs": s * (X["vb"] - X["vs"]),
            "vgd": s * (X["vg"] - X["vd"]),
            "vbd": s * (X["vb"] - X["vd"]),
            "abs_vds": vds.abs(),
            "vgs_over_vt": vgs / v_t,
            "vds_over_vt": vds / v_t,
            "overdrive_over_vt": (vgs - vth) / v_t,
            "drain_factor": -np.expm1(-vds.abs() / v_t),
            "vth": proc["vth0"],
            "toxe": proc["toxe"],
            "inv_toxe": 1.0 / proc["toxe"],
            "ndep": proc["ndep"],
            "u0": proc["u0"],
            "xj": proc["xj"],
        },
        index=df.index,
    )
    return F


@dataclass
class TargetTransform:
    """Invertible map between currents (A) and the regression target space.

    Chosen default for the models: kind="asinh" with a per-target scale i0 learned from the
    training data, z = asinh(I / i0_t). Reasons (for the report):
      - id and is change sign in every case (about half the rows in the ON cases, where the
        current goes through zero at vd = vs), and asinh is smooth through zero, so no separate
        sign model is needed;
      - for |I| >> i0 it equals sign(I) * ln(2|I| / i0), i.e. log-like, so an error dz in target
        space is a relative error of about dz in amperes, which is what we are judged on (SPEC §7);
      - the targets live on very different scales (OFF ig ~1e-22 A, ON id up to ~1e-4 A), so one
        fixed i0 cannot be "small" for all of them. i0_t = the `quantile` of |I_t| on the
        training rows keeps ~99 % of every target in the log-like regime.

    kind="log10_abs" (z = log10|I|) is also available, but only for targets whose sign is
    constant in the training data (the sign is then stored and restored). It raises on a target
    with mixed signs instead of silently guessing the sign.

    shape_channel=True (used for the ON cases): before the transform, id and is are divided by
    channel_shape(X) = tanh((vd - vs) / 2vT), and multiplied back on the way out. The channel
    current crosses zero at vd = vs, with the shape (1 - exp(-vds/vT)) in subthreshold and ~vds
    in the linear region; dividing by tanh(vds/2vT) removes that zero crossing and the sign, so
    the model learns a smooth, one-signed remainder. On validation this cut the ON-case mlp p95
    error for id/is from 164/206 % to 34/48 % (nmos_on). It is NOT used for the OFF cases: there
    id/is near vd = vs are dominated by junction leakage, which does not flip with vds, and
    dividing made them worse. Needs X_raw in fit/transform/inverse_transform.
    """

    kind: str = "log10_abs"  # "log10_abs" | "asinh"
    i0: float = 1e-15  # lower bound for the learned per-target asinh scale
    quantile: float = 0.01  # asinh: i0_t = max(i0_min, quantile of |I_t|)
    i0_min: float = 1e-30
    i0_: np.ndarray | None = field(default=None, repr=False)  # fitted, shape (4,)
    sign_: np.ndarray | None = field(default=None, repr=False)  # fitted (log10_abs), shape (4,)
    shape_channel: bool = False  # divide id/is by channel_shape(X) (ON cases)

    @property
    def _shaped(self) -> bool:
        # getattr: models pickled before this field existed load with it missing -> False
        return bool(getattr(self, "shape_channel", False))

    def _y(self, y: pd.DataFrame, X_raw: pd.DataFrame | None = None) -> np.ndarray:
        Y = y.loc[:, list(TARGET_COLUMNS)].to_numpy(dtype=float)
        if self._shaped:
            Y = Y.copy()
            Y[:, _CHANNEL_IDX] /= channel_shape(_require_x(X_raw))[:, None]
        return Y

    def fit(self, y: pd.DataFrame, X_raw: pd.DataFrame | None = None) -> TargetTransform:
        """Learn the per-target scale (asinh) or the per-target sign (log10_abs).

        X_raw (raw inputs, same rows as y) is required when shape_channel is True; y itself
        works too, since the training DataFrame holds both inputs and currents.
        """
        Y = self._y(y, X_raw if X_raw is not None else (y if self._shaped else None))
        if self.kind == "asinh":
            q = np.quantile(np.abs(Y), self.quantile, axis=0)
            self.i0_ = np.maximum(q, self.i0_min)
        elif self.kind == "log10_abs":
            pos, neg = (Y > 0).all(axis=0), (Y < 0).all(axis=0)
            mixed = [t for t, p, n in zip(TARGET_COLUMNS, pos, neg, strict=True) if not (p or n)]
            if mixed:
                raise ValueError(
                    f"log10_abs needs a constant sign per target, but {mixed} change sign "
                    "(or contain zeros) in the training data; use kind='asinh'"
                )
            self.sign_ = np.where(pos, 1.0, -1.0)
        else:
            raise ValueError(f"unknown target transform kind {self.kind!r}")
        return self

    def _check_fitted(self) -> None:
        if (self.kind == "asinh" and self.i0_ is None) or (
            self.kind == "log10_abs" and self.sign_ is None
        ):
            raise RuntimeError("TargetTransform is not fitted; call fit() first")

    def transform(self, y: pd.DataFrame, X_raw: pd.DataFrame | None = None) -> np.ndarray:
        """Currents (A) to target space, shape (n, 4) in TARGET_COLUMNS order."""
        self._check_fitted()
        Y = self._y(y, X_raw if X_raw is not None else (y if self._shaped else None))
        if self.kind == "asinh":
            return np.arcsinh(Y / self.i0_)
        return np.log10(np.abs(Y))

    def inverse_transform(self, z: np.ndarray, X_raw: pd.DataFrame | None = None) -> pd.DataFrame:
        """Target space back to currents (A), DataFrame[TARGET_COLUMNS] (index of X_raw if given).

        X_raw provides the index, and the vd/vs/temp needed to undo shape_channel.
        """
        self._check_fitted()
        Z = np.asarray(z, dtype=float)
        if Z.ndim != 2 or Z.shape[1] != len(TARGET_COLUMNS):
            raise ValueError(f"expected shape (n, {len(TARGET_COLUMNS)}), got {Z.shape}")
        if self.kind == "asinh":
            Y = np.sinh(Z) * self.i0_
        else:
            Y = self.sign_ * np.power(10.0, Z)
        if self._shaped:
            Y = Y.copy()
            Y[:, _CHANNEL_IDX] *= channel_shape(_require_x(X_raw))[:, None]
        index = X_raw.index if X_raw is not None else None
        return pd.DataFrame(Y, index=index, columns=list(TARGET_COLUMNS))


_CHANNEL_IDX = [TARGET_COLUMNS.index("id"), TARGET_COLUMNS.index("is")]
_SHAPE_MIN = 1e-9  # |tanh| floor so vd == vs exactly cannot divide by zero


def channel_shape(X_raw: pd.DataFrame) -> np.ndarray:
    """tanh((vd - vs) / 2vT): the sign and zero crossing of the channel current.

    Raw vd - vs for both devices: channel current flows into the drain terminal when vd > vs,
    for NMOS and PMOS alike. Magnitude floored at 1e-9 (sign kept).
    """
    v_t = K_OVER_Q * (X_raw["temp"].to_numpy(dtype=float) + T_ZERO_C)
    t = np.tanh((X_raw["vd"].to_numpy(dtype=float) - X_raw["vs"].to_numpy(dtype=float)) / (2 * v_t))
    return np.where(t >= 0, 1.0, -1.0) * np.maximum(np.abs(t), _SHAPE_MIN)


def _require_x(X_raw: pd.DataFrame | None) -> pd.DataFrame:
    if X_raw is None:
        raise ValueError("shape_channel=True needs X_raw (vd, vs, temp) for the same rows")
    return X_raw
