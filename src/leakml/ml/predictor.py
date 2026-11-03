"""Inference API for the transistor models.  IMPLEMENTED (interface P2 -> P3/P4, SPEC §4.6).

Model file format ("bundle", SPEC §4.6) = joblib dump of a dict:

    {
      "format":    "leakml.transistor_model/v1",
      "case":      "nmos_off",
      "regressor": <object with .predict(X: DataFrame[INPUT_COLUMNS]) -> ndarray (n, 4)>,
      "meta":      {"model_name": ..., "train_ranges": {col: [lo, hi]}, ...},
    }

The regressor owns ALL preprocessing (features, target transform, inverse transform) and returns
currents in amperes, TARGET_COLUMNS order (id, ig, is, ib), with the into-device sign. Its class must
live inside the leakml package (not in a script) so that joblib can unpickle it.

P2 writes bundles with save_bundle(). P3/P4 only ever do:

    bank = TransistorModelBank.load_all(cfg.paths.models_dir)
    cases = TransistorModelBank.select_cases(devices, vg, vdd)
    I = bank.predict(X, cases)          # DataFrame[id, ig, is, ib], same index as X
"""

from __future__ import annotations

import logging
import warnings
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol

import joblib
import numpy as np
import pandas as pd

from leakml.constants import CASES, INPUT_COLUMNS, TARGET_COLUMNS, Case

log = logging.getLogger(__name__)

MODEL_FORMAT = "leakml.transistor_model/v1"
RANGE_TOL = 1e-3  # relative slack before an input counts as extrapolation


class CurrentRegressor(Protocol):
    def predict(self, X: pd.DataFrame) -> np.ndarray: ...


def input_ranges(X: pd.DataFrame) -> dict[str, list[float]]:
    """{col: [min, max]} over INPUT_COLUMNS. Store it in meta["train_ranges"] for extrapolation warnings."""
    return {c: [float(X[c].min()), float(X[c].max())] for c in INPUT_COLUMNS}


def save_bundle(
    path: Path | str, case: Case | str, regressor: CurrentRegressor, meta: Mapping[str, Any]
) -> Path:
    """Write a model bundle (see module docstring). Used by P2's train.save_model and the baseline."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    bundle = {
        "format": MODEL_FORMAT,
        "case": Case(case).value,
        "regressor": regressor,
        "meta": dict(meta),
    }
    joblib.dump(bundle, path)
    return path


class TransistorModel:
    """One trained model for one case. It behaves as a pure function of its inputs."""

    def __init__(
        self, case: Case | str, regressor: CurrentRegressor, meta: Mapping[str, Any] | None = None
    ):
        self.case = Case(case)
        self.regressor = regressor
        self.meta = dict(meta or {})

    @classmethod
    def load(cls, path: Path | str) -> TransistorModel:
        path = Path(path)
        bundle = joblib.load(path)
        if not isinstance(bundle, dict) or bundle.get("format") != MODEL_FORMAT:
            raise ValueError(f"{path} is not a {MODEL_FORMAT} bundle (use predictor.save_bundle)")
        return cls(bundle["case"], bundle["regressor"], bundle.get("meta"))

    def _warn_extrapolation(self, X: pd.DataFrame) -> None:
        ranges = self.meta.get("train_ranges")
        if not ranges:
            return
        for col, (lo, hi) in ranges.items():
            slack = RANGE_TOL * max(abs(hi - lo), abs(hi), 1e-30)
            out = ((X[col] < lo - slack) | (X[col] > hi + slack)).mean()
            if out > 0:
                warnings.warn(
                    f"{self.case.value}: {out:.1%} of rows have {col} outside the training range "
                    f"[{lo:.4g}, {hi:.4g}] (extrapolation)",
                    stacklevel=3,
                )

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        """INPUT_COLUMNS (raw values, this case's device) -> DataFrame[id, ig, is, ib] in A, into-device sign."""
        missing = [c for c in INPUT_COLUMNS if c not in X.columns]
        if missing:
            raise KeyError(f"missing input columns {missing}")
        Xin = X.loc[:, list(INPUT_COLUMNS)].astype(float)
        if len(Xin) == 0:
            return pd.DataFrame(columns=list(TARGET_COLUMNS), index=X.index, dtype=float)
        self._warn_extrapolation(Xin)
        Y = np.asarray(self.regressor.predict(Xin), dtype=float)
        if Y.shape != (len(Xin), len(TARGET_COLUMNS)):
            raise ValueError(f"regressor returned shape {Y.shape}, expected {(len(Xin), 4)}")
        if not np.isfinite(Y).all():
            raise ValueError(f"{self.case.value}: regressor returned non-finite currents")
        return pd.DataFrame(Y, index=X.index, columns=list(TARGET_COLUMNS))


class TransistorModelBank:
    """All four case models together, with vectorised prediction over mixed cases."""

    def __init__(self, models: Mapping[Case, TransistorModel]):
        self.models = {Case(k): v for k, v in models.items()}

    @classmethod
    def load_all(cls, model_dir: Path | str) -> TransistorModelBank:
        model_dir = Path(model_dir)
        paths = {case: model_dir / f"{case.value}.joblib" for case in CASES}
        missing = [str(p) for p in paths.values() if not p.is_file()]
        if missing:
            raise FileNotFoundError(
                f"missing model files: {missing}. Train them (scripts/04-05) or, for development, "
                "create stand-ins with scripts/dev_train_baseline.py"
            )
        models = {case: TransistorModel.load(p) for case, p in paths.items()}
        for case, m in models.items():
            if m.case != case:
                raise ValueError(f"{paths[case]} contains a model for {m.case}, not {case}")
            log.info("loaded %s: %s", case.value, m.meta.get("model_name", "?"))
        return cls(models)

    def predict(
        self, X: pd.DataFrame, cases: Sequence[Case | str] | pd.Series | np.ndarray
    ) -> pd.DataFrame:
        """Predict rows that belong to different cases in one call (one model call per case)."""
        codes = np.asarray([Case(c).value for c in cases])
        if len(codes) != len(X):
            raise ValueError("len(cases) must equal len(X)")
        out = np.full((len(X), len(TARGET_COLUMNS)), np.nan)
        for case in np.unique(codes):
            mask = codes == case
            out[mask] = self.models[Case(case)].predict(X.iloc[mask]).to_numpy()
        return pd.DataFrame(out, index=X.index, columns=list(TARGET_COLUMNS))

    @staticmethod
    def select_case(device: str, vg: float, vdd: float) -> Case:
        """Case from the device type and the logic level of its gate voltage (threshold vdd/2)."""
        high = vg > 0.5 * vdd
        if device == "nmos":
            return Case.NMOS_ON if high else Case.NMOS_OFF
        if device == "pmos":
            return Case.PMOS_OFF if high else Case.PMOS_ON
        raise ValueError(f"device must be 'nmos' or 'pmos', got {device!r}")

    @staticmethod
    def select_cases(
        devices: Sequence[str], vg: Sequence[float], vdd: Sequence[float]
    ) -> list[Case]:
        """Vectorised select_case."""
        return [
            TransistorModelBank.select_case(d, g, v)
            for d, g, v in zip(devices, vg, vdd, strict=True)
        ]


def snapped_gate_voltage(case: Case | str, vdd: float) -> float:
    """The exact vg the model for `case` was trained with (0 or vdd)."""
    gate_high = Case(case) in (Case.NMOS_ON, Case.PMOS_OFF)
    return float(vdd) if gate_high else 0.0
