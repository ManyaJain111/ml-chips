"""Error metrics shared by Part A, B and C.  IMPLEMENTED (shared foundation, maintained by P2).

Definitions: SPEC §7. Every table in the report comes from these functions, so "median error"
means the same thing everywhere.

    summarize(pred, true, i_floor=1e-15)            -> dict with one table row
    results_table(df, ["gate", "inputs"])           -> DataFrame (+ ALL row), cols i_ml / i_spice
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

METRIC_KEYS = (
    "n",
    "n_excluded",
    "spice_median",
    "ml_median",
    "bias_pct",
    "median_err_pct",
    "p95_err_pct",
    "max_err_pct",
    "mean_acc_pct",
    "r2_log",
)


def relative_error(pred: np.ndarray, true: np.ndarray, i_floor: float | None = None) -> np.ndarray:
    """Signed relative error (pred - true) / |true|.

    NaN where true == 0, where either value is non-finite, or where |true| < i_floor.
    """
    pred = np.asarray(pred, dtype=float)
    true = np.asarray(true, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        e = (pred - true) / np.abs(true)
    bad = ~np.isfinite(e)
    if i_floor is not None:
        bad |= np.abs(true) < i_floor
    return np.where(bad, np.nan, e)


def r2_log(pred: np.ndarray, true: np.ndarray) -> float:
    """R^2 of log10|pred| against log10|true| (rows with a zero or non-finite value are ignored)."""
    pred = np.abs(np.asarray(pred, dtype=float))
    true = np.abs(np.asarray(true, dtype=float))
    ok = np.isfinite(pred) & np.isfinite(true) & (pred > 0) & (true > 0)
    if ok.sum() < 2:
        return float("nan")
    lp, lt = np.log10(pred[ok]), np.log10(true[ok])
    ss_tot = np.sum((lt - lt.mean()) ** 2)
    if ss_tot == 0:
        return float("nan")
    return float(1.0 - np.sum((lt - lp) ** 2) / ss_tot)


def summarize(
    pred: np.ndarray, true: np.ndarray, *, i_floor: float | None = None
) -> dict[str, float]:
    """One results-table row (keys = METRIC_KEYS). Percentages are in %."""
    pred = np.asarray(pred, dtype=float)
    true = np.asarray(true, dtype=float)
    e = relative_error(pred, true, i_floor)
    valid = np.isfinite(e)
    ae = np.abs(e[valid])
    has = ae.size > 0
    nan = float("nan")
    return {
        "n": int(true.size),
        "n_excluded": int((~valid).sum()),
        "spice_median": float(np.median(true)) if true.size else nan,
        "ml_median": float(np.median(pred)) if pred.size else nan,
        "bias_pct": 100.0 * float(np.median(e[valid])) if has else nan,
        "median_err_pct": 100.0 * float(np.median(ae)) if has else nan,
        "p95_err_pct": 100.0 * float(np.percentile(ae, 95)) if has else nan,
        "max_err_pct": 100.0 * float(ae.max()) if has else nan,
        "mean_acc_pct": 100.0 * (1.0 - float(ae.mean())) if has else nan,
        "r2_log": r2_log(pred[valid], true[valid]),
    }


def results_table(
    df: pd.DataFrame,
    group_cols: Sequence[str],
    pred_col: str = "i_ml",
    true_col: str = "i_spice",
    i_floor: float | None = None,
    add_all: bool = True,
) -> pd.DataFrame:
    """summarize() per group (first-appearance order), plus an 'ALL' row over every row of df.

    Deliverable 6: results_table(gate_results, ["gate", "inputs"])
    Deliverable 8: results_table(circuit_results, ["circuit"])
    Part A:        results_table(long_df, ["case", "model", "target"], "pred", "true")
    """
    group_cols = list(group_cols)
    rows = []
    for key, g in df.groupby(group_cols, sort=False):
        key = key if isinstance(key, tuple) else (key,)
        rows.append(
            {
                **dict(zip(group_cols, key, strict=True)),
                **summarize(g[pred_col].to_numpy(), g[true_col].to_numpy(), i_floor=i_floor),
            }
        )
    if add_all:
        rows.append(
            {
                **{c: "ALL" for c in group_cols},
                **summarize(df[pred_col].to_numpy(), df[true_col].to_numpy(), i_floor=i_floor),
            }
        )
    return pd.DataFrame(rows, columns=[*group_cols, *METRIC_KEYS])
