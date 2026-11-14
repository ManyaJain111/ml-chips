"""Deliverables 6 and 7: gate results table and gate parity plot.  Owner: P3.

  merge_spice_ml      gate_spice + gate_ml -> data/gates/gate_results.csv (1:1, checked)
  gate_results_table  D6: one row per (gate, inputs) + ALL, metrics per SPEC §7
  gate_parity_plot    D7: ML vs SPICE i_vdd on log-log axes, coloured by gate
  worst_cases         the k rows with the largest |relative error|, for the error analysis
  stand_in_sources    why a gate_ml.csv / model folder must not become a deliverable (hard rule 13)

gate_results.csv keeps the SPEC §5.4 column names (i_vdd_spice, i_vdd_ml, ...) rather than renaming
to i_spice / i_ml: it carries several current pairs (KCL, direct, i_in), and circuit/report.py
accepts either naming.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from leakml.constants import GATE_INPUT_COMBOS, GATES
from leakml.metrics import relative_error, results_table
from leakml.ml.predictor import TransistorModel
from leakml.plotting import parity_plot_log

KEY = ["gate", "inputs", "corner_id"]
N_GATE_CASES = sum(len(GATE_INPUT_COMBOS[g]) for g in GATES)  # 18

# Columns of the D6 table, in order (SPEC §7 metrics + the extras below).
TABLE_COLUMNS = [
    "gate",
    "inputs",
    "n",
    "n_excluded",
    "spice_median",
    "ml_median",
    "bias_pct",
    "median_err_pct",
    "p95_err_pct",
    "max_err_pct",
    "mean_acc_pct",
    "within_1pct",  # % of (non-excluded) rows with |e| <= 1 %
    "within_10pct",  # % of (non-excluded) rows with |e| <= 10 %
    "median_err_direct_pct",  # same metric for i_vdd_ml_direct (method comparison)
    "median_err_i_in_pct",  # i_in_ml vs i_in_spice, rows with |i_in_spice| >= i_floor only
]


def _key_list(df: pd.DataFrame, n: int = 10) -> str:
    rows = df[KEY].head(n).to_records(index=False).tolist()
    more = f" ... (+{len(df) - n} more)" if len(df) > n else ""
    return f"{rows}{more}"


def _check_keys(df: pd.DataFrame, name: str) -> None:
    missing = [c for c in KEY if c not in df.columns]
    if missing:
        raise ValueError(f"{name} has no column(s) {missing}")
    if df["inputs"].map(type).ne(str).any():
        bad = df.loc[df["inputs"].map(type).ne(str), KEY]
        raise TypeError(
            f"{name}.inputs must be str (read CSVs with leakml.io.read_csv, hard rule 12); "
            f"non-str keys: {_key_list(bad)}"
        )
    dup = df[df.duplicated(KEY, keep=False)]
    if len(dup):
        raise ValueError(f"{name} has {len(dup)} rows with duplicated keys: {_key_list(dup)}")


def merge_spice_ml(
    gate_spice: pd.DataFrame, gate_ml: pd.DataFrame, n_corners: int | None = None
) -> pd.DataFrame:
    """Join on (gate, inputs, corner_id) -> data/gates/gate_results.csv.

    Strict 1:1: `inputs` must be str on both sides (an int 1 never silently matches "01"), no
    duplicated keys, every key present on both sides, and exactly 18 * n_corners rows
    (n_corners defaults to the number of distinct corner_ids in gate_spice). Raises with the
    offending keys otherwise. Rows come out in GATES / GATE_INPUT_COMBOS / corner_id order.
    """
    _check_keys(gate_spice, "gate_spice")
    _check_keys(gate_ml, "gate_ml")
    both = [c for c in gate_ml.columns if c in gate_spice.columns and c not in KEY]
    if both:
        raise ValueError(f"gate_spice and gate_ml both have column(s) {both}")

    m = gate_spice.merge(gate_ml, on=KEY, how="outer", indicator=True)
    for side, label in (("left_only", "gate_ml"), ("right_only", "gate_spice")):
        miss = m[m["_merge"] == side]
        if len(miss):
            raise ValueError(f"{len(miss)} keys missing from {label}: {_key_list(miss)}")
    m = m.drop(columns="_merge")

    if n_corners is None:
        n_corners = int(gate_spice["corner_id"].nunique())
    expected = N_GATE_CASES * n_corners
    if len(m) != expected:
        raise ValueError(
            f"merged gate results have {len(m)} rows, expected {N_GATE_CASES} x {n_corners} = "
            f"{expected}"
        )
    valid = {(g, b) for g in GATES for b in GATE_INPUT_COMBOS[g]}
    unknown = m[[(g, b) not in valid for g, b in zip(m["gate"], m["inputs"], strict=True)]]
    if len(unknown):
        raise ValueError(
            f"{len(unknown)} rows with an unknown (gate, inputs): {_key_list(unknown)}"
        )
    per_case = m.groupby(["gate", "inputs"]).size()
    if (per_case != n_corners).any():
        raise ValueError(f"(gate, inputs) without exactly {n_corners} corners:\n{per_case}")

    order = {(g, b): i for i, (g, b) in enumerate(sorted(valid, key=_case_rank))}
    m["_o"] = [order[(g, b)] for g, b in zip(m["gate"], m["inputs"], strict=True)]
    return m.sort_values(["_o", "corner_id"]).drop(columns="_o").reset_index(drop=True)


def _case_rank(gb: tuple[str, str]) -> tuple[int, int]:
    g, b = gb
    return GATES.index(g), GATE_INPUT_COMBOS[g].index(b)


def _extras(g: pd.DataFrame, i_floor: float | None) -> dict[str, float]:
    e = np.abs(relative_error(g["i_vdd_ml"], g["i_vdd_spice"], i_floor))
    e = e[np.isfinite(e)]
    out = {
        "within_1pct": 100.0 * float(np.mean(e <= 0.01)) if e.size else np.nan,
        "within_10pct": 100.0 * float(np.mean(e <= 0.10)) if e.size else np.nan,
    }
    for col, pred, true in (
        ("median_err_direct_pct", "i_vdd_ml_direct", "i_vdd_spice"),
        ("median_err_i_in_pct", "i_in_ml", "i_in_spice"),
    ):
        if pred in g and true in g:
            ee = np.abs(relative_error(g[pred], g[true], i_floor))
            ee = ee[np.isfinite(ee)]
            out[col] = 100.0 * float(np.median(ee)) if ee.size else np.nan
        else:
            out[col] = np.nan
    return out


def gate_results_table(merged: pd.DataFrame, i_floor: float | None = None) -> pd.DataFrame:
    """Deliverable 6: one row per (gate, inputs), 18 rows plus ALL (columns: TABLE_COLUMNS).

    The SPEC §7 metrics compare i_vdd_ml (KCL) with i_vdd_spice; rows with |i_vdd_spice| < i_floor
    are excluded and counted in n_excluded. The extra columns: % of rows within 1 % / 10 %, the
    median error of i_vdd_ml_direct, and the median error of i_in_ml (rows with
    |i_in_spice| >= i_floor; NaN where there are none, e.g. if every gate input current is tiny).
    """
    base = results_table(merged, ["gate", "inputs"], "i_vdd_ml", "i_vdd_spice", i_floor=i_floor)
    extras = [_extras(g, i_floor) for _, g in merged.groupby(["gate", "inputs"], sort=False)] + [
        _extras(merged, i_floor)
    ]
    table = pd.concat([base, pd.DataFrame(extras)], axis=1)
    return table[TABLE_COLUMNS]


def gate_parity_plot(
    merged: pd.DataFrame,
    out_path: Path,
    pred_col: str = "i_vdd_ml",
    i_floor: float | None = None,
) -> dict[str, float]:
    """Deliverable 7: ML vs SPICE leakage for ALL gate simulations on log axes, coloured by gate.

    pred_col="i_vdd_ml" is the KCL composition (D7); "i_vdd_ml_direct" draws the direct-sum
    comparison. y = x plus ±1 % and ±10 % bands. Writes <out_path>.png and .pdf, returns the
    metrics printed in the plot's corner box.
    """
    method = {"i_vdd_ml": "KCL", "i_vdd_ml_direct": "direct sum"}.get(pred_col, pred_col)
    n_corners = merged["corner_id"].nunique()
    return parity_plot_log(
        merged["i_vdd_spice"].to_numpy(),
        merged[pred_col].to_numpy(),
        out_path,
        title=(
            f"Gate leakage: ML ({method}) vs SPICE\n"
            f"5 gates, {N_GATE_CASES} input cases x {n_corners} PVT corners"
        ),
        hue=merged["gate"].to_numpy(),
        hue_label="gate",
        bands_pct=(1.0, 10.0),
        i_floor=i_floor,
        quantity="leakage |I_VDD| (A)",
    )


def worst_cases(merged: pd.DataFrame, k: int = 20, i_floor: float | None = None) -> pd.DataFrame:
    """Top-k |relative error| rows (KCL i_vdd_ml) with their corner conditions.

    Columns: gate, inputs, corner_id, temp, vdd, i_vdd_spice, i_vdd_ml, err_pct (signed,
    100 * (ML - SPICE) / |SPICE|), abs_err_pct. Sorted by abs_err_pct, largest first.
    """
    e = relative_error(merged["i_vdd_ml"], merged["i_vdd_spice"], i_floor)
    out = merged[[*KEY, "temp", "vdd", "i_vdd_spice", "i_vdd_ml"]].copy()
    out["err_pct"] = 100.0 * e
    out["abs_err_pct"] = np.abs(out["err_pct"])
    out = out[np.isfinite(out["abs_err_pct"])]
    return (
        out.sort_values("abs_err_pct", ascending=False, kind="stable")
        .head(k)
        .reset_index(drop=True)
    )


def stand_in_sources(gate_ml: pd.DataFrame, models_dir: Path) -> list[str]:
    """Reasons these gate results come from stand-ins (empty list = OK to publish).

    A placeholder gate_ml.csv (scripts/dev_placeholder_gate_ml.py) has a `source` column; a
    stand-in model (scripts/dev_train_baseline.py) has meta["stand_in"]. Hard rule 13.
    """
    reasons = []
    if "source" in gate_ml.columns:
        reasons.append(f"gate_ml.csv is a placeholder (source={sorted(set(gate_ml['source']))})")
    if "i_vdd_ml_direct" not in gate_ml.columns:
        reasons.append("gate_ml.csv has no i_vdd_ml_direct column (not written by step 08)")
    for f in sorted(Path(models_dir).glob("*.joblib")):
        if (TransistorModel.load(f).meta or {}).get("stand_in"):
            reasons.append(f"stand-in model {f.name}")
    return reasons
