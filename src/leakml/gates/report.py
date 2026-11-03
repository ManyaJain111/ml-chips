"""Deliverables 6 and 7: gate results table and gate parity plot.  Owner: P3."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def merge_spice_ml(gate_spice: pd.DataFrame, gate_ml: pd.DataFrame) -> pd.DataFrame:
    """Join on (gate, inputs, corner_id) -> data/gates/gate_results.csv.

    TODO(P3): assert a 1:1 join, with exactly 18*200 rows (or 18*quick_n_corners).
    Rename to i_spice / i_ml for metrics.results_table.
    """
    raise NotImplementedError


def gate_results_table(merged: pd.DataFrame, i_floor: float | None = None) -> pd.DataFrame:
    """Deliverable 6: one row per (gate, inputs), 18 rows plus ALL.

    TODO(P3): metrics.results_table(merged, ["gate", "inputs"]). Columns: SPICE median, ML median,
    bias, median error, p95 error (+ mean accuracy). Write results/tables/gate_results.{csv,md}.
    """
    raise NotImplementedError


def gate_parity_plot(merged: pd.DataFrame, out_path: Path) -> None:
    """Deliverable 7: ML vs SPICE leakage for ALL gate simulations on log axes, coloured by gate.

    TODO(P3): plotting.parity_plot_log(..., hue=merged["gate"]).
    Output: results/plots/gate_parity.{png,pdf}
    """
    raise NotImplementedError


def worst_cases(merged: pd.DataFrame, k: int = 20) -> pd.DataFrame:
    """Top-k |relative error| rows with their corner conditions (temp, vdd, vth0...), for the
    error-analysis section.

    TODO(P3): write results/tables/gate_worst_cases.csv
    """
    raise NotImplementedError
