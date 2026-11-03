"""Deliverable 8: test-circuit results table + parity plot.  Owner: P4."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def merge_spice_ml(circuit_spice: pd.DataFrame, circuit_ml: pd.DataFrame) -> pd.DataFrame:
    """TODO(P4): join on (circuit, vector, corner_id) and assert 1:1 -> data/circuit/circuit_results.csv."""
    raise NotImplementedError


def circuit_results_table(merged: pd.DataFrame, i_floor: float | None = None) -> pd.DataFrame:
    """Same metrics as the gate table (SPICE median, ML median, bias, median err, p95 err).

    TODO(P4): overall row + optionally per-vector-class rows (e.g. by number of ones in the
    vector). Write results/tables/circuit_results.{csv,md} and include the gate-level ALL row
    next to it for comparison.
    """
    raise NotImplementedError


def circuit_parity_plot(merged: pd.DataFrame, out_path: Path) -> None:
    """TODO(P4): plotting.parity_plot_log, coloured by temperature -> results/plots/circuit_parity.{png,pdf}."""
    raise NotImplementedError


def loading_error_analysis(merged: pd.DataFrame) -> pd.DataFrame:
    """Error analysis: how much of the circuit-level error comes from ignoring inter-gate loading
    and non-ideal (not exactly 0/VDD) internal logic levels?

    TODO(P4, nice-to-have): correlate the error with corner temp/vdd/vth0 and with the gate mix
    of each vector.
    """
    raise NotImplementedError
