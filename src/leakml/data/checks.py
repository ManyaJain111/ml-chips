"""Deliverable 3: Kirchhoff check and the three trend plots.  Owner: P1."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from leakml.constants import Case


def kirchhoff_residual(df: pd.DataFrame) -> pd.Series:
    """Per-row KCL residual  r = id + ig + is + ib  (should be ~0, since currents are INTO the device).

    TODO(P1): also return, or make available, the normalised residual
    |r| / max(|id|, |ig|, |is|, |ib|).
    """
    raise NotImplementedError


def kirchhoff_table(datasets: dict[Case, pd.DataFrame]) -> pd.DataFrame:
    """One row per case: median, p95 and max of |r| and of the normalised residual.

    TODO(P1): write results/tables/kirchhoff.csv and .md. A large residual means a sign or
    parse bug (most often a sign flip on one terminal).
    """
    raise NotImplementedError


def plot_off_current_vs_temperature(datasets: dict[Case, pd.DataFrame], out_dir: Path) -> None:
    """OFF current (|id| of nmos_off and pmos_off) against temperature, log y axis.

    TODO(P1): to show a clean trend, restrict to |vd - vs| > 0.5*vdd, or colour by |vds|.
    Expect roughly exponential growth with T (subthreshold + junction leakage).
    Use plotting.trend_scatter. Output: results/plots/trend_ioff_vs_temp.{png,pdf}
    """
    raise NotImplementedError


def plot_off_current_vs_vth0(datasets: dict[Case, pd.DataFrame], out_dir: Path) -> None:
    """OFF current against vth0 (use |vth0| for PMOS). Expect exponential decrease.

    TODO(P1): output results/plots/trend_ioff_vs_vth0.{png,pdf}
    """
    raise NotImplementedError


def plot_gate_current_vs_toxe(datasets: dict[Case, pd.DataFrame], out_dir: Path) -> None:
    """|ig| against toxe for all four cases. Expect a strong decrease as the oxide gets thicker
    (tunnelling).

    TODO(P1): output results/plots/trend_ig_vs_toxe.{png,pdf}. If ig is ~0 everywhere, check
    whether the model card enables gate tunnelling (igcmod / igbmod in BSIM4) and mention it in
    the report.
    """
    raise NotImplementedError
