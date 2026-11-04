"""Deliverable 3: Kirchhoff check and the three trend plots.  Owner: P1."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from leakml import plotting
from leakml.constants import Case


def kirchhoff_residual(df: pd.DataFrame) -> pd.Series:
    """Per-row KCL residual  r = id + ig + is + ib  (should be ~0, since currents are INTO the device)."""
    return df["id"] + df["ig"] + df["is"] + df["ib"]


def kirchhoff_relative_residual(df: pd.DataFrame, i_floor: float = 1e-18) -> pd.Series:
    """Per-row normalised KCL residual  |r| / max(|id|, |ig|, |is|, |ib|)."""
    r = kirchhoff_residual(df)
    i_max = df[["id", "ig", "is", "ib"]].abs().max(axis=1)
    return r.abs() / i_max.clip(lower=i_floor)


def kirchhoff_table(datasets: dict[Case, pd.DataFrame]) -> pd.DataFrame:
    """One row per case: median, p95 and max of |r| and of the normalised residual."""
    rows = []
    for case in Case:
        if case not in datasets:
            continue
        df = datasets[case]
        abs_r = kirchhoff_residual(df).abs()
        rel_r = kirchhoff_relative_residual(df)
        rows.append(
            {
                "case": case.value,
                "median_abs_r_a": float(abs_r.median()),
                "p95_abs_r_a": float(abs_r.quantile(0.95)),
                "max_abs_r_a": float(abs_r.max()),
                "median_rel_r": float(rel_r.median()),
                "p95_rel_r": float(rel_r.quantile(0.95)),
                "max_rel_r": float(rel_r.max()),
            }
        )
    return pd.DataFrame(rows)


def plot_off_current_vs_temperature(datasets: dict[Case, pd.DataFrame], out_dir: Path) -> None:
    """OFF current (|id| of nmos_off and pmos_off) against temperature, log y axis.

    Restricts to |vd - vs| > 0.5*vdd to show clean exponential subthreshold + junction growth.
    Output: <out_dir>/trend_ioff_vs_temp.{png,pdf}
    """
    nmos = datasets[Case.NMOS_OFF]
    pmos = datasets[Case.PMOS_OFF]
    n_mask = (nmos["vd"] - nmos["vs"]).abs() > 0.5 * nmos["vdd"]
    p_mask = (pmos["vd"] - pmos["vs"]).abs() > 0.5 * pmos["vdd"]
    n_sub = nmos[n_mask]
    p_sub = pmos[p_mask]

    temps = np.concatenate([n_sub["temp"].to_numpy(), p_sub["temp"].to_numpy()])
    ioffs = np.concatenate([n_sub["id"].abs().to_numpy(), p_sub["id"].abs().to_numpy()])
    labels = np.array(["NMOS OFF"] * len(n_sub) + ["PMOS OFF"] * len(p_sub))

    plotting.trend_scatter(
        x=temps,
        y=ioffs,
        out_path=out_dir / "trend_ioff_vs_temp",
        xlabel="Temperature (°C)",
        ylabel="|Id| (A)",
        title="OFF Current vs Temperature (|Vds| > 0.5*VDD)",
        hue=labels,
        hue_label="Device",
        logy=True,
    )


def plot_off_current_vs_vth0(datasets: dict[Case, pd.DataFrame], out_dir: Path) -> None:
    """OFF current against vth0 (use |vth0| for PMOS). Expect exponential decrease.

    Restricts to |vd - vs| > 0.5*vdd. Output: <out_dir>/trend_ioff_vs_vth0.{png,pdf}
    """
    nmos = datasets[Case.NMOS_OFF]
    pmos = datasets[Case.PMOS_OFF]
    n_mask = (nmos["vd"] - nmos["vs"]).abs() > 0.5 * nmos["vdd"]
    p_mask = (pmos["vd"] - pmos["vs"]).abs() > 0.5 * pmos["vdd"]
    n_sub = nmos[n_mask]
    p_sub = pmos[p_mask]

    vth0s = np.concatenate([n_sub["vth0"].to_numpy(), p_sub["vth0"].abs().to_numpy()])
    ioffs = np.concatenate([n_sub["id"].abs().to_numpy(), p_sub["id"].abs().to_numpy()])
    labels = np.array(["NMOS OFF"] * len(n_sub) + ["PMOS OFF"] * len(p_sub))

    plotting.trend_scatter(
        x=vth0s,
        y=ioffs,
        out_path=out_dir / "trend_ioff_vs_vth0",
        xlabel="|vth0| (V)",
        ylabel="|Id| (A)",
        title="OFF Current vs Threshold Voltage (|Vds| > 0.5*VDD)",
        hue=labels,
        hue_label="Device",
        logy=True,
    )


def plot_gate_current_vs_toxe(datasets: dict[Case, pd.DataFrame], out_dir: Path) -> None:
    """|ig| against toxe for all four cases. Expect a strong decrease as oxide gets thicker.

    Output: <out_dir>/trend_ig_vs_toxe.{png,pdf}
    """
    all_toxe = []
    all_ig = []
    all_labels = []
    for case in Case:
        if case not in datasets:
            continue
        df = datasets[case]
        all_toxe.append(df["toxe"].to_numpy())
        all_ig.append(df["ig"].abs().to_numpy())
        label = case.value.replace("_", " ").upper()
        all_labels.extend([label] * len(df))

    toxe = np.concatenate(all_toxe)
    ig = np.concatenate(all_ig)
    labels = np.array(all_labels)

    plotting.trend_scatter(
        x=toxe * 1e9,
        y=ig,
        out_path=out_dir / "trend_ig_vs_toxe",
        xlabel="toxe (nm)",
        ylabel="|Ig| (A)",
        title="Gate Tunneling Current vs Oxide Thickness",
        hue=labels,
        hue_label="Case",
        logy=True,
    )
