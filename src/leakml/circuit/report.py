"""Deliverable 8: test-circuit results table + parity plot.  Owner: P4."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from leakml import metrics, plotting

KEY = ["circuit", "vector", "corner_id"]


def merge_spice_ml(circuit_spice: pd.DataFrame, circuit_ml: pd.DataFrame) -> pd.DataFrame:
    """Join on (circuit, vector, corner_id), asserting a 1:1 match -> data/circuit/circuit_results.csv."""
    for name, df in (("circuit_spice", circuit_spice), ("circuit_ml", circuit_ml)):
        if df.duplicated(KEY).any():
            raise ValueError(f"{name} has duplicate {KEY} rows")
    merged = circuit_spice.merge(circuit_ml, on=KEY, how="outer", validate="1:1", indicator=True)
    if (merged["_merge"] != "both").any():
        raise ValueError(
            f"SPICE and ML tables do not match 1:1 ({merged['_merge'].value_counts().to_dict()}). "
            "Re-run steps 10 and 11 with the same --config/--quick."
        )
    return merged.drop(columns="_merge")


def _n_ones(vector: str) -> int:
    return vector.count("1")


def circuit_results_table(
    merged: pd.DataFrame,
    i_floor: float | None = None,
    gate_results: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Metrics (SPEC §7) for the circuit overall, by number of ones in the vector, and (if given)
    the gate-level ALL row from `gate_results` (needs i_vdd_spice / i_vdd_ml) for comparison.
    """
    df = merged.assign(n_ones=merged["vector"].map(_n_ones))
    rows = []
    overall = metrics.results_table(
        df, ["circuit"], "i_vdd_ml", "i_vdd_spice", i_floor, add_all=False
    )
    for _, r in overall.iterrows():
        rows.append({"scope": f"{r['circuit']} (all vectors)", **r.drop("circuit").to_dict()})
    by_ones = metrics.results_table(df, ["n_ones"], "i_vdd_ml", "i_vdd_spice", i_floor, False)
    for _, r in by_ones.sort_values("n_ones").iterrows():
        rows.append({"scope": f"  ones = {int(r['n_ones'])}", **r.drop("n_ones").to_dict()})
    if "i_vdd_ml_gates" in df.columns:
        # Ablation: the plain sum of isolated-gate leakages, without the loading term (SPEC §6.3)
        s = metrics.summarize(
            df["i_vdd_ml_gates"].to_numpy(), df["i_vdd_spice"].to_numpy(), i_floor=i_floor
        )
        rows.append({"scope": "no loading term (sum of gate i_vdd)", **s})
    if {"i_in_ml", "i_in_spice"} <= set(df.columns) and np.isfinite(df["i_in_ml"]).all():
        # secondary metric: current delivered by the primary-input sources
        s = metrics.summarize(
            df["i_in_ml"].to_numpy(), df["i_in_spice"].to_numpy(), i_floor=i_floor
        )
        rows.append({"scope": "primary-input current i_in", **s})
    if gate_results is not None and len(gate_results):
        # SPEC §5.4 names are i_vdd_*; accept the shorter i_spice / i_ml too
        ml_c = "i_vdd_ml" if "i_vdd_ml" in gate_results else "i_ml"
        sp_c = "i_vdd_spice" if "i_vdd_spice" in gate_results else "i_spice"
        s = metrics.summarize(
            gate_results[ml_c].to_numpy(), gate_results[sp_c].to_numpy(), i_floor=i_floor
        )
        rows.append({"scope": "gate level (ALL gates, Part B)", **s})
    return pd.DataFrame(rows, columns=["scope", *metrics.METRIC_KEYS])


def circuit_parity_plot(merged: pd.DataFrame, out_path: Path, i_floor: float | None = None) -> None:
    """Log-log ML vs SPICE leakage, coloured by temperature -> results/plots/circuit_parity.{png,pdf}."""
    name = str(merged["circuit"].iloc[0])
    plotting.parity_plot_log(
        merged["i_vdd_spice"].to_numpy(),
        merged["i_vdd_ml"].to_numpy(),
        out_path,
        title=f"{name}: total leakage, ML vs SPICE",
        hue=merged["temp"].to_numpy(),
        hue_label="temperature (°C)",
        i_floor=i_floor,
        quantity="leakage (A)",
    )


def loading_error_analysis(merged: pd.DataFrame) -> pd.DataFrame:
    """Where does the circuit-level error come from? Spearman correlation of the signed relative
    error with corner vdd / temp, the number of ones in the vector and the gate mix (n_* counts).
    Large |rho| for a gate-mix column points at inter-gate loading / non-ideal logic levels
    in that gate configuration (ignored by the hierarchical model).
    """
    e = metrics.relative_error(merged["i_vdd_ml"].to_numpy(), merged["i_vdd_spice"].to_numpy())
    feats = {"vdd": merged["vdd"], "temp": merged["temp"]}
    feats["n_ones"] = merged["vector"].map(_n_ones)
    feats.update({c: merged[c] for c in merged.columns if c.startswith("n_") and c != "n_ones"})
    rows = []
    ok = np.isfinite(e)
    for name, x in feats.items():
        x = pd.Series(np.asarray(x, dtype=float))[ok]
        rho = x.corr(pd.Series(e[ok]), method="spearman") if x.nunique() > 1 else np.nan
        rows.append({"feature": name, "spearman_rho_vs_signed_error": rho})
    out = pd.DataFrame(rows)
    return out.reindex(
        out["spearman_rho_vs_signed_error"].abs().sort_values(ascending=False).index
    ).reset_index(drop=True)
