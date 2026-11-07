"""Step 08: ML gate leakage from the transistor models (Part B step 4).  Owner: P3.

Inputs:  artifacts/models/*.joblib, data/gates/gate_spice.csv (ONLY the node voltages are used),
         data/corners.csv
Outputs: data/gates/gate_ml.csv  (SPEC §5.4)

The summary compares against i_vdd_spice for information only. run_gate_ml itself never sees the
SPICE currents (hard rule 1). Exits non-zero if the output keys do not match gate_spice 1:1, or
if stand-in models would end up in a deliverable folder (hard rule 13).
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

from leakml import io
from leakml.cli import setup_logging
from leakml.config import DEFAULT_CONFIG, ensure_dirs, load_config
from leakml.constants import GATES
from leakml.gates.ml_model import run_gate_ml
from leakml.ml.predictor import TransistorModelBank

KEY = ["gate", "inputs", "corner_id"]


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default=DEFAULT_CONFIG)
    p.add_argument("--quick", action="store_true", help="pilot run with small sample counts")
    return p.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    setup_logging()
    cfg = load_config(args.config, quick=args.quick)
    ensure_dirs(cfg)

    for path, step in ((cfg.paths.gate_spice_csv, "07"), (cfg.paths.corners_csv, "06")):
        if not path.exists():
            sys.exit(f"{path} not found: run step {step} with the same --config/--quick first")
    gate_spice = io.read_csv(cfg.paths.gate_spice_csv)
    corners = io.read_csv(cfg.paths.corners_csv)
    bank = TransistorModelBank.load_all(cfg.paths.models_dir)
    stand_ins = [c.value for c, m in bank.models.items() if m.meta.get("stand_in")]
    if stand_ins and cfg.paths.is_deliverable_location():
        sys.exit(f"REFUSING: stand-in models {stand_ins} in {cfg.paths.models_dir} (hard rule 13)")
    print(
        f"{len(gate_spice)} gate cases, {len(corners)} corners, models from {cfg.paths.models_dir}"
        + (f"  [STAND-IN models: {stand_ins}]" if stand_ins else "")
    )

    df = run_gate_ml(cfg, gate_spice, corners, bank)

    if len(df) != len(gate_spice) or not df[KEY].equals(
        gate_spice[KEY].astype({"corner_id": int}).reset_index(drop=True)
    ):
        sys.exit(
            f"FAILED: gate_ml has {len(df)} rows / keys that do not match gate_spice "
            f"({len(gate_spice)} rows). Nothing written."
        )
    bad = ~np.isfinite(df["i_vdd_ml"])
    if bad.any():
        sys.exit(f"FAILED: {int(bad.sum())} non-finite i_vdd_ml values. Nothing written.")

    io.write_csv(df, cfg.paths.gate_ml_csv)

    # ---- summary (SPICE currents used for reporting only)
    j = df.merge(gate_spice[[*KEY, "i_vdd_spice"]], on=KEY, validate="1:1")
    j["rel_err"] = (j["i_vdd_ml"] - j["i_vdd_spice"]).abs() / j["i_vdd_spice"].abs()
    summary = (
        j.groupby("gate", sort=False)
        .agg(
            max_vg_snap_err=("max_vg_snap_err", "max"),
            i_vdd_ml_min=("i_vdd_ml", "min"),
            i_vdd_ml_median=("i_vdd_ml", "median"),
            i_vdd_ml_max=("i_vdd_ml", "max"),
            median_rel_err=("rel_err", "median"),
        )
        .reindex(GATES)
    )
    print(f"\nrows: {len(df)} (matches gate_spice)")
    print(summary.to_string(float_format=lambda x: f"{x:.3e}"))
    print(f"\nwrote {len(df)} rows -> {cfg.paths.gate_ml_csv}")


if __name__ == "__main__":
    main()
