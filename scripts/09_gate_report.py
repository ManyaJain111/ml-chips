"""Step 09: gate results table + parity plot (Deliverables 6, 7).  Owner: P3.

Inputs:  data/gates/gate_spice.csv, data/gates/gate_ml.csv, artifacts/models/*.joblib (stand-in check)
Outputs: data/gates/gate_results.csv
         results/tables/gate_results.{csv,md}        D6
         results/tables/gate_worst_cases.{csv,md}
         results/plots/gate_parity.{png,pdf}         D7 (KCL i_vdd_ml vs SPICE)
         results/plots/gate_parity_direct.{png,pdf}  direct-sum method vs SPICE, for comparison

Refuses to write into the deliverable folders when gate_ml.csv is a placeholder or the models it
came from are stand-ins (hard rule 13).
"""

from __future__ import annotations

import argparse
import sys

from leakml import io
from leakml.cli import setup_logging
from leakml.config import DEFAULT_CONFIG, ensure_dirs, load_config
from leakml.gates.report import (
    gate_parity_plot,
    gate_results_table,
    merge_spice_ml,
    stand_in_sources,
    worst_cases,
)

TARGET_MEAN_ACC_PCT = 99.0  # brief / SPEC §7


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
    p = cfg.paths

    for path, step in ((p.gate_spice_csv, "07"), (p.gate_ml_csv, "08")):
        if not path.exists():
            sys.exit(f"{path} not found: run step {step} with the same --config/--quick first")
    gate_spice = io.read_csv(p.gate_spice_csv)
    gate_ml = io.read_csv(p.gate_ml_csv)

    stand_ins = stand_in_sources(gate_ml, p.models_dir)
    if stand_ins and p.is_deliverable_location():
        sys.exit("REFUSING to write deliverables (hard rule 13): " + "; ".join(stand_ins))
    if stand_ins:
        print("[STAND-IN inputs, dev/pilot output only] " + "; ".join(stand_ins))

    try:
        merged = merge_spice_ml(gate_spice, gate_ml, n_corners=cfg.gates.n_corners)
    except (ValueError, TypeError) as e:
        sys.exit(f"FAILED to merge gate_spice and gate_ml: {e}")
    io.write_csv(merged, p.gate_results_csv)

    i_floor = cfg.ml.i_floor_a
    table = gate_results_table(merged, i_floor)
    io.save_table(table, p.tables_dir / "gate_results")
    worst = worst_cases(merged, k=20, i_floor=i_floor)
    io.save_table(worst, p.tables_dir / "gate_worst_cases")
    stats = gate_parity_plot(merged, p.plots_dir / "gate_parity", "i_vdd_ml", i_floor)
    gate_parity_plot(merged, p.plots_dir / "gate_parity_direct", "i_vdd_ml_direct", i_floor)

    print(f"\nD6 gate results ({len(merged)} gate simulations, i_floor = {i_floor:g} A):")
    print(table.to_string(index=False, float_format=lambda x: f"{x:.4g}"))
    print(f"\nworst {len(worst)} cases (KCL):")
    print(worst.to_string(index=False, float_format=lambda x: f"{x:.4g}"))
    acc = stats["mean_acc_pct"]
    print(
        f"\noverall mean accuracy {acc:.2f} % vs target {TARGET_MEAN_ACC_PCT:g} % "
        + ("(met)" if acc >= TARGET_MEAN_ACC_PCT else "(NOT met)")
    )
    print(f"wrote {p.gate_results_csv}")
    print(f"wrote {p.tables_dir / 'gate_results'}.{{csv,md}}, gate_worst_cases.{{csv,md}}")
    print(f"wrote {p.plots_dir / 'gate_parity'}{{,_direct}}.{{png,pdf}}")


if __name__ == "__main__":
    main()
