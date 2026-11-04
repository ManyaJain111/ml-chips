"""Step 03: data check, Kirchhoff + trend plots (Deliverable 3).  Owner: P1.

Outputs: results/tables/kirchhoff.{csv,md}
         results/plots/trend_ioff_vs_temp, trend_ioff_vs_vth0, trend_ig_vs_toxe (.png/.pdf)
"""

from __future__ import annotations

import argparse

from leakml import io
from leakml.config import DEFAULT_CONFIG, load_config
from leakml.constants import Case
from leakml.data import checks, generate


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default=DEFAULT_CONFIG)
    p.add_argument("--quick", action="store_true", help="pilot run with small sample counts")
    return p.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    cfg = load_config(args.config, quick=args.quick)

    print(f"Loading transistor datasets for {cfg.assignment.node} (quick={args.quick})...")
    datasets = {case: generate.load_dataset(case, cfg) for case in Case}

    print("Computing Kirchhoff (KCL) residual table...")
    table = checks.kirchhoff_table(datasets)
    table_csv, table_md = io.save_table(table, cfg.paths.tables_dir / "kirchhoff")
    print(f"Saved KCL table -> {table_md}")
    print()
    print(table.to_string(index=False))
    print()

    print("Generating trend plots...")
    checks.plot_off_current_vs_temperature(datasets, cfg.paths.plots_dir)
    checks.plot_off_current_vs_vth0(datasets, cfg.paths.plots_dir)
    checks.plot_gate_current_vs_toxe(datasets, cfg.paths.plots_dir)
    print(f"Saved trend plots -> {cfg.paths.plots_dir}")

    # Check that p95 relative KCL residual is < 1e-5 for all cases
    all_ok = True
    print("\n--- Physical Consistency Verdict ---")
    for _, row in table.iterrows():
        case = row["case"]
        p95_rel = row["p95_rel_r"]
        status = "PASS" if p95_rel < 1e-5 else "WARN"
        print(f"  {case:10s} : p95 rel KCL = {p95_rel:.2e}  [{status}]")
        if status == "WARN":
            all_ok = False

    if all_ok:
        print("\nAll cases passed physical Kirchhoff conservation check!")


if __name__ == "__main__":
    main()
