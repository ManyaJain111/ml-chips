"""Step 12: test-circuit results + parity plot (Deliverable 8).  Owner: P4.

Outputs: data/circuit/circuit_results.csv
         results/tables/circuit_results.{csv,md}
         results/plots/circuit_parity.{png,pdf}
"""

from __future__ import annotations

import argparse

from leakml import io
from leakml.circuit import report
from leakml.cli import setup_logging
from leakml.config import DEFAULT_CONFIG, ensure_dirs, load_config


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
    merged = report.merge_spice_ml(io.read_csv(p.circuit_spice_csv), io.read_csv(p.circuit_ml_csv))
    io.write_csv(merged, p.circuit_results_csv)
    gate = io.read_csv(p.gate_results_csv) if p.gate_results_csv.exists() else None
    table = report.circuit_results_table(merged, cfg.ml.i_floor_a, gate)
    io.save_table(table, p.tables_dir / "circuit_results")
    report.loading_error_analysis(merged).pipe(
        io.save_table, p.tables_dir / "circuit_error_drivers"
    )
    report.circuit_parity_plot(merged, p.plots_dir / "circuit_parity", cfg.ml.i_floor_a)
    print(
        table[["scope", "n", "median_err_pct", "p95_err_pct", "mean_acc_pct"]].to_string(
            index=False
        )
    )
    circ = table.iloc[0]["mean_acc_pct"]
    msg = f"circuit mean accuracy {circ:.3f} %"
    if gate is not None:
        msg += f" vs gate-level {table.iloc[-1]['mean_acc_pct']:.3f} %"
    print(msg)


if __name__ == "__main__":
    main()
