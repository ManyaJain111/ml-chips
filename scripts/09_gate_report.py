"""Step 09: gate results table + parity plot (Deliverables 6, 7).  Owner: P3.

Outputs: data/gates/gate_results.csv
         results/tables/gate_results.{csv,md}, results/tables/gate_worst_cases.csv
         results/plots/gate_parity.{png,pdf}
"""

from __future__ import annotations

import argparse

from leakml.config import DEFAULT_CONFIG


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default=DEFAULT_CONFIG)
    p.add_argument("--quick", action="store_true", help="pilot run with small sample counts")
    return p.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)  # noqa: F841  (remove the noqa once main() is implemented)
    # TODO(P3):
    # 1. merged = gates.report.merge_spice_ml(gate_spice, gate_ml)
    # 2. gates.report.gate_results_table(merged, cfg.ml.i_floor_a)
    # 3. gates.report.gate_parity_plot(merged, results/plots/gate_parity)
    # 4. gates.report.worst_cases(merged)
    # 5. print the overall mean accuracy vs the 99 % target
    raise NotImplementedError


if __name__ == "__main__":
    main()
