"""Step 12: test-circuit results + parity plot (Deliverable 8).  Owner: P4.

Outputs: data/circuit/circuit_results.csv
         results/tables/circuit_results.{csv,md}
         results/plots/circuit_parity.{png,pdf}
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
    # TODO(P4):
    # 1. merged = circuit.report.merge_spice_ml(circuit_spice, circuit_ml)
    # 2. circuit.report.circuit_results_table(merged, cfg.ml.i_floor_a), with the gate-level ALL row
    #    alongside
    # 3. circuit.report.circuit_parity_plot(merged, results/plots/circuit_parity)
    # 4. print circuit mean accuracy vs gate-level mean accuracy
    raise NotImplementedError


if __name__ == "__main__":
    main()
