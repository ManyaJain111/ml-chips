"""Step 11: hierarchical ML prediction for the test circuit (Part C step 7).  Owner: P4.

Inputs:  data/gates/gate_ml.csv (via GateLeakageModel), data/corners.csv
Outputs: data/circuit/circuit_ml.csv
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
    # 1. gate_model = gates.ml_model.GateLeakageModel.from_results(read gate_ml.csv)
    # 2. df = circuit.ml_model.run_circuit_ml(c, cfg, corners, gate_model)
    # 3. save data/circuit/circuit_ml.csv
    raise NotImplementedError


if __name__ == "__main__":
    main()
