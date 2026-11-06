"""Step 11: hierarchical ML prediction for the test circuit (Part C step 7).  Owner: P4.

Inputs:  data/gates/gate_ml.csv (via GateLeakageModel), data/corners.csv
Outputs: data/circuit/circuit_ml.csv
"""

from __future__ import annotations

import argparse

from leakml import io
from leakml.circuit.definition import get_circuit
from leakml.circuit.ml_model import run_circuit_ml
from leakml.cli import setup_logging
from leakml.config import DEFAULT_CONFIG, ensure_dirs, load_config
from leakml.gates.ml_model import GateLeakageModel


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
    c = get_circuit(cfg.assignment.test_circuit)
    corners = io.read_csv(cfg.paths.corners_csv)
    gate_model = GateLeakageModel.from_csv(cfg.paths.gate_ml_csv)
    df = run_circuit_ml(c, cfg, corners, gate_model)
    io.write_csv(df, cfg.paths.circuit_ml_csv)
    print(f"wrote {len(df)} rows -> {cfg.paths.circuit_ml_csv}")


if __name__ == "__main__":
    main()
