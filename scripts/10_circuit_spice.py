"""Step 10: SPICE reference for the full test circuit (Part C step 6).  Owner: P4.

Outputs: data/circuit/circuit_spice.csv
         netlists/examples/circuit_{cla3|parity8}.sp   (Deliverable 2)
"""

from __future__ import annotations

import argparse
import os
import sys

from leakml import io
from leakml.circuit.definition import get_circuit
from leakml.circuit.netlist import render_circuit_netlist
from leakml.circuit.spice_ref import run_circuit_reference
from leakml.cli import setup_logging
from leakml.config import DEFAULT_CONFIG, ensure_dirs, load_config
from leakml.pdk import load_model_card


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default=DEFAULT_CONFIG)
    p.add_argument("--quick", action="store_true", help="pilot run with small sample counts")
    p.add_argument(
        "--resume", action="store_true", help="skip (vector, corner) pairs already on disk"
    )
    return p.parse_args(argv)


def main(argv=None) -> None:
    # parallel ngspice processes starve each other's OpenMP threads otherwise (30-40x slower)
    os.environ.setdefault("OMP_WAIT_POLICY", "passive")
    args = parse_args(argv)
    setup_logging()
    cfg = load_config(args.config, quick=args.quick)
    ensure_dirs(cfg)
    c = get_circuit(cfg.assignment.test_circuit)
    corners = io.read_csv(cfg.paths.corners_csv)  # the SAME file as Part B
    card = load_model_card(cfg.assignment.model_file)

    # Deliverable 2: corner-0, all-zeros deck
    c0 = corners.iloc[0].to_dict()
    deck = render_circuit_netlist(c, "0" * len(c.primary_inputs), c0, card, cfg)
    ex = cfg.paths.netlist_examples / f"circuit_{c.name}.sp"
    ex.write_text(deck + "\n")

    df = run_circuit_reference(c, cfg, corners, card, resume=args.resume)
    bad = df[~df["logic_ok"]]
    if len(bad):
        print(
            f"FAILED: {len(bad)}/{len(df)} rows failed or have wrong logic levels", file=sys.stderr
        )
        print(bad.head(10).to_string(), file=sys.stderr)
        sys.exit(1)
    io.write_csv(df, cfg.paths.circuit_spice_csv)
    print(f"wrote {len(df)} rows -> {cfg.paths.circuit_spice_csv} (example deck: {ex})")


if __name__ == "__main__":
    main()
