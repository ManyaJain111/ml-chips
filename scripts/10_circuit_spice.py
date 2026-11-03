"""Step 10: SPICE reference for the full test circuit (Part C step 6).  Owner: P4.

Outputs: data/circuit/circuit_spice.csv
         netlists/examples/circuit_{cla3|parity8}.sp   (Deliverable 2)
"""

from __future__ import annotations

import argparse

from leakml.config import DEFAULT_CONFIG


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default=DEFAULT_CONFIG)
    p.add_argument("--quick", action="store_true", help="pilot run with small sample counts")
    p.add_argument(
        "--resume", action="store_true", help="skip (vector, corner) pairs already on disk"
    )
    return p.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)  # noqa: F841  (remove the noqa once main() is implemented)
    # TODO(P4):
    # 1. c = circuit.definition.get_circuit(cfg.assignment.test_circuit)
    # 2. corners = data/corners.csv (the SAME file as Part B)
    # 3. df = circuit.spice_ref.run_circuit_reference(c, cfg, corners, card), with checkpointing
    #    (this is the longest SPICE job in the project)
    # 4. assert logic_ok everywhere, then save data/circuit/circuit_spice.csv
    raise NotImplementedError


if __name__ == "__main__":
    main()
