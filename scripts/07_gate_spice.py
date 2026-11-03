"""Step 07: SPICE reference for 18 gate cases x 200 corners (Part B step 3).  Owner: P3.

Outputs: data/gates/gate_spice.csv
         netlists/examples/gate_{INV,NAND2,NOR2,AND2,OR2}.sp   (Deliverable 2)
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
    # 1. corners = read data/corners.csv; card = pdk.load_model_card(...)
    # 2. df = gates.spice_ref.run_gate_reference(cfg, corners, card)
    # 3. assert every row has logic_ok (else a wiring bug in topology/netlist). Log any failures
    # 4. save data/gates/gate_spice.csv
    raise NotImplementedError


if __name__ == "__main__":
    main()
