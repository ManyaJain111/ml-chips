"""Step 08: ML gate leakage from the transistor models (Part B step 4).  Owner: P3.

Inputs:  artifacts/models/*.joblib, data/gates/gate_spice.csv (ONLY the node voltages are used),
         data/corners.csv
Outputs: data/gates/gate_ml.csv
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
    # 1. bank = ml.predictor.TransistorModelBank.load_all(artifacts/models)
    # 2. df = gates.ml_model.run_gate_ml(cfg, gate_spice, corners, bank)
    #    In "spice" mode it reads v_out/v_x/v_y/v_n1 from gate_spice. It must NOT read i_*_spice.
    # 3. save data/gates/gate_ml.csv
    raise NotImplementedError


if __name__ == "__main__":
    main()
