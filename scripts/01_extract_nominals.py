"""Step 01: extract nominal process parameters from the assigned model file.  IMPLEMENTED.

Outputs: artifacts/nominals.json
"""

from __future__ import annotations

import argparse

from leakml.cli import setup_logging
from leakml.config import DEFAULT_CONFIG, ensure_dirs, load_config
from leakml.constants import PROCESS_PARAMS
from leakml.pdk import dump_nominals, load_model_card


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
    card = load_model_card(cfg.assignment.model_file)
    dump_nominals(card, cfg.paths.nominals_json)

    tol = cfg.sampling.process_tol
    print(f"{'param':6} {'device':6} {'nominal':>12} {'low':>12} {'high':>12}")
    for dev in ("nmos", "pmos"):
        for p in PROCESS_PARAMS:
            v = card.nominal[dev][p]
            lo, hi = sorted((v * (1 - tol), v * (1 + tol)))
            print(f"{p:6} {dev:6} {v:12.5g} {lo:12.5g} {hi:12.5g}")
    print(f"wrote {cfg.paths.nominals_json}")


if __name__ == "__main__":
    main()
