"""Step 02: generate the four transistor datasets (Deliverable 1) + example FET netlists (Deliverable 2).
IMPLEMENTED.

Outputs: data/transistor/{nmos_on,nmos_off,pmos_on,pmos_off}.csv
         netlists/examples/fet_{case}.sp
"""

from __future__ import annotations

import argparse
import logging

from leakml.cli import setup_logging
from leakml.config import DEFAULT_CONFIG, ensure_dirs, load_config
from leakml.constants import CASES, TARGET_COLUMNS, Case
from leakml.data.generate import generate_case_dataset, save_dataset
from leakml.pdk import load_model_card

log = logging.getLogger("02_generate")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default=DEFAULT_CONFIG)
    p.add_argument("--quick", action="store_true", help="pilot run with small sample counts")
    p.add_argument("--cases", nargs="*", default=None, help="subset of cases to (re)generate")
    return p.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    setup_logging()
    cfg = load_config(args.config, quick=args.quick)
    ensure_dirs(cfg)
    card = load_model_card(cfg.assignment.model_file)
    cases = [Case(c) for c in args.cases] if args.cases else list(CASES)

    for case in cases:
        df = generate_case_dataset(case, cfg, card)
        path = save_dataset(df, case, cfg)
        rng = "  ".join(
            f"|{t}| {df[t].abs().min():.2e}..{df[t].abs().max():.2e}" for t in TARGET_COLUMNS
        )
        print(f"{case.value:9} {len(df):6d} rows -> {path}\n          {rng}")


if __name__ == "__main__":
    main()
