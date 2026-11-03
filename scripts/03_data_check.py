"""Step 03: data check, Kirchhoff + trend plots (Deliverable 3).  Owner: P1.

Outputs: results/tables/kirchhoff.{csv,md}
         results/plots/trend_ioff_vs_temp, trend_ioff_vs_vth0, trend_ig_vs_toxe (.png/.pdf)
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
    # TODO(P1):
    # 1. datasets = {case: data.generate.load_dataset(case, cfg) for case in CASES}
    # 2. data.checks.kirchhoff_table(datasets) -> results/tables/kirchhoff.{csv,md}
    # 3. the three trend plots from data.checks
    # 4. print a one-line verdict per case (KCL residual p95 relative < 1e-6 ?) for the report
    raise NotImplementedError


if __name__ == "__main__":
    main()
