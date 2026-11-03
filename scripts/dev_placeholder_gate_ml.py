"""DEV ONLY: write a schema-correct PLACEHOLDER gate_ml.csv so P4 can build/run Part C (steps 11-12)
before P3's real gate ML exists. The numbers are a made-up smooth function of the corner, NOT physics.

Refuses to touch the real data/ folder, and never overwrites a gate_ml.csv that is not itself a
placeholder (so P3's real results are safe).

    uv run python scripts/dev_placeholder_gate_ml.py --config configs/dev.yaml --quick
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd

from leakml import io
from leakml.cli import setup_logging
from leakml.config import DEFAULT_CONFIG, ensure_dirs, load_config
from leakml.constants import GATE_INPUT_COMBOS, GATES


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default=DEFAULT_CONFIG)
    p.add_argument("--quick", action="store_true", help="pilot run with small sample counts")
    return p.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    setup_logging()
    cfg = load_config(args.config, quick=args.quick)
    if cfg.paths.is_deliverable_location():
        print("REFUSING: use --quick or --config configs/dev.yaml", file=sys.stderr)
        sys.exit(1)
    ensure_dirs(cfg)
    out = cfg.paths.gate_ml_csv
    if out.exists() and "source" not in io.read_csv(out).columns:
        print(f"REFUSING: {out} holds real gate-ML results", file=sys.stderr)
        sys.exit(1)

    corners = io.read_csv(cfg.paths.corners_csv)  # run scripts/06_generate_corners.py first
    rows = []
    for gi, gate in enumerate(GATES):
        for bi, bits in enumerate(GATE_INPUT_COMBOS[gate]):
            scale = 1e-9 * (1 + 0.3 * gi + 0.1 * bi)
            i_vdd = scale * corners.vdd * np.exp((corners.temp - 27) / 40) * (0.47 / corners.n_vth0)
            rows.append(pd.DataFrame({
                "gate": gate, "inputs": bits, "corner_id": corners.corner_id,
                "i_vdd_ml": i_vdd, "i_in_ml": 1e-3 * i_vdd, "source": "placeholder",
            }))  # fmt: skip
    df = pd.concat(rows, ignore_index=True)
    io.write_csv(df, out)
    print(f"wrote PLACEHOLDER {len(df)} rows -> {out}")


if __name__ == "__main__":
    main()
