"""Step 06: draw the PVT corners shared by Part B and Part C.  IMPLEMENTED.

Outputs: data/corners.csv  (columns = constants.CORNER_COLUMNS)

Refuses to overwrite an existing, DIFFERENT corners.csv once Part B/C SPICE results exist (every
downstream result keys on corner_id). Use --force only if you will re-run steps 07-12.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

from leakml import io
from leakml.cli import setup_logging
from leakml.config import DEFAULT_CONFIG, ensure_dirs, load_config
from leakml.pdk import load_model_card
from leakml.sampling import draw_corners, make_rng


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default=DEFAULT_CONFIG)
    p.add_argument("--quick", action="store_true", help="pilot run with small sample counts")
    p.add_argument(
        "--force", action="store_true", help="overwrite even if downstream results exist"
    )
    return p.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    setup_logging()
    cfg = load_config(args.config, quick=args.quick)
    ensure_dirs(cfg)
    card = load_model_card(cfg.assignment.model_file)
    corners = draw_corners(cfg.gates.n_corners, cfg, card, make_rng(cfg.seeds.corners, "corners"))

    out = cfg.paths.corners_csv
    if out.exists():
        old = io.read_csv(out)
        same = old.shape == corners.shape and np.allclose(
            old.to_numpy(float), corners.to_numpy(float), rtol=1e-9, atol=0
        )
        if same:
            print(f"{out} already up to date ({len(old)} corners)")
            return
        downstream = [
            p for p in (cfg.paths.gate_spice_csv, cfg.paths.circuit_spice_csv) if p.exists()
        ]
        if downstream and not args.force:
            print(
                f"REFUSING: {out} would change but downstream results exist: "
                f"{[str(p) for p in downstream]}. Re-run with --force and then redo steps 07-12.",
                file=sys.stderr,
            )
            sys.exit(1)
    io.write_csv(corners, out)
    print(f"wrote {len(corners)} corners -> {out}")
    print(corners.describe().T[["min", "max"]].to_string())


if __name__ == "__main__":
    main()
