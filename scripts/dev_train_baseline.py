"""DEV ONLY: train STAND-IN transistor models (leakml.ml.baseline) so Part B/C can run before
P2's real models exist. NOT part of `make all`, NOT a deliverable.

Writes cfg.paths.models_dir/{case}.joblib in the bundle format of leakml.ml.predictor, and refuses
to write into the real artifacts/models/ (use --quick or configs/dev.yaml).

    uv run python scripts/dev_train_baseline.py --config configs/dev.yaml --quick
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

from leakml.cli import setup_logging
from leakml.config import DEFAULT_CONFIG, ensure_dirs, load_config
from leakml.constants import CASES, INPUT_COLUMNS, TARGET_COLUMNS
from leakml.data.generate import load_dataset
from leakml.metrics import summarize
from leakml.ml.baseline import AsinhExtraTrees
from leakml.ml.predictor import input_ranges, save_bundle


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default=DEFAULT_CONFIG)
    p.add_argument("--quick", action="store_true", help="pilot run with small sample counts")
    p.add_argument("--n-estimators", type=int, default=200)
    return p.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    setup_logging()
    cfg = load_config(args.config, quick=args.quick)
    if cfg.paths.is_deliverable_location():
        print(
            "REFUSING: stand-in models must not be written to artifacts/models/. "
            "Use --quick or --config configs/dev.yaml.",
            file=sys.stderr,
        )
        sys.exit(1)
    ensure_dirs(cfg)

    for case in CASES:
        df = load_dataset(case, cfg)
        rng = np.random.default_rng(cfg.seeds.training)
        hold = rng.random(len(df)) < 0.2
        X, Y = df.loc[:, list(INPUT_COLUMNS)], df.loc[:, list(TARGET_COLUMNS)]
        reg = AsinhExtraTrees(n_estimators=args.n_estimators, seed=cfg.seeds.training)
        reg.fit(X[~hold], Y[~hold])
        P = reg.predict(X[hold])
        stats = {
            t: summarize(P[:, i], Y[hold][t].to_numpy(), i_floor=cfg.ml.i_floor_a)
            for i, t in enumerate(TARGET_COLUMNS)
        }
        meta = {
            "model_name": "STAND-IN baseline (leakml.ml.baseline.AsinhExtraTrees)",
            "stand_in": True,
            "n_train": int((~hold).sum()),
            "train_ranges": input_ranges(X[~hold]),
            "holdout_metrics": stats,
            "config": cfg.snapshot(),
        }
        path = save_bundle(cfg.paths.model_file(case), case, reg, meta)
        errs = "  ".join(f"{t}: med {s['median_err_pct']:.2g}%" for t, s in stats.items())
        print(f"{case.value:9} -> {path}   holdout {errs}")


if __name__ == "__main__":
    main()
