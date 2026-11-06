"""Step 04: split, train the 4 model types per case, compare on validation (Deliverable 4).  Owner: P2.

Outputs: artifacts/splits/{case}.npz
         results/tables/model_comparison.{csv,md}          (long: case x model x target)
         results/tables/model_comparison_summary.{csv,md}  (compact: case x model, target-averaged)
         results/tables/model_search.{csv,md}              (every hyperparameter grid point)
         artifacts/candidates/{case}_{model}.joblib   (all fitted candidates, so step 05 needn't refit)

The test split is never looked at here (hard rule 2).

    uv run python scripts/04_train_compare.py                       # full, tuned
    uv run python scripts/04_train_compare.py --quick --no-tune     # pilot smoke run
"""

from __future__ import annotations

import argparse
import json
import sys
import time

import joblib
import pandas as pd

from leakml import io
from leakml.cli import setup_logging
from leakml.config import DEFAULT_CONFIG, ensure_dirs, load_config
from leakml.constants import CASE_DEVICE, CASES, Case
from leakml.data.generate import load_dataset
from leakml.ml import zoo
from leakml.ml.split import save_split_indices, split_dataset
from leakml.ml.train import compare_models, comparison_summary, comparison_table


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default=DEFAULT_CONFIG)
    p.add_argument("--quick", action="store_true", help="pilot run with small sample counts")
    p.add_argument("--cases", nargs="*", default=None)
    p.add_argument("--models", nargs="*", default=None, help="subset of zoo.MODEL_TYPES")
    p.add_argument("--no-tune", action="store_true", help="defaults only, skip the grid search")
    return p.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    setup_logging()
    cfg = load_config(args.config, quick=args.quick)
    ensure_dirs(cfg)

    cases = [Case(c) for c in args.cases] if args.cases else list(CASES)
    models = tuple(args.models) if args.models else tuple(cfg.ml.model_types)
    unknown = [m for m in models if m not in zoo.MODEL_TYPES]
    if unknown:
        sys.exit(f"unknown model types {unknown}, expected some of {zoo.MODEL_TYPES}")
    if not cfg.paths.nominals_json.is_file():
        sys.exit(f"{cfg.paths.nominals_json} missing: run step 01 first (make nominals)")
    nominals = json.loads(cfg.paths.nominals_json.read_text())["nominal"]
    tune = not args.no_tune
    print(f"step 04: cases={[c.value for c in cases]} models={list(models)} tune={tune}")

    results = {}
    t_all = time.perf_counter()
    for case in cases:
        df = load_dataset(case, cfg)
        splits = split_dataset(df, cfg.split_scheme, cfg.seeds.split)
        save_split_indices(splits, cfg.paths.split_file(case))
        print(
            f"{case.value:9} n={len(df)}  " + " ".join(f"{k}={len(v)}" for k, v in splits.items())
        )

        fits = compare_models(
            case, splits, models, cfg.seeds.training,
            nominal=nominals[CASE_DEVICE[case]], i_floor=cfg.ml.i_floor_a, tune=tune,
            selection_metric=cfg.ml.selection_metric,
        )  # fmt: skip
        for r in fits:
            joblib.dump(r, cfg.paths.candidates_dir / f"{case.value}_{r.model_name}.joblib",
                        compress=3)  # fmt: skip
            print(
                f"   {r.model_name:13} val p95 {r.score('p95_err_pct'):8.3g} %   "
                f"median {r.score('median_err_pct'):7.3g} %   fit {r.fit_seconds:6.1f} s   "
                f"{len(r.search)} grid point(s)"
            )
        results[case] = fits

    long = comparison_table(results)
    summary = comparison_summary(long)
    search = pd.concat(
        [r.search.assign(case=c.value, model=r.model_name) for c, fits in results.items()
         for r in fits],
        ignore_index=True,
    )  # fmt: skip
    search = search[["case", "model", *[c for c in search.columns if c not in ("case", "model")]]]
    tables = cfg.paths.tables_dir
    for name, table in (
        ("model_comparison", long),
        ("model_comparison_summary", summary),
        (
            "model_search",
            search.astype({c: str for c in search.columns if search[c].dtype == object}),
        ),
    ):
        _, md = io.save_table(table, tables / name)
        print(f"Saved {md}")
    if len(cases) < len(CASES) or len(models) < len(cfg.ml.model_types):
        print("NOTE: subset run, so the tables cover only the cases/models above")
    print(f"step 04 done in {time.perf_counter() - t_all:.0f} s")


if __name__ == "__main__":
    main()
