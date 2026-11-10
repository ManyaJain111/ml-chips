"""Step 05: select the best model per case, evaluate on TEST, save (Deliverable 5).  Owner: P2.

Outputs: artifacts/models/{case}.joblib            (the 4 saved model files)
         results/tables/chosen_models.{csv,md}      (chosen model per case + why, validation)
         results/tables/chosen_models_test.{csv,md} (test median + p95 error per case/target)
         results/plots/pred_vs_spice_{case}.{png,pdf}

This is the ONLY place the test split is used (hard rule 2). Selection uses validation only.
"""

from __future__ import annotations

import argparse
import sys

import joblib
import numpy as np
import pandas as pd

from leakml import io
from leakml.cli import setup_logging
from leakml.config import DEFAULT_CONFIG, ensure_dirs, load_config
from leakml.constants import CASES, INPUT_COLUMNS
from leakml.data.generate import load_dataset
from leakml.ml.predictor import TransistorModelBank
from leakml.ml.split import load_split_indices
from leakml.ml.train import evaluate_on_test, save_model, select_best


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
    metric = cfg.ml.selection_metric

    chosen_rows, test_tables, test_sets = [], [], {}
    for case in CASES:
        paths = {
            m: cfg.paths.candidates_dir / f"{case.value}_{m}.joblib" for m in cfg.ml.model_types
        }
        missing = [str(p) for p in paths.values() if not p.is_file()]
        if missing:
            sys.exit(f"missing candidates {missing}: run step 04 first (make train)")
        cands = [joblib.load(p) for p in paths.values()]

        best = select_best(cands, metric)
        ranking = sorted(cands, key=lambda r: r.score(metric))
        reason = "; ".join(
            f"{r.model_name} val {metric} {r.score(metric):.3g} / median "
            f"{r.score('median_err_pct'):.3g}" for r in ranking
        )  # fmt: skip
        tuned = len(best.search) > 1
        print(f"{case.value:9} -> {best.model_name}  ({reason})")

        df = load_dataset(case, cfg)
        test = df.loc[load_split_indices(cfg.paths.split_file(case))["test"]]
        test_sets[case] = test
        tm = evaluate_on_test(
            best, test, i_floor=cfg.ml.i_floor_a,
            plot_path=cfg.paths.plots_dir / f"pred_vs_spice_{case.value}",
        )  # fmt: skip
        test_tables.append(tm)
        save_model(
            best, cfg.paths.model_file(case),
            extra_meta={
                "config": cfg.snapshot(),
                "selection": {"metric": metric, "ranking": reason, "tuned": tuned},
                "test_metrics": tm.to_dict("records"),
            },
        )  # fmt: skip
        chosen_rows.append({
            "case": case.value, "model": best.model_name, "tuned": tuned,
            f"val_{metric}": best.score(metric), "val_median_err_pct": best.score("median_err_pct"),
            "hyperparams": str(best.hyperparams), "ranking": reason,
        })  # fmt: skip
        print("          test p95 % " + "  ".join(
            f"{t}={v:.3g}" for t, v in zip(tm["target"], tm["p95_err_pct"], strict=True)))  # fmt: skip

    tables = cfg.paths.tables_dir
    for name, table in (
        ("chosen_models", pd.DataFrame(chosen_rows)),
        ("chosen_models_test", pd.concat(test_tables, ignore_index=True)),
    ):
        _, md = io.save_table(table, tables / name)
        print(f"Saved {md}")

    # Smoke test: the files load through the P3/P4 API and reproduce the in-memory predictions.
    bank = TransistorModelBank.load_all(cfg.paths.models_dir)
    for case, test in test_sets.items():
        X = test.loc[:, list(INPUT_COLUMNS)].iloc[:5]
        via_bank = bank.predict(X, [case] * len(X)).to_numpy()
        direct = bank.models[case].regressor.predict(X)
        if not np.allclose(via_bank, direct, rtol=1e-12, atol=0):
            sys.exit(f"smoke test FAILED for {case.value}")
    print(f"smoke test OK: {cfg.paths.models_dir} loads via TransistorModelBank")
    if not all(r["tuned"] for r in chosen_rows):
        print("WARNING: some chosen models were NOT tuned (step 04 --no-tune)")


if __name__ == "__main__":
    main()
