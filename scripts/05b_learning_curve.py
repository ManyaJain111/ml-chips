"""Step 05b: learning curve with the CHOSEN models (brief Part A.1, "smallest dataset possible").

Owner: P1 (with P2). Run after step 05: it reads the model type, hyperparameters and options
from artifacts/models/{case}.joblib, refits that exact recipe on nested subsets of each case's
training split, and scores every n on the SAME validation rows. The test split is never read.

Outputs: results/tables/learning_curve.{csv,md}
         results/plots/learning_curve.{png,pdf}

    uv run python scripts/05b_learning_curve.py
    uv run python scripts/05b_learning_curve.py --sizes 1000 3500 7000 14000 21000
"""

from __future__ import annotations

import argparse
import json
import sys

import pandas as pd

from leakml import io, plotting
from leakml.cli import setup_logging
from leakml.config import DEFAULT_CONFIG, ensure_dirs, load_config
from leakml.constants import CASE_DEVICE, CASES, TARGET_COLUMNS
from leakml.data.generate import load_dataset
from leakml.data.learning_curve import fit_fixed_on_subsets
from leakml.ml.predictor import TransistorModel
from leakml.ml.split import load_split_indices


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default=DEFAULT_CONFIG)
    p.add_argument("--quick", action="store_true", help="pilot run with small sample counts")
    # sizes above the training split are capped at it, so the last point is the full split
    p.add_argument(
        "--sizes", type=int, nargs="+", default=[500, 1000, 2000, 3500, 7000, 14000, 21000]
    )
    p.add_argument("--replot", action="store_true", help="only redraw the plots from the table")
    return p.parse_args(argv)


def plot(table: pd.DataFrame, metric: str, ylabel: str, out_path) -> str:
    fig, axes = plotting.plt.subplots(1, len(CASES), figsize=(4 * len(CASES), 3.5), sharey=True)
    for ax, case in zip(axes, CASES, strict=True):
        g = table[table["case"] == case.value]
        for i, t in enumerate(TARGET_COLUMNS):  # fixed colour per current (OFF ig is all NaN)
            gt = g[g["target"] == t].dropna(subset=[metric])
            ax.plot(gt["n_train"], gt[metric], marker="o", color=f"C{i}", label=t)
        ns = sorted(g["n_train"].unique())
        ax.set_xscale("log")
        ax.set_xticks(ns, [f"{n / 1000:g}k" for n in ns])
        ax.minorticks_off()
        ax.set_yscale("log")
        ax.set_title(case.value)
        ax.set_xlabel("training rows n")
        ax.grid(True, which="both", alpha=0.3)
    axes[0].set_ylabel(ylabel)
    axes[0].legend()
    png, _ = plotting.save_fig(fig, out_path)
    return png


def main(argv=None) -> None:
    args = parse_args(argv)
    setup_logging()
    cfg = load_config(args.config, quick=args.quick)
    ensure_dirs(cfg)
    table_path = cfg.paths.tables_dir / "learning_curve"
    if args.replot:
        return save_plots(io.read_csv(table_path.with_suffix(".csv")), cfg)
    nominals = json.loads(cfg.paths.nominals_json.read_text())["nominal"]

    parts = []
    for case in CASES:
        model_file = cfg.paths.model_file(case)
        if not model_file.is_file():
            sys.exit(f"{model_file} missing: run step 05 first (make finalize)")
        meta = TransistorModel.load(model_file).meta
        df = load_dataset(case, cfg)
        idx = load_split_indices(cfg.paths.split_file(case))
        part = fit_fixed_on_subsets(
            case, df, idx["train"], idx["val"], args.sizes,
            model_name=meta["model_name"], hyperparams=meta["hyperparams"],
            target_kind=meta["target_transform"]["kind"],
            shape_channel=meta["target_transform"]["shape_channel"], is_kcl=meta["is_kcl"],
            nominal=nominals[CASE_DEVICE[case]], split_seed=cfg.seeds.split,
            train_seed=cfg.seeds.training, i_floor=cfg.ml.i_floor_a,
        )  # fmt: skip
        parts.append(part)
        for n, g in part.groupby("n_train"):
            print(f"{case.value:9} {meta['model_name']} n={n:5d}  val p95 % " + "  ".join(
                f"{t}={v:.3g}" for t, v in zip(g["target"], g["p95_err_pct"], strict=True)))  # fmt: skip

    table = pd.concat(parts, ignore_index=True)
    _, md = io.save_table(table, table_path)
    print(f"Saved {md}")
    save_plots(table, cfg)


def save_plots(table: pd.DataFrame, cfg) -> None:
    png = plot(table, "p95_err_pct", "validation p95 relative error (%)",
               cfg.paths.plots_dir / "learning_curve")  # fmt: skip
    print(f"Saved {png}")
    png = plot(table, "median_err_pct", "validation median relative error (%)",
               cfg.paths.plots_dir / "learning_curve_median")  # fmt: skip
    print(f"Saved {png}")


if __name__ == "__main__":
    main()
