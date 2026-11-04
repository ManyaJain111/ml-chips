"""DEV ONLY: learning curve with the STAND-IN model (brief Part A.1, "smallest dataset possible").

Trains leakml.ml.baseline.AsinhExtraTrees on nested subsets of each case's training split
(n = 1k, 2k, 5k, all) and scores every n on the SAME validation rows. The test split is never
read. NOT part of `make all`: the final curve must be re-run with P2's chosen model.

Stand-in numbers never reach deliverable folders (hard rule 13), so outputs go to
<results>/_dev_learning_curve/ (gitignored): learning_curve.{csv,md} and learning_curve.{png,pdf}.

    uv run python scripts/dev_learning_curve.py
    uv run python scripts/dev_learning_curve.py --config configs/dev.yaml --quick --sizes 100 200 350
"""

from __future__ import annotations

import argparse

import pandas as pd

from leakml import io, plotting
from leakml.cli import setup_logging
from leakml.config import DEFAULT_CONFIG, load_config
from leakml.constants import CASES, INPUT_COLUMNS, TARGET_COLUMNS
from leakml.data.generate import load_dataset
from leakml.data.learning_curve import learning_curve_subsets
from leakml.metrics import summarize
from leakml.ml.baseline import AsinhExtraTrees
from leakml.ml.split import load_split_indices, split_dataset


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default=DEFAULT_CONFIG)
    p.add_argument("--quick", action="store_true", help="pilot run with small sample counts")
    p.add_argument("--sizes", type=int, nargs="+", default=[1000, 2000, 5000, 10000])
    p.add_argument("--n-estimators", type=int, default=200)
    return p.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    setup_logging()
    cfg = load_config(args.config, quick=args.quick)
    out_dir = cfg.paths.results / "_dev_learning_curve"

    rows = []
    for case in CASES:
        df = load_dataset(case, cfg)
        split_file = cfg.paths.split_file(case)
        if split_file.exists():
            idx = load_split_indices(split_file)
        else:  # same function and seed step 04 uses, so the rows are identical
            idx = {
                k: v.index.tolist()
                for k, v in split_dataset(df, cfg.split_scheme, cfg.seeds.split).items()
            }
        val = df.loc[idx["val"]]
        X_val, Y_val = val.loc[:, list(INPUT_COLUMNS)], val.loc[:, list(TARGET_COLUMNS)]

        subsets = learning_curve_subsets(idx["train"], args.sizes, cfg.seeds.split, case)
        for n, rows_n in subsets.items():
            train = df.loc[rows_n]
            reg = AsinhExtraTrees(n_estimators=args.n_estimators, seed=cfg.seeds.training)
            reg.fit(train.loc[:, list(INPUT_COLUMNS)], train.loc[:, list(TARGET_COLUMNS)])
            P = reg.predict(X_val)
            for i, t in enumerate(TARGET_COLUMNS):
                s = summarize(P[:, i], Y_val[t].to_numpy(), i_floor=cfg.ml.i_floor_a)
                rows.append({"case": case.value, "n_train": n, "target": t, **s})
            p95 = {r["target"]: r["p95_err_pct"] for r in rows[-len(TARGET_COLUMNS) :]}
            print(
                f"{case.value:9} n={n:6d}  val p95 % "
                + "  ".join(f"{t} {v:.3g}" for t, v in p95.items())
            )

    table = pd.DataFrame(rows)
    _, md = io.save_table(table, out_dir / "learning_curve")
    print(f"Saved table -> {md}")

    fig, axes = plotting.plt.subplots(1, len(CASES), figsize=(4 * len(CASES), 3.5), sharey=True)
    for ax, case in zip(axes, CASES, strict=True):
        g = table[table["case"] == case.value]
        for t in TARGET_COLUMNS:
            gt = g[g["target"] == t]
            ax.plot(gt["n_train"], gt["p95_err_pct"], marker="o", label=t)
        ax.set_xscale("log")
        ns = sorted(g["n_train"].unique())
        ax.set_xticks(ns, [f"{n // 1000}k" if n % 1000 == 0 else str(n) for n in ns])
        ax.minorticks_off()
        ax.set_yscale("log")
        ax.set_title(f"{case.value} (STAND-IN model)")
        ax.set_xlabel("training rows n")
        ax.grid(True, which="both", alpha=0.3)
    axes[0].set_ylabel("validation p95 relative error (%)")
    axes[0].legend()
    png, _ = plotting.save_fig(fig, out_dir / "learning_curve")
    print(f"Saved plot  -> {png}")


if __name__ == "__main__":
    main()
