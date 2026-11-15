import sys
from pathlib import Path

import numpy as np
import pandas as pd

from leakml import io
from leakml.metrics import summarize
from leakml.ml.predictor import TransistorModelBank

out_dir = Path(sys.argv[1])
o, f = (
    io.read_csv(out_dir / "gate_results_table_orig.csv"),
    io.read_csv(out_dir / "gate_results_table_fresh.csv"),
)
cols = [
    "median_err_pct",
    "p95_err_pct",
    "max_err_pct",
    "mean_acc_pct",
    "within_1pct",
    "within_10pct",
    "median_err_direct_pct",
    "median_err_i_in_pct",
]
side = o[["gate", "inputs"]].copy()
for c in cols:
    side[c + "_orig"] = o[c]
    side[c + "_fresh"] = f[c]
io.save_table(side, out_dir / "gate_results_orig_vs_fresh", floatfmt=".3g")
pd.set_option("display.width", 250)
short = {
    "median_err_pct": "med",
    "p95_err_pct": "p95",
    "max_err_pct": "max",
    "mean_acc_pct": "acc",
    "within_1pct": "<1%",
    "within_10pct": "<10%",
    "median_err_direct_pct": "dirmed",
    "median_err_i_in_pct": "i_in",
}
print(
    side.rename(
        columns=lambda c: next((short[k] + c[len(k) :] for k in short if c.startswith(k)), c)
    ).to_string(index=False, float_format=lambda x: f"{x:.2f}")
)
ro, rf = (
    io.read_csv(out_dir / "gate_results_orig.csv"),
    io.read_csv(out_dir / "gate_results_fresh.csv"),
)
rows = []
for name, r in (("orig", ro), ("fresh", rf)):
    for meth, col in (("KCL", "i_vdd_ml"), ("direct", "i_vdd_ml_direct")):
        s = summarize(r[col].to_numpy(), r.i_vdd_spice.to_numpy(), i_floor=1e-15)
        e = np.abs((r[col] - r.i_vdd_spice) / r.i_vdd_spice)
        rows.append(
            {
                "corners": name,
                "method": meth,
                **{
                    k: s[k]
                    for k in (
                        "median_err_pct",
                        "p95_err_pct",
                        "max_err_pct",
                        "mean_acc_pct",
                        "bias_pct",
                    )
                },
                "within_1pct": 100 * (e <= 0.01).mean(),
                "within_10pct": 100 * (e <= 0.1).mean(),
            }
        )
T = pd.DataFrame(rows)
io.save_table(T, out_dir / "overall_kcl_direct_orig_vs_fresh", floatfmt=".4g")
print("\n", T.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
# corner-level bootstrap of the mean-accuracy difference (corners are the sampling unit)
rng = np.random.default_rng(0)


def per_corner(r, col):
    return (np.abs((r[col] - r.i_vdd_spice) / r.i_vdd_spice)).groupby(r.corner_id).mean().to_numpy()


for meth, col in (("KCL", "i_vdd_ml"), ("direct", "i_vdd_ml_direct")):
    a, b = per_corner(ro, col), per_corner(rf, col)
    d = [100 * (rng.choice(a, 200).mean() - rng.choice(b, 200).mean()) for _ in range(5000)]
    print(
        f"{meth}: mean acc orig-fresh = {100 * (b.mean() - a.mean()) * -1:+.3f} pts, 95% bootstrap CI [{np.percentile(d, 2.5):+.3f}, {np.percentile(d, 97.5):+.3f}]"
    )
# what drives fresh max
print("\nfresh worst 10:")
print(
    io.read_csv(out_dir / "gate_worst_cases_fresh.csv")
    .head(10)
    .to_string(index=False, float_format=lambda x: f"{x:.4g}")
)
# corner ranges: fresh vs orig and vs model training ranges
c_o, c_f = io.read_csv("data/corners.csv"), io.read_csv(out_dir / "corners_fresh.csv")
bank = TransistorModelBank.load_all("artifacts/models")
tr = bank.models[list(bank.models)[0]].meta["train_ranges"]
print("\ncorner param ranges (fresh outside orig range?):")
for c in c_o.columns[1:]:
    print(
        f"  {c:7s} orig [{c_o[c].min():.4g}, {c_o[c].max():.4g}]  fresh [{c_f[c].min():.4g}, {c_f[c].max():.4g}]"
    )
