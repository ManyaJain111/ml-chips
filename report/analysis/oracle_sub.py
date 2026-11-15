"""Per-case error attribution for the worst gate cases (report analysis).

Runs oracle.py (same directory) to get the per-FET ML and SPICE currents, then, for each of the
worst gate cases, replaces one (transistor case, current) group of ML outputs by its SPICE value
and recomputes the case's gate accuracy with the real KCL composition.

Run from the repo root:  uv run python report/analysis/oracle_sub.py report/analysis/oracle
Writes <out_dir>/worst_case_attribution.csv.
"""

from __future__ import annotations

import contextlib
import io as _io
import runpy
import sys
from pathlib import Path

import pandas as pd

CASES = ("nmos_off", "nmos_on", "pmos_off", "pmos_on")
CURRENTS = ("id", "ig", "is", "ib")
WORST = (("NAND2", "10"), ("NOR2", "10"), ("NOR2", "11"))

out_dir = Path(sys.argv[1])
with contextlib.redirect_stdout(_io.StringIO()):
    ns = runpy.run_path(str(Path(__file__).with_name("oracle.py")))
res, i_vdd_all = ns["res"], ns["i_vdd_all"]


def accuracy(pred: pd.Series, mask: pd.Series) -> tuple[float, float, float]:
    e = ((pred - res["i_vdd_spice"]) / res["i_vdd_spice"].abs())[mask].abs()
    return 100 * (1 - e.mean()), 100 * e.median(), 100 * e.max()


rows = []
for gate, bits in WORST:
    mask = (res["gate"] == gate) & (res["inputs"] == bits)
    base = accuracy(i_vdd_all(lambda c, t: False), mask)
    rows.append({"gate": gate, "inputs": bits, "fixed": "none",
                 "mean_acc": base[0], "median": base[1], "max": base[2]})  # fmt: skip
    print(
        f"{gate} {bits}: baseline mean acc {base[0]:.2f}, median {base[1]:.2f}, max {base[2]:.1f}"
    )
    for case in CASES:
        for cur in CURRENTS:
            a = accuracy(i_vdd_all(lambda c, t, case=case, cur=cur: c == case and t == cur), mask)
            rows.append({"gate": gate, "inputs": bits, "fixed": f"{case}.{cur}",
                         "mean_acc": a[0], "median": a[1], "max": a[2]})  # fmt: skip
            if a[0] - base[0] > 0.05:
                print(f"    {case}.{cur}: mean acc {a[0]:.2f} (+{a[0] - base[0]:.2f})")

pd.DataFrame(rows).to_csv(out_dir / "worst_case_attribution.csv", index=False, float_format="%.10e")
