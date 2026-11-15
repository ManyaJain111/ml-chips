"""Per-FET SPICE oracle for the gate ML (report analysis; writes only to <out_dir>).

Every FET of every (gate, inputs, corner) is simulated ALONE in ngspice at exactly the feature row
the ML model sees (snapped vg). That gives the true per-terminal current for each model input, so
the gate i_vdd error can be attributed to (transistor case, current):
  - per (case, current): ML vs SPICE error on the terminals that enter i_vdd
  - ablation: replace ONE (case, current) group by its SPICE value, recompute i_vdd with the real
    KCL code, and see how much the gate mean accuracy improves.
Run from the repo root:  uv run python report/analysis/oracle.py report/analysis/oracle
(reuses <out_dir>/oracle_spice.csv if present, else runs ngspice on every FET row)
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from leakml import io, sampling
from leakml.config import load_config
from leakml.constants import TARGET_COLUMNS, Case
from leakml.gates import ml_model as mm
from leakml.gates.topology import NODE_VDD, get_gate
from leakml.ml.predictor import RANGE_TOL, TransistorModelBank
from leakml.pdk import load_model_card
from leakml.spice.fet_netlist import FET_EXPECT, FET_SOURCES, render_fet_netlist
from leakml.spice.parse import currents_into_device
from leakml.spice.runner import run_many_cfg

OUT = Path(sys.argv[1])
cfg = load_config("configs/project.yaml")
card = load_model_card(cfg.assignment.model_file)
gs = io.read_csv(cfg.paths.gate_spice_csv)
corners = io.read_csv(cfg.paths.corners_csv)
res = io.read_csv(cfg.paths.gate_results_csv)
bank = TransistorModelBank.load_all(cfg.paths.models_dir)
cn = corners.set_index("corner_id")
params = {
    d: pd.DataFrame(
        [sampling.corner_process_params(r, d) for r in cn.reset_index().to_dict("records")],
        index=cn.index,
    )
    for d in ("nmos", "pmos")
}

# ---- every FET row, grouped exactly like run_gate_ml
groups, parts = [], []
for (g, bits), pos in gs.groupby(["gate", "inputs"], sort=False).indices.items():
    gate = get_gate(g)
    sub = gs.iloc[pos]
    cid = sub["corner_id"].to_numpy()
    vdd = cn.loc[cid, "vdd"].to_numpy(float)
    temp = cn.loc[cid, "temp"].to_numpy(float)
    rec = {
        n: sub[c].to_numpy(float)
        for n, c in mm._RECORDED_COLUMNS.items()
        if n == gate.output or n in gate.internal_nodes
    }
    node_v = mm._node_arrays(gate, bits, vdd, rec)
    blocks = mm._fet_arrays(gate, node_v, vdd, temp, {d: params[d].loc[cid] for d in params}, cfg)
    for k, (X, cases, _) in enumerate(blocks):
        X = X.reset_index(drop=True).copy()
        X["case"] = [Case(c).value for c in cases]
        X["gate"], X["inputs"], X["corner_id"], X["k"] = g, bits, cid, k
        X["fet"] = gate.fets[k].name
        parts.append(X)
    groups.append((g, bits, pos, cid))
F = pd.concat(parts, ignore_index=True)
print(f"{len(F)} FET rows")

# ---- ML currents (one batched call, like step 08)
feat = [c for c in F.columns if c not in ("case", "gate", "inputs", "corner_id", "k", "fet")]
pred = bank.predict(F[feat], F["case"].tolist())
for t in TARGET_COLUMNS:
    F[f"{t}_ml"] = pred[t].to_numpy()

# ---- SPICE currents (cached)
cache = OUT / "oracle_spice.csv"
if cache.exists():
    S = io.read_csv(cache)
    assert len(S) == len(F)
else:
    t0 = time.time()
    decks = [
        render_fet_netlist(r, c, card, cfg.assignment.lmin)
        for r, c in zip(F[feat].to_dict("records"), F["case"], strict=True)
    ]
    out = run_many_cfg(decks, cfg, expect=FET_EXPECT, progress=False)
    bad = [i for i, x in enumerate(out) if not x.ok]
    assert not bad, f"{len(bad)} failed sims, e.g. {bad[:5]}"
    S = pd.DataFrame([currents_into_device(x.values, FET_SOURCES) for x in out])
    io.write_csv(S, cache)
    print(f"ngspice: {len(F)} sims in {time.time() - t0:.0f} s")
for t in TARGET_COLUMNS:
    F[f"{t}_sp"] = S[t].to_numpy()

# ---- recompose i_vdd per (gate, inputs) with the real KCL code, for a given current source
idx = {(g, b, k): grp.index.to_numpy() for (g, b, k), grp in F.groupby(["gate", "inputs", "k"])}


def i_vdd_all(use_spice) -> pd.Series:
    """use_spice(case, t) -> bool: which (case, current) groups take the SPICE value."""
    out = pd.Series(np.nan, index=res.index)
    key = res.set_index(["gate", "inputs", "corner_id"]).index
    for g, bits, _pos, cid in groups:
        gate = get_gate(g)
        cols = {}
        for k in range(len(gate.fets)):
            ii = idx[(g, bits, k)]
            case = F.at[ii[0], "case"]
            cols[k] = {
                t: F.loc[ii, f"{t}_sp" if use_spice(case, t) else f"{t}_ml"].to_numpy()
                for t in TARGET_COLUMNS
            }
        free = np.vstack(
            [
                mm._free_mask([F.at[idx[(g, bits, k)][0], "case"]] * len(cid),
                              F.loc[idx[(g, bits, k)], "vd"].to_numpy(),
                              F.loc[idx[(g, bits, k)], "vs"].to_numpy())
                for k in range(len(gate.fets))
            ]
        )  # fmt: skip
        fixed = mm._kcl_arrays(gate, lambda k, t, cols=cols: cols[k][t], free)
        i_vdd, _ = mm._leakage_arrays(gate, bits, lambda k, t, fixed=fixed: fixed[k][t])
        rows = key.get_indexer(pd.MultiIndex.from_arrays([[g] * len(cid), [bits] * len(cid), cid]))
        out.iloc[rows] = i_vdd
    return out


def stats(pred: pd.Series) -> dict:
    e = (pred - res["i_vdd_spice"]) / res["i_vdd_spice"].abs()
    a = e.abs()
    return {"mean_acc": 100 * (1 - a.mean()), "median": 100 * a.median(),
            "p95": 100 * a.quantile(0.95), "max": 100 * a.max(), "bias": 100 * e.median()}  # fmt: skip


base = i_vdd_all(lambda c, t: False)
print("check: recomposed ML i_vdd vs step-09 i_vdd_ml, max rel diff",
      f"{((base - res['i_vdd_ml']).abs() / res['i_vdd_ml'].abs()).max():.2e}")  # fmt: skip
oracle = i_vdd_all(lambda c, t: True)
print("check: all-SPICE FET currents vs gate SPICE i_vdd:", stats(oracle))

# ---- which terminal currents enter i_vdd (directly), per FET row
for t in TARGET_COLUMNS:
    F[f"on_vdd_{t}"] = False
for g in F["gate"].unique():
    gate = get_gate(g)
    for k, f in enumerate(gate.fets):
        m = (F["gate"] == g) & (F["k"] == k)
        for t in "dgsb":
            if getattr(f, t) == NODE_VDD:
                F.loc[m, f"on_vdd_i{t}"] = True
F["free"] = F["case"].isin(["nmos_on", "pmos_on"]) & ((F["vd"] - F["vs"]).abs() < mm.KCL_VDS_MAX)

rows = []
for case in ["nmos_off", "nmos_on", "pmos_off", "pmos_on"]:
    for t in TARGET_COLUMNS:
        m = F["case"] == case
        x, y = F.loc[m, f"{t}_ml"], F.loc[m, f"{t}_sp"]
        ok = y.abs() >= cfg.ml.i_floor_a
        e = ((x - y) / y.abs())[ok]
        on_vdd = F.loc[m, f"on_vdd_{t}"]
        abl = stats(i_vdd_all(lambda c, tt, case=case, t=t: c == case and tt == t))
        rows.append({
            "case": case, "current": t, "n_fet_rows": int(m.sum()),
            "on_vdd_rows": int(on_vdd.sum()),
            "median_abs_spice": float(y.abs().median()),
            "fet_median_err_pct": 100 * float(e.abs().median()),
            "fet_p95_err_pct": 100 * float(e.abs().quantile(0.95)),
            "fet_bias_pct": 100 * float(e.median()),
            "gate_mean_acc_if_exact": abl["mean_acc"],
            "gate_median_if_exact": abl["median"],
            "gate_p95_if_exact": abl["p95"],
        })  # fmt: skip
T = pd.DataFrame(rows)
b = stats(base)
T["mean_acc_gain"] = T["gate_mean_acc_if_exact"] - b["mean_acc"]
T = T.sort_values("mean_acc_gain", ascending=False)
pd.set_option("display.width", 250)
print("\nbaseline gate stats:", {k: round(v, 3) for k, v in b.items()})
print(T.to_string(index=False, float_format=lambda v: f"{v:.3g}"))
io.write_csv(T, OUT / "error_sources.csv")
io.write_csv(
    pd.DataFrame(
        [{"composition": "ml_kcl", **stats(base)}, {"composition": "spice_oracle", **stats(oracle)}]
    ),
    OUT / "oracle_summary.csv",
)  # report: gate i_vdd error with ML vs with per-FET SPICE currents (same KCL composition)

# ---- worst rows: which FET terminal error drives each
w = res.assign(err=100 * (res.i_vdd_ml - res.i_vdd_spice) / res.i_vdd_spice.abs())
w = w.reindex(w["err"].abs().sort_values(ascending=False).index).head(10)
print("\nworst 10 rows, with the per-FET terminal error on the vdd net (ML-SPICE, A):")
for r in w.itertuples():
    m = (F.gate == r.gate) & (F.inputs == r.inputs) & (F.corner_id == r.corner_id)
    terms = []
    for _, f in F[m].iterrows():
        for t in "dgsb":
            if f[f"on_vdd_i{t}"]:
                d = f[f"i{t}_ml"] - f[f"i{t}_sp"]
                terms.append((abs(d), f"{f.fet}({f.case}).i{t}{'*' if f.free and t in 'ds' else ''} "
                              f"{d:+.2e} ({100 * d / f[f'i{t}_sp']:+.0f}%)"))  # fmt: skip
    terms.sort(reverse=True)
    print(f"  {r.gate} {r.inputs} c{r.corner_id} T={r.temp:.1f} VDD={r.vdd:.3f}  "
          f"err {r.err:+.1f}%  |  " + "; ".join(t for _, t in terms[:3]))  # fmt: skip

# ---- extrapolation (same rule as the predictor warning)
F["extrap"] = False
ext = []
for case, m in bank.models.items():
    sel = F["case"] == case.value
    for col, (lo, hi) in m.meta["train_ranges"].items():
        if col not in F:
            continue
        slack = RANGE_TOL * max(abs(hi - lo), abs(hi), 1e-30)
        x = F.loc[sel, col]
        fl = (x < lo - slack) | (x > hi + slack)
        if fl.any():
            ov = np.maximum(lo - x, x - hi).clip(lower=0)[fl]
            ext.append({"case": case.value, "col": col, "train_lo": lo, "train_hi": hi,
                        "n_fet_rows": int(fl.sum()), "pct": 100 * fl.mean(),
                        "n_high": int((x[fl] > hi).sum()), "max_over_V": ov.max(),
                        "max_over_pct_vdd": 100 * (ov / F.loc[fl[fl].index, "vdd"]).max(),
                        "fets": ", ".join(sorted(set(F.loc[fl[fl].index, "gate"] + "/" + F.loc[fl[fl].index, "inputs"] + ":" + F.loc[fl[fl].index, "fet"]))),
                        "free_rows": int(F.loc[fl[fl].index, "free"].sum())})  # fmt: skip
            F.loc[fl[fl].index, "extrap"] = True
E = pd.DataFrame(ext)
print("\nextrapolation:")
print(E.drop(columns="fets").to_string(index=False, float_format=lambda v: f"{v:.3g}"))
for r in E.itertuples():
    print(f"  {r.case}.{r.col}: {r.fets}")
x = F[F.extrap]
for t in TARGET_COLUMNS:
    xm, xs = x[f"{t}_ml"], x[f"{t}_sp"]
    nm = F[~F.extrap & F.case.isin(x.case.unique())]
    ok, okn = xs.abs() >= 1e-15, nm[f"{t}_sp"].abs() >= 1e-15
    print(f"  {t}: FET median err extrapolating {100 * ((xm - xs) / xs.abs())[ok].abs().median():.2f}% "
          f"vs same-case others {100 * ((nm[f'{t}_ml'] - nm[f'{t}_sp']) / nm[f'{t}_sp'].abs())[okn].abs().median():.2f}%")  # fmt: skip
K = ["gate", "inputs", "corner_id"]
gx = F.groupby(K)["extrap"].any().rename("extrap").reset_index()
j = res.merge(gx, on=K, validate="1:1")
j["ae"] = 100 * (j.i_vdd_ml / j.i_vdd_spice - 1).abs()
print("\ngate rows with an extrapolating FET:", int(j.extrap.sum()))
print(
    j.groupby("extrap")["ae"]
    .describe(percentiles=[0.5, 0.95])[["count", "50%", "95%", "max"]]
    .round(2)
    .to_string()
)
j["vdd_bin"] = pd.cut(j.vdd, [0.80, 0.90, 0.95, 0.97, 0.978, 1.0])
print(
    j.groupby(["vdd_bin", "extrap"], observed=True)["ae"]
    .agg(["count", "median"])
    .round(2)
    .to_string()
)
print("extrapolating corners:", sorted(set(j.loc[j.extrap, "corner_id"])))
