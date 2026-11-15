"""Error analysis for the report (Section 6): error vs case, gate, input and corner conditions.

    uv run python report/make_analysis.py        (also called by report/make_numbers.py)

Reads only committed pipeline outputs (data/, artifacts/, results/, report/analysis/) and writes
only to report/generated/analysis/: one CSV per analysis plus the figure error_vs_corner.pdf.
make_numbers.py turns the CSVs into macros/tables and copies the figure into report/figures/.

Hard rules respected: the transistor-model errors are measured on the VALIDATION split (the test
split stays untouched, hard rule 2); gate and circuit SPICE results are used for evaluation only.
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from leakml import io, plotting, sampling
from leakml.circuit.definition import get_circuit
from leakml.circuit.ml_model import run_circuit_ml
from leakml.config import load_config
from leakml.constants import CASES, TARGET_COLUMNS, Case
from leakml.gates import ml_model as gm
from leakml.gates.topology import get_gate
from leakml.metrics import summarize
from leakml.ml.predictor import RANGE_TOL, TransistorModelBank
from leakml.pdk import load_model_card

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "report" / "generated" / "analysis"
TEMP_BINS = [-40.0, 0.0, 40.0, 80.0, 125.0]
DEV_BINS = [-10.0, -5.0, 0.0, 5.0, 10.0]  # process deviation from nominal, %
VDS_BINS = [0.0, 1e-3, 0.1, 0.3, np.inf]
VDS_LABELS = ["<1 mV", "1 mV-0.1 V", "0.1-0.3 V", ">0.3 V"]

log = logging.getLogger("make_analysis")


def _rel_err(ml: pd.Series, spice: pd.Series) -> pd.Series:
    return (ml - spice) / spice.abs()


def _binned(df: pd.DataFrame, col: str, bins, err: str = "err") -> pd.DataFrame:
    """median/p95 |err|, median signed err (bias) and mean accuracy per bin of `col` (in %)."""
    b = pd.cut(df[col], bins, include_lowest=True)
    g = df.assign(bin=b, ae=df[err].abs()).groupby("bin", observed=True)
    out = pd.DataFrame({
        "lo": [iv.left for iv in g.groups], "hi": [iv.right for iv in g.groups],
        "n": g.size().to_numpy(),
        "median_err_pct": 100 * g["ae"].median().to_numpy(),
        "p95_err_pct": 100 * g["ae"].quantile(0.95).to_numpy(),
        "bias_pct": 100 * g[err].median().to_numpy(),
        "mean_acc_pct": 100 * (1 - g["ae"].mean().to_numpy()),
    })  # fmt: skip
    out["lo"] = [bins[0] if i == 0 else v for i, v in enumerate(out["lo"])]  # undo include_lowest
    return out


# ------------------------------------------------------------------ gates vs corner conditions
def gate_vs_corner(cfg, card) -> pd.DataFrame:
    g = io.read_csv(cfg.paths.gate_results_csv)
    c = io.read_csv(cfg.paths.corners_csv)
    d = g.drop(columns=["vdd", "temp"], errors="ignore").merge(c, on="corner_id")
    d["err"] = _rel_err(d["i_vdd_ml"], d["i_vdd_spice"])
    d["n_vth0_dev"] = 100 * (d["n_vth0"] / card.nominal["nmos"]["vth0"] - 1)
    d["p_vth0_dev"] = 100 * (d["p_vth0"] / card.nominal["pmos"]["vth0"] - 1)  # same sign: ratio
    io.write_csv(_binned(d, "temp", TEMP_BINS), OUT / "gate_vs_temp.csv")
    io.write_csv(_binned(d, "n_vth0_dev", DEV_BINS), OUT / "gate_vs_nvth0.csv")
    io.write_csv(_binned(d, "p_vth0_dev", DEV_BINS), OUT / "gate_vs_pvth0.csv")
    params = ["temp", "vdd", "n_vth0", "n_toxe", "n_ndep", "n_u0", "n_xj",
              "p_vth0", "p_toxe", "p_ndep", "p_u0", "p_xj"]  # fmt: skip
    rho = [{"feature": p, "rho_signed": d[p].corr(d["err"], method="spearman"),
            "rho_abs": d[p].corr(d["err"].abs(), method="spearman")} for p in params]  # fmt: skip
    io.write_csv(pd.DataFrame(rho), OUT / "gate_error_drivers.csv")
    return d


# ------------------------------------------------------------------ transistor models (validation)
def transistor_errors(cfg, bank: TransistorModelBank) -> None:
    rows_t, rows_v, rows_f = [], [], []
    for case in CASES:
        df = io.read_csv(cfg.paths.transistor_csv(case.value))
        with np.load(cfg.paths.artifacts / "splits" / f"{case.value}.npz") as z:
            val = df.loc[z["val"]]
        pred = bank.predict(val, [case] * len(val))
        vds = (val["vd"] - val["vs"]).abs()
        for t in TARGET_COLUMNS:
            y, p = val[t], pred[t]
            ok = y.abs() >= cfg.ml.i_floor_a
            rows_f.append({"case": case.value, "target": t, "n": len(y),
                           "below_floor_pct": 100 * (~ok).mean()})  # fmt: skip
            if ok.sum() == 0:
                continue
            e = _rel_err(p[ok], y[ok])
            sub = pd.DataFrame({"err": e, "temp": val.loc[ok, "temp"], "vds": vds[ok]})
            bt = _binned(sub, "temp", TEMP_BINS)
            bt.insert(0, "target", t)
            bt.insert(0, "case", case.value)
            rows_t.append(bt)
            bv = _binned(sub, "vds", VDS_BINS)
            bv.insert(0, "band", VDS_LABELS[: len(bv)] if len(bv) == 4 else
                      [VDS_LABELS[VDS_BINS.index(lo)] for lo in bv["lo"]])  # fmt: skip
            bv.insert(0, "target", t)
            bv.insert(0, "case", case.value)
            rows_v.append(bv)
    io.write_csv(pd.concat(rows_t), OUT / "transistor_vs_temp.csv")
    io.write_csv(pd.concat(rows_v), OUT / "transistor_vs_vds.csv")
    io.write_csv(pd.DataFrame(rows_f), OUT / "transistor_floor.csv")


# ------------------------------------------------------------------ extrapolation at gate level
def gate_extrapolation(cfg, bank: TransistorModelBank, gate_df: pd.DataFrame) -> None:
    gs = io.read_csv(cfg.paths.gate_spice_csv)
    cn = io.read_csv(cfg.paths.corners_csv).set_index("corner_id")
    params = {d: pd.DataFrame([sampling.corner_process_params(r, d)
                               for r in cn.reset_index().to_dict("records")], index=cn.index)
              for d in ("nmos", "pmos")}  # fmt: skip
    rec_cols = {"out": "v_out_spice", "x": "v_x_spice", "y": "v_y_spice", "n1": "v_n1_spice"}
    rows, flags = [], []
    for (name, bits), sub in gs.groupby(["gate", "inputs"], sort=False):
        gate = get_gate(name)
        cid = sub["corner_id"].to_numpy()
        vdd = cn.loc[cid, "vdd"].to_numpy(float)
        temp = cn.loc[cid, "temp"].to_numpy(float)
        rec = {n: sub[c].to_numpy(float) for n, c in rec_cols.items()
               if n == gate.output or n in gate.internal_nodes}  # fmt: skip
        node_v = gm._node_arrays(gate, bits, vdd, rec)
        blocks = gm._fet_arrays(gate, node_v, vdd, temp,
                                {d: params[d].loc[cid] for d in params}, cfg)  # fmt: skip
        out_any = np.zeros(len(sub), bool)
        for fet, (X, cases, _) in zip(gate.fets, blocks, strict=True):
            case = Case(cases[0])
            ranges = bank.models[case].meta.get("train_ranges", {})
            for col, (lo, hi) in ranges.items():
                slack = RANGE_TOL * max(abs(hi - lo), abs(hi), 1e-30)
                x = X[col].to_numpy(float)
                over = np.maximum(lo - slack - x, x - hi - slack)
                m = over > 0
                if m.any():
                    rows.append({"gate": name, "inputs": bits, "fet": fet.name, "case": case.value,
                                 "column": col, "n_rows": int(m.sum()),
                                 "max_over": float((np.maximum(lo - x, x - hi))[m].max()),
                                 "range_width": hi - lo})  # fmt: skip
                    out_any |= m
        flags.append(pd.DataFrame({"gate": name, "inputs": bits, "corner_id": cid,
                                   "extrap": out_any}))  # fmt: skip
    cols = ["gate", "inputs", "fet", "case", "column", "n_rows", "max_over", "range_width"]
    io.write_csv(pd.DataFrame(rows, columns=cols), OUT / "gate_extrapolation.csv")
    f = gate_df.merge(pd.concat(flags), on=["gate", "inputs", "corner_id"])
    summ = []
    for flag, grp in f.groupby("extrap"):
        ae = grp["err"].abs()
        summ.append({"extrap": bool(flag), "n": len(grp), "median_err_pct": 100 * ae.median(),
                     "p95_err_pct": 100 * ae.quantile(0.95), "max_err_pct": 100 * ae.max(),
                     "n_corners": grp["corner_id"].nunique()})  # fmt: skip
    io.write_csv(pd.DataFrame(summ), OUT / "gate_extrapolation_summary.csv")


# ------------------------------------------------------------------ gate SPICE physics
def gate_spice_physics(cfg) -> None:
    gs = io.read_csv(cfg.paths.gate_spice_csv)
    piv = gs.pivot_table(index="corner_id", columns=["gate", "inputs"], values="i_vdd_spice")
    temp = gs.groupby("corner_id")["temp"].first().reindex(piv.index)

    def ratio(a, b):
        return piv[a] / piv[b]

    comps = {"nand01_inv0": (("NAND2", "01"), ("INV", "0")),
             "nor11_nor01": (("NOR2", "11"), ("NOR2", "01")),
             "nor11_nor10": (("NOR2", "11"), ("NOR2", "10")),
             "nand00_nand01": (("NAND2", "00"), ("NAND2", "01")),
             "nand00_nand10": (("NAND2", "00"), ("NAND2", "10"))}  # fmt: skip
    rows = []
    for key, (a, b) in comps.items():
        r = ratio(a, b)
        rows.append({"comparison": key, "median_ratio": r.median(),
                     "max_abs_dev": (r - 1).abs().max(), "n_lower": int((r < 1).sum()),
                     "n": len(r)})  # fmt: skip
    io.write_csv(pd.DataFrame(rows), OUT / "spice_ratios.csv")

    r = ratio(("NAND2", "00"), ("NAND2", "01"))
    tb = pd.cut(temp, [-40, 0, 40, 80, 130], include_lowest=True)
    t = pd.DataFrame({"r": r, "bin": tb}).groupby("bin", observed=True)["r"]
    nt = pd.DataFrame({"lo": [-40, 0, 40, 80], "hi": [0, 40, 80, 125],
                  "median_ratio": t.median().to_numpy(), "n_lower": t.apply(lambda s: (s < 1).sum()).to_numpy(),
                  "n": t.size().to_numpy()})  # fmt: skip
    io.write_csv(nt, OUT / "spice_nand00_vs_temp.csv")

    # temperature trend per case: hot 10 % vs cold 10 % of corners, Spearman rho(T, log I)
    hot = temp >= temp.quantile(0.9)
    cold = temp <= temp.quantile(0.1)
    trend = []
    for col in piv.columns:
        x = piv[col]
        trend.append({"gate": col[0], "inputs": col[1], "median_i_vdd": x.median(),
                      "hot_cold": x[hot].median() / x[cold].median(),
                      "rho_t_logi": temp.corr(np.log(x), method="spearman")})  # fmt: skip
    order = {g: i for i, g in enumerate(["INV", "NAND2", "NOR2", "AND2", "OR2"])}
    trend = pd.DataFrame(trend).sort_values(["gate", "inputs"], key=lambda s: s.map(order) if
                                            s.name == "gate" else s)  # fmt: skip
    io.write_csv(trend, OUT / "spice_temp_trend.csv")
    groups = pd.DataFrame([{"t_hot_min": temp[hot].min(), "t_hot_max": temp[hot].max(),
                   "t_cold_min": temp[cold].min(), "t_cold_max": temp[cold].max(),
                   "n_hot": int(hot.sum()), "n_cold": int(cold.sum())}])  # fmt: skip
    io.write_csv(groups, OUT / "spice_temp_groups.csv")

    # internal node voltages (as fraction of VDD)
    nodes = []
    for gate, node in (("NAND2", "x"), ("NOR2", "y"), ("AND2", "n1"), ("OR2", "n1")):
        sub = gs[gs["gate"] == gate]
        for bits, s in sub.groupby("inputs"):
            v = s[f"v_{node}_spice"]
            frac = v / s["vdd"]
            nodes.append({"gate": gate, "node": node, "inputs": bits, "v_min": v.min(),
                          "v_max": v.max(), "frac_median": frac.median(),
                          "dist_rail_max": np.minimum(v, s["vdd"] - v).max()})  # fmt: skip
    io.write_csv(pd.DataFrame(nodes), OUT / "spice_nodes.csv")
    mn = pd.DataFrame([{"min_i_vdd": gs["i_vdd_spice"].min(),
                   "case_min": "{} {}".format(*gs.loc[gs["i_vdd_spice"].idxmin(), ["gate", "inputs"]])}]
                 )  # fmt: skip
    io.write_csv(mn, OUT / "spice_min.csv")


# ------------------------------------------------------------------ circuit
def circuit_analysis(cfg) -> None:
    cr = io.read_csv(cfg.paths.circuit_dir / "circuit_results.csv")
    cr["err"] = _rel_err(cr["i_vdd_ml"], cr["i_vdd_spice"])
    io.write_csv(_binned(cr, "temp", TEMP_BINS), OUT / "circuit_vs_temp.csv")

    # composition with the SPICE gate values (Part B SPICE i_vdd and i_in in place of the ML
    # ones): isolates the composition/loading error from the ML error. Evaluation only.
    gsp = io.read_csv(cfg.paths.gate_spice_csv)
    lut = gm.GateLeakageModel.from_results(
        gsp.rename(columns={"i_vdd_spice": "i_vdd_ml", "i_in_spice": "i_in_ml"})
        [["gate", "inputs", "corner_id", "i_vdd_ml", "i_in_ml"]])  # fmt: skip
    corners = io.read_csv(cfg.paths.corners_csv)
    comp = run_circuit_ml(get_circuit(cfg.assignment.test_circuit), cfg, corners, lut)
    m = cr.merge(comp[["vector", "corner_id", "i_vdd_ml", "i_vdd_ml_gates"]],
                 on=["vector", "corner_id"], suffixes=("", "_spicegates"))  # fmt: skip
    rows = []
    for label, pred, ref in (
        ("spice_gates_with_loading", m["i_vdd_ml_spicegates"], m["i_vdd_spice"]),
        ("spice_gates_no_loading", m["i_vdd_ml_gates_spicegates"], m["i_vdd_spice"]),
        ("ml_vs_spice_gate_composition", m["i_vdd_ml"], m["i_vdd_ml_spicegates"]),
        ("ml_final", m["i_vdd_ml"], m["i_vdd_spice"]),
    ):
        s = summarize(pred.to_numpy(), ref.to_numpy(), i_floor=cfg.ml.i_floor_a)
        rows.append({"composition": label, **{k: s[k] for k in
                     ("median_err_pct", "p95_err_pct", "max_err_pct", "bias_pct", "mean_acc_pct")}})  # fmt: skip
    io.write_csv(pd.DataFrame(rows), OUT / "circuit_decomposition.csv")


# ------------------------------------------------------------------ figure
def figure(gate_df: pd.DataFrame) -> None:
    gt = io.read_csv(OUT / "gate_vs_temp.csv")
    gn = io.read_csv(OUT / "gate_vs_nvth0.csv")
    gp = io.read_csv(OUT / "gate_vs_pvth0.csv")
    ct = io.read_csv(OUT / "circuit_vs_temp.csv")
    fig, axes = plotting.plt.subplots(1, 3, figsize=(12, 3.6), sharey=True)
    for ax, d, xl in ((axes[0], gt, "temperature (°C)"), (axes[1], gn, "NMOS vth0 deviation (%)"),
                      (axes[2], gp, "PMOS |vth0| deviation (%)")):  # fmt: skip
        x = (d["lo"] + d["hi"]) / 2
        ax.plot(x, d["median_err_pct"], "o-", label="gate median |err|")
        ax.plot(x, d["p95_err_pct"], "s--", label="gate p95 |err|")
        ax.plot(x, d["bias_pct"], "^:", label="gate bias")
        ax.set_xlabel(xl)
        ax.grid(alpha=0.3)
        ax.axhline(0, color="k", lw=0.6)
    xc = (ct["lo"] + ct["hi"]) / 2
    axes[0].plot(xc, ct["median_err_pct"], "o-", color="0.4", label="circuit median |err|")
    axes[0].plot(xc, ct["bias_pct"], "^:", color="0.4", label="circuit bias")
    axes[0].set_ylabel("relative error (%)")
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT / "error_vs_corner.pdf", bbox_inches="tight")
    plotting.plt.close(fig)


def run() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = load_config(ROOT / "configs" / "project.yaml")
    card = load_model_card(cfg.assignment.model_file)
    gate_df = gate_vs_corner(cfg, card)
    gate_spice_physics(cfg)
    circuit_analysis(cfg)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # extrapolation warnings are what we measure below
        bank = TransistorModelBank.load_all(cfg.paths.models_dir)
        transistor_errors(cfg, bank)
        gate_extrapolation(cfg, bank, gate_df)
    figure(gate_df)
    print(f"analysis -> {OUT.relative_to(ROOT)}/ ({len(list(OUT.glob('*.csv')))} tables)")


if __name__ == "__main__":
    run()
