"""Generate everything the LaTeX report needs from the pipeline outputs, inside report/.

    uv run python report/make_numbers.py

Writes report/generated/numbers.tex (one macro per quoted number), report/generated/tab_*.tex
(booktabs tabulars; the section file adds caption and label), copies the plots into
report/figures/ and packs report/overleaf.zip for Overleaf's "Upload Project".

Reads results/tables/*.csv, artifacts/env.json, artifacts/splits/*.npz, configs/project.yaml and the
data CSVs, and writes one LaTeX macro per number the report quotes plus booktabs tables. No number
in the report is typed by hand: rerun this after `make all` and re-upload the zip.

A missing source file or column never crashes the script: the macro (or table) becomes
\\todo{missing: ...} and the script prints what is missing at the end.
"""

from __future__ import annotations

import json
import math
import shutil
import sys
import zipfile
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from leakml import io
from leakml.metrics import summarize

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "report"
OUT = REPORT / "generated"
FIGS = REPORT / "figures"
ZIP = REPORT / "overleaf.zip"
PLOTS = ROOT / "results" / "plots"
# Plots the report includes (results/plots/<name>.pdf -> report/figures/<name>.pdf).
FIGURES = [
    "trend_ioff_vs_temp", "trend_ioff_vs_vth0", "trend_ig_vs_toxe",  # D3
    "pred_vs_spice_nmos_on", "pred_vs_spice_nmos_off",  # D5
    "pred_vs_spice_pmos_on", "pred_vs_spice_pmos_off",
    "learning_curve", "learning_curve_median",
    "gate_parity", "gate_parity_direct",  # D7
    "circuit_parity",  # D8
]  # fmt: skip
TABLES = ROOT / "results" / "tables"
CASES = ["nmos_on", "nmos_off", "pmos_on", "pmos_off"]
TARGETS = ["id", "ig", "is", "ib"]
MODELS = ["poly_ridge", "random_forest", "gbdt", "mlp"]
MODEL_NAMES = {
    "poly_ridge": "Polynomial ridge",
    "random_forest": "Random forest",
    "gbdt": "GBDT",
    "mlp": "MLP",
}
TARGET_TEX = {"id": "$I_D$", "ig": "$I_G$", "is": "$I_S$", "ib": "$I_B$"}
CIRCUIT_NAMES = {"parity8": "8-bit parity checker", "cla3": "3-bit carry-lookahead adder"}

missing: list[str] = []
OUT.mkdir(parents=True, exist_ok=True)
for _old in OUT.glob("tab_*.tex"):  # drop tables from earlier runs / renamed tables
    _old.unlink()
macros: list[tuple[str, str, str]] = []  # (name, value, comment)


# ------------------------------------------------------------------ formatting


_TEX_SPECIAL = {
    "\\": r"\textbackslash{}", "_": r"\_", "%": r"\%", "&": r"\&", "#": r"\#", "$": r"\$",
    "{": r"\{", "}": r"\}", "^": r"\textasciicircum{}", "~": r"\textasciitilde{}",
}  # fmt: skip


def tex_escape(s: str) -> str:
    return "".join(_TEX_SPECIAL.get(ch, ch) for ch in str(s))


def camel(s: str) -> str:
    """nmos_on -> NmosOn, id -> Id (macro names may only contain letters)."""
    return "".join(p.capitalize() for p in str(s).split("_"))


def _finite(x) -> bool:
    try:
        return math.isfinite(float(x))
    except (TypeError, ValueError):
        return False


def pct(x, d: int = 2, plus: bool = False) -> str:
    if not _finite(x):
        return "--"
    x = float(x)
    if round(x, d) == 0:
        x = 0.0  # no "-0.00"
    opt = "[retain-explicit-plus]" if plus else ""
    return rf"\qty{opt}{{{x:+.{d}f}}}{{\percent}}" if plus else rf"\qty{{{x:.{d}f}}}{{\percent}}"


def num(x, d: int = 2) -> str:
    return rf"\num{{{x:.{d}f}}}" if _finite(x) else "--"


def sci(x, d: int = 3) -> str:
    return rf"\num{{{x:.{d - 1}e}}}" if _finite(x) else "--"


def amp(x, d: int = 3) -> str:
    return rf"\qty{{{x:.{d - 1}e}}}{{\ampere}}" if _finite(x) else "--"


def count(x) -> str:
    return rf"\num{{{int(x)}}}" if _finite(x) else "--"


def tt(s: str) -> str:
    return rf"\texttt{{{tex_escape(s)}}}"


def gate_case(gate: str, inputs: str) -> str:
    return rf"{tex_escape(gate)}~\texttt{{{tex_escape(inputs)}}}"


# ------------------------------------------------------------------ macro registry


def define(name: str, value: Callable[[], str] | str, what: str, comment: str = "") -> None:
    """Register \\name. `value` may be a thunk; any exception -> \\todo{missing: what}."""
    assert name.isalpha(), name
    try:
        v = value() if callable(value) else value
        if v is None or v == "--":
            raise ValueError("no value")
    except Exception as e:  # noqa: BLE001  (any missing file/column/row must not crash)
        missing.append(f"\\{name}: {what} ({type(e).__name__}: {e})")
        v = rf"\todo{{missing: {tex_escape(what)}}}"
    macros.append((name, v, comment))


def load_csv(name: str) -> pd.DataFrame | None:
    p = TABLES / f"{name}.csv"
    if not p.is_file():
        missing.append(f"file results/tables/{name}.csv")
        return None
    return io.read_csv(p)


def need(df: pd.DataFrame | None, *cols: str) -> pd.DataFrame:
    if df is None:
        raise FileNotFoundError("source table missing")
    absent = [c for c in cols if c not in df.columns]
    if absent:
        raise KeyError(f"columns {absent}")
    return df


def one(df: pd.DataFrame, mask) -> pd.Series:
    sub = df[mask]
    if len(sub) != 1:
        raise LookupError(f"expected 1 row, found {len(sub)}")
    return sub.iloc[0]


# ------------------------------------------------------------------ tables


def write_table(name: str, build: Callable[[], str], what: str) -> None:
    path = OUT / f"tab_{name}.tex"
    try:
        body = build()
    except Exception as e:  # noqa: BLE001
        missing.append(f"tab_{name}.tex: {what} ({type(e).__name__}: {e})")
        body = rf"\todo{{missing table: {tex_escape(what)}}}" + "\n"
    path.write_text(f"% Generated by report/make_numbers.py from {what}. Do not edit.\n{body}")


def tabular(colspec: str, header: list[str], rows: list[list[str]], midrules=()) -> str:
    lines = [
        rf"\begin{{tabular}}{{{colspec}}}",
        r"\toprule",
        " & ".join(header) + r" \\",
        r"\midrule",
    ]
    for i, r in enumerate(rows):
        if i in midrules:
            lines.append(r"\midrule")
        lines.append(" & ".join(r) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


# ================================================================== sources

cfg_raw: dict = {}
try:
    cfg_raw = yaml.safe_load((ROOT / "configs" / "project.yaml").read_text())
except Exception as e:  # noqa: BLE001
    missing.append(f"configs/project.yaml ({e})")
env: dict = {}
try:
    env = json.loads((ROOT / "artifacts" / "env.json").read_text())
except Exception as e:  # noqa: BLE001
    missing.append(f"artifacts/env.json ({e})")


def cfg(dotted: str):
    v = cfg_raw
    for k in dotted.split("."):
        v = v[k]
    return v


# ------------------------------------------------------------------ setup
define("TeamName", lambda: tex_escape(cfg("team.name")), "configs team.name")
define("TechNode", lambda: tex_escape(cfg("assignment.node")), "configs assignment.node")
define("ModelFile", lambda: tt(cfg("assignment.model_file")), "configs assignment.model_file")
define("VDDnom", lambda: rf"\qty{{{float(cfg('assignment.vdd_nominal')):g}}}{{\volt}}", "VDD")
define("VDDtol", lambda: pct(100 * float(cfg("sampling.vdd_tol")), 0), "sampling.vdd_tol")
define(
    "VDDmin",
    lambda: (
        rf"\qty{{{cfg('assignment.vdd_nominal') * (1 - cfg('sampling.vdd_tol')):.2f}}}{{\volt}}"
    ),
    "VDD range",
)
define(
    "VDDmax",
    lambda: (
        rf"\qty{{{cfg('assignment.vdd_nominal') * (1 + cfg('sampling.vdd_tol')):.2f}}}{{\volt}}"
    ),
    "VDD range",
)
define("Lmin", lambda: rf"\qty{{{float(cfg('assignment.lmin')) * 1e9:g}}}{{\nano\metre}}", "lmin")
define(
    "TestCircuit",
    lambda: CIRCUIT_NAMES[cfg("assignment.test_circuit")],
    "configs assignment.test_circuit",
)
define("TestCircuitKey", lambda: tt(cfg("assignment.test_circuit")), "test_circuit key")
define("TempMin", lambda: rf"\qty{{{cfg('sampling.temp_range_c')[0]:g}}}{{\degreeCelsius}}", "temp")
define("TempMax", lambda: rf"\qty{{{cfg('sampling.temp_range_c')[1]:g}}}{{\degreeCelsius}}", "temp")
define("WMultMin", lambda: f"{cfg('sampling.w_mult_range')[0]:g}", "w_mult_range")
define("WMultMax", lambda: f"{cfg('sampling.w_mult_range')[1]:g}", "w_mult_range")
define("ProcessTol", lambda: pct(100 * float(cfg("sampling.process_tol")), 0), "process_tol")
define("NmosWMult", lambda: f"{cfg('gates.sizing.nmos_w_mult'):g}", "gates.sizing")
define("PmosWMult", lambda: f"{cfg('gates.sizing.pmos_w_mult'):g}", "gates.sizing")
define("NumCorners", lambda: count(cfg("gates.n_corners")), "gates.n_corners")
define("IFloor", lambda: amp(float(cfg("ml.i_floor_a")), 1), "ml.i_floor_a")
define("SplitScheme", lambda: cfg("split.scheme").replace("_", "/"), "split.scheme")
for k in ("transistor_data", "split", "training", "corners"):
    define(f"Seed{camel(k)}", lambda k=k: str(cfg(f"seeds.{k}")), f"seeds.{k}")
define("SelectionMetric", lambda: tt(cfg("ml.selection_metric")), "ml.selection_metric")
define("SimVersion", lambda: tex_escape(env["simulator"]), "env.json simulator")
define("PythonVersion", lambda: env["python"], "env.json python")
for pkg in ("numpy", "pandas", "scikit-learn", "lightgbm"):
    define(f"Ver{camel(pkg.replace('-', '_'))}", lambda pkg=pkg: env["packages"][pkg], f"env {pkg}")
define("EnvCommit", lambda: tt(env["git_commit"][:7]), "env.json git_commit")
define("EnvDate", lambda: env["timestamp_utc"][:10], "env.json timestamp_utc")


def rows_per_case() -> str:
    n = {}
    for c in CASES:
        with open(ROOT / "data" / "transistor" / f"{c}.csv") as f:
            n[c] = sum(1 for _ in f) - 1
    if len(set(n.values())) != 1:
        raise ValueError(f"cases differ: {n}")
    return count(next(iter(n.values())))


define("RowsPerCase", rows_per_case, "data/transistor/*.csv row counts")


def split_size(part: str) -> str:
    sizes = {c: len(np.load(ROOT / "artifacts" / "splits" / f"{c}.npz")[part]) for c in CASES}
    if len(set(sizes.values())) != 1:
        raise ValueError(f"cases differ: {sizes}")
    return count(next(iter(sizes.values())))


for part in ("train", "val", "test"):
    define(f"N{camel(part)}", lambda p=part: split_size(p), f"artifacts/splits/*.npz {part}")


def gate_rows() -> int:
    with open(ROOT / "data" / "gates" / "gate_spice.csv") as f:
        return sum(1 for _ in f) - 1


define("NumGateSims", lambda: count(gate_rows()), "data/gates/gate_spice.csv rows")

# ------------------------------------------------------------------ Kirchhoff (D3)
kir = load_csv("kirchhoff")
for c in CASES:
    C = camel(c)
    define(
        f"KirchhoffMaxAbs{C}",
        lambda c=c: amp(one(need(kir, "case", "max_abs_r_a"), kir["case"] == c)["max_abs_r_a"], 2),
        f"kirchhoff {c} max_abs_r_a",
    )
    define(
        f"KirchhoffMaxRel{C}",
        lambda c=c: sci(one(need(kir, "case", "max_rel_r"), kir["case"] == c)["max_rel_r"], 2),
        f"kirchhoff {c} max_rel_r",
    )
define(
    "KirchhoffWorstAbs",
    lambda: amp(need(kir, "max_abs_r_a")["max_abs_r_a"].max(), 2),
    "kirchhoff max_abs_r_a",
)


def _kir_table() -> str:
    d = need(kir, "case", "median_abs_r_a", "p95_abs_r_a", "max_abs_r_a", "median_rel_r",
             "p95_rel_r", "max_rel_r")  # fmt: skip
    head = (
        r"\begin{tabular}{lrrrrrr}" "\n" r"\toprule" "\n"
        r" & \multicolumn{3}{c}{$|I_D+I_G+I_S+I_B|$ (A)} & \multicolumn{3}{c}{relative to $\max|I|$} \\"
        "\n" r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}" "\n"
        r"Case & median & p95 & max & median & p95 & max \\" "\n" r"\midrule" "\n"
    )  # fmt: skip
    body = "".join(
        " & ".join(
            [
                tt(r.case),
                sci(r.median_abs_r_a, 2),
                sci(r.p95_abs_r_a, 2),
                sci(r.max_abs_r_a, 2),
                sci(r.median_rel_r, 2),
                sci(r.p95_rel_r, 2),
                sci(r.max_rel_r, 2),
            ]
        )
        + r" \\"
        + "\n"  # fmt: skip
        for r in d.itertuples()
    )
    return head + body + r"\bottomrule" "\n" r"\end{tabular}" "\n"


write_table("kirchhoff", _kir_table, "results/tables/kirchhoff.csv")

# ------------------------------------------------------------------ models (D4, D5)
chosen = load_csv("chosen_models")
test = load_csv("chosen_models_test")
comp = load_csv("model_comparison")
comp_sum = load_csv("model_comparison_summary")


def test_value(c: str, t: str, col: str, fmt: Callable) -> str:
    """Test-set metric; "n/a" when every row is below i_floor (e.g. OFF-state I_G ~ 1e-23 A)."""
    r = one(need(test, "case", "target", col, "n", "n_excluded"),
            (test["case"] == c) & (test["target"] == t))  # fmt: skip
    if int(r["n_excluded"]) == int(r["n"]):
        return r"n/a"
    return fmt(r[col])


for c in CASES:
    C = camel(c)
    define(
        f"ChosenModel{C}",
        lambda c=c: MODEL_NAMES.get(m := one(need(chosen, "case", "model"), chosen["case"] == c)["model"], tex_escape(m)),
        f"chosen_models {c} model",
    )  # fmt: skip
    for col, nm in (
        ("val_median_err_pct", "ValMedianErr"),
        ("val_p95_err_pct", "ValPninetyfiveErr"),
    ):
        define(
            f"{nm}{C}",
            lambda c=c, col=col: pct(one(need(chosen, "case", col), chosen["case"] == c)[col], 1),
            f"chosen_models {c} {col}",
        )
    for t in TARGETS:
        T = camel(t)
        for col, nm, fmt in (
            ("median_err_pct", "TestMedianErr", lambda x: pct(x, 2)),
            ("p95_err_pct", "TestPninetyfiveErr", lambda x: pct(x, 1)),
            ("r2_log", "TestRtwo", lambda x: num(x, 4)),
            ("mean_acc_pct", "TestMeanAcc", lambda x: pct(x, 1)),
        ):
            define(
                f"{nm}{C}{T}",
                lambda c=c, t=t, col=col, fmt=fmt: test_value(c, t, col, fmt),
                f"chosen_models_test {c} {t} {col}",
            )  # fmt: skip


def _comparison_summary() -> str:
    d = need(comp_sum, "case", "model", "r2_log", "median_err_pct", "p95_err_pct", "mean_acc_pct")
    best = d.loc[d.groupby("case")["p95_err_pct"].idxmin()].set_index("case")["model"]
    rows, rules = [], []
    for c in CASES:
        sub = d[d["case"] == c].set_index("model").reindex(MODELS)
        if rows:
            rules.append(len(rows))
        for m, r in sub.iterrows():
            b = (lambda s: rf"\textbf{{{s}}}") if best.get(c) == m else (lambda s: s)
            rows.append([tt(c) if m == MODELS[0] else "", b(MODEL_NAMES[m]), b(num(r.r2_log, 4)),
                         b(pct(r.median_err_pct, 2)), b(pct(r.p95_err_pct, 1)),
                         b(pct(r.mean_acc_pct, 1))])  # fmt: skip
    return tabular("llrrrr", ["Case", "Model", r"$R^2$ ($\log|I|$)", "Median err.", "p95 err.",
                              "Mean acc."], rows, rules)  # fmt: skip


write_table("models", _comparison_summary,
            "results/tables/model_comparison_summary.csv (validation)")  # fmt: skip


def _comparison_full() -> str:
    d = need(comp, "case", "model", "target", "r2_log", "median_err_pct", "p95_err_pct")
    rows, rules = [], []
    for c in CASES:
        if rows:
            rules.append(len(rows))
        for m in MODELS:
            sub = d[(d["case"] == c) & (d["model"] == m)].set_index("target").reindex(TARGETS)
            cells = []
            for t in TARGETS:
                r = sub.loc[t]
                cells += [num(r.r2_log, 3), pct(r.median_err_pct, 1), pct(r.p95_err_pct, 0)]
            rows.append([tt(c) if m == MODELS[0] else "", MODEL_NAMES[m], *cells])
    head = (
        r"\begin{tabular}{ll" + "rrr" * 4 + "}\n" r"\toprule" "\n & & "
        + " & ".join(rf"\multicolumn{{3}}{{c}}{{{TARGET_TEX[t]}}}" for t in TARGETS) + r" \\" "\n"
        + "".join(rf"\cmidrule(lr){{{3 + 3 * i}-{5 + 3 * i}}}" for i in range(4)) + "\n"
        + "Case & Model" + r" & $R^2$ & med. & p95" * 4 + r" \\" "\n" r"\midrule" "\n"
    )  # fmt: skip
    body = ""
    for i, r in enumerate(rows):
        if i in rules:
            body += r"\midrule" "\n"
        body += " & ".join(r) + r" \\" "\n"
    return head + body + r"\bottomrule" "\n" r"\end{tabular}" "\n"


write_table("models_full", _comparison_full,
            "results/tables/model_comparison.csv (validation, per target)")  # fmt: skip


def _chosen_test() -> str:
    d = need(test, "case", "model", "target", "median_err_pct", "p95_err_pct", "r2_log",
             "mean_acc_pct", "bias_pct", "n_excluded")  # fmt: skip
    rows, rules = [], []
    for c in CASES:
        sub = d[d["case"] == c].set_index("target").reindex(TARGETS)
        if rows:
            rules.append(len(rows))
        for t, r in sub.iterrows():
            if int(r.n_excluded) == int(r.n):  # every row below i_floor: no relative error exists
                rows.append([tt(c) if t == TARGETS[0] else "",
                             MODEL_NAMES.get(r.model, tex_escape(r.model)) if t == TARGETS[0] else "",
                             TARGET_TEX[t], rf"\multicolumn{{5}}{{c}}{{all {count(r.n)} rows "
                             rf"below \IFloor{{}} (median $|I|$ = {sci(abs(r.spice_median), 2)}\,A)}}",
                             count(r.n_excluded)])  # fmt: skip
                continue
            rows.append([
                tt(c) if t == TARGETS[0] else "",
                MODEL_NAMES.get(r.model, tex_escape(r.model)) if t == TARGETS[0] else "",
                TARGET_TEX[t], pct(r.median_err_pct, 2), pct(r.p95_err_pct, 1),
                num(r.r2_log, 4), pct(r.mean_acc_pct, 1), pct(r.bias_pct, 2, plus=True),
                count(r.n_excluded),
            ])  # fmt: skip
    return tabular("lllrrrrrr", ["Case", "Model", "Current", "Median err.", "p95 err.",
                                 r"$R^2$", "Mean acc.", "Bias", "Excl."], rows, rules)  # fmt: skip


write_table("chosen", _chosen_test, "results/tables/chosen_models_test.csv (test set)")

# ------------------------------------------------------------------ learning curve
lc = load_csv("learning_curve")
lc_lines: list[str] = []
try:
    d = need(lc, "case", "target", "n_train", "median_err_pct", "p95_err_pct")
    sizes = sorted(d["n_train"].astype(int).unique())
    define("LCSizes", ", ".join(rf"\num{{{s}}}" for s in sizes), "learning curve sizes")
    define("LCMaxSize", count(max(sizes)), "learning curve max size")
    for r in d.itertuples():
        key = f"{r.case}@{r.target}@{int(r.n_train)}"
        lc_lines.append(
            rf"\expandafter\def\csname lcmed@{key}\endcsname{{{pct(r.median_err_pct, 2)}}}"
        )
        lc_lines.append(
            rf"\expandafter\def\csname lcpnf@{key}\endcsname{{{pct(r.p95_err_pct, 1)}}}"
        )
except Exception as e:  # noqa: BLE001
    missing.append(f"learning curve macros ({type(e).__name__}: {e})")


def _learning_curve() -> str:
    d = need(lc, "case", "target", "n_train", "median_err_pct")
    sizes = sorted(d["n_train"].astype(int).unique())
    piv = d.pivot_table(index=["case", "target"], columns="n_train", values="median_err_pct")
    rows, rules = [], []
    for c in CASES:
        if rows:
            rules.append(len(rows))
        for t in TARGETS:
            r = piv.loc[(c, t)] if (c, t) in piv.index else pd.Series(dtype=float)
            rows.append([tt(c) if t == TARGETS[0] else "", TARGET_TEX[t],
                         *[num(r.get(s, np.nan), 1) for s in sizes]])  # fmt: skip
    head = (
        r"\begin{tabular}{ll" + "r" * len(sizes) + "}\n" r"\toprule" "\n"
        + rf" & & \multicolumn{{{len(sizes)}}}{{c}}{{Training rows}} \\" "\n"
        + rf"\cmidrule(lr){{3-{2 + len(sizes)}}}" "\n"
        + "Case & Current & " + " & ".join(rf"\num{{{s}}}" for s in sizes) + r" \\" "\n"
        + r"\midrule" "\n"
    )  # fmt: skip
    body = ""
    for i, r in enumerate(rows):
        if i in rules:
            body += r"\midrule" "\n"
        body += " & ".join(r) + r" \\" "\n"
    return head + body + r"\bottomrule" "\n" r"\end{tabular}" "\n"


write_table("learning", _learning_curve,
            "results/tables/learning_curve.csv (validation median error, %)")  # fmt: skip

# ------------------------------------------------------------------ gates (D6)
gt = load_csv("gate_results")
gw = load_csv("gate_worst_cases")


def gate_all(col: str):
    d = need(gt, "gate", "inputs", col)
    return one(d, (d["gate"] == "ALL") & (d["inputs"] == "ALL"))[col]


for nm, col, fmt in (
    ("GateMeanAcc", "mean_acc_pct", lambda x: pct(x, 2)),
    ("GateMedianErr", "median_err_pct", lambda x: pct(x, 2)),
    ("GatePninetyfiveErr", "p95_err_pct", lambda x: pct(x, 2)),
    ("GateMaxErr", "max_err_pct", lambda x: pct(x, 1)),
    ("GateBias", "bias_pct", lambda x: pct(x, 2, plus=True)),
    ("GateWithinOne", "within_1pct", lambda x: pct(x, 1)),
    ("GateWithinTen", "within_10pct", lambda x: pct(x, 2)),
    ("GateDirectMedianErr", "median_err_direct_pct", lambda x: pct(x, 2)),
    ("GateIinMedianErr", "median_err_i_in_pct", lambda x: pct(x, 2)),
    ("GateSpiceMedian", "spice_median", lambda x: amp(x, 3)),
    ("GateNumRows", "n", count),
):
    define(nm, lambda col=col, fmt=fmt: fmt(gate_all(col)), f"gate_results ALL {col}")


def gate_extreme(best: bool):
    d = need(gt, "gate", "inputs", "mean_acc_pct")
    d = d[d["gate"] != "ALL"]
    return d.loc[d["mean_acc_pct"].idxmax() if best else d["mean_acc_pct"].idxmin()]


for nm, best in (("Best", True), ("Worst", False)):
    define(f"Gate{nm}Case", lambda b=best: gate_case(*gate_extreme(b)[["gate", "inputs"]]),
           f"gate_results {nm.lower()} case")  # fmt: skip
    define(f"Gate{nm}Acc", lambda b=best: pct(gate_extreme(b)["mean_acc_pct"], 2),
           f"gate_results {nm.lower()} mean_acc")  # fmt: skip


def gate_direct_overall(col: str):
    """Direct-method metrics over all gate rows (not in the D6 csv, computed from gate_results)."""
    g = io.read_csv(ROOT / "data" / "gates" / "gate_results.csv")
    s = summarize(g["i_vdd_ml_direct"].to_numpy(), g["i_vdd_spice"].to_numpy(),
                  i_floor=float(cfg("ml.i_floor_a")))  # fmt: skip
    return s[col]


define("GateDirectMeanAcc", lambda: pct(gate_direct_overall("mean_acc_pct"), 2),
       "data/gates/gate_results.csv direct mean acc")  # fmt: skip
define("GateDirectPninetyfiveErr", lambda: pct(gate_direct_overall("p95_err_pct"), 1),
       "data/gates/gate_results.csv direct p95")  # fmt: skip

define("GateWorstRowCase", lambda: gate_case(*need(gw, "gate", "inputs").iloc[0][["gate", "inputs"]]),
       "gate_worst_cases first row")  # fmt: skip
define("GateWorstRowErr", lambda: pct(need(gw, "err_pct").iloc[0]["err_pct"], 1, plus=True),
       "gate_worst_cases err_pct")  # fmt: skip
define("GateWorstRowTemp",
       lambda: rf"\qty{{{need(gw, 'temp').iloc[0]['temp']:.1f}}}{{\degreeCelsius}}",
       "gate_worst_cases temp")  # fmt: skip
define("GateWorstRowVDD", lambda: rf"\qty{{{need(gw, 'vdd').iloc[0]['vdd']:.3f}}}{{\volt}}",
       "gate_worst_cases vdd")  # fmt: skip


def _gate_table() -> str:
    d = need(gt, "gate", "inputs", "spice_median", "ml_median", "bias_pct", "median_err_pct",
             "p95_err_pct", "mean_acc_pct")  # fmt: skip
    rows, rules, prev = [], [], None
    for r in d.itertuples():
        if prev is not None and r.gate != prev:
            rules.append(len(rows))
        first = r.gate != prev
        prev = r.gate
        rows.append([
            r"\textbf{All}" if r.gate == "ALL" else (tex_escape(r.gate) if first else ""),
            "" if r.gate == "ALL" else tt(r.inputs),
            sci(r.spice_median, 3), sci(r.ml_median, 3), pct(r.bias_pct, 2, plus=True),
            pct(r.median_err_pct, 2), pct(r.p95_err_pct, 2), pct(r.mean_acc_pct, 2),
        ])  # fmt: skip
    return tabular("llrrrrrr", ["Gate", "Inputs", "SPICE median (A)", "ML median (A)", "Bias",
                                "Median err.", "p95 err.", "Mean acc."], rows, rules)  # fmt: skip


write_table("gates", _gate_table, "results/tables/gate_results.csv")


def _gate_worst() -> str:
    d = need(gw, "gate", "inputs", "corner_id", "temp", "vdd", "i_vdd_spice", "i_vdd_ml", "err_pct")
    rows = [
        [
            tex_escape(r.gate),
            tt(r.inputs),
            str(int(r.corner_id)),
            rf"\num{{{r.temp:.1f}}}",
            rf"\num{{{r.vdd:.3f}}}",
            sci(r.i_vdd_spice, 3),
            sci(r.i_vdd_ml, 3),
            pct(r.err_pct, 1, plus=True),
        ]  # fmt: skip
        for r in d.itertuples()
    ]
    return tabular("llrrrrrr", ["Gate", "Inputs", "Corner", r"$T$ (\si{\degreeCelsius})",
                                r"$V_{DD}$ (V)", "SPICE (A)", "ML (A)", "Error"], rows)  # fmt: skip


write_table("gate_worst", _gate_worst, "results/tables/gate_worst_cases.csv")

# ------------------------------------------------------------------ circuit (D8)
cr = load_csv("circuit_results")


def circ_row(pattern: str) -> pd.Series:
    d = need(cr, "scope")
    return one(d, d["scope"].str.contains(pattern, regex=False))


for prefix, pattern in (("Circuit", "(all vectors)"), ("CircuitNoLoad", "no loading term")):
    for nm, col, fmt in (
        ("MeanAcc", "mean_acc_pct", lambda x: pct(x, 2)),
        ("MedianErr", "median_err_pct", lambda x: pct(x, 2)),
        ("PninetyfiveErr", "p95_err_pct", lambda x: pct(x, 2)),
        ("MaxErr", "max_err_pct", lambda x: pct(x, 1)),
        ("Bias", "bias_pct", lambda x: pct(x, 2, plus=True)),
    ):
        define(f"{prefix}{nm}", lambda p=pattern, col=col, fmt=fmt: fmt(circ_row(p)[col]),
               f"circuit_results '{pattern}' {col}")  # fmt: skip
define("CircuitNumSims", lambda: count(circ_row("(all vectors)")["n"]), "circuit_results n")
define("CircuitSpiceMedian", lambda: amp(circ_row("(all vectors)")["spice_median"], 3),
       "circuit_results spice_median")  # fmt: skip
define("CircuitIinMedianErr", lambda: pct(circ_row("primary-input current")["median_err_pct"], 2),
       "circuit_results primary-input current row")  # fmt: skip


def _circuit_table() -> str:
    d = need(cr, "scope", "n", "spice_median", "ml_median", "bias_pct", "median_err_pct",
             "p95_err_pct", "max_err_pct", "mean_acc_pct")  # fmt: skip
    main = d[~d["scope"].str.contains("no loading|primary-input|gate level", regex=True)]
    extra = d[d["scope"].str.contains("no loading|primary-input|gate level", regex=True)]
    rows = []
    for r in pd.concat([main, extra]).itertuples():
        scope = (
            rf"{r.scope.split('=')[1].strip()} ones in the input vector"
            if r.scope.strip().startswith("ones =")
            else tex_escape(r.scope.strip())
        )
        rows.append([scope, count(r.n), sci(r.spice_median, 3), sci(r.ml_median, 3),
                     pct(r.bias_pct, 2, plus=True), pct(r.median_err_pct, 2),
                     pct(r.p95_err_pct, 2), pct(r.max_err_pct, 1), pct(r.mean_acc_pct, 2)])  # fmt: skip
    rules = [1, len(main)] if len(main) > 1 else [len(main)]
    return tabular("lrrrrrrrr", ["Scope", "$n$", "SPICE median (A)", "ML median (A)", "Bias",
                                 "Median err.", "p95 err.", "Max err.", "Mean acc."],
                   rows, rules)  # fmt: skip


write_table("circuit", _circuit_table, "results/tables/circuit_results.csv")

# ================================================================== write numbers.tex
lines = [
    "% Generated by report/make_numbers.py. Do not edit: rerun the script after `make all`.",
    "% Sources: results/tables/*.csv, artifacts/env.json, configs/project.yaml, data/*.csv",
    "",
]
for name, value, comment in macros:
    lines.append(rf"\newcommand{{\{name}}}{{{value}}}" + (f"  % {comment}" if comment else ""))
lines += [
    "",
    "% Learning curve: \\LCMedian{case}{target}{n_train} and \\LCPninetyfive{case}{target}{n_train},",
    "% e.g. \\LCMedian{nmos_off}{id}{7000} (validation error of the chosen model type).",
    r"\makeatletter",
    r"\newcommand{\LCMedian}[3]{\@ifundefined{lcmed@#1@#2@#3}{\todo{missing: learning curve \detokenize{#1 #2 #3}}}{\csname lcmed@#1@#2@#3\endcsname}}",
    r"\newcommand{\LCPninetyfive}[3]{\@ifundefined{lcpnf@#1@#2@#3}{\todo{missing: learning curve \detokenize{#1 #2 #3}}}{\csname lcpnf@#1@#2@#3\endcsname}}",
    r"\makeatother",
    *lc_lines,
    "",
]
(OUT / "numbers.tex").write_text("\n".join(lines))

# ================================================================== figures
if FIGS.exists():
    shutil.rmtree(FIGS)
FIGS.mkdir()
for fig in FIGURES:
    src = PLOTS / f"{fig}.pdf"
    if src.is_file():
        shutil.copy2(src, FIGS / src.name)
    else:
        missing.append(f"figure results/plots/{fig}.pdf (not copied; \\includegraphics will fail)")

# ================================================================== Overleaf zip
ZIP_CONTENT = ["main.tex", "sections", "generated", "figures"]
ZIP.unlink(missing_ok=True)
with zipfile.ZipFile(ZIP, "w", zipfile.ZIP_DEFLATED) as zf:
    for item in ZIP_CONTENT:
        p = REPORT / item
        if not p.exists():
            missing.append(f"report/{item} (not in overleaf.zip)")
            continue
        files = [p] if p.is_file() else sorted(f for f in p.rglob("*") if f.is_file())
        for f in files:
            if f.name.startswith("."):
                continue
            zf.write(f, f.relative_to(REPORT).as_posix())
    n_zip = len(zf.namelist())

n_tab = len(list(OUT.glob("tab_*.tex")))
print(f"wrote {len(macros)} macros + {len(lc_lines) // 2} learning-curve entries -> "
      f"{(OUT / 'numbers.tex').relative_to(ROOT)}, {n_tab} tables -> {OUT.relative_to(ROOT)}/")  # fmt: skip
print(f"copied {len(list(FIGS.glob('*.pdf')))} figures -> {FIGS.relative_to(ROOT)}/")
print(f"packed {n_zip} files -> {ZIP}")
if missing:
    print(f"\nMISSING ({len(missing)}):")
    for m in missing:
        print("  -", m)
    sys.exit(0)
print("nothing missing")
