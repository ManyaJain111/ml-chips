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
import subprocess
import sys
import zipfile
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from leakml import io, sampling
from leakml.constants import Case
from leakml.metrics import summarize
from leakml.ml import zoo
from leakml.ml.features import TargetTransform, build_features
from leakml.ml.train import SHAPED_CASES, TARGET_KIND

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

# Section 6 analysis (report/make_analysis.py) -> report/generated/analysis/*.csv + figure
AN = OUT / "analysis"
try:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import make_analysis  # noqa: E402

    make_analysis.run()
except Exception as _e:  # noqa: BLE001
    missing.append(f"report/make_analysis.py failed ({type(_e).__name__}: {_e})")
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
KIR_STATS = (("median_abs_r_a", "MedianAbs", True), ("p95_abs_r_a", "PninetyfiveAbs", True),
             ("median_rel_r", "MedianRel", False), ("p95_rel_r", "PninetyfiveRel", False))  # fmt: skip
for c in CASES:
    for col, nm, is_amp in KIR_STATS:
        define(
            f"Kirchhoff{nm}{camel(c)}",
            lambda c=c, col=col, a=is_amp: (amp if a else sci)(
                one(need(kir, "case", col), kir["case"] == c)[col], 2),
            f"kirchhoff {c} {col}",
        )  # fmt: skip
define("KirchhoffWorstRel", lambda: sci(need(kir, "max_rel_r")["max_rel_r"].max(), 2),
       "kirchhoff max_rel_r")  # fmt: skip
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


def _circuit_scope(raw: str) -> str:
    """Short row labels so the table fits the text width."""
    t = raw.strip()
    if t.startswith("ones ="):
        return rf"\quad {t.split('=')[1].strip()} ones"
    for key, label in (("(all vectors)", "all vectors"), ("no loading", "without loading term"),
                       ("primary-input", r"primary-input current $I_{in}$"),
                       ("gate level", "gate level (all gates)")):  # fmt: skip
        if key in t:
            return label
    return tex_escape(t)


def _circuit_table() -> str:
    d = need(cr, "scope", "n", "spice_median", "ml_median", "bias_pct", "median_err_pct",
             "p95_err_pct", "max_err_pct", "mean_acc_pct")  # fmt: skip
    main = d[~d["scope"].str.contains("no loading|primary-input|gate level", regex=True)]
    extra = d[d["scope"].str.contains("no loading|primary-input|gate level", regex=True)]
    rows = []
    for r in pd.concat([main, extra]).itertuples():
        scope = _circuit_scope(r.scope)
        rows.append([scope, count(r.n), sci(r.spice_median, 3), sci(r.ml_median, 3),
                     pct(r.bias_pct, 2, plus=True), pct(r.median_err_pct, 2),
                     pct(r.p95_err_pct, 2), pct(r.max_err_pct, 1), pct(r.mean_acc_pct, 2)])  # fmt: skip
    rules = [1, len(main)] if len(main) > 1 else [len(main)]
    return tabular("lrrrrrrrr", ["Scope", "$n$", "SPICE median (A)", "ML median (A)", "Bias",
                                 "Median err.", "p95 err.", "Max err.", "Mean acc."],
                   rows, rules)  # fmt: skip


write_table("circuit", _circuit_table, "results/tables/circuit_results.csv")


# ------------------------------------------------------------------ sampling rules (code constants)
# Design constants of leakml.sampling (not results), generated so the text cannot drift from code.
define("LowVdsFrac", lambda: pct(100 * sampling.LOW_VDS_FRAC, 0), "sampling.LOW_VDS_FRAC")
define("LowVdsMin", lambda: rf"\qty{{{sampling.LOW_VDS_RANGE[0]:.0e}}}{{\volt}}", "LOW_VDS_RANGE")
define("LowVdsMax", lambda: rf"\qty{{{sampling.LOW_VDS_RANGE[1]:.0e}}}{{\volt}}", "LOW_VDS_RANGE")
define("SourcePinFrac", lambda: pct(100 * sampling.SOURCE_PIN_FRAC, 0), "SOURCE_PIN_FRAC")
define(
    "PinMin", lambda: rf"\qty{{{sampling.SOURCE_PIN_RANGE[0]:.0e}}}{{\volt}}", "SOURCE_PIN_RANGE"
)
define(
    "PinMax", lambda: rf"\qty{{{sampling.SOURCE_PIN_RANGE[1]:.0e}}}{{\volt}}", "SOURCE_PIN_RANGE"
)
define("DrainPinFrac", lambda: pct(100 * sampling.DRAIN_PIN_FRAC, 0), "DRAIN_PIN_FRAC")
define("BothPinFrac", lambda: pct(100 * sampling.DRAIN_PIN_FRAC * sampling.SOURCE_PIN_FRAC, 0),
       "DRAIN_PIN_FRAC * SOURCE_PIN_FRAC")  # fmt: skip
define("BoundaryFrac", lambda: pct(100 * sampling.BOUNDARY_FRAC, 0), "BOUNDARY_FRAC")
define("BoundaryPerInput", lambda: pct(100 * sampling.BOUNDARY_PER_INPUT, 0), "BOUNDARY_PER_INPUT")
define("BoundaryBand", lambda: pct(100 * sampling.BOUNDARY_BAND, 0), "BOUNDARY_BAND")
define("BoundaryEdgeShare",
       lambda: pct(100 * (sampling.BOUNDARY_FRAC * sampling.BOUNDARY_PER_INPUT
                          + (1 - sampling.BOUNDARY_FRAC * sampling.BOUNDARY_PER_INPUT)
                          * 2 * sampling.BOUNDARY_BAND), 0),
       "share of rows within BOUNDARY_BAND of an edge, per input")  # fmt: skip
define("BoundaryEdgeShareUniform", lambda: pct(100 * 2 * sampling.BOUNDARY_BAND, 0),
       "uniform share within BOUNDARY_BAND of an edge")  # fmt: skip
define("BoundaryInputs", lambda: ", ".join(tt(c) for c in sampling.BOUNDARY_INPUTS),
       "BOUNDARY_INPUTS")  # fmt: skip


# ------------------------------------------------------------------ features / hyperparameters
def _num_features() -> str:
    head = io.read_csv(ROOT / "data" / "transistor" / "nmos_off.csv").head(3)
    return str(build_features(head, "nmos_off").shape[1])


define("NumFeatures", _num_features, "build_features column count")
define("AsinhQuantile", lambda: pct(100 * TargetTransform().quantile, 0),
       "TargetTransform.quantile")  # fmt: skip
define("TargetKind", lambda: tt(TARGET_KIND), "train.TARGET_KIND")
define("ShapedCases", lambda: " and ".join(tt(c) for c in CASES if Case(c) in SHAPED_CASES),
       "train.SHAPED_CASES")  # fmt: skip


def _grid_size(name: str) -> int:
    return int(np.prod([len(v) for v in zoo.search_space(name).values()]))


define("GridPoints", lambda: str(_grid_size(MODELS[0])) if len({_grid_size(m) for m in MODELS}) == 1
       else "/".join(str(_grid_size(m)) for m in MODELS), "zoo.search_space sizes")  # fmt: skip


def _hp_value(k: str, v) -> str:
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, tuple):
        return r"$" + r"\times".join(str(x) for x in v) + r"$"
    if isinstance(v, float) and v != 0 and (abs(v) < 1e-2 or abs(v) >= 1e4):
        return rf"\num{{{v:.0e}}}"
    return tex_escape(f"{v:g}" if isinstance(v, float) else v)


HP_NAMES = {
    "degree": "degree", "alpha": r"$\alpha$", "n_estimators": "trees",
    "max_features": "max.\\ features", "min_samples_leaf": "min.\\ leaf",
    "learning_rate": "learning rate", "num_leaves": "leaves",
    "min_child_samples": "min.\\ child samples", "subsample": "row subsample",
    "colsample_bytree": "feature subsample", "linear_tree": "linear leaves",
    "linear_lambda": r"linear-leaf $\lambda$", "hidden_layer_sizes": "layers",
    "learning_rate_init": "Adam lr", "batch_size": "batch",
    "n_iter_no_change": "early-stop patience (epochs)",
    "validation_fraction": "early-stop hold-out",
}  # fmt: skip
HP_SKIP = {"n_jobs", "subsample_freq", "early_stopping", "max_iter"}


def _hyper_table() -> str:
    rows = []
    for m in MODELS:
        d = zoo.default_hyperparams(m)
        grid = zoo.search_space(m)
        defaults = ", ".join(f"{HP_NAMES.get(k, tex_escape(k))} {_hp_value(k, v)}"
                             for k, v in d.items() if k not in HP_SKIP and k not in grid)  # fmt: skip
        g = "; ".join(f"{HP_NAMES.get(k, tex_escape(k))} $\\in$ \\{{"
                      + ", ".join(_hp_value(k, x) for x in v) + "\\}" for k, v in grid.items())  # fmt: skip
        rows.append([MODEL_NAMES[m], defaults or "--", g])
    colspec = (r">{\raggedright\arraybackslash}p{0.17\linewidth}"
               r">{\raggedright\arraybackslash}p{0.43\linewidth}"
               r">{\raggedright\arraybackslash}p{0.34\linewidth}")  # fmt: skip
    return tabular(colspec, ["Model", "Fixed hyperparameters", "Validation grid"], rows)


write_table("hyper", _hyper_table, "leakml.ml.zoo defaults and search_space")


def _chosen_hp(c: str) -> str:
    import ast

    r = one(need(chosen, "case", "model", "hyperparams"), chosen["case"] == c)
    hp = ast.literal_eval(r["hyperparams"])
    grid = zoo.search_space(r["model"])
    return ", ".join(f"{HP_NAMES.get(k, tex_escape(k))} {_hp_value(k, hp[k])}" for k in grid)


for c in CASES:
    define(f"ChosenHyper{camel(c)}", lambda c=c: _chosen_hp(c), f"chosen_models {c} hyperparams")


def _chosen_summary() -> str:
    d = need(chosen, "case", "model")
    models = d.set_index("case")["model"].reindex(CASES)
    if models.nunique() == 1:
        return f"{MODEL_NAMES.get(models.iloc[0], tex_escape(models.iloc[0]))} in all four cases"
    return "; ".join(f"{tt(c)}: {MODEL_NAMES.get(m, tex_escape(m))}" for c, m in models.items())


define("ChosenSummary", _chosen_summary, "chosen_models model column")


def _kcl_summary() -> str:
    d = need(chosen, "case", "is_kcl")
    k = d.set_index("case")["is_kcl"].astype(str).str.lower().eq("true").reindex(CASES)
    if k.all():
        return "all four cases"
    if not k.any():
        return "none of the cases"
    return ", ".join(tt(c) for c in CASES if k[c])


define("KclCases", _kcl_summary, "chosen_models is_kcl")

# Direct vs derived I_S: best grid point of the chosen model type, mean val p95 over currents.
search = load_csv("model_search")


def _kcl_variant(c: str, kcl: bool) -> str:
    d = need(search, "case", "model", "is_kcl", "p95_err_pct")
    m = one(need(chosen, "case", "model"), chosen["case"] == c)["model"]
    sel = d[(d["case"] == c) & (d["model"] == m)
            & (d["is_kcl"].astype(str).str.lower() == str(kcl).lower())]  # fmt: skip
    if sel.empty:
        raise LookupError("no rows")
    return pct(sel["p95_err_pct"].min(), 1)


for c in CASES:
    define(f"KclDirectPninetyfive{camel(c)}", lambda c=c: _kcl_variant(c, False),
           f"model_search {c} is_kcl=False")  # fmt: skip
    define(f"KclDerivedPninetyfive{camel(c)}", lambda c=c: _kcl_variant(c, True),
           f"model_search {c} is_kcl=True")  # fmt: skip

# Linear-leaf improvement: the last run with constant-leaf GBDT is commit CONST_LEAF_COMMIT
# (same data and grid; the only code change up to the linear-leaf run is linear_tree in zoo.py).
CONST_LEAF_COMMIT = "e7eb562"


def git_csv(path: str) -> pd.DataFrame:
    import io as _io

    out = subprocess.run(["git", "-C", str(ROOT), "show", f"{CONST_LEAF_COMMIT}:{path}"],
                         capture_output=True, text=True, check=True).stdout  # fmt: skip
    return pd.read_csv(_io.StringIO(out), dtype={"inputs": str})


try:
    const_chosen = git_csv("results/tables/chosen_models.csv")
    const_gate = git_csv("results/tables/gate_results.csv")
except Exception as e:  # noqa: BLE001
    missing.append(f"constant-leaf baseline (git show {CONST_LEAF_COMMIT}): {e}")
    const_chosen = const_gate = None

define("ConstLeafCommit", tt(CONST_LEAF_COMMIT), "constant-leaf commit")
for c in CASES:
    for col, nm, d in (
        ("val_median_err_pct", "ValMedianErr", 1),
        ("val_p95_err_pct", "ValPninetyfiveErr", 1),
    ):
        define(f"ConstLeaf{nm}{camel(c)}",
               lambda c=c, col=col, d=d: pct(one(need(const_chosen, "case", col),
                                                 const_chosen["case"] == c)[col], d),
               f"{CONST_LEAF_COMMIT} chosen_models {c} {col}")  # fmt: skip
define("ConstLeafGateMeanAcc",
       lambda: pct(one(need(const_gate, "gate", "mean_acc_pct"),
                       const_gate["gate"] == "ALL")["mean_acc_pct"], 2),
       f"{CONST_LEAF_COMMIT} gate_results ALL mean_acc_pct")  # fmt: skip


def _linear_table() -> str:
    a = need(const_chosen, "case", "model", "val_median_err_pct", "val_p95_err_pct").set_index(
        "case"
    )
    b = need(chosen, "case", "model", "val_median_err_pct", "val_p95_err_pct").set_index("case")
    rows = [[tt(c), pct(a.loc[c, "val_median_err_pct"], 2), pct(b.loc[c, "val_median_err_pct"], 2),
             pct(a.loc[c, "val_p95_err_pct"], 1), pct(b.loc[c, "val_p95_err_pct"], 1)]
            for c in CASES]  # fmt: skip
    g = need(const_gate, "gate", "mean_acc_pct")
    rows.append([r"Gate mean acc.\ (all)", r"\multicolumn{2}{c}{"
                 + pct(one(g, g["gate"] == "ALL")["mean_acc_pct"], 2) + r" $\to$ "
                 + pct(gate_all("mean_acc_pct"), 2) + "}", "", ""])  # fmt: skip
    head = (r"\begin{tabular}{lrrrr}" "\n" r"\toprule" "\n"
            r" & \multicolumn{2}{c}{Val.\ median err.} & \multicolumn{2}{c}{Val.\ p95 err.} \\"
            "\n" r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}" "\n"
            r"Case & constant & linear & constant & linear \\" "\n" r"\midrule" "\n")  # fmt: skip
    body = "".join(" & ".join(r) + r" \\" "\n" + (r"\midrule" "\n" if i == 3 else "")
                   for i, r in enumerate(rows))  # fmt: skip
    return head + body + r"\bottomrule" "\n" r"\end{tabular}" "\n"


write_table("linear_leaves", _linear_table,
            f"chosen_models.csv and gate_results.csv at {CONST_LEAF_COMMIT} vs now")  # fmt: skip


# ------------------------------------------------------------------ example netlists (verbatim)
NETLISTS = ROOT / "netlists" / "examples"


def _netlist_excerpt(name: str) -> str:
    """The deck with every .model block cut down to the overridden/sampled parameters."""
    import re

    keep = {"vth0", "toxe", "ndep", "u0", "xj", "rgatemod"}
    out, in_model = [], False
    for ln in (NETLISTS / name).read_text().splitlines():
        if in_model and not ln.startswith("+"):
            out.append("+ ... (all other parameters as in the model card)")
            in_model = False
        if ln.lower().startswith(".model"):
            in_model = True
            out.append(ln)
        elif in_model:
            hit = [
                f"{k}={v}" for k, v in re.findall(r"(\w+)\s*=\s*(\S+)", ln[1:]) if k.lower() in keep
            ]
            if hit:
                out.append("+ " + " ".join(hit))
        else:
            out.append(ln)
    if any(len(x) > 95 for x in out):
        raise ValueError("netlist line too long for verbatim")
    return "\n".join(out)


for _src, _dst in (("fet_nmos_off.sp", "netlist_fet_nmos_off.tex"),
                   ("gate_NAND2.sp", "netlist_gate_nand2.tex")):  # fmt: skip
    try:
        (OUT / _dst).write_text(
            f"% Generated by report/make_numbers.py from netlists/examples/{_src}.\n"
            "\\begin{verbatim}\n" + _netlist_excerpt(_src) + "\n\\end{verbatim}\n"
        )
    except Exception as e:  # noqa: BLE001
        missing.append(f"netlists/examples/{_src} excerpt ({type(e).__name__}: {e})")
        (OUT / _dst).write_text(rf"\todo{{missing: example netlist {tex_escape(_src)}}}" + "\n")

NETLIST_WHAT = {"fet": "single transistor", "gate": "gate", "circuit": "test circuit"}


def _netlist_table() -> str:
    files = sorted(NETLISTS.glob("*.sp"), key=lambda f: (["fet", "gate", "circuit"].index(
        f.stem.split("_")[0]), f.stem))  # fmt: skip
    if not files:
        raise FileNotFoundError("no netlists/examples/*.sp")
    rows = []
    for f in files:
        kind, rest = f.stem.split("_", 1)
        first = f.read_text().splitlines()[0].lstrip("* ").strip()
        rows.append([tt(f.name), NETLIST_WHAT.get(kind, kind), tex_escape(rest), tex_escape(first),
                     count(len(f.read_text().splitlines()))])  # fmt: skip
    return tabular("lllll", ["File", "Level", "Case", "Deck title", "Lines"], rows)


write_table("netlists", _netlist_table, "netlists/examples/*.sp")


# ------------------------------------------------------------------ gates: extra (Section 4)
GATE_DATA = ROOT / "data" / "gates"
ANALYSIS = ROOT / "report" / "analysis"


def _gate_rows() -> pd.DataFrame:
    return io.read_csv(GATE_DATA / "gate_results.csv")


def _logic_ok(path: Path) -> str:
    d = io.read_csv(path)
    ok = d["logic_ok"].astype(str).str.lower().eq("true")
    return rf"\num{{{int(ok.sum())}}}/\num{{{len(d)}}}"


define("GateLogicOk", lambda: _logic_ok(GATE_DATA / "gate_spice.csv"), "gate_spice logic_ok")
define("GateSnapMax",
       lambda: rf"\qty{{{_gate_rows()['max_vg_snap_err'].max():.1e}}}{{\volt}}",
       "gate_results max_vg_snap_err")  # fmt: skip


def _iin_summary() -> dict:
    g = _gate_rows()
    return summarize(g["i_in_ml"].to_numpy(), g["i_in_spice"].to_numpy(),
                     i_floor=float(cfg("ml.i_floor_a")))  # fmt: skip


define("GateIinMeanAcc", lambda: pct(_iin_summary()["mean_acc_pct"], 2), "gate i_in mean acc")
define("GateIinPninetyfiveErr", lambda: pct(_iin_summary()["p95_err_pct"], 1), "gate i_in p95")
define("GateIinBias", lambda: pct(_iin_summary()["bias_pct"], 2, plus=True), "gate i_in bias")


def _iin_share() -> float:
    g = _gate_rows()
    h = g[g["i_in_spice"] > float(cfg("ml.i_floor_a"))]
    return 100 * float(np.median(h["i_in_spice"] / h["i_vdd_spice"]))


define("GateIinShare", lambda: pct(_iin_share(), 0), "median i_in/i_vdd, gates with a high input")


def _gate_subset(gate: str) -> dict:
    g = _gate_rows()
    g = g[g["gate"] == gate]
    return summarize(g["i_vdd_ml"].to_numpy(), g["i_vdd_spice"].to_numpy(),
                     i_floor=float(cfg("ml.i_floor_a")))  # fmt: skip


define("GateNandMeanAcc", lambda: pct(_gate_subset("NAND2")["mean_acc_pct"], 2), "NAND2 mean acc")
define("GateNandMedianErr", lambda: pct(_gate_subset("NAND2")["median_err_pct"], 2), "NAND2 med")
define("GateNandPninetyfiveErr", lambda: pct(_gate_subset("NAND2")["p95_err_pct"], 2), "NAND2 p95")
define("GateNandMaxErr", lambda: pct(_gate_subset("NAND2")["max_err_pct"], 1), "NAND2 max")
define("GateNandBias", lambda: pct(_gate_subset("NAND2")["bias_pct"], 2, plus=True), "NAND2 bias")

define("ConstLeafGateMedianErr",
       lambda: pct(one(need(const_gate, "gate", "median_err_pct"),
                       const_gate["gate"] == "ALL")["median_err_pct"], 2),
       f"{CONST_LEAF_COMMIT} gate_results ALL median_err_pct")  # fmt: skip


def _on_vds() -> dict[str, float]:
    """|vds| of every ON FET in every gate row, from the SPICE node voltages (as step 08 sees them)."""
    from leakml.gates.ml_model import KCL_VDS_MAX
    from leakml.gates.topology import get_gate, input_levels

    g = io.read_csv(GATE_DATA / "gate_spice.csv")
    free, biased = [], []
    for (name, bits), sub in g.groupby(["gate", "inputs"]):
        gate = get_gate(name)
        vdd = sub["vdd"].to_numpy(dtype=float)
        v = {"vdd": vdd, "0": np.zeros_like(vdd)}
        v |= {pin: vdd * float(lv) for pin, lv in input_levels(gate, bits).items()}
        for node in (gate.output, *gate.internal_nodes):
            v[node] = sub[f"v_{node}_spice"].to_numpy(dtype=float)
        for f in gate.fets:
            vg = v[f.g]
            on = (vg > vdd / 2) if f.device == "nmos" else (vg < vdd / 2)
            vds = np.abs(v[f.d] - v[f.s])[on]
            free.extend(vds[vds < KCL_VDS_MAX])
            biased.extend(vds[vds >= KCL_VDS_MAX])
    free, biased = np.array(free), np.array(biased)
    return {"free_med": float(np.median(free)), "free_max": float(free.max()),
            "free_n": len(free), "biased_min": float(biased.min()), "thr": KCL_VDS_MAX}  # fmt: skip


define("OnVdsMedian", lambda: rf"\qty{{{_on_vds()['free_med']:.0e}}}{{\volt}}", "in-gate ON |vds|")
define("OnVdsMax", lambda: rf"\qty{{{_on_vds()['free_max']:.0e}}}{{\volt}}", "in-gate ON |vds|")
define("OnVdsCount", lambda: count(_on_vds()["free_n"]), "number of KCL-corrected FET rows")
define("OnBiasedVdsMin", lambda: rf"\qty{{{_on_vds()['biased_min']:.2f}}}{{\volt}}",
       "smallest |vds| of an ON FET above the KCL threshold")  # fmt: skip
define("KclVdsMax", lambda: rf"\qty{{{_on_vds()['thr'] * 1e3:g}}}{{\milli\volt}}",
       "ml_model.KCL_VDS_MAX")  # fmt: skip


# Report analyses (report/analysis/): guarded against staleness by the ML gate mean accuracy they
# were computed against, which must equal the current results/tables/gate_results.csv value.
def _analysis(path: str, ref_acc: Callable[[pd.DataFrame], float]) -> pd.DataFrame:
    d = pd.read_csv(ANALYSIS / path, dtype={"inputs": str})
    now = float(gate_all("mean_acc_pct"))
    if abs(ref_acc(d) - now) > 1e-6:
        raise ValueError(f"stale: computed for gate mean acc {ref_acc(d):.4f}, now {now:.4f}; "
                         "rerun report/analysis (see its README)")  # fmt: skip
    return d


def _oracle() -> pd.DataFrame:
    return _analysis("oracle/oracle_summary.csv",
                     lambda d: float(d.loc[d["composition"] == "ml_kcl", "mean_acc"].iloc[0])
                     ).set_index("composition")  # fmt: skip


for nm, col, d in (("MeanAcc", "mean_acc", 3), ("MedianErr", "median", 4),
                   ("PninetyfiveErr", "p95", 3), ("MaxErr", "max", 2)):  # fmt: skip
    define(f"Oracle{nm}", lambda col=col, d=d: pct(_oracle().loc["spice_oracle", col], d),
           f"report/analysis/oracle/oracle_summary.csv {col}")  # fmt: skip


def _attr(gate: str, inputs: str, fixed: str) -> str:
    _oracle()  # staleness guard: computed in the same oracle run
    d = pd.read_csv(ANALYSIS / "oracle" / "worst_case_attribution.csv", dtype={"inputs": str})
    r = d[(d["gate"] == gate) & (d["inputs"] == inputs) & (d["fixed"] == fixed)]
    if len(r) != 1:
        raise LookupError(f"{gate} {inputs} {fixed}: {len(r)} rows")
    return pct(r["mean_acc"].iloc[0], 2)


for nm, g, b, fx in (("NandTenBase", "NAND2", "10", "none"),
                     ("NandTenNmosOnId", "NAND2", "10", "nmos_on.id"),
                     ("NandTenPmosOnIg", "NAND2", "10", "pmos_on.ig"),
                     ("NorTenBase", "NOR2", "10", "none"),
                     ("NorTenPmosOffIs", "NOR2", "10", "pmos_off.is"),
                     ("NorElevenBase", "NOR2", "11", "none"),
                     ("NorElevenPmosOffIs", "NOR2", "11", "pmos_off.is")):  # fmt: skip
    define(f"Attr{nm}", lambda g=g, b=b, fx=fx: _attr(g, b, fx),
           f"oracle/worst_case_attribution.csv {g} {b} {fx}")  # fmt: skip


def _error_source(rank: int, what: str) -> str:
    _oracle()
    d = pd.read_csv(ANALYSIS / "oracle" / "error_sources.csv")
    r = d.sort_values("mean_acc_gain", ascending=False).iloc[rank]
    if what == "name":
        return rf"{tt(r['case'])} {TARGET_TEX[r['current']]}"
    return pct(r["gate_mean_acc_if_exact"], 2)


for rank, nm in ((0, "First"), (1, "Second")):
    define(f"ErrorSource{nm}", lambda r=rank: _error_source(r, "name"), "error_sources rank")
    define(f"ErrorSource{nm}Acc", lambda r=rank: _error_source(r, "acc"), "error_sources rank")


def _fresh() -> pd.DataFrame:
    return _analysis("fresh_corners/overall_kcl_direct_orig_vs_fresh.csv",
                     lambda d: float(d.loc[(d["corners"] == "orig") & (d["method"] == "KCL"),
                                           "mean_acc_pct"].iloc[0]))  # fmt: skip


def _fresh_val(corners: str, method: str, col: str) -> float:
    d = _fresh()
    return float(d.loc[(d["corners"] == corners) & (d["method"] == method), col].iloc[0])


for nm, col, d, plus in (("MeanAcc", "mean_acc_pct", 2, False), ("MedianErr", "median_err_pct", 2, False),
                         ("PninetyfiveErr", "p95_err_pct", 2, False), ("MaxErr", "max_err_pct", 1, False),
                         ("Bias", "bias_pct", 2, True)):  # fmt: skip
    define(f"Fresh{nm}", lambda col=col, d=d, plus=plus: pct(_fresh_val("fresh", "KCL", col), d, plus),
           f"fresh_corners overall {col}")  # fmt: skip
define("FreshDirectMeanAcc", lambda: pct(_fresh_val("fresh", "direct", "mean_acc_pct"), 2),
       "fresh_corners overall direct mean acc")  # fmt: skip
define("FreshNumCorners",
       lambda: count(len(pd.read_csv(ANALYSIS / "fresh_corners" / "corners_fresh.csv"))),
       "corners_fresh.csv rows")  # fmt: skip
define("FreshLogicOk", lambda: _logic_ok(ANALYSIS / "fresh_corners" / "gate_spice_fresh.csv"),
       "gate_spice_fresh logic_ok")  # fmt: skip


def _fresh_seed() -> str:
    import re

    src = (ANALYSIS / "fresh_corners.py").read_text()
    m = re.search(r"cfg\.seeds\.corners \+ (\d+)", src)
    if not m:
        raise LookupError("seed offset not found in fresh_corners.py")
    return str(int(cfg("seeds.corners")) + int(m.group(1)))


define("FreshSeed", _fresh_seed, "fresh_corners.py seed")


def _fresh_table() -> str:
    d = _fresh()
    rows = []
    for corners, label in (("orig", "original"), ("fresh", "fresh")):
        for method, mlabel in (("KCL", "KCL"), ("direct", "direct")):
            r = d[(d["corners"] == corners) & (d["method"] == method)].iloc[0]
            rows.append([label if method == "KCL" else "", mlabel, pct(r.median_err_pct, 2),
                         pct(r.p95_err_pct, 2), pct(r.max_err_pct, 1), pct(r.bias_pct, 2, plus=True),
                         pct(r.mean_acc_pct, 2)])  # fmt: skip
    return tabular("llrrrrr", ["Corners", "Composition", "Median err.", "p95 err.", "Max err.",
                               "Bias", "Mean acc."], rows, [2])  # fmt: skip


write_table(
    "fresh", _fresh_table, "report/analysis/fresh_corners/overall_kcl_direct_orig_vs_fresh.csv"
)


# ------------------------------------------------------------------ circuit: extra (Section 5)
def _circuit_def():
    from leakml.circuit.definition import get_circuit

    return get_circuit(cfg("assignment.test_circuit"))


def _gate_mix() -> str:
    from collections import Counter

    cnt = Counter(i.gate for i in _circuit_def().instances)
    return ", ".join(rf"\num{{{n}}} {g}" for g, n in sorted(cnt.items()))


define("CircuitNumGates", lambda: count(len(_circuit_def().instances)), "circuit instances")
define("CircuitGateMix", _gate_mix, "circuit gate types")
define("CircuitNumXor",
       lambda: count(len({i.name.rsplit("_g", 1)[0] for i in _circuit_def().instances
                          if "_g" in i.name})), "XOR2 blocks")  # fmt: skip
define("CircuitNumInputs", lambda: count(len(_circuit_def().primary_inputs)), "primary inputs")
define("CircuitNumVectors", lambda: count(2 ** len(_circuit_def().primary_inputs)), "vectors")
define("CircuitLogicOk", lambda: _logic_ok(ROOT / "data" / "circuit" / "circuit_spice.csv"),
       "circuit_spice logic_ok")  # fmt: skip


def _load_share() -> float:
    c = io.read_csv(ROOT / "data" / "circuit" / "circuit_ml.csv")
    return 100 * float(np.median(c["i_load_ml"] / c["i_vdd_ml"]))


define("CircuitLoadShare", lambda: pct(_load_share(), 0), "median i_load_ml / i_vdd_ml")


def _ones_range(col: str, lo: bool, plus: bool = False) -> str:
    d = need(cr, "scope", col)
    o = d[d["scope"].str.strip().str.startswith("ones =")][col]
    return pct(o.min() if lo else o.max(), 2, plus)


define("CircuitOnesAccMin", lambda: _ones_range("mean_acc_pct", True), "circuit ones rows")
define("CircuitOnesAccMax", lambda: _ones_range("mean_acc_pct", False), "circuit ones rows")
define("CircuitOnesBiasMin", lambda: _ones_range("bias_pct", True, True), "circuit ones rows")
define("CircuitOnesBiasMax", lambda: _ones_range("bias_pct", False, True), "circuit ones rows")
define("CircuitIinMeanAcc", lambda: pct(circ_row("primary-input current")["mean_acc_pct"], 2),
       "circuit_results primary-input current mean acc")  # fmt: skip


def _driver(feature: str) -> str:
    d = load_csv("circuit_error_drivers")
    r = one(need(d, "feature", "spearman_rho_vs_signed_error"), d["feature"] == feature)
    return num(r["spearman_rho_vs_signed_error"], 2)


define("CircuitTempRho", lambda: _driver("temp"), "circuit_error_drivers temp")


# ------------------------------------------------------------------ error analysis (Section 6)
def an(name: str) -> pd.DataFrame:
    return io.read_csv(AN / f"{name}.csv")


CASE_TEX = {c: tt(c) for c in CASES}
BAND = {"Low": "<1 mV", "High": ">0.3 V"}
TBIN = {"Cold": -40.0, "Hot": 80.0}
MIN_BIN_ROWS = 20  # bins with fewer rows are shown as "--"
define("MinBinRows", str(MIN_BIN_ROWS), "make_numbers MIN_BIN_ROWS")


def _tr(kind: str, case: str, target: str, key: str) -> str:
    d = an("transistor_vs_vds" if kind == "vds" else "transistor_vs_temp")
    sel = (d["case"] == case) & (d["target"] == target)
    sel &= (d["band"] == BAND[key]) if kind == "vds" else (d["lo"].astype(float) == TBIN[key])
    r = d[sel]
    if len(r) != 1 or int(r["n"].iloc[0]) < MIN_BIN_ROWS:
        return "n/a"
    return pct(r["p95_err_pct"].iloc[0], 1)


for c in CASES:
    for t in TARGETS:
        for key in BAND:
            define(f"TrVds{key}{camel(c)}{camel(t)}", lambda c=c, t=t, k=key: _tr("vds", c, t, k),
                   f"transistor_vs_vds {c} {t} {key}")  # fmt: skip
        for key in TBIN:
            define(f"TrTemp{key}{camel(c)}{camel(t)}",
                   lambda c=c, t=t, k=key: _tr("temp", c, t, k),
                   f"transistor_vs_temp {c} {t} {key}")  # fmt: skip
        define(f"BelowFloor{camel(c)}{camel(t)}",
               lambda c=c, t=t: pct(one(an("transistor_floor"), (an("transistor_floor")["case"] == c)
                                        & (an("transistor_floor")["target"] == t))["below_floor_pct"], 0),
               f"transistor_floor {c} {t}")  # fmt: skip

TR_ROWS = [(c, t) for c in CASES for t in TARGETS if not (c.endswith("off") and t == "ig")]


def _tr_table(kind: str) -> str:
    d = an("transistor_vs_vds" if kind == "vds" else "transistor_vs_temp")
    keys = (
        list(dict.fromkeys(d["band"])) if kind == "vds" else sorted(d["lo"].astype(float).unique())
    )
    rows, rules = [], []
    for c, t in TR_ROWS:
        if rows and t == "id":
            rules.append(len(rows))
        cells = []
        for k in keys:
            sel = (d["case"] == c) & (d["target"] == t)
            sel &= (d["band"] == k) if kind == "vds" else (d["lo"].astype(float) == k)
            r = d[sel]
            ok = len(r) == 1 and int(r["n"].iloc[0]) >= MIN_BIN_ROWS
            cells.append(pct(r["p95_err_pct"].iloc[0], 1) if ok else "--")
        rows.append([CASE_TEX[c] if t == "id" else "", TARGET_TEX[t], *cells])
    if kind == "vds":
        head = [
            tex_escape(k).replace("<", "$<$").replace(">", "$>$").replace("-", "--") for k in keys
        ]
    else:
        his = {-40.0: 0, 0.0: 40, 40.0: 80, 80.0: 125}
        head = [rf"\qtyrange{{{k:g}}}{{{his[k]:g}}}{{\degreeCelsius}}" for k in keys]
    return tabular("ll" + "r" * len(keys), ["Case", "Current", *head], rows, rules)


write_table("trans_vds", lambda: _tr_table("vds"), "generated/analysis/transistor_vs_vds.csv")
write_table("trans_temp", lambda: _tr_table("temp"), "generated/analysis/transistor_vs_temp.csv")


# --- gate SPICE physics
def _ratio(key: str, col: str) -> float:
    d = an("spice_ratios")
    return float(one(d, d["comparison"] == key)[col])


define("NandZeroOneInvDev", lambda: pct(100 * _ratio("nand01_inv0", "max_abs_dev"), 2),
       "spice_ratios nand01_inv0")  # fmt: skip
for nm, key in (("NorStackA", "nor11_nor01"), ("NorStackB", "nor11_nor10"),
                ("NandZeroA", "nand00_nand01"), ("NandZeroB", "nand00_nand10")):  # fmt: skip
    define(f"{nm}Ratio", lambda k=key: num(_ratio(k, "median_ratio"), 3 if "nor" in k else 2),
           f"spice_ratios {key}")  # fmt: skip
    define(f"{nm}Lower",
           lambda k=key: rf"\num{{{int(_ratio(k, 'n_lower'))}}}/\num{{{int(_ratio(k, 'n'))}}}",
           f"spice_ratios {key} n_lower")  # fmt: skip


def _nand_temp() -> str:
    d = an("spice_nand00_vs_temp")
    rows = [[rf"\qtyrange{{{r.lo:g}}}{{{r.hi:g}}}{{\degreeCelsius}}", num(r.median_ratio, 2),
             rf"\num{{{int(r.n_lower)}}}/\num{{{int(r.n)}}}"] for r in d.itertuples()]  # fmt: skip
    return tabular("lrr", ["Temperature", r"median $I_{00}/I_{01}$", r"corners with $I_{00}<I_{01}$"],
                   rows)  # fmt: skip


write_table("nand_temp", _nand_temp, "generated/analysis/spice_nand00_vs_temp.csv")


def _node(gate: str, inputs: str, col: str):
    d = an("spice_nodes")
    return one(d, (d["gate"] == gate) & (d["inputs"] == inputs))[col]


define("VxZeroZeroMin", lambda: rf"\qty{{{_node('NAND2', '00', 'v_min'):.3f}}}{{\volt}}", "v_x 00")
define("VxZeroZeroMax", lambda: rf"\qty{{{_node('NAND2', '00', 'v_max'):.3f}}}{{\volt}}", "v_x 00")
define("VxZeroZeroFrac", lambda: num(_node("NAND2", "00", "frac_median"), 2), "v_x 00 frac")
define("VxOneZeroFrac", lambda: num(_node("NAND2", "10", "frac_median"), 2), "v_x 10 frac")
define("VyOneZeroFrac", lambda: num(_node("NOR2", "10", "frac_median"), 2), "v_y 10 frac")
define("VyOneOneFrac", lambda: num(_node("NOR2", "11", "frac_median"), 2), "v_y 11 frac")
define("VxOnRailMax",
       lambda: rf"\qty{{{max(_node('NAND2', b, 'dist_rail_max') for b in ('01', '11')):.0e}}}{{\volt}}",
       "v_x when tied to ground")  # fmt: skip


def _n1_rail() -> float:
    d = an("spice_nodes")
    return float(d.loc[d["node"] == "n1", "dist_rail_max"].max())


define("NOneRailMax",
       lambda: rf"\qty{{{_n1_rail():.0e}}}{{\volt}}",
       "n1 distance from rail")  # fmt: skip

_tg = lambda c: an("spice_temp_groups")[c].iloc[0]  # noqa: E731
define("TempHotMin", lambda: rf"\qty{{{_tg('t_hot_min'):.0f}}}{{\degreeCelsius}}", "hot corners")
define("TempHotMax", lambda: rf"\qty{{{_tg('t_hot_max'):.0f}}}{{\degreeCelsius}}", "hot corners")
define("TempColdMin", lambda: rf"\qty{{{_tg('t_cold_min'):.0f}}}{{\degreeCelsius}}", "cold corners")
define("TempColdMax", lambda: rf"\qty{{{_tg('t_cold_max'):.0f}}}{{\degreeCelsius}}", "cold corners")
define("TempGroupSize", lambda: count(_tg("n_hot")), "hot/cold group size")


def _trend_table() -> str:
    d = an("spice_temp_trend")
    rows, rules, prev = [], [], None
    for r in d.itertuples():
        if prev is not None and r.gate != prev:
            rules.append(len(rows))
        rows.append([tex_escape(r.gate) if r.gate != prev else "", tt(r.inputs),
                     sci(r.median_i_vdd, 3), num(r.hot_cold, 1), num(r.rho_t_logi, 2)])  # fmt: skip
        prev = r.gate
    return tabular("llrrr", ["Gate", "Inputs", "SPICE median (A)", "hot/cold",
                             r"$\rho(T, \log I_{VDD})$"], rows, rules)  # fmt: skip


write_table("temp_trend", _trend_table, "generated/analysis/spice_temp_trend.csv")


def _trend_range(strong: bool, col: str, lo: bool) -> str:
    d = an("spice_temp_trend")
    sub = d[d["hot_cold"] > 50] if strong else d[d["hot_cold"] <= 50]
    v = sub[col].min() if lo else sub[col].max()
    return num(v, 0 if col == "hot_cold" and strong else 1)


define("StrongHotColdMin", lambda: _trend_range(True, "hot_cold", True), "strong-rise cases")
define("StrongHotColdMax", lambda: _trend_range(True, "hot_cold", False), "strong-rise cases")
define("WeakHotColdMin", lambda: _trend_range(False, "hot_cold", True), "weak-rise cases")
define("WeakHotColdMax", lambda: _trend_range(False, "hot_cold", False), "weak-rise cases")
define("StrongCases",
       lambda: ", ".join(gate_case(r.gate, r.inputs) for r in an("spice_temp_trend").itertuples()
                         if r.hot_cold > 50), "strong-rise cases")  # fmt: skip
define("GateSpiceMin", lambda: amp(an("spice_min")["min_i_vdd"].iloc[0], 2), "spice_min")
define("GateSpiceMinCase", lambda: gate_case(*an("spice_min")["case_min"].iloc[0].split()),
       "spice_min case")  # fmt: skip


def _floor_rel() -> float:
    k = need(kir, "case", "p95_abs_r_a")
    floor = float(k[k["case"].str.startswith("pmos")]["p95_abs_r_a"].max())
    return 100 * floor / float(an("spice_min")["min_i_vdd"].iloc[0])


define("SpiceFloorRel", lambda: pct(_floor_rel(), 1), "PMOS Kirchhoff p95 / smallest gate i_vdd")


# --- gate and circuit error vs corner conditions
def _bin(name: str, lo: float, col: str):
    d = an(name)
    return one(d, d["lo"].astype(float) == lo)[col]


for prefix, name in (("Gate", "gate_vs_temp"), ("Circuit", "circuit_vs_temp")):
    for key, lo in TBIN.items():
        define(f"{prefix}{key}MedianErr", lambda n=name, lo=lo: pct(_bin(n, lo, "median_err_pct"), 2),
               f"{name} {key}")  # fmt: skip
        define(f"{prefix}{key}Bias", lambda n=name, lo=lo: pct(_bin(n, lo, "bias_pct"), 2, True),
               f"{name} {key}")  # fmt: skip
        define(f"{prefix}{key}MeanAcc", lambda n=name, lo=lo: pct(_bin(n, lo, "mean_acc_pct"), 2),
               f"{name} {key}")  # fmt: skip

FEATURE_TEX = {"temp": "temperature", "vdd": "$V_{DD}$", **{
    f"{d[0]}_{p}": f"{d.upper()} \\texttt{{{p}}}" for d in ("nmos", "pmos") for p in
    ("vth0", "toxe", "ndep", "u0", "xj")}}  # fmt: skip
FEATURE_TEX = {k.replace("nmos_", "n_").replace("pmos_", "p_"): v for k, v in FEATURE_TEX.items()}


def _drivers(rank: int, what: str) -> str:
    d = an("gate_error_drivers")
    d = d.iloc[d["rho_signed"].abs().sort_values(ascending=False).index]
    r = d.iloc[rank]
    return (
        FEATURE_TEX.get(r["feature"], tex_escape(r["feature"]))
        if what == "name"
        else num(r["rho_signed"], 2)
    )


for rank, nm in ((0, "First"), (1, "Second"), (2, "Third")):
    define(f"GateDriver{nm}", lambda r=rank: _drivers(r, "name"), "gate_error_drivers")
    define(f"GateDriver{nm}Rho", lambda r=rank: _drivers(r, "rho"), "gate_error_drivers")


def _corner_table() -> str:
    g, c = an("gate_vs_temp"), an("circuit_vs_temp")
    rows = []
    for (_, a), (_, b) in zip(g.iterrows(), c.iterrows(), strict=True):
        rows.append([rf"\qtyrange{{{a.lo:g}}}{{{a.hi:g}}}{{\degreeCelsius}}",
                     pct(a.median_err_pct, 2), pct(a.p95_err_pct, 2), pct(a.bias_pct, 2, True),
                     pct(b.median_err_pct, 2), pct(b.p95_err_pct, 2), pct(b.bias_pct, 2, True)])  # fmt: skip
    head = (r"\begin{tabular}{lrrrrrr}" "\n" r"\toprule" "\n"
            r" & \multicolumn{3}{c}{Gates (all 18 cases)} & \multicolumn{3}{c}{Test circuit} \\"
            "\n" r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}" "\n"
            r"Temperature & median & p95 & bias & median & p95 & bias \\" "\n" r"\midrule" "\n")  # fmt: skip
    return (
        head + "".join(" & ".join(r) + r" \\" "\n" for r in rows) + r"\bottomrule"
        "\n"
        r"\end{tabular}"
        "\n"
    )


write_table("err_temp", _corner_table, "generated/analysis/gate_vs_temp.csv, circuit_vs_temp.csv")


def _vth_table() -> str:
    n, p = an("gate_vs_nvth0"), an("gate_vs_pvth0")
    rows = []
    for (_, a), (_, b) in zip(n.iterrows(), p.iterrows(), strict=True):
        rows.append([rf"\numrange{{{a.lo:+g}}}{{{a.hi:+g}}}\,\%", pct(a.median_err_pct, 2),
                     pct(a.bias_pct, 2, True), pct(b.median_err_pct, 2), pct(b.bias_pct, 2, True)])  # fmt: skip
    head = (r"\begin{tabular}{lrrrr}" "\n" r"\toprule" "\n"
            r" & \multicolumn{2}{c}{NMOS \texttt{vth0}} & \multicolumn{2}{c}{PMOS $|$\texttt{vth0}$|$} \\"
            "\n" r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}" "\n"
            r"Deviation from nominal & median & bias & median & bias \\" "\n" r"\midrule" "\n")  # fmt: skip
    return (
        head + "".join(" & ".join(r) + r" \\" "\n" for r in rows) + r"\bottomrule"
        "\n"
        r"\end{tabular}"
        "\n"
    )


write_table("err_vth", _vth_table, "generated/analysis/gate_vs_nvth0.csv, gate_vs_pvth0.csv")


# --- extrapolation
def _ext(flag: bool, col: str):
    d = an("gate_extrapolation_summary")
    d = d[d["extrap"].astype(str).str.lower() == str(flag).lower()]
    if d.empty:
        return 0 if col in ("n", "n_corners") else float("nan")
    return d[col].iloc[0]


define("ExtrapRows", lambda: count(_ext(True, "n")), "gate rows with an extrapolating FET")
define("ExtrapCorners", lambda: count(_ext(True, "n_corners")), "corners with extrapolation")
define("ExtrapMedianErr", lambda: pct(_ext(True, "median_err_pct"), 2), "extrap rows median")
define("ExtrapMaxErr", lambda: pct(_ext(True, "max_err_pct"), 1), "extrap rows max")
define("NoExtrapMedianErr", lambda: pct(_ext(False, "median_err_pct"), 2), "other rows median")
define("ExtrapMaxOver",
       lambda: rf"\qty{{{1e3 * an('gate_extrapolation')['max_over'].max():.1f}}}{{\milli\volt}}",
       "gate_extrapolation max_over")  # fmt: skip
define("ExtrapMaxOverPct",
       lambda: pct(100 * (an("gate_extrapolation")["max_over"]
                          / an("gate_extrapolation")["range_width"]).max(), 1),
       "gate_extrapolation max_over / range")  # fmt: skip
define("ExtrapWhat",
       lambda: "; ".join(f"{tt(c)} " + ", ".join(sorted(f"${{{col[0].upper()}}}_{{{col[1:].upper()}}}$"
                                                        for col in g["column"].unique()))
                         for c, g in an("gate_extrapolation").groupby("case")),
       "gate_extrapolation case/column")  # fmt: skip


# --- circuit decomposition
def _dec(label: str, col: str) -> float:
    d = an("circuit_decomposition")
    return float(one(d, d["composition"] == label)[col])


define("CompSpiceMeanAcc", lambda: pct(_dec("spice_gates_with_loading", "mean_acc_pct"), 3), "dec")
define(
    "CompSpiceMedianErr", lambda: pct(_dec("spice_gates_with_loading", "median_err_pct"), 3), "dec"
)
define("CompSpiceMaxErr", lambda: pct(_dec("spice_gates_with_loading", "max_err_pct"), 2), "dec")
define(
    "CompSpiceNoLoadBias", lambda: pct(_dec("spice_gates_no_loading", "bias_pct"), 1, True), "dec"
)
define("CompMlMeanAcc", lambda: pct(_dec("ml_vs_spice_gate_composition", "mean_acc_pct"), 2), "dec")


# --- oracle error sources (report/analysis), as a table
def _sources_table() -> str:
    _oracle()  # staleness guard
    d = pd.read_csv(ANALYSIS / "oracle" / "error_sources.csv")
    d = d.sort_values("mean_acc_gain", ascending=False).head(6)
    rows = [[tt(r.case), TARGET_TEX[r.current], pct(r.fet_median_err_pct, 1),
             pct(r.fet_p95_err_pct, 1), pct(r.gate_mean_acc_if_exact, 2),
             pct(r.mean_acc_gain, 2, True)] for r in d.itertuples()]  # fmt: skip
    return tabular("llrrrr", ["Case", "Current", "FET median err.", "FET p95 err.",
                              "Gate mean acc.\\ if exact", "Gain"], rows)  # fmt: skip


write_table("error_sources", _sources_table, "report/analysis/oracle/error_sources.csv")

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
for fig in ("error_vs_corner",):
    src = AN / f"{fig}.pdf"
    if src.is_file():
        shutil.copy2(src, FIGS / src.name)
    else:
        missing.append(f"figure report/generated/analysis/{fig}.pdf (make_analysis.py)")

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
            if f.name.startswith(".") or "analysis" in f.relative_to(REPORT).parts[:2]:
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
