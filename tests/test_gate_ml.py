"""Tests for leakml.gates.ml_model (Part B step 4: ML gate leakage from the transistor models).

Unit tests use a STUB model bank whose currents are simple functions of the terminal voltages, so
every expected value can be written down by hand. The requires_ngspice test is the "SPICE oracle":
each FET of each gate is simulated alone at its in-gate terminal voltages, and the bookkeeping in
leakage_from_currents must then reproduce the gate's own -i(vvdd).
"""

from __future__ import annotations

import dataclasses
import math
import shutil

import numpy as np
import pandas as pd
import pytest

from leakml import io
from leakml.config import DEFAULT_CONFIG, load_config
from leakml.constants import GATE_INPUT_COMBOS, GATES, INPUT_COLUMNS, TARGET_COLUMNS, Case
from leakml.gates.ml_model import (
    GATE_ML_COLUMNS,
    KCL_VDS_MAX,
    SNAP_ERR_ATTR,
    GateLeakageModel,
    _kcl_arrays,
    fet_feature_rows,
    kcl_corrected_currents,
    leakage_from_currents,
    predict_gate_leakage,
    resolve_node_voltages,
    run_gate_ml,
)
from leakml.gates.netlist import gate_widths
from leakml.gates.spice_ref import run_gate_reference
from leakml.gates.topology import (
    Fet,
    GateTopology,
    get_gate,
    input_levels,
    logic_function,
    terminals_on,
)
from leakml.ml.predictor import TransistorModel, TransistorModelBank
from leakml.sampling import corner_process_params, draw_corners, make_rng
from leakml.spice.fet_netlist import FET_EXPECT, FET_SOURCES, render_fet_netlist
from leakml.spice.parse import currents_into_device
from leakml.spice.runner import run_many_cfg

requires_ngspice = pytest.mark.skipif(shutil.which("ngspice") is None, reason="ngspice not on PATH")

# Stub currents: (id, ig, is, ib) = SCALE[case] * (vd, vg, vs, vb). Easy to check by hand, and the
# per-case scale shows which model each FET was sent to.
SCALE = {Case.NMOS_ON: 1.0, Case.NMOS_OFF: 2.0, Case.PMOS_ON: 3.0, Case.PMOS_OFF: 4.0}


class _StubRegressor:
    def __init__(self, scale: float):
        self.scale = scale

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.scale * X.loc[:, ["vd", "vg", "vs", "vb"]].to_numpy(dtype=float)


class _CountingBank(TransistorModelBank):
    def __init__(self):
        super().__init__({c: TransistorModel(c, _StubRegressor(s)) for c, s in SCALE.items()})
        self.calls = 0
        self.rows = 0

    def predict(self, X, cases):
        self.calls += 1
        self.rows += len(X)
        return super().predict(X, cases)


@pytest.fixture
def bank() -> _CountingBank:
    return _CountingBank()


@pytest.fixture(scope="module")
def corners(dev_cfg, card) -> pd.DataFrame:
    return draw_corners(3, dev_cfg, card, make_rng(dev_cfg.seeds.corners, "corners"))


@pytest.fixture(scope="module")
def corner0(corners) -> dict:
    return corners.iloc[0].to_dict()


def _recorded(gate_name: str, bits: str, vdd: float, n1_offset: float = 0.0) -> dict[str, float]:
    """Plausible recorded node voltages: out at its logic level, x/y in between, n1 = first-stage
    logic level moved by n1_offset towards the rail's inside (non-zero snap error)."""
    rec = {"out": vdd * logic_function(gate_name, bits), "x": 0.11 * vdd, "y": 0.87 * vdd}
    if gate_name in ("AND2", "OR2"):
        first = 1 - logic_function(gate_name, bits)
        rec["n1"] = vdd - n1_offset if first else n1_offset
    return rec


def _fake_gate_spice(corners: pd.DataFrame, n1_offset: float = 2e-5) -> pd.DataFrame:
    """A gate_spice-shaped table (voltages only; the current columns are NaN on purpose)."""
    rows = []
    for g in GATES:
        gate = get_gate(g)
        for bits in GATE_INPUT_COMBOS[g]:
            for c in corners.to_dict("records"):
                rec = _recorded(g, bits, c["vdd"], n1_offset)
                rows.append(
                    {"gate": g, "inputs": bits, "corner_id": int(c["corner_id"])}
                    | {"vdd": c["vdd"], "temp": c["temp"]}
                    | {"i_vdd_spice": np.nan, "i_in_spice": np.nan, "v_out_spice": rec["out"]}
                    | {
                        f"v_{n}_spice": rec[n] if n in gate.internal_nodes else np.nan
                        for n in ("x", "y", "n1")
                    }
                    | {"logic_ok": True}
                )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- resolve_node_voltages


def test_resolve_inv():
    nv = resolve_node_voltages(get_gate("INV"), "1", 0.9, {"out": 1e-6, "x": 123.0})
    assert nv == {"vdd": 0.9, "0": 0.0, "a": 0.9, "out": 1e-6}  # extra keys ignored


def test_resolve_nand2_input_order():
    nv = resolve_node_voltages(get_gate("NAND2"), "01", 0.9, {"out": 0.9, "x": 0.01})
    assert nv["a"] == 0.0 and nv["b"] == 0.9  # first bit drives "a"
    assert nv["x"] == 0.01 and nv["out"] == 0.9 and nv["0"] == 0.0 and nv["vdd"] == 0.9


def test_resolve_and2():
    nv = resolve_node_voltages(get_gate("AND2"), "11", 0.95, {"out": 0.95, "x": 1e-7, "n1": 2e-6})
    assert nv == {"vdd": 0.95, "0": 0.0, "a": 0.95, "b": 0.95, "out": 0.95, "x": 1e-7, "n1": 2e-6}


@pytest.mark.parametrize(
    ("gate_name", "recorded", "exc"),
    [
        ("NAND2", {"out": 0.9}, KeyError),  # x missing
        ("AND2", {"out": 0.9, "x": 0.1}, KeyError),  # n1 missing
        ("INV", {}, KeyError),  # out missing
        ("NOR2", {"out": 0.0, "y": math.nan}, ValueError),  # NaN = node not recorded
    ],
)
def test_resolve_missing_node_raises(gate_name, recorded, exc):
    with pytest.raises(exc):
        resolve_node_voltages(get_gate(gate_name), GATE_INPUT_COMBOS[gate_name][0], 0.9, recorded)


def test_resolve_wrong_bit_count():
    with pytest.raises(ValueError):
        resolve_node_voltages(get_gate("NAND2"), "1", 0.9, {"out": 0.9, "x": 0.0})


# ---------------------------------------------------------------- fet_feature_rows


def _rows(gate_name, bits, corner, cfg, n1_offset=0.0):
    gate = get_gate(gate_name)
    vdd = corner["vdd"]
    nv = resolve_node_voltages(gate, bits, vdd, _recorded(gate_name, bits, vdd, n1_offset))
    return gate, nv, *fet_feature_rows(gate, bits, corner, nv, cfg)


@pytest.mark.parametrize("gate_name", GATES)
def test_rows_columns_and_snapped_vg(gate_name, corner0, dev_cfg):
    for bits in GATE_INPUT_COMBOS[gate_name]:
        gate, _, X, cases = _rows(gate_name, bits, corner0, dev_cfg, n1_offset=3e-5)
        assert list(X.columns) == list(INPUT_COLUMNS)
        assert list(X.index) == [f.name for f in gate.fets]
        assert len(cases) == len(gate.fets)
        vdd = corner0["vdd"]
        assert X["vg"].isin([0.0, vdd]).all()  # exactly 0 or VDD, as in training
        assert (X["vdd"] == vdd).all() and (X["temp"] == corner0["temp"]).all()


def test_rows_nand2_01_cases_and_terminals(corner0, dev_cfg):
    gate, nv, X, cases = _rows("NAND2", "01", corner0, dev_cfg)
    assert cases == [Case.PMOS_ON, Case.PMOS_OFF, Case.NMOS_OFF, Case.NMOS_ON]  # MPA MPB MNA MNB
    vdd = corner0["vdd"]
    assert X.loc["MNA", ["vd", "vs", "vb"]].tolist() == [nv["out"], nv["x"], 0.0]
    assert X.loc["MNB", ["vd", "vs", "vb"]].tolist() == [nv["x"], 0.0, 0.0]
    assert X.loc["MPA", ["vd", "vs", "vb"]].tolist() == [nv["out"], vdd, vdd]
    assert X.loc["MPB", "vg"] == vdd and X.loc["MPA", "vg"] == 0.0
    assert X.attrs[SNAP_ERR_ATTR] == [0.0] * 4  # gates on inputs: nothing to snap


def test_rows_widths_and_process_params(corner0, dev_cfg):
    gate, _, X, _ = _rows("NOR2", "10", corner0, dev_cfg)
    wn, wp = gate_widths(dev_cfg)
    for f in gate.fets:
        assert X.loc[f.name, "w"] == (wn if f.device == "nmos" else wp)
        for p, v in corner_process_params(corner0, f.device).items():
            assert X.loc[f.name, p] == v  # n_* for NMOS, p_* for PMOS
    assert X.loc["MPA", "vth0"] < 0 < X.loc["MNA", "vth0"]  # PMOS vth0 keeps its sign


@pytest.mark.parametrize(("gate_name", "bits"), [("AND2", "00"), ("AND2", "11"), ("OR2", "01")])
def test_rows_snap_error_only_on_n1_stage(gate_name, bits, corner0, dev_cfg):
    gate, nv, X, cases = _rows(gate_name, bits, corner0, dev_cfg, n1_offset=2e-5)
    errs = dict(zip(X.index, X.attrs[SNAP_ERR_ATTR], strict=True))
    for f in gate.fets:
        if f.g == "n1":
            assert errs[f.name] == pytest.approx(2e-5, rel=1e-6)
        else:
            assert errs[f.name] == 0.0
    # the INV stage is classified from the raw n1 level
    first_high = 1 - logic_function(gate_name, bits)
    mp2 = cases[[f.name for f in gate.fets].index("MP2")]
    assert mp2 == (Case.PMOS_OFF if first_high else Case.PMOS_ON)


# ---------------------------------------------------------------- leakage_from_currents


def _table(gate, values: dict[str, tuple[float, float, float, float]]) -> pd.DataFrame:
    return pd.DataFrame(
        [values[f.name] for f in gate.fets],
        index=[f.name for f in gate.fets],
        columns=list(TARGET_COLUMNS),
    )


def test_leakage_inv():
    gate = get_gate("INV")
    cur = _table(gate, {"MP": (1, 2, 3, 4), "MN": (10, 20, 30, 40)})
    assert leakage_from_currents(gate, cur, "0") == {"i_vdd_ml": 3 + 4, "i_in_ml": 0.0}
    assert leakage_from_currents(gate, cur, "1") == {"i_vdd_ml": 3 + 4, "i_in_ml": 2 + 20}


def test_leakage_signs_kept():
    """Currents INTO the vdd-net terminals are summed as they are; a negative one subtracts."""
    gate = get_gate("INV")
    cur = _table(gate, {"MP": (-5e-12, 1e-15, 4e-12, -1e-13), "MN": (5e-12, 0, -5e-12, 0)})
    assert leakage_from_currents(gate, cur, "0")["i_vdd_ml"] == pytest.approx(4e-12 - 1e-13)


def test_leakage_nand2():
    gate = get_gate("NAND2")
    cur = _table(
        gate,
        {"MPA": (1, 2, 3, 4), "MPB": (10, 20, 30, 40), "MNA": (1e2, 2e2, 3e2, 4e2), "MNB": (1e3, 2e3, 3e3, 4e3)},
    )  # fmt: skip
    expected_vdd = 3 + 4 + 30 + 40  # both PMOS sources and bulks
    assert leakage_from_currents(gate, cur, "10") == {
        "i_vdd_ml": expected_vdd,
        "i_in_ml": 2 + 2e2,  # a high: MPA, MNA gates
    }
    assert leakage_from_currents(gate, cur, "01")["i_in_ml"] == 20 + 2e3  # b high: MPB, MNB
    assert leakage_from_currents(gate, cur, "11")["i_in_ml"] == 2 + 20 + 2e2 + 2e3


def test_leakage_nor2_series_pmos():
    """MPB's source is y, not vdd: only MPA.s, MPA.b and MPB.b count."""
    gate = get_gate("NOR2")
    cur = _table(
        gate,
        {"MPA": (1, 2, 3, 4), "MPB": (10, 20, 30, 40), "MNA": (1e2, 2e2, 3e2, 4e2), "MNB": (1e3, 2e3, 3e3, 4e3)},
    )  # fmt: skip
    assert leakage_from_currents(gate, cur, "00")["i_vdd_ml"] == 3 + 4 + 40


def test_leakage_and2_n1_gate_is_not_an_input():
    gate = get_gate("AND2")
    vals = {f.name: (0.0, 1.0, 0.0, 0.0) for f in gate.fets}  # ig = 1 everywhere
    out = leakage_from_currents(gate, _table(gate, vals), "11")
    assert out["i_in_ml"] == 4.0  # MPA MPB MNA MNB; MP2/MN2 are gated by n1, not by an input
    assert out["i_vdd_ml"] == 0.0


def test_leakage_positional_and_bad_shape():
    gate = get_gate("INV")
    cur = _table(gate, {"MP": (1, 2, 3, 4), "MN": (10, 20, 30, 40)}).reset_index(drop=True)
    assert leakage_from_currents(gate, cur, "1")["i_vdd_ml"] == 7
    with pytest.raises(ValueError):
        leakage_from_currents(gate, cur.iloc[:1], "1")


# ---------------------------------------------------------------- predict_gate_leakage (stub)


def test_predict_gate_leakage_by_hand(corner0, bank, dev_cfg):
    """NAND2 "01" with stub currents SCALE[case] * (vd, vg, vs, vb)."""
    vdd = corner0["vdd"]
    rec = {"out": vdd, "x": 0.01}
    out = predict_gate_leakage(get_gate("NAND2"), "01", corner0, rec, bank, dev_cfg)
    # i_vdd: MPA (pmos_on, x3) is + ib, MPB (pmos_off, x4) is + ib; all at vdd
    assert out["i_vdd_ml"] == pytest.approx(3 * 2 * vdd + 4 * 2 * vdd)
    # i_in: b high -> ig of MPB (pmos_off, vg=vdd) and MNB (nmos_on, vg=vdd)
    assert out["i_in_ml"] == pytest.approx(4 * vdd + 1 * vdd)
    assert out["max_vg_snap_err"] == 0.0
    assert bank.calls == 1


# ---------------------------------------------------------------- run_gate_ml (stub)


def test_run_gate_ml_schema_and_lookup(corners, bank, dev_cfg, tmp_path):
    gs = _fake_gate_spice(corners)
    df = run_gate_ml(dev_cfg, gs, corners, bank)

    assert list(df.columns) == list(GATE_ML_COLUMNS)
    assert len(df) == len(gs) == 18 * len(corners)
    pd.testing.assert_frame_equal(
        df[["gate", "inputs", "corner_id"]], gs[["gate", "inputs", "corner_id"]]
    )
    assert df["inputs"].map(type).eq(str).all()
    assert bank.calls == 1  # ONE batched call
    assert bank.rows == sum(len(get_gate(g).fets) * len(GATE_INPUT_COMBOS[g]) for g in GATES) * len(
        corners
    )

    # node columns: equal to SPICE where the gate has the node, NaN otherwise
    for node in ("x", "y", "n1"):
        has = df["gate"].map(lambda g, n=node: n in get_gate(g).internal_nodes)
        assert df.loc[~has, f"v_{node}_ml"].isna().all()
        np.testing.assert_array_equal(df.loc[has, f"v_{node}_ml"], gs.loc[has, f"v_{node}_spice"])

    # snap error: only AND2/OR2, = the n1 offset
    two_stage = df["gate"].isin(["AND2", "OR2"])
    assert (df.loc[~two_stage, "max_vg_snap_err"] == 0).all()
    np.testing.assert_allclose(df.loc[two_stage, "max_vg_snap_err"], 2e-5, rtol=1e-6)

    # vectorised path == single-case path, row by row
    cn = corners.set_index("corner_id")
    for r, g in zip(gs.to_dict("records"), df.to_dict("records"), strict=True):
        gate = get_gate(r["gate"])
        rec = {"out": r["v_out_spice"]} | {n: r[f"v_{n}_spice"] for n in gate.internal_nodes}
        corner = cn.loc[r["corner_id"]].to_dict() | {"corner_id": r["corner_id"]}
        one = predict_gate_leakage(
            gate, r["inputs"], corner, rec, TransistorModelBank(bank.models), dev_cfg
        )
        assert g["i_vdd_ml"] == pytest.approx(one["i_vdd_ml"], rel=1e-12)
        assert g["i_in_ml"] == pytest.approx(one["i_in_ml"], rel=1e-12, abs=0)
        assert g["max_vg_snap_err"] == pytest.approx(one["max_vg_snap_err"], rel=1e-9)
        assert g["i_vdd_ml_direct"] == pytest.approx(one["i_vdd_ml_direct"], rel=1e-12)

    # Part C reads it back through GateLeakageModel
    path = io.write_csv(df, tmp_path / "gate_ml.csv")
    gm = GateLeakageModel.from_csv(path)
    cid = int(corners["corner_id"].iloc[1])
    row = df[(df.gate == "NAND2") & (df.inputs == "01") & (df.corner_id == cid)].iloc[0]
    assert gm.predict("NAND2", "01", cid) == pytest.approx(row["i_vdd_ml"], rel=1e-9)
    assert gm.predict("NAND2", "01", cid, metric="i_in_ml") == pytest.approx(
        row["i_in_ml"], rel=1e-9
    )
    assert set(gm.metrics) == {"i_vdd_ml", "i_in_ml", "i_vdd_ml_direct"}
    assert gm.predict("NAND2", "01", cid, metric="i_vdd_ml_direct") == pytest.approx(
        row["i_vdd_ml_direct"], rel=1e-9
    )


def test_run_gate_ml_never_needs_spice_currents(corners, bank, dev_cfg):
    gs = _fake_gate_spice(corners).drop(columns=["i_vdd_spice", "i_in_spice", "logic_ok"])
    assert len(run_gate_ml(dev_cfg, gs, corners, bank)) == len(gs)


def test_run_gate_ml_order_independent(corners, bank, dev_cfg):
    gs = _fake_gate_spice(corners)
    shuffled = gs.sample(frac=1.0, random_state=0).reset_index(drop=True)
    a = run_gate_ml(dev_cfg, gs, corners, bank)
    b = run_gate_ml(dev_cfg, shuffled, corners, bank)
    key = ["gate", "inputs", "corner_id"]
    pd.testing.assert_frame_equal(shuffled[key], b[key])  # keeps the input order
    merged = a.merge(b, on=key, suffixes=("_a", "_b"))
    np.testing.assert_allclose(merged["i_vdd_ml_a"], merged["i_vdd_ml_b"], rtol=1e-12)


def test_run_gate_ml_rejects_bad_input(corners, bank, dev_cfg):
    gs = _fake_gate_spice(corners)
    with pytest.raises(ValueError, match="duplicate"):
        run_gate_ml(dev_cfg, pd.concat([gs, gs.iloc[:1]]), corners, bank)
    with pytest.raises(KeyError, match="corner_id"):
        run_gate_ml(dev_cfg, gs, corners.iloc[1:], bank)
    mangled = gs.copy()
    mangled["inputs"] = mangled["inputs"].map(lambda s: str(int(s)))  # "01" -> "1"
    with pytest.raises(ValueError, match="bit string"):
        run_gate_ml(dev_cfg, mangled, corners, bank)
    missing = gs.copy()
    missing.loc[missing["gate"] == "NAND2", "v_x_spice"] = np.nan
    with pytest.raises(ValueError, match="'x'"):
        run_gate_ml(dev_cfg, missing, corners, bank)


def test_run_gate_ml_only_spice_mode(corners, bank, dev_cfg):
    cfg = dataclasses.replace(
        dev_cfg, gates=dataclasses.replace(dev_cfg.gates, node_voltage_source="ml_solve")
    )
    with pytest.raises(NotImplementedError, match="ml_solve"):
        run_gate_ml(cfg, _fake_gate_spice(corners), corners, bank)


# ---------------------------------------------------------------- SPICE oracle (real ngspice)


@requires_ngspice
def test_spice_oracle_bookkeeping(corners, card, dev_cfg):
    """Simulate every FET ALONE at its in-gate terminal voltages (unsnapped vg), combine with
    leakage_from_currents, and compare with the gate's own SPICE i_vdd / i_in (dev card, one
    corner, all 18 cases). Proves the vdd-net terminal sum, signs and node mapping; says nothing
    about the ML models."""
    c = corners.iloc[[0]]
    corner = c.iloc[0].to_dict()
    ref = run_gate_reference(dev_cfg, c, card)
    assert ref["logic_ok"].all()

    jobs, decks, rows = [], [], {}
    for r in ref.to_dict("records"):
        gate = get_gate(r["gate"])
        rec = {"out": r["v_out_spice"]} | {n: r[f"v_{n}_spice"] for n in gate.internal_nodes}
        nv = resolve_node_voltages(gate, r["inputs"], corner["vdd"], rec)
        X, cases = fet_feature_rows(gate, r["inputs"], corner, nv, dev_cfg)
        rows[(r["gate"], r["inputs"])] = (X, cases)
        for (name, row), case, fet in zip(X.iterrows(), cases, gate.fets, strict=True):
            sample = row.to_dict() | {"vg": nv[fet.g]}  # unsnapped
            decks.append(render_fet_netlist(sample, case, card, dev_cfg.assignment.lmin))
            jobs.append((r["gate"], r["inputs"], name))
    res = run_many_cfg(decks, dev_cfg, expect=FET_EXPECT, progress=False)
    assert all(x.ok for x in res), [j for j, x in zip(jobs, res, strict=True) if not x.ok]
    cur = pd.DataFrame(
        [currents_into_device(x.values, FET_SOURCES) for x in res],
        index=pd.MultiIndex.from_tuples(jobs, names=["gate", "inputs", "fet"]),
    ).sort_index()

    err_vdd, err_in, err_kcl, n_free = [], [], [], 0
    for r in ref.to_dict("records"):
        gate = get_gate(r["gate"])
        fc = cur.loc[(r["gate"], r["inputs"])]
        out = leakage_from_currents(gate, fc, r["inputs"])
        err_vdd.append(abs(out["i_vdd_ml"] - r["i_vdd_spice"]) / r["i_vdd_spice"])
        # KCL composition on SPICE currents: must reproduce SPICE just as well
        X, cases = rows[(r["gate"], r["inputs"])]
        n_free += int(
            sum(
                c in (Case.NMOS_ON, Case.PMOS_ON) and abs(v) < KCL_VDS_MAX
                for c, v in zip(cases, X["vd"] - X["vs"], strict=True)
            )
        )
        kcl = leakage_from_currents(gate, kcl_corrected_currents(gate, fc, X, cases), r["inputs"])
        err_kcl.append(abs(kcl["i_vdd_ml"] - r["i_vdd_spice"]) / r["i_vdd_spice"])
        if any(input_levels(gate, r["inputs"]).values()):
            err_in.append(abs(out["i_in_ml"] - r["i_in_spice"]) / abs(r["i_in_spice"]))
        else:
            assert out["i_in_ml"] == 0.0 and r["i_in_spice"] == 0.0
    assert np.median(err_vdd) < 1e-4, err_vdd
    assert max(err_vdd) < 1e-3, err_vdd
    assert max(err_in) < 1e-6, err_in
    assert n_free >= 18  # the KCL correction really was exercised
    assert np.median(err_kcl) < 1e-4, err_kcl
    assert max(err_kcl) < 1e-3, err_kcl


# ---------------------------------------------------------------- KCL composition (fake currents)


def _fake_currents(gate, seed: int) -> pd.DataFrame:
    """Arbitrary model-like currents that do NOT satisfy KCL (like real ML predictions)."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        rng.uniform(-1e-11, 1e-11, (len(gate.fets), 4)),
        index=pd.Index([f.name for f in gate.fets], name="fet"),
        columns=list(TARGET_COLUMNS),
    )


def _node_kcl(gate, cur: pd.DataFrame, node: str) -> float:
    return sum(cur.loc[f.name, f"i{t}"] for f, t in terminals_on(gate, node))


def test_kcl_parallel_on_nand2_00(corner0, dev_cfg):
    """MPA and MPB are both ON between vdd and out: 2 unknowns, 1 equation (node out). Only the
    sum of their channel currents is determined; i_vdd depends only on that sum."""
    vdd = corner0["vdd"]
    gate, _, X, cases = _rows("NAND2", "00", corner0, dev_cfg)
    X.loc[["MPA", "MPB"], "vd"] = vdd - 3e-8  # in-gate out voltage: |vds| = 30 nV
    assert cases[:2] == [Case.PMOS_ON, Case.PMOS_ON]
    raw = _fake_currents(gate, 1)
    fixed = kcl_corrected_currents(gate, raw, X, cases)

    # untouched: OFF FETs, and ig/ib of the ON ones
    pd.testing.assert_frame_equal(fixed.loc[["MNA", "MNB"]], raw.loc[["MNA", "MNB"]])
    pd.testing.assert_frame_equal(fixed[["ig", "ib"]], raw[["ig", "ib"]])
    # FET-level KCL for the corrected FETs, node KCL at out
    for n in ("MPA", "MPB"):
        assert fixed.loc[n, TARGET_COLUMNS].sum() == pytest.approx(0.0, abs=1e-22)
    assert _node_kcl(gate, fixed, "out") == pytest.approx(0.0, abs=1e-22)
    # minimum-norm split: equal channel currents
    assert fixed.loc["MPA", "id"] == pytest.approx(fixed.loc["MPB", "id"], rel=1e-12)

    # i_vdd by hand: -(idA + idB) - igA - igB, with idA + idB = -MNA.id (node out)
    expected = raw.loc["MNA", "id"] - raw.loc["MPA", "ig"] - raw.loc["MPB", "ig"]
    i_vdd = leakage_from_currents(gate, fixed, "00")["i_vdd_ml"]
    assert i_vdd == pytest.approx(expected, rel=1e-12)

    # any other split of the same sum gives the same i_vdd
    for shift in (3e-12, -7e-12, 1e-9):
        other = fixed.copy()
        other.loc["MPA", "id"] += shift
        other.loc["MPA", "is"] -= shift
        other.loc["MPB", "id"] -= shift
        other.loc["MPB", "is"] += shift
        assert _node_kcl(gate, other, "out") == pytest.approx(0.0, abs=1e-22)
        assert leakage_from_currents(gate, other, "00")["i_vdd_ml"] == pytest.approx(
            i_vdd, rel=1e-9
        )


def test_kcl_series_on_nand2_11(corner0, dev_cfg):
    """MNA (out-x) and MNB (x-0) both ON in series: 2 unknowns, 2 equations (out, x)."""
    gate, _, X, cases = _rows("NAND2", "11", corner0, dev_cfg)
    X.loc["MNA", ["vd", "vs"]] = [2e-7, 1e-7]
    X.loc["MNB", ["vd", "vs"]] = [1e-7, 0.0]
    assert cases[2:] == [Case.NMOS_ON, Case.NMOS_ON]
    raw = _fake_currents(gate, 2)
    fixed = kcl_corrected_currents(gate, raw, X, cases)

    id_a = -(raw.loc["MPA", "id"] + raw.loc["MPB", "id"])  # node out
    is_a = -(id_a + raw.loc["MNA", "ig"] + raw.loc["MNA", "ib"])
    id_b = -is_a  # node x
    assert fixed.loc["MNA", "id"] == pytest.approx(id_a, rel=1e-12)
    assert fixed.loc["MNA", "is"] == pytest.approx(is_a, rel=1e-12)
    assert fixed.loc["MNB", "id"] == pytest.approx(id_b, rel=1e-12)
    for node in ("out", "x"):
        assert _node_kcl(gate, fixed, node) == pytest.approx(0.0, abs=1e-22)
    # nothing ON touches vdd here: the KCL result equals the direct sum
    assert leakage_from_currents(gate, fixed, "11") == leakage_from_currents(gate, raw, "11")


def test_kcl_threshold(corner0, dev_cfg):
    """ON devices at |vds| >= KCL_VDS_MAX, and OFF devices, keep their model currents."""
    gate, _, X, cases = _rows("NAND2", "00", corner0, dev_cfg)
    X.loc["MPA", "vd"] = X.loc["MPA", "vs"] - 2 * KCL_VDS_MAX
    X.loc["MPB", "vd"] = X.loc["MPB", "vs"] - 0.5 * KCL_VDS_MAX
    raw = _fake_currents(gate, 3)
    fixed = kcl_corrected_currents(gate, raw, X, cases)
    pd.testing.assert_frame_equal(fixed.drop(index="MPB"), raw.drop(index="MPB"))
    assert fixed.loc["MPB", "id"] != raw.loc["MPB", "id"]


def test_kcl_rejects_on_path_vdd_to_ground():
    """Not static CMOS: an ON path vdd -> out -> 0 would make i_vdd depend on the split."""
    bad = GateTopology(
        "BAD",
        ("a",),
        "out",
        (),
        (Fet("MP", "pmos", "out", "a", "vdd", "vdd"), Fet("MN", "nmos", "out", "a", "0", "0")),
    )
    with pytest.raises(ValueError, match="vdd-to-ground"):
        _kcl_arrays(bad, lambda k, t: np.array([1e-12]), np.array([[True], [True]]))


def test_run_gate_ml_kcl_vs_direct(corners, bank, dev_cfg):
    """With the stub bank, KCL changes exactly the cases with an ON FET whose source is on vdd at
    |vds| ~ 0 (the recorded out of _fake_gate_spice is exactly 0 or VDD)."""
    df = run_gate_ml(dev_cfg, _fake_gate_spice(corners), corners, bank)
    changed = ~np.isclose(df["i_vdd_ml"], df["i_vdd_ml_direct"], rtol=1e-12, atol=0)
    no_on_pmos = df["gate"].isin(["INV", "NAND2"]) & df["inputs"].isin(["1", "11"])
    no_on_pmos |= (df["gate"] == "NOR2") & df["inputs"].isin(["10", "11"])
    assert not changed[no_on_pmos].any()
    assert changed[(df["gate"] == "INV") & (df["inputs"] == "0")].all()


# ---------------------------------------------------------------- real models (if available)


def _real_bank_or_skip():
    cfg = load_config(DEFAULT_CONFIG)
    if not cfg.paths.gate_spice_csv.exists():
        pytest.skip("no real gate_spice.csv")
    try:
        bank = TransistorModelBank.load_all(cfg.paths.models_dir)
    except (FileNotFoundError, OSError, ImportError) as e:  # models missing / libomp missing
        pytest.skip(f"real transistor models not loadable: {e}")
    if any(m.meta.get("stand_in") for m in bank.models.values()):
        pytest.skip("stand-in models")
    return cfg, bank


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_real_run_kcl_leakage_positive():
    """On the real config with the real models: the KCL i_vdd_ml is never negative."""
    cfg, bank = _real_bank_or_skip()
    df = run_gate_ml(
        cfg, io.read_csv(cfg.paths.gate_spice_csv), io.read_csv(cfg.paths.corners_csv), bank
    )
    assert len(df) == 3600
    assert (df["i_vdd_ml"] > 0).all(), df.nsmallest(5, "i_vdd_ml")
