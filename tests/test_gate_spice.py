"""Tests for leakml.gates.netlist.render_gate_netlist and leakml.gates.spice_ref (Part B step 3).

Unit tests use hand-made SPICE values and need no simulator. The tests marked requires_ngspice run
the real simulator on the dev (fixture) card and skip where ngspice is not installed.
"""

from __future__ import annotations

import math
import re
import shutil

import numpy as np
import pandas as pd
import pytest

from leakml.constants import GATE_INPUT_COMBOS, GATES, SPICE_OPTIONS
from leakml.gates.netlist import render_gate_netlist, subckt_pins
from leakml.gates.spice_ref import (
    GATE_SPICE_COLUMNS,
    extract_gate_result,
    gate_print_exprs,
    run_gate_reference,
)
from leakml.gates.topology import get_gate, input_levels, logic_function
from leakml.sampling import draw_corners, make_rng

requires_ngspice = pytest.mark.skipif(shutil.which("ngspice") is None, reason="ngspice not on PATH")

VDD = 0.9


@pytest.fixture(scope="module")
def corners(dev_cfg, card) -> pd.DataFrame:
    return draw_corners(2, dev_cfg, card, make_rng(dev_cfg.seeds.corners, "corners"))


@pytest.fixture(scope="module")
def corner0(corners) -> dict:
    return corners.iloc[0].to_dict()


def _values(gate_name: str, bits: str, *, v_out: float | None = None) -> dict[str, float]:
    """Plausible parsed SPICE output: supply delivers 2 nA, each input source -1 pA (high) or
    +5 pA (low), output at the correct logic level unless v_out is given."""
    gate = get_gate(gate_name)
    vals = {"i(vvdd)": -2e-9}
    for pin, lv in input_levels(gate, bits).items():
        vals[f"i(v{pin})"] = -1e-12 if lv else 5e-12
    vals["v(out)"] = VDD * logic_function(gate_name, bits) if v_out is None else v_out
    for n in gate.internal_nodes:
        vals[f"v(x1.{n})"] = 0.3
    return vals


# ---------------------------------------------------------------- extract_gate_result (no ngspice)


def test_extract_supply_sign():
    row = extract_gate_result(_values("INV", "0"), get_gate("INV"), VDD, inputs="0")
    assert row["i_vdd_spice"] == pytest.approx(2e-9)  # -i(vvdd): supply delivering is positive


@pytest.mark.parametrize(
    ("bits", "expected"), [("00", 0.0), ("01", 1e-12), ("10", 1e-12), ("11", 2e-12)]
)
def test_extract_input_current_high_inputs_only(bits, expected):
    row = extract_gate_result(_values("NAND2", bits), get_gate("NAND2"), VDD, inputs=bits)
    # low inputs carry +5 pA in _values; they must not contribute
    assert row["i_in_spice"] == pytest.approx(expected, abs=1e-24)


def test_extract_low_inputs_excluded():
    vals = _values("NOR2", "10")
    vals["i(vb)"] = -1.0  # a huge current on the LOW input must be ignored
    row = extract_gate_result(vals, get_gate("NOR2"), VDD, inputs="10")
    assert row["i_in_spice"] == pytest.approx(1e-12)


@pytest.mark.parametrize(
    ("gate_name", "present"),
    [("INV", ()), ("NAND2", ("x",)), ("NOR2", ("y",)), ("AND2", ("x", "n1")), ("OR2", ("y", "n1"))],
)
def test_extract_internal_nodes_nan_when_missing(gate_name, present):
    bits = GATE_INPUT_COMBOS[gate_name][0]
    row = extract_gate_result(_values(gate_name, bits), get_gate(gate_name), VDD, inputs=bits)
    for node in ("x", "y", "n1"):
        v = row[f"v_{node}_spice"]
        assert (v == pytest.approx(0.3)) if node in present else math.isnan(v)


@pytest.mark.parametrize("gate_name", GATES)
def test_extract_logic_ok_correct_output(gate_name):
    for bits in GATE_INPUT_COMBOS[gate_name]:
        row = extract_gate_result(_values(gate_name, bits), get_gate(gate_name), VDD, inputs=bits)
        assert row["logic_ok"] is True
        assert row["v_out_spice"] == pytest.approx(VDD * logic_function(gate_name, bits))


@pytest.mark.parametrize(
    ("gate_name", "bits", "v_out"),
    [
        ("INV", "0", 0.0),  # should be VDD
        ("NAND2", "11", VDD),  # should be 0
        ("AND2", "11", 0.5 * VDD),  # mid-rail
        ("NOR2", "00", 0.85 * VDD),  # 0.15*VDD off: outside the 0.1*VDD window
        ("OR2", "00", 0.1 * VDD + 1e-6),  # just outside
    ],
)
def test_extract_logic_ok_wrong_output(gate_name, bits, v_out):
    vals = _values(gate_name, bits, v_out=v_out)
    row = extract_gate_result(vals, get_gate(gate_name), VDD, inputs=bits)
    assert row["logic_ok"] is False


def test_extract_logic_ok_threshold_is_strict():
    ok = extract_gate_result(
        _values("INV", "1", v_out=0.0999 * VDD), get_gate("INV"), VDD, inputs="1"
    )
    assert ok["logic_ok"] is True


# ---------------------------------------------------------------- render_gate_netlist (no ngspice)


@pytest.mark.parametrize("gate_name", GATES)
def test_deck_header_and_temp(gate_name, corner0, card, dev_cfg):
    bits = GATE_INPUT_COMBOS[gate_name][-1]
    deck = render_gate_netlist(get_gate(gate_name), bits, corner0, card, dev_cfg)
    lines = deck.splitlines()
    assert lines[0].startswith("*") and gate_name in lines[0] and bits in lines[0]
    assert lines[1] == SPICE_OPTIONS
    (temp,) = re.findall(r"^\.temp (\S+)$", deck, flags=re.M)
    assert float(temp) == pytest.approx(corner0["temp"], rel=1e-9)
    assert deck.count(".model") == 2  # one corner draw per device type
    assert deck.rstrip().endswith(".end")


@pytest.mark.parametrize("gate_name", GATES)
def test_deck_sources_and_instance(gate_name, corner0, card, dev_cfg):
    gate = get_gate(gate_name)
    vdd = corner0["vdd"]
    for bits in GATE_INPUT_COMBOS[gate_name]:
        deck = render_gate_netlist(gate, bits, corner0, card, dev_cfg)
        sources = re.findall(r"^V(\w+) (\w+) 0 dc (\S+)$", deck, flags=re.M)
        # one supply + exactly one source per input, each named after (and driving) its pin
        assert [s[0].lower() for s in sources] == ["vdd", *gate.inputs]
        assert all(name.lower() == node for name, node, _ in sources)
        assert float(sources[0][2]) == pytest.approx(vdd, rel=1e-9)
        for (_, _pin, v), bit in zip(sources[1:], bits, strict=True):
            if bit == "0":
                assert v == "0" and float(v) == 0.0
            else:
                assert float(v) == pytest.approx(vdd, rel=1e-9)
        # X1 with nets in subckt_pins order, subckt gnd pin tied to node 0
        nets = ["0" if p == "gnd" else p for p in subckt_pins(gate)]
        x_lines = re.findall(r"^X.*$", deck, flags=re.M)
        assert x_lines == [f"X1 {' '.join(nets)} {gate_name}"]
        assert f".subckt {gate_name} {' '.join(subckt_pins(gate))}" in deck


@pytest.mark.parametrize("gate_name", GATES)
def test_deck_print_list(gate_name, corner0, card, dev_cfg):
    gate = get_gate(gate_name)
    deck = render_gate_netlist(gate, GATE_INPUT_COMBOS[gate_name][0], corner0, card, dev_cfg)
    printed = [e for line in re.findall(r"^print (.*)$", deck, flags=re.M) for e in line.split()]
    assert printed == gate_print_exprs(gate)
    expected_nodes = {f"v(x1.{n})" for n in gate.internal_nodes}
    assert {e for e in printed if e.startswith("v(x1.")} == expected_nodes  # nothing extra
    assert {"i(vvdd)", "v(out)"} <= set(printed)
    assert {f"i(v{p})" for p in gate.inputs} <= set(printed)


@pytest.mark.parametrize("gate_name", GATES)
def test_deck_output_unloaded(gate_name, corner0, card, dev_cfg):
    gate = get_gate(gate_name)
    deck = render_gate_netlist(gate, GATE_INPUT_COMBOS[gate_name][0], corner0, card, dev_cfg)
    top_level = deck.split(f".ends {gate_name}", 1)[1].split(".control", 1)[0]
    # outside the subckt: only V sources and the X1 instance; `out` appears only on X1
    elements = [ln for ln in top_level.splitlines() if ln.strip()]
    assert all(ln[0] in "VX" for ln in elements), elements
    assert [ln for ln in elements if re.search(r"\bout\b", ln)] == [
        ln for ln in elements if ln.startswith("X1")
    ]


def test_deck_rejects_wrong_bit_count(corner0, card, dev_cfg):
    with pytest.raises(ValueError):
        render_gate_netlist(get_gate("NAND2"), "1", corner0, card, dev_cfg)


# ---------------------------------------------------------------- real ngspice


@requires_ngspice
def test_gate_reference_one_corner(corners, card, dev_cfg):
    """All 18 cases at one dev corner: correct logic, positive supply current, schema intact."""
    df = run_gate_reference(dev_cfg, corners.iloc[[0]], card)
    assert list(df.columns) == list(GATE_SPICE_COLUMNS)
    assert len(df) == 18
    assert sorted(zip(df["gate"], df["inputs"], strict=True)) == sorted(
        (g, b) for g in GATES for b in GATE_INPUT_COMBOS[g]
    )
    assert df["inputs"].map(type).eq(str).all()  # bit strings, never ints
    assert df["logic_ok"].all(), df[~df["logic_ok"]]
    assert (df["i_vdd_spice"] > 0).all(), df[["gate", "inputs", "i_vdd_spice"]]
    assert np.isfinite(df["i_in_spice"]).all()
    for g in GATES:
        sub = df[df["gate"] == g]
        for node in ("x", "y", "n1"):
            col = sub[f"v_{node}_spice"]
            assert (col.notna() if node in get_gate(g).internal_nodes else col.isna()).all()
