"""leakml.gates.topology (implemented foundation). SPICE-level checks live in test_spice.py."""

import pytest

from leakml.constants import GATE_INPUT_COMBOS, GATES
from leakml.gates.netlist import render_gate_subckt, subckt_pins
from leakml.gates.topology import (
    GATE_LIBRARY,
    get_gate,
    logic_function,
    terminals_on,
    validate_topology,
)

TRUTH = {
    "INV": {"0": 1, "1": 0},
    "NAND2": {"00": 1, "01": 1, "10": 1, "11": 0},
    "NOR2": {"00": 1, "01": 0, "10": 0, "11": 0},
    "AND2": {"00": 0, "01": 0, "10": 0, "11": 1},
    "OR2": {"00": 0, "01": 1, "10": 1, "11": 1},
}


def test_library_complete():
    assert set(GATE_LIBRARY) == set(GATES)
    for g in GATES:
        validate_topology(get_gate(g))


@pytest.mark.parametrize("gate", GATES)
def test_truth_tables(gate):
    assert {b: logic_function(gate, b) for b in GATE_INPUT_COMBOS[gate]} == TRUTH[gate]


def test_bad_input_bits():
    with pytest.raises(ValueError):
        logic_function("NAND2", "1")


def test_fet_counts_and_internal_nodes():
    assert {g: len(get_gate(g).fets) for g in GATES} == {
        "INV": 2,
        "NAND2": 4,
        "NOR2": 4,
        "AND2": 6,
        "OR2": 6,
    }
    assert get_gate("AND2").internal_nodes == ("x", "n1")
    assert get_gate("OR2").internal_nodes == ("y", "n1")


def test_vdd_terminals():
    inv = [(f.name, t) for f, t in terminals_on(get_gate("INV"), "vdd")]
    assert inv == [("MP", "s"), ("MP", "b")]
    nor = [(f.name, t) for f, t in terminals_on(get_gate("NOR2"), "vdd")]
    assert nor == [("MPA", "s"), ("MPA", "b"), ("MPB", "b")]  # MPB source is y, not vdd


def test_subckt_text():
    g = get_gate("NAND2")
    txt = render_gate_subckt(g, 9e-8, 1.8e-7, 4.5e-8)
    assert txt.splitlines()[0] == ".subckt NAND2 a b out vdd gnd"
    assert "MNB x b gnd gnd nmos W=9e-08 L=4.5e-08" in txt
    assert txt.strip().endswith(".ends NAND2")
    assert subckt_pins(get_gate("INV")) == ("a", "out", "vdd", "gnd")
