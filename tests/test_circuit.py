"""Owner: P4."""

import pytest

from leakml.circuit import logic
from leakml.circuit.definition import get_circuit
from leakml.constants import GATES

NAMES = ["cla3", "parity8"]


@pytest.mark.parametrize("name", NAMES)
def test_validate_both_circuits(name):
    c = get_circuit(name)  # runs validate_circuit
    assert {i.gate for i in c.instances} <= set(GATES)


@pytest.mark.parametrize("name,n", [("cla3", 128), ("parity8", 256)])
def test_logic_matches_reference_exhaustively(name, n):
    c = get_circuit(name)
    vectors = logic.all_input_vectors(c)
    assert len(vectors) == n == len(set(vectors))
    for v in vectors:
        nets = logic.evaluate(c, v)
        ref = logic.reference_function(c, v)
        assert {po: nets[po] for po in c.primary_outputs} == ref, v


def test_gate_counts():
    c = get_circuit("parity8")
    assert [i.gate for i in c.instances] == ["NAND2"] * 28
    cla = get_circuit("cla3")
    kinds = {g: sum(i.gate == g for i in cla.instances) for g in GATES}
    assert kinds == {"INV": 0, "NAND2": 24, "NOR2": 0, "AND2": 12, "OR2": 6}
