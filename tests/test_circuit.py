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


def _toy_gate_model(i_in_scale: float = 1.0):
    """NAND2 gate ML: i_vdd = 1 + corner_id, i_in = scale * (number of high pins)."""
    import pandas as pd

    from leakml.gates.ml_model import GateLeakageModel

    rows = [
        {"gate": "NAND2", "inputs": b, "corner_id": cid, "i_vdd_ml": 1.0 + cid,
         "i_in_ml": i_in_scale * b.count("1")}
        for b in ("00", "01", "10", "11") for cid in (0, 1)
    ]  # fmt: skip
    return GateLeakageModel.from_results(pd.DataFrame(rows)), pd.DataFrame({"corner_id": [0, 1]})


def test_internal_high_fraction():
    from leakml.circuit.ml_model import internal_high_fraction

    c = get_circuit("parity8")
    g1, g2 = c.instances[0], c.instances[1]  # NAND(a,b), NAND(a, n1)
    assert internal_high_fraction(c, g1, "11") == 0.0  # both primary
    assert internal_high_fraction(c, g1, "00") == 0.0
    assert internal_high_fraction(c, g2, "11") == 0.5
    assert internal_high_fraction(c, g2, "01") == 1.0


def test_circuit_ml_loading_term():
    from leakml.circuit.ml_model import predict_circuit_leakage, run_circuit_ml

    c = get_circuit("parity8")
    gm, corners = _toy_gate_model()
    df = run_circuit_ml(c, None, corners, gm)
    assert len(df) == 256 * 2
    # isolated-gate sum: 28 gates x (1 + corner_id)
    assert (df["i_vdd_ml_gates"] == 28 * (1.0 + df["corner_id"])).all()
    # every high pin carries 1 A: loading + primary share = total high pins of all gates
    for v in ("00000000", "10110010", "11111111"):
        nets = logic.evaluate(c, v)
        high = sum(nets[n] for i in c.instances for n in i.inputs)
        prim_high = sum(nets[n] for i in c.instances for n in i.inputs if n in c.primary_inputs)
        r = df[(df["vector"] == v) & (df["corner_id"] == 1)].iloc[0]
        assert r["i_load_ml"] + r["i_in_ml"] == pytest.approx(high)
        assert r["i_in_ml"] == pytest.approx(prim_high)
        ref = predict_circuit_leakage(c, v, 1, gm)
        assert r["i_vdd_ml"] == pytest.approx(ref["total"])
        assert r["i_load_ml"] == pytest.approx(ref["loading"])


def test_circuit_ml_without_input_current_is_plain_sum():
    from leakml.circuit.ml_model import run_circuit_ml

    gm, corners = _toy_gate_model(i_in_scale=0.0)
    df = run_circuit_ml(get_circuit("parity8"), None, corners, gm)
    assert (df["i_vdd_ml"] == df["i_vdd_ml_gates"]).all()
    assert (df["i_load_ml"] == 0).all()
