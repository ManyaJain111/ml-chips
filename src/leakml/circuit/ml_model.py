"""Part C step 7: hierarchical ML leakage = sum of gate-model leakages + inter-gate loading.  Owner: P4.

Hierarchy: circuit -> gate instances (logic sim gives each one's input bits) -> GateLeakageModel
(P3) -> TransistorModelBank (P2). Every gate input is assumed to sit at exactly 0 or VDD.

Loading term (SPEC §6.3). In Part B each gate's inputs are ideal sources, so the gate-tunnelling
current drawn by an input at logic 1 (`i_in_ml`) is delivered by that source and is NOT in the
gate's `i_vdd_ml`. Inside the circuit a high internal net is held up by its driver's pull-up, so
that same current is drawn from VDD. A low net sinks its current into the driver's pull-down,
which adds nothing to VDD. Hence

    i_vdd_ml = sum_gates i_vdd_ml(gate, bits) + sum_gates f_int * i_in_ml(gate, bits)

where f_int is the fraction of a gate's high input pins that are driven by another gate (high
pins fed by primary inputs are supplied by the testbench sources, as in SPICE). The gate ML only
gives `i_in_ml` summed over a gate's high pins, so it is split evenly between them (exact for
symmetric gates such as NAND2/NOR2, an approximation for the AND2/OR2 first stage).
"""

from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd

from leakml.circuit import logic
from leakml.circuit.definition import Circuit, GateInstance
from leakml.config import ProjectConfig
from leakml.gates.ml_model import GateLeakageModel


def gate_counts(c: Circuit, vector: str) -> Counter[tuple[str, str]]:
    """How many instances of each (gate, input bits) the vector exercises."""
    pairs = logic.gate_input_bits(c, logic.evaluate(c, vector))
    return Counter((inst.gate, bits) for inst, bits in pairs)


def internal_high_fraction(c: Circuit, inst: GateInstance, bits: str) -> float:
    """Share of `inst`'s logic-1 input pins that are driven by another gate (0.0 if none high)."""
    primary = set(c.primary_inputs)
    high = [net for net, b in zip(inst.inputs, bits, strict=True) if b == "1"]
    if not high:
        return 0.0
    return sum(net not in primary for net in high) / len(high)


def loading_weights(c: Circuit, vector: str) -> Counter[tuple[str, str]]:
    """Sum of `internal_high_fraction` per (gate, input bits): the multiplier on `i_in_ml`."""
    out: Counter[tuple[str, str]] = Counter()
    for inst, bits in logic.gate_input_bits(c, logic.evaluate(c, vector)):
        out[(inst.gate, bits)] += internal_high_fraction(c, inst, bits)
    return out


def predict_circuit_leakage(
    c: Circuit, vector: str, corner_id: int, gate_model: GateLeakageModel
) -> dict[str, float]:
    """Total ML leakage for one input vector at one corner (reference implementation).

    Returns {"total": gates + loading, "gates": sum of gate i_vdd_ml, "loading": loading term,
    "<GATE>": per-gate-type i_vdd_ml contribution, ...} in amperes.
    """
    out: dict[str, float] = {"total": 0.0, "gates": 0.0, "loading": 0.0}
    for (gate, bits), n in gate_counts(c, vector).items():
        val = n * gate_model.predict(gate, bits, corner_id, "i_vdd_ml")
        out["gates"] += val
        out[gate] = out.get(gate, 0.0) + val
    for (gate, bits), w in loading_weights(c, vector).items():
        if w:
            out["loading"] += w * gate_model.predict(gate, bits, corner_id, "i_in_ml")
    out["total"] = out["gates"] + out["loading"]
    return out


def run_circuit_ml(
    c: Circuit, cfg: ProjectConfig, corners: pd.DataFrame, gate_model: GateLeakageModel
) -> pd.DataFrame:
    """All vectors x corners -> data/circuit/circuit_ml.csv.

    Gate input bits are computed once per vector, then each sum over gates is a matrix product
    (weights [vectors x keys] @ gate result [keys x corners]).
    Columns: circuit, vector, corner_id,
      i_vdd_ml        gates + loading (the prediction compared with i_vdd_spice)
      i_vdd_ml_gates  sum of isolated-gate i_vdd_ml (no loading term)
      i_load_ml       loading term (internal high inputs' i_in_ml)
      i_in_ml         primary-input share of i_in_ml (comparable with i_in_spice)
      n_<gate>_<bits> instance counts.
    """
    vectors = logic.all_input_vectors(c)
    counts = [gate_counts(c, v) for v in vectors]
    loads = [loading_weights(c, v) for v in vectors]
    keys = sorted({k for cnt in counts for k in cnt})
    cids = [int(x) for x in corners["corner_id"]]
    count_mat = np.array([[cnt.get(k, 0) for k in keys] for cnt in counts], dtype=float)
    load_mat = np.array([[ld.get(k, 0.0) for k in keys] for ld in loads], dtype=float)
    # high-pin share fed by primary inputs: (number of instances with >=1 high pin) - load share
    any_high = np.array([[0.0 if "1" not in b else 1.0 for _, b in keys]])
    prim_mat = count_mat * any_high - load_mat

    def lookup(metric: str) -> np.ndarray:
        return np.array([[gate_model.predict(g, b, cid, metric) for cid in cids] for g, b in keys])

    i_gates = count_mat @ lookup("i_vdd_ml")
    i_in_gate = lookup("i_in_ml")
    i_load = load_mat @ i_in_gate
    i_prim = prim_mat @ i_in_gate

    nv, nc = len(vectors), len(cids)
    df = pd.DataFrame(
        {
            "circuit": c.name,
            "vector": np.repeat(vectors, nc),
            "corner_id": np.tile(cids, nv),
            "i_vdd_ml": (i_gates + i_load).ravel(),
            "i_vdd_ml_gates": i_gates.ravel(),
            "i_load_ml": i_load.ravel(),
            "i_in_ml": i_prim.ravel(),
        }
    )
    for j, (g, b) in enumerate(keys):
        df[f"n_{g}_{b}"] = np.repeat(count_mat[:, j].astype(int), nc)
    return df
