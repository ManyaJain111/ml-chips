"""Part C step 7: hierarchical ML leakage = sum of gate-model leakages.  Owner: P4.

Hierarchy: circuit -> gate instances (logic sim gives each one's input bits) -> GateLeakageModel
(P3) -> TransistorModelBank (P2). Assumes every gate input sits at exactly 0 or VDD and ignores
loading between gates (brief §5).
"""

from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd

from leakml.circuit import logic
from leakml.circuit.definition import Circuit
from leakml.config import ProjectConfig
from leakml.gates.ml_model import GateLeakageModel


def gate_counts(c: Circuit, vector: str) -> Counter[tuple[str, str]]:
    """How many instances of each (gate, input bits) the vector exercises."""
    pairs = logic.gate_input_bits(c, logic.evaluate(c, vector))
    return Counter((inst.gate, bits) for inst, bits in pairs)


def predict_circuit_leakage(
    c: Circuit, vector: str, corner_id: int, gate_model: GateLeakageModel, metric: str = "i_vdd_ml"
) -> dict[str, float]:
    """Total ML leakage for one input vector at one corner.

    Returns {"total": sum, "<GATE>": per-gate-type contribution, ...} in amperes.
    """
    out: dict[str, float] = {"total": 0.0}
    for (gate, bits), n in gate_counts(c, vector).items():
        val = n * gate_model.predict(gate, bits, corner_id, metric)
        out["total"] += val
        out[gate] = out.get(gate, 0.0) + val
    return out


def run_circuit_ml(
    c: Circuit, cfg: ProjectConfig, corners: pd.DataFrame, gate_model: GateLeakageModel
) -> pd.DataFrame:
    """All vectors x corners -> data/circuit/circuit_ml.csv.

    Gate input bits are computed once per vector, then the sum over gates is a matrix product
    (counts [vectors x keys] @ gate leakage [keys x corners]).
    Columns: circuit, vector, corner_id, i_vdd_ml, i_in_ml, n_<gate>_<bits>.
    i_in_ml sums the gate-input currents of ALL gates (internal gate inputs are driven by other
    gates, so unlike SPICE's primary-input current it is only indicative; i_vdd is the metric).
    """
    vectors = logic.all_input_vectors(c)
    counts = [gate_counts(c, v) for v in vectors]
    keys = sorted({k for cnt in counts for k in cnt})
    cids = [int(x) for x in corners["corner_id"]]
    count_mat = np.array([[cnt.get(k, 0) for k in keys] for cnt in counts], dtype=float)

    def lookup(metric: str) -> np.ndarray:
        return np.array([[gate_model.predict(g, b, cid, metric) for cid in cids] for g, b in keys])

    i_vdd = count_mat @ lookup("i_vdd_ml")
    i_in = count_mat @ lookup("i_in_ml") if "i_in_ml" in gate_model.metrics else None

    nv, nc = len(vectors), len(cids)
    df = pd.DataFrame(
        {
            "circuit": c.name,
            "vector": np.repeat(vectors, nc),
            "corner_id": np.tile(cids, nv),
            "i_vdd_ml": i_vdd.ravel(),
            "i_in_ml": i_in.ravel() if i_in is not None else np.nan,
        }
    )
    for j, (g, b) in enumerate(keys):
        df[f"n_{g}_{b}"] = np.repeat(count_mat[:, j].astype(int), nc)
    return df
