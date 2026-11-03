"""Part C step 7: hierarchical ML leakage = sum of gate-model leakages.  Owner: P4.

Hierarchy: circuit -> gate instances (logic sim gives each one's input bits) -> GateLeakageModel
(P3) -> TransistorModelBank (P2). Assumes every gate input sits at exactly 0 or VDD and ignores
loading between gates (brief §5).
"""

from __future__ import annotations

import pandas as pd

from leakml.circuit.definition import Circuit
from leakml.config import ProjectConfig
from leakml.gates.ml_model import GateLeakageModel


def predict_circuit_leakage(
    c: Circuit, vector: str, corner_id: int, gate_model: GateLeakageModel, metric: str = "i_vdd_ml"
) -> dict[str, float]:
    """Total ML leakage for one input vector at one corner.

    TODO(P4):
      - net_values = logic.evaluate(c, vector)
      - for (inst, bits) in logic.gate_input_bits(c, net_values):
            total += gate_model.predict(inst.gate, bits, corner_id, metric)
      - also return the per-gate-type breakdown (useful for error analysis: which gates dominate
        the leakage)
    """
    raise NotImplementedError


def run_circuit_ml(
    c: Circuit, cfg: ProjectConfig, corners: pd.DataFrame, gate_model: GateLeakageModel
) -> pd.DataFrame:
    """All vectors x corners -> data/circuit/circuit_ml.csv.

    TODO(P4): precompute gate_input_bits per vector ONCE (it doesn't depend on the corner), then
    loop over the corners. Columns: circuit, vector, corner_id, i_vdd_ml, i_in_ml,
    n_<gate>_<bits> counts.
    """
    raise NotImplementedError
