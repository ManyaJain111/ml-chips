"""Part B step 3: SPICE reference for the 18 gate cases x 200 corners.  Owner: P3.

These results are for EVALUATION ONLY. They must never be used to train or tune the transistor
models (brief Part B step 3).
"""

from __future__ import annotations

from collections.abc import Mapping

import pandas as pd

from leakml.config import ProjectConfig
from leakml.gates.topology import GateTopology
from leakml.pdk import ModelCard


def extract_gate_result(
    values: Mapping[str, float], gate: GateTopology, vdd: float
) -> dict[str, float]:
    """Turn parsed SPICE values into one result row (SPEC §5.4 columns).

    TODO(P3):
      - i_vdd_spice   = -i(vvdd)                 (current delivered by the supply, SPEC §3.5)
      - i_in_spice    = sum over inputs at logic 1 of -i(v<inp>)  (secondary metric)
      - v_out_spice, v_<node>_spice for every internal node
      - flag logic_ok = |v_out - expected_level| < 0.1*vdd, using topology.logic_function
    """
    raise NotImplementedError


def run_gate_reference(cfg: ProjectConfig, corners: pd.DataFrame, card: ModelCard) -> pd.DataFrame:
    """Simulate every (gate, inputs, corner) and return the long table -> data/gates/gate_spice.csv.

    TODO(P3):
      - loop over constants.GATES x GATE_INPUT_COMBOS x corners (18 x 200 = 3600 decks)
      - build all decks first, then one spice.runner.run_many call (parallel)
      - AND2/OR2 are simulated as COMPLETE gates (do not add NAND2 + INV results, brief §5)
      - columns: gate, inputs, corner_id, vdd, temp, i_vdd_spice, i_in_spice, v_out_spice,
        v_x_spice, v_y_spice, v_n1_spice (NaN where a node doesn't exist), logic_ok
    """
    raise NotImplementedError
