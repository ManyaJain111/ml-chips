"""Part C step 6: SPICE reference for the complete test circuit.  Owner: P4."""

from __future__ import annotations

import pandas as pd

from leakml.circuit.definition import Circuit
from leakml.config import ProjectConfig
from leakml.pdk import ModelCard


def run_circuit_reference(
    c: Circuit, cfg: ProjectConfig, corners: pd.DataFrame, card: ModelCard
) -> pd.DataFrame:
    """Every input vector x the SAME corners as Part B -> data/circuit/circuit_spice.csv.

    TODO(P4):
      - cla3: 128 x 200 = 25,600 decks. parity8: 256 x 200 = 51,200 decks. Use runner.run_many
        and consider chunking or checkpointing to disk every N runs, so a crash doesn't lose
        hours of work
      - i_vdd_spice = -i(vvdd), i_in_spice = sum over primary inputs at 1 of -i(v<pi>)
      - logic_ok: every primary output voltage matches logic.reference_function (|v - level| < 0.1 vdd)
      - columns: circuit, vector, corner_id, vdd, temp, i_vdd_spice, i_in_spice, logic_ok
    """
    raise NotImplementedError
