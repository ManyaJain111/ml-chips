"""Part B step 3: SPICE reference for the 18 gate cases x 200 corners.  Owner: P3.

These results are for EVALUATION ONLY. They must never be used to train or tune the transistor
models (brief Part B step 3).
"""

from __future__ import annotations

import logging
from collections.abc import Mapping

import numpy as np
import pandas as pd

from leakml.config import ProjectConfig
from leakml.constants import GATE_INPUT_COMBOS, GATES
from leakml.gates.netlist import render_gate_netlist
from leakml.gates.topology import GateTopology, get_gate, input_levels, logic_function
from leakml.pdk import ModelCard
from leakml.spice.runner import run_many_cfg

log = logging.getLogger(__name__)

# SPEC §5.4, in this order.
GATE_SPICE_COLUMNS: tuple[str, ...] = (
    "gate",
    "inputs",
    "corner_id",
    "vdd",
    "temp",
    "i_vdd_spice",
    "i_in_spice",
    "v_out_spice",
    "v_x_spice",
    "v_y_spice",
    "v_n1_spice",
    "logic_ok",
)
NODE_COLUMNS: dict[str, str] = {"x": "v_x_spice", "y": "v_y_spice", "n1": "v_n1_spice"}
# Printed by every gate deck, whatever the gate (the rest is checked per gate after the run).
COMMON_EXPECT: tuple[str, ...] = ("i(vvdd)", "v(out)")


def gate_print_exprs(gate: GateTopology) -> list[str]:
    """Quantities printed by render_gate_netlist for this gate (keys of SpiceResult.values)."""
    return [
        "i(vvdd)",
        *[f"i(v{pin})" for pin in gate.inputs],
        f"v({gate.output})",
        *[f"v(x1.{n})" for n in gate.internal_nodes],
    ]


def extract_gate_result(
    values: Mapping[str, float], gate: GateTopology, vdd: float, *, inputs: str
) -> dict[str, float]:
    """Turn parsed SPICE values into one result row (the measured SPEC §5.4 columns).

      - i_vdd_spice = -i(vvdd)                  (current delivered by the supply, SPEC §3.5)
      - i_in_spice  = sum of -i(v<pin>) over the inputs at logic 1, 0.0 if none are high
      - v_out_spice, v_x_spice / v_y_spice / v_n1_spice (NaN for nodes the gate doesn't have)
      - logic_ok    = |v_out - vdd*logic_function(gate, inputs)| < 0.1*vdd

    `inputs` (the bit string, e.g. "01") is needed for i_in and logic_ok; it is keyword-only so
    the original (values, gate, vdd) positional order is unchanged.
    """
    levels = input_levels(gate, inputs)
    v_out = float(values[f"v({gate.output})"])
    expected = vdd * logic_function(gate.name, inputs)
    row: dict[str, float] = {
        "i_vdd_spice": -float(values["i(vvdd)"]),
        "i_in_spice": float(sum(-float(values[f"i(v{pin})"]) for pin, lv in levels.items() if lv)),
        "v_out_spice": v_out,
    }
    for node, col in NODE_COLUMNS.items():
        row[col] = float(values[f"v(x1.{node})"]) if node in gate.internal_nodes else np.nan
    row["logic_ok"] = bool(abs(v_out - expected) < 0.1 * vdd)
    return row


def run_gate_reference(cfg: ProjectConfig, corners: pd.DataFrame, card: ModelCard) -> pd.DataFrame:
    """Simulate every (gate, inputs, corner) and return the long table -> data/gates/gate_spice.csv.

    18 cases (constants.GATES x GATE_INPUT_COMBOS) x every corner, gate-major then inputs then
    corner. All decks are built first and run in ONE parallel run_many call. AND2/OR2 are
    simulated as COMPLETE gates (never NAND2/NOR2 + INV added together, brief §5).

    A deck that fails (ngspice error, non-convergence, or a missing printed value) is kept as a
    row with NaN measurements and logic_ok=False, and logged; nothing is dropped. Callers can
    tell a failed simulation from a logic failure by i_vdd_spice being NaN.
    Columns: GATE_SPICE_COLUMNS (SPEC §5.4).
    """
    jobs = [
        (gate_name, bits, corner)
        for gate_name in GATES
        for bits in GATE_INPUT_COMBOS[gate_name]
        for corner in corners.to_dict("records")
    ]
    decks = [render_gate_netlist(get_gate(g), bits, cr, card, cfg) for g, bits, cr in jobs]
    results = run_many_cfg(decks, cfg, expect=list(COMMON_EXPECT), desc="gate decks")

    rows = []
    for (gate_name, bits, corner), res in zip(jobs, results, strict=True):
        gate = get_gate(gate_name)
        vdd = float(corner["vdd"])
        cid = int(corner["corner_id"])
        error = res.error
        if res.ok:
            missing = [k for k in gate_print_exprs(gate) if k not in res.values]
            if missing:
                error = f"missing outputs: {missing}"
        if res.ok and not error:
            meas = extract_gate_result(res.values, gate, vdd, inputs=bits)
        else:
            log.error(
                "%s %s corner %d: simulation failed: %s (deck: %s)",
                gate_name,
                bits,
                cid,
                error,
                res.netlist_path,
            )
            meas = {c: np.nan for c in GATE_SPICE_COLUMNS[5:-1]} | {"logic_ok": False}
        rows.append(
            {"gate": gate_name, "inputs": bits, "corner_id": cid, "vdd": vdd}
            | {"temp": float(corner["temp"])}
            | meas
        )
    return pd.DataFrame(rows, columns=list(GATE_SPICE_COLUMNS))
