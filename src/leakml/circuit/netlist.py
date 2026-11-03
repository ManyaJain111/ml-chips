"""Transistor-level SPICE deck for the full test circuit (Deliverable 2).  Owner: P4."""

from __future__ import annotations

from collections.abc import Mapping

from leakml.circuit.definition import Circuit
from leakml.config import ProjectConfig
from leakml.pdk import ModelCard


def render_circuit_netlist(
    c: Circuit,
    vector: str,
    corner: Mapping[str, float],
    card: ModelCard,
    cfg: ProjectConfig,
) -> str:
    """One complete deck: all gates expanded to transistors, one supply, one source per input.

    TODO(P4). Structure:
        common.netlist_header(f"{c.name} vec {vector} corner {corner_id}", corner["temp"])
        common.model_block(card, n_* params, p_* params)       # ONE draw per device type per corner
        gates.netlist.render_gate_subckt(...) for each gate type used (same sizing as Part B!)
        Vvdd vdd 0 dc {vdd}
        V<pi> <pi> 0 dc {0 or vdd}                             # one source per primary input
        X<inst> <inputs...> <out> vdd 0 <GATE>                 # one line per GateInstance
        common.op_control_block(["i(vvdd)", *input source currents, *primary output voltages])
        .end
      - AND2/OR2 instances use the AND2/OR2 subckts (the same transistor topology as Part B)
      - outputs are unloaded except by the gates they drive (loading IS present here, unlike the
        ML composition, which is the expected source of extra error, brief §5)
      - save the corner-0, all-zeros-vector deck to netlists/examples/circuit_{name}.sp
    """
    raise NotImplementedError
