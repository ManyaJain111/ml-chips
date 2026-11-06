"""Transistor-level SPICE deck for the full test circuit (Deliverable 2).  Owner: P4."""

from __future__ import annotations

from collections.abc import Mapping

from leakml import sampling
from leakml.circuit.definition import Circuit
from leakml.config import ProjectConfig
from leakml.gates.netlist import gate_widths, render_gate_subckt, subckt_pins
from leakml.gates.topology import get_gate
from leakml.pdk import ModelCard
from leakml.spice import common


def input_source(pi: str) -> str:
    """SPICE source name driving primary input `pi`."""
    return f"v{pi}"


def circuit_print_exprs(c: Circuit) -> list[str]:
    """Quantities printed from every circuit deck (parsed by circuit.spice_ref)."""
    return [
        "i(vvdd)",
        *[f"i({input_source(pi)})" for pi in c.primary_inputs],
        *[f"v({po})" for po in c.primary_outputs],
    ]


def render_circuit_netlist(
    c: Circuit,
    vector: str,
    corner: Mapping[str, float],
    card: ModelCard,
    cfg: ProjectConfig,
) -> str:
    """One complete deck: all gates expanded to transistors, one supply, one source per input.

    Gate subckts are those of Part B (same topology, sizing and model draw). Primary inputs sit at
    exactly 0 or VDD; outputs are loaded only by the gates they drive.
    """
    if len(vector) != len(c.primary_inputs):
        raise ValueError(f"vector {vector!r} does not match {len(c.primary_inputs)} inputs")
    vdd = float(corner["vdd"])
    cid = int(corner.get("corner_id", -1))
    nmos_w, pmos_w = gate_widths(cfg)
    used = sorted({i.gate for i in c.instances})
    lines = [
        common.netlist_header(f"{c.name} vec {vector} corner {cid}", float(corner["temp"])),
        common.model_block(
            card,
            sampling.corner_process_params(corner, "nmos"),
            sampling.corner_process_params(corner, "pmos"),
        ),
    ]
    for g in used:
        lines.append(
            render_gate_subckt(
                get_gate(g),
                nmos_w,
                pmos_w,
                cfg.assignment.lmin,
                nmos_model=card.model_names["nmos"],
                pmos_model=card.model_names["pmos"],
            )
        )
    lines.append(f"Vvdd vdd 0 dc {common.fmt(vdd)}")
    for pi, bit in zip(c.primary_inputs, vector, strict=True):
        lines.append(f"{input_source(pi)} {pi} 0 dc {common.fmt(vdd if bit == '1' else 0.0)}")
    for inst in c.instances:
        pins = subckt_pins(get_gate(inst.gate))
        assert len(pins) == len(inst.inputs) + 3
        lines.append(f"X{inst.name} {' '.join(inst.inputs)} {inst.output} vdd 0 {inst.gate}")
    lines.append(common.op_control_block(circuit_print_exprs(c)))
    lines.append(common.END)
    return "\n".join(lines)
