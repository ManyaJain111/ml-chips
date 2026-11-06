"""SPICE netlists for gates (Deliverable 2: one netlist per gate).

render_gate_subckt / subckt_pins / gate_widths: IMPLEMENTED (P4 reuses them for the circuit).
render_gate_netlist: Owner P3.
"""

from __future__ import annotations

from collections.abc import Mapping

from leakml import sampling
from leakml.config import ProjectConfig
from leakml.constants import NODE_GND
from leakml.gates.topology import TERMINALS, GateTopology, input_levels
from leakml.pdk import ModelCard
from leakml.spice import common
from leakml.spice.common import fmt

SUBCKT_GND = "gnd"  # topology node "0" becomes this pin inside the subckt


def subckt_pins(gate: GateTopology) -> tuple[str, ...]:
    """Stable pin order of every gate subckt: inputs..., out, vdd, gnd. E.g. NAND2 -> (a, b, out, vdd, gnd).

    Instantiate as:  X<name> <in1> [<in2>] <out> vdd 0 <GATE>
    """
    return (*gate.inputs, gate.output, "vdd", SUBCKT_GND)


def gate_widths(cfg: ProjectConfig) -> tuple[float, float]:
    """(nmos_w, pmos_w) in metres from cfg.gates sizing. The same for every gate and the circuit."""
    lmin = cfg.assignment.lmin
    return cfg.gates.nmos_w_mult * lmin, cfg.gates.pmos_w_mult * lmin


def render_gate_subckt(
    gate: GateTopology,
    nmos_w: float,
    pmos_w: float,
    lmin: float,
    *,
    nmos_model: str = "nmos",
    pmos_model: str = "pmos",
) -> str:
    """`.subckt <NAME> <pins> ... .ends` with one M-line per topology Fet (model names from card.model_names).

    Internal nodes keep their topology names, so inside instance X1 they are probed as v(x1.x),
    v(x1.y), v(x1.n1) (verified on ngspice 47).
    """

    def node(n: str) -> str:
        return SUBCKT_GND if n == NODE_GND else n

    lines = [f".subckt {gate.name} {' '.join(subckt_pins(gate))}"]
    for f in gate.fets:
        model, w = (nmos_model, nmos_w) if f.device == "nmos" else (pmos_model, pmos_w)
        nodes = " ".join(node(f.node(t)) for t in TERMINALS)
        lines.append(f"{f.name} {nodes} {model} W={fmt(w)} L={fmt(lmin)}")
    lines.append(f".ends {gate.name}")
    return "\n".join(lines)


def render_gate_netlist(
    gate: GateTopology,
    inputs: str,
    corner: Mapping[str, float],
    card: ModelCard,
    cfg: ProjectConfig,
) -> str:
    """Complete gate test-bench for one input combination at one corner.

    Structure (SPEC §4.2, §5.2):
        * <GATE> <inputs> corner <id>
        .options ...  /  .temp <corner temp>            (common.netlist_header)
        .model nmos ... / .model pmos ...               (corner n_* / p_* process draw)
        .subckt <GATE> <pins> ... .ends                 (render_gate_subckt, fixed sizing)
        Vvdd vdd 0 dc <vdd>
        Va a 0 dc <0 or vdd>                            (one source per input, exactly 0 or VDD)
        X1 a b out vdd 0 <GATE>                         (pin order = subckt_pins)
        .control op, print i(vvdd) i(va) ... v(out) v(x1.<node>) ... .endc
        .end

    The printed quantities are exactly spice_ref.gate_print_exprs(gate): i(vvdd) < 0 means the
    supply delivers current, so i_vdd_spice = -i(vvdd). The output is left unloaded (no load is
    needed for a DC operating point). Saving the corner-0 example decks is done by step 07.
    """
    vdd = float(corner["vdd"])
    cid = int(corner.get("corner_id", -1))
    levels = input_levels(gate, inputs)  # validates the bit string length
    nets = [NODE_GND if p == SUBCKT_GND else p for p in subckt_pins(gate)]
    probes = [
        "i(vvdd)",
        *[f"i(v{pin})" for pin in gate.inputs],
        f"v({gate.output})",
        *[f"v(x1.{n})" for n in gate.internal_nodes],
    ]
    return "\n".join(
        [
            common.netlist_header(f"{gate.name} {inputs} corner {cid}", float(corner["temp"])),
            common.model_block(
                card,
                sampling.corner_process_params(corner, "nmos"),
                sampling.corner_process_params(corner, "pmos"),
            ),
            render_gate_subckt(
                gate,
                *gate_widths(cfg),
                cfg.assignment.lmin,
                nmos_model=card.model_names["nmos"],
                pmos_model=card.model_names["pmos"],
            ),
            f"Vvdd vdd 0 dc {fmt(vdd)}",
            *[f"V{pin} {pin} 0 dc {fmt(vdd) if lv else '0'}" for pin, lv in levels.items()],
            f"X1 {' '.join(nets)} {gate.name}",
            common.op_control_block(probes),
            common.END,
        ]
    )
