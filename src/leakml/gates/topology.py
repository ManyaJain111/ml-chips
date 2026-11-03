"""Transistor-level gate topologies (SPEC §5.2).  IMPLEMENTED (shared by P3 and P4).

This is the single source of truth for how each gate is wired. The SPICE subckts
(gates/netlist.py), the ML terminal-voltage mapping (gates/ml_model.py) and the circuit expansion
(circuit/netlist.py) are ALL generated from it, which keeps SPICE and ML structurally identical.

    g = get_gate("NAND2"); g.fets, g.internal_nodes          # -> ("x",)
    logic_function("NAND2", "11")                            # -> 0
    terminals_on(g, "vdd")                                   # [(Fet MPA, "s"), (Fet MPA, "b"), ...]
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal

from leakml.constants import GATE_INPUT_COMBOS, GATES, NODE_GND, NODE_VDD

TERMINALS = ("d", "g", "s", "b")


@dataclass(frozen=True)
class Fet:
    name: str  # SPICE instance name, must start with "M"
    device: Literal["nmos", "pmos"]
    d: str  # node names: "vdd", "0", input pins ("a", "b"), "out", internal nodes ("x", "y", "n1")
    g: str
    s: str
    b: str

    def node(self, terminal: str) -> str:
        return getattr(self, terminal)


@dataclass(frozen=True)
class GateTopology:
    name: str  # "INV", "NAND2", ...
    inputs: tuple[str, ...]  # ("a",) or ("a", "b"). Bit i of the input string drives inputs[i]
    output: str  # "out"
    # nodes whose SPICE voltage is recorded, e.g. ("x",) or ("y", "n1")
    internal_nodes: tuple[str, ...]
    fets: tuple[Fet, ...]


def _rename(fets: tuple[Fet, ...], old: str, new: str) -> tuple[Fet, ...]:
    return tuple(replace(f, **{t: new for t in TERMINALS if f.node(t) == old}) for f in fets)


_V, _G = NODE_VDD, NODE_GND

_INV_FETS = (
    Fet("MP", "pmos", "out", "a", _V, _V),
    Fet("MN", "nmos", "out", "a", _G, _G),
)
_NAND2_FETS = (
    Fet("MPA", "pmos", "out", "a", _V, _V),
    Fet("MPB", "pmos", "out", "b", _V, _V),
    Fet("MNA", "nmos", "out", "a", "x", _G),  # A is the NMOS nearest the output
    Fet("MNB", "nmos", "x", "b", _G, _G),
)
_NOR2_FETS = (
    Fet("MPA", "pmos", "y", "a", _V, _V),  # A is the PMOS nearest VDD
    Fet("MPB", "pmos", "out", "b", "y", _V),
    Fet("MNA", "nmos", "out", "a", _G, _G),
    Fet("MNB", "nmos", "out", "b", _G, _G),
)
_OUT_INV = (
    Fet("MP2", "pmos", "out", "n1", _V, _V),
    Fet("MN2", "nmos", "out", "n1", _G, _G),
)

GATE_LIBRARY: dict[str, GateTopology] = {
    "INV": GateTopology("INV", ("a",), "out", (), _INV_FETS),
    "NAND2": GateTopology("NAND2", ("a", "b"), "out", ("x",), _NAND2_FETS),
    "NOR2": GateTopology("NOR2", ("a", "b"), "out", ("y",), _NOR2_FETS),
    "AND2": GateTopology(
        "AND2", ("a", "b"), "out", ("x", "n1"), _rename(_NAND2_FETS, "out", "n1") + _OUT_INV
    ),
    "OR2": GateTopology(
        "OR2", ("a", "b"), "out", ("y", "n1"), _rename(_NOR2_FETS, "out", "n1") + _OUT_INV
    ),
}

_TRUTH = {
    "INV": lambda a: 1 - a,
    "NAND2": lambda a, b: 1 - (a & b),
    "NOR2": lambda a, b: 1 - (a | b),
    "AND2": lambda a, b: a & b,
    "OR2": lambda a, b: a | b,
}


def get_gate(name: str) -> GateTopology:
    try:
        return GATE_LIBRARY[name]
    except KeyError:
        raise KeyError(f"unknown gate {name!r}; valid: {list(GATE_LIBRARY)}") from None


def logic_function(gate: str, bits: str) -> int:
    """Expected logic output for an input bit string, e.g. logic_function("NAND2", "11") == 0."""
    if gate not in _TRUTH:
        raise KeyError(f"unknown gate {gate!r}")
    if bits not in GATE_INPUT_COMBOS[gate]:
        raise ValueError(
            f"{gate}: invalid input bits {bits!r}, expected one of {GATE_INPUT_COMBOS[gate]}"
        )
    return _TRUTH[gate](*(int(c) for c in bits))


def input_levels(gate: GateTopology, bits: str) -> dict[str, int]:
    """{"a": 0, "b": 1} for bits "01"."""
    if len(bits) != len(gate.inputs):
        raise ValueError(f"{gate.name}: expected {len(gate.inputs)} bits, got {bits!r}")
    return {pin: int(c) for pin, c in zip(gate.inputs, bits, strict=True)}


def nets_of(gate: GateTopology) -> set[str]:
    """Every node name used by the gate."""
    return {f.node(t) for f in gate.fets for t in TERMINALS}


def terminals_on(gate: GateTopology, net: str) -> list[tuple[Fet, str]]:
    """Every (fet, terminal) connected to `net`. E.g. terminals_on(g, "vdd") for the leakage sum (SPEC §3.5)."""
    return [(f, t) for f in gate.fets for t in TERMINALS if f.node(t) == net]


def validate_topology(gate: GateTopology) -> None:
    """Static checks, raising ValueError on the first problem."""
    names = [f.name for f in gate.fets]
    if len(set(names)) != len(names):
        raise ValueError(f"{gate.name}: duplicate FET names {names}")
    if any(not n.upper().startswith("M") for n in names):
        raise ValueError(f"{gate.name}: SPICE MOSFET names must start with 'M'")
    for f in gate.fets:
        expected_bulk = _V if f.device == "pmos" else _G
        if f.b != expected_bulk:
            raise ValueError(f"{gate.name}.{f.name}: {f.device} bulk must be {expected_bulk!r}")
    nets = nets_of(gate)
    allowed = {_V, _G, gate.output, *gate.inputs, *gate.internal_nodes}
    if nets - allowed:
        raise ValueError(f"{gate.name}: undeclared nodes {sorted(nets - allowed)}")
    if not any(f.d == gate.output or f.s == gate.output for f in gate.fets):
        raise ValueError(f"{gate.name}: output {gate.output!r} is not driven")
    for node in gate.internal_nodes:
        if len(terminals_on(gate, node)) < 2:
            raise ValueError(f"{gate.name}: internal node {node!r} is floating")
    for pin in gate.inputs:
        if not all(t == "g" for _, t in terminals_on(gate, pin)) or not terminals_on(gate, pin):
            raise ValueError(f"{gate.name}: input {pin!r} must connect only to gates")


for _g in GATES:  # fail at import time if the library and constants ever diverge
    validate_topology(GATE_LIBRARY[_g])
assert set(GATE_LIBRARY) == set(GATES)
