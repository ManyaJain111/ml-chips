"""Gate-level definitions of the test circuits (SPEC §6.1).  Owner: P4.

ONLY gates from constants.GATES may be used (INV, NAND2, NOR2, AND2, OR2), because the ML side can
only compose gates that have a gate model. Anything else (XOR, 3-input gates) must be decomposed.
"""

from __future__ import annotations

from dataclasses import dataclass

from leakml.constants import GATE_INPUT_COMBOS, GATES


@dataclass(frozen=True)
class GateInstance:
    name: str  # unique instance name, e.g. "U12"
    gate: str  # one of constants.GATES
    inputs: tuple[str, ...]  # net names, in the gate's pin order (a, b)
    output: str  # net name


@dataclass(frozen=True)
class Circuit:
    name: str  # "cla3" | "parity8"
    primary_inputs: tuple[str, ...]  # order defines the input-vector bit order
    primary_outputs: tuple[str, ...]
    instances: tuple[GateInstance, ...]  # in topological order


def xor2(prefix: str, a: str, b: str, out: str) -> list[GateInstance]:
    """XOR2 from 4 NAND2: n1=NAND(a,b), n2=NAND(a,n1), n3=NAND(b,n1), out=NAND(n2,n3)."""
    n1, n2, n3 = f"{prefix}_n1", f"{prefix}_n2", f"{prefix}_n3"
    return [
        GateInstance(f"{prefix}_g1", "NAND2", (a, b), n1),
        GateInstance(f"{prefix}_g2", "NAND2", (a, n1), n2),
        GateInstance(f"{prefix}_g3", "NAND2", (b, n1), n3),
        GateInstance(f"{prefix}_g4", "NAND2", (n2, n3), out),
    ]


def build_cla3() -> Circuit:
    """3-bit carry-lookahead adder: inputs a0..a2, b0..b2, cin (7 bits -> 128 vectors).

    g_i = AND2(a_i, b_i), p_i = XOR2(a_i, b_i); carries are flattened sums of products over
    g, p, cin (not rippled) built from 2-input AND2/OR2 trees with shared product terms:
        c1 = g0 + p0*c0
        c2 = g1 + p1*g0 + p1*p0*c0
        c3 = g2 + p2*g1 + p2*p1*g0 + p2*p1*p0*c0        (named `cout`)
    s_i = XOR2(p_i, c_i) with c0 = cin. Outputs: s0, s1, s2, cout.
    Gate count: 42 = 24 NAND2 (6 XOR2) + 12 AND2 + 6 OR2.
    """
    inst: list[GateInstance] = []
    n = [0]

    def gate(kind: str, a: str, b: str, out: str) -> str:
        n[0] += 1
        inst.append(GateInstance(f"u{n[0]}", kind, (a, b), out))
        return out

    for i in range(3):
        gate("AND2", f"a{i}", f"b{i}", f"g{i}")
        inst.extend(xor2(f"xp{i}", f"a{i}", f"b{i}", f"p{i}"))
    # shared products
    gate("AND2", "p0", "cin", "p0c")  # p0*c0
    gate("AND2", "p1", "g0", "p1g0")
    gate("AND2", "p1", "p0", "p1p0")
    gate("AND2", "p1p0", "cin", "p1p0c")
    gate("AND2", "p2", "g1", "p2g1")
    gate("AND2", "p2", "p1", "p2p1")
    gate("AND2", "p2p1", "g0", "p2p1g0")
    gate("AND2", "p2p1", "p0", "p2p1p0")
    gate("AND2", "p2p1p0", "cin", "p2p1p0c")
    # carries
    gate("OR2", "g0", "p0c", "c1")
    gate("OR2", "g1", "p1g0", "c2a")
    gate("OR2", "c2a", "p1p0c", "c2")
    gate("OR2", "g2", "p2g1", "c3a")
    gate("OR2", "c3a", "p2p1g0", "c3b")
    gate("OR2", "c3b", "p2p1p0c", "cout")
    # sums
    inst.extend(xor2("xs0", "p0", "cin", "s0"))
    inst.extend(xor2("xs1", "p1", "c1", "s1"))
    inst.extend(xor2("xs2", "p2", "c2", "s2"))
    return Circuit(
        "cla3",
        ("a0", "a1", "a2", "b0", "b1", "b2", "cin"),
        ("s0", "s1", "s2", "cout"),
        tuple(inst),
    )


def build_parity8() -> Circuit:
    """8-bit parity checker: inputs d0..d7 (8 bits -> 256 vectors).

    Balanced XOR tree, 4 + 2 + 1 = 7 XOR2 = 28 NAND2. Output `parity` = XOR of all bits, i.e.
    1 for an ODD number of ones (odd-parity indicator).
    """
    inst: list[GateInstance] = []
    level = [f"d{i}" for i in range(8)]
    stage = 0
    while len(level) > 1:
        nxt = []
        for j in range(0, len(level), 2):
            out = "parity" if len(level) == 2 else f"x{stage}_{j // 2}"
            inst.extend(xor2(f"xr{stage}_{j // 2}", level[j], level[j + 1], out))
            nxt.append(out)
        level = nxt
        stage += 1
    return Circuit("parity8", tuple(f"d{i}" for i in range(8)), ("parity",), tuple(inst))


def get_circuit(name: str) -> Circuit:
    """Build and validate the named circuit ("cla3" / "parity8")."""
    builders = {"cla3": build_cla3, "parity8": build_parity8}
    if name not in builders:
        raise ValueError(f"unknown circuit {name!r}; expected one of {sorted(builders)}")
    c = builders[name]()
    validate_circuit(c)
    return c


def validate_circuit(c: Circuit) -> None:
    """Structural checks: unique names, single drivers, topological order, legal gates/arity."""
    names = [i.name for i in c.instances]
    if len(set(names)) != len(names):
        raise ValueError(f"{c.name}: duplicate instance names")
    if len(set(c.primary_inputs)) != len(c.primary_inputs):
        raise ValueError(f"{c.name}: duplicate primary inputs")
    driven = set(c.primary_inputs)
    for i in c.instances:
        if i.gate not in GATES:
            raise ValueError(f"{c.name}: {i.name} uses {i.gate!r}, not in {GATES}")
        if len(i.inputs) != len(GATE_INPUT_COMBOS[i.gate][0]):
            raise ValueError(f"{c.name}: {i.name} ({i.gate}) has wrong arity {i.inputs}")
        for net in i.inputs:
            if net not in driven:
                raise ValueError(
                    f"{c.name}: {i.name} input {net!r} is undriven or out of topological order"
                )
        if i.output in driven:
            raise ValueError(f"{c.name}: net {i.output!r} has more than one driver")
        driven.add(i.output)
    for po in c.primary_outputs:
        if po not in driven:
            raise ValueError(f"{c.name}: primary output {po!r} is undriven")
