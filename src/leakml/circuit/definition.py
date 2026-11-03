"""Gate-level definitions of the test circuits (SPEC §6.1).  Owner: P4.

ONLY gates from constants.GATES may be used (INV, NAND2, NOR2, AND2, OR2), because the ML side can
only compose gates that have a gate model. Anything else (XOR, 3-input gates) must be decomposed.
"""

from __future__ import annotations

from dataclasses import dataclass


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
    """XOR2 from 4 NAND2: n1=NAND(a,b), n2=NAND(a,n1), n3=NAND(b,n1), out=NAND(n2,n3).

    TODO(P4): return the 4 GateInstances with unique internal net names (f"{prefix}_n1", ...).
    Shared by both circuits.
    """
    raise NotImplementedError


def build_cla3() -> Circuit:
    """3-bit carry-lookahead adder: inputs a0..a2, b0..b2, cin (7 bits -> 128 vectors).

    TODO(P4):
      - g_i = AND2(a_i, b_i),  p_i = XOR2(a_i, b_i)                (generate / propagate)
      - lookahead carries, computed DIRECTLY from g, p and cin (not rippled):
            c1 = g0 + p0*c0
            c2 = g1 + p1*g0 + p1*p0*c0
            c3 = g2 + p2*g1 + p2*p1*g0 + p2*p1*p0*c0
        decomposed into 2-input AND2/OR2 trees (or NAND-NAND, which is cheaper. Pick one and
        document it in the report with a gate count)
      - s_i = XOR2(p_i, c_i), cout = c3
      - outputs: s0, s1, s2, cout
    """
    raise NotImplementedError


def build_parity8() -> Circuit:
    """8-bit parity checker: inputs d0..d7 (8 bits -> 256 vectors).

    TODO(P4):
      - balanced XOR tree: 4 + 2 + 1 = 7 XOR2 = 28 NAND2
      - output `parity` = XOR of all bits (1 if there is an odd number of ones). State the
        convention in the report. If the team wants even-parity generation, add an INV
    """
    raise NotImplementedError


def get_circuit(name: str) -> Circuit:
    """TODO(P4): dispatch on name ("cla3" / "parity8") and run validate_circuit on the result."""
    raise NotImplementedError


def validate_circuit(c: Circuit) -> None:
    """Structural checks.

    TODO(P4): instance names unique, every net has exactly one driver (a primary input or one
    gate output), no undriven gate inputs, instances in topological order, only constants.GATES
    used, arity matches the gate, every primary output is driven.
    """
    raise NotImplementedError
