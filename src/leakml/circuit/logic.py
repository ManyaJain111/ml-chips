"""Logic simulation of the test circuit.  Owner: P4."""

from __future__ import annotations

from leakml.circuit.definition import Circuit, GateInstance
from leakml.gates.topology import logic_function


def all_input_vectors(c: Circuit) -> list[str]:
    """Every input combination as a bit string (bit i -> c.primary_inputs[i]). 2^n vectors."""
    n = len(c.primary_inputs)
    return [format(k, f"0{n}b") for k in range(2**n)]


def evaluate(c: Circuit, vector: str) -> dict[str, int]:
    """Logic value of EVERY net for one input vector."""
    if len(vector) != len(c.primary_inputs) or set(vector) - {"0", "1"}:
        raise ValueError(f"bad vector {vector!r} for {c.name} ({len(c.primary_inputs)} inputs)")
    nets = {pi: int(b) for pi, b in zip(c.primary_inputs, vector, strict=True)}
    for inst in c.instances:
        bits = "".join(str(nets[n]) for n in inst.inputs)
        nets[inst.output] = logic_function(inst.gate, bits)
    return nets


def gate_input_bits(c: Circuit, net_values: dict[str, int]) -> list[tuple[GateInstance, str]]:
    """For each gate instance, the input bit string it sees (e.g. ("U3", "10"))."""
    return [(i, "".join(str(net_values[n]) for n in i.inputs)) for i in c.instances]


def reference_function(c: Circuit, vector: str) -> dict[str, int]:
    """Golden behavioural model (primary outputs only) used to unit-test `evaluate`."""
    v = {pi: int(b) for pi, b in zip(c.primary_inputs, vector, strict=True)}
    if c.name == "cla3":
        a = v["a0"] + 2 * v["a1"] + 4 * v["a2"]
        b = v["b0"] + 2 * v["b1"] + 4 * v["b2"]
        t = a + b + v["cin"]
        return {"s0": t & 1, "s1": (t >> 1) & 1, "s2": (t >> 2) & 1, "cout": (t >> 3) & 1}
    if c.name == "parity8":
        return {"parity": sum(v.values()) % 2}
    raise ValueError(f"no reference function for circuit {c.name!r}")
