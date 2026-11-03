"""Logic simulation of the test circuit.  Owner: P4."""

from __future__ import annotations

from leakml.circuit.definition import Circuit, GateInstance


def all_input_vectors(c: Circuit) -> list[str]:
    """Every input combination as a bit string (bit i -> c.primary_inputs[i]). 2^n vectors.

    TODO(P4)
    """
    raise NotImplementedError


def evaluate(c: Circuit, vector: str) -> dict[str, int]:
    """Logic value of EVERY net for one input vector.

    TODO(P4): walk c.instances in order and use gates.topology.logic_function.
    """
    raise NotImplementedError


def gate_input_bits(c: Circuit, net_values: dict[str, int]) -> list[tuple[GateInstance, str]]:
    """For each gate instance, the input bit string it sees (e.g. ("U3", "10")).

    TODO(P4): this is what the hierarchical ML model needs: (gate type, input combo) per instance.
    """
    raise NotImplementedError


def reference_function(c: Circuit, vector: str) -> dict[str, int]:
    """Golden behavioural model used to unit-test `evaluate`.

    TODO(P4): cla3 -> integer add (a + b + cin) split into s0..s2 and cout. parity8 -> sum(bits) % 2.
    """
    raise NotImplementedError
