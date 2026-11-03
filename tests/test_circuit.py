"""Owner: P4."""

import pytest

pytestmark = pytest.mark.skip(reason="scaffold: implement leakml.circuit first")


def test_validate_both_circuits():
    # get_circuit("cla3") and get_circuit("parity8") pass validate_circuit and use only constants.GATES.
    ...


def test_logic_matches_reference_exhaustively():
    # For EVERY input vector, evaluate() primary outputs == reference_function().
    # (128 vectors for cla3, 256 for parity8.)
    ...


def test_gate_counts():
    # parity8 = 28 NAND2. cla3 count matches the number documented in the report.
    ...
