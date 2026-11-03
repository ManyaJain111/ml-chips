"""Project-wide constants and data contracts (SPEC.md §3, §4).

These are the contract, not implementation, so they are filled in. Change them only together with
SPEC.md, and tell the other owners.
"""

from __future__ import annotations

from enum import StrEnum


class Case(StrEnum):
    """The four single-transistor bias cases of Part A."""

    NMOS_ON = "nmos_on"
    NMOS_OFF = "nmos_off"
    PMOS_ON = "pmos_on"
    PMOS_OFF = "pmos_off"


CASES: tuple[Case, ...] = tuple(Case)

# Device type for each case.
CASE_DEVICE: dict[Case, str] = {
    Case.NMOS_ON: "nmos",
    Case.NMOS_OFF: "nmos",
    Case.PMOS_ON: "pmos",
    Case.PMOS_OFF: "pmos",
}

# Gate / body bias for each case as a symbolic level, "vdd" or "gnd" (brief §4, table on p.3).
# Resolve "vdd" to the *sample's* drawn VDD, not the nominal one.
CASE_BIAS: dict[Case, dict[str, str]] = {
    Case.NMOS_ON: {"vg": "vdd", "vb": "gnd"},
    Case.NMOS_OFF: {"vg": "gnd", "vb": "gnd"},
    Case.PMOS_ON: {"vg": "gnd", "vb": "vdd"},
    Case.PMOS_OFF: {"vg": "vdd", "vb": "vdd"},
}

# BSIM4 parameters varied by +/-10 % (separate draws for NMOS and PMOS).
PROCESS_PARAMS: tuple[str, ...] = ("vth0", "toxe", "ndep", "u0", "xj")

# Deliverable 1: the CSV columns, in this exact order.
INPUT_COLUMNS: tuple[str, ...] = ("vdd", "temp", "w", "vd", "vs", "vg", "vb", *PROCESS_PARAMS)
TARGET_COLUMNS: tuple[str, ...] = ("id", "ig", "is", "ib")
DATASET_COLUMNS: tuple[str, ...] = INPUT_COLUMNS + TARGET_COLUMNS

# Mandatory in EVERY netlist (transistor, gate, test circuit), brief §4 "Netlist requirements".
SPICE_OPTIONS = ".options gmin=1e-18 abstol=1e-18 reltol=1e-6 vntol=1e-9"

# Part B gate library, with input combinations as bit strings (first char = input A).
GATES: tuple[str, ...] = ("INV", "NAND2", "NOR2", "AND2", "OR2")
GATE_INPUT_COMBOS: dict[str, tuple[str, ...]] = {
    "INV": ("0", "1"),
    "NAND2": ("00", "01", "10", "11"),
    "NOR2": ("00", "01", "10", "11"),
    "AND2": ("00", "01", "10", "11"),
    "OR2": ("00", "01", "10", "11"),
}  # 2 + 4*4 = 18 gate cases

# Corner table columns (SPEC §5.1). n_* = NMOS draw, p_* = PMOS draw.
CORNER_COLUMNS: tuple[str, ...] = (
    "corner_id",
    "vdd",
    "temp",
    *(f"n_{p}" for p in PROCESS_PARAMS),
    *(f"p_{p}" for p in PROCESS_PARAMS),
)

TEST_CIRCUITS: tuple[str, ...] = ("cla3", "parity8")

# Canonical SPICE node names (SPEC §3.3)
NODE_VDD = "vdd"
NODE_GND = "0"
