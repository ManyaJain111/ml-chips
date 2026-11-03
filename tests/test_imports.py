"""Smoke test: every module imports, and the constants contract holds. This should pass from day 1."""

import importlib
import pkgutil

import leakml
from leakml import constants as C


def test_all_modules_import():
    for mod in pkgutil.walk_packages(leakml.__path__, prefix="leakml."):
        importlib.import_module(mod.name)


def test_dataset_column_contract():
    # Deliverable 1 column order. Change it only together with SPEC.md §4.3.
    assert C.DATASET_COLUMNS == (
        "vdd",
        "temp",
        "w",
        "vd",
        "vs",
        "vg",
        "vb",
        "vth0",
        "toxe",
        "ndep",
        "u0",
        "xj",
        "id",
        "ig",
        "is",
        "ib",
    )


def test_eighteen_gate_cases():
    assert sum(len(C.GATE_INPUT_COMBOS[g]) for g in C.GATES) == 18
