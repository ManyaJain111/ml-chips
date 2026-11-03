"""Shared fixtures: the dev config (synthetic model card) so tests never need the assigned PDK."""

from __future__ import annotations

from pathlib import Path

import pytest

from leakml.config import DEV_CONFIG, load_config
from leakml.pdk import load_model_card

FIXTURE_CARD = Path(__file__).parent / "fixtures" / "fixture_45nm.pm"


@pytest.fixture(scope="session")
def dev_cfg():
    return load_config(DEV_CONFIG, quick=True)


@pytest.fixture(scope="session")
def card():
    return load_model_card(FIXTURE_CARD)
