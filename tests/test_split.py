"""Tests for leakml.ml.split (SPEC §4.5)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from leakml.ml.split import load_split_indices, save_split_indices, split_dataset


@pytest.fixture
def df() -> pd.DataFrame:
    return pd.DataFrame({"x": np.arange(10_000, dtype=float)})


@pytest.mark.parametrize(
    ("scheme", "sizes"), [("70_15_15", (7000, 1500, 1500)), ("80_10_10", (8000, 1000, 1000))]
)
def test_sizes(df, scheme, sizes):
    s = split_dataset(df, scheme, seed=42)
    assert tuple(len(s[k]) for k in ("train", "val", "test")) == sizes


def test_disjoint_and_complete(df):
    s = split_dataset(df, "70_15_15", seed=42)
    idx = [set(s[k].index) for k in ("train", "val", "test")]
    assert not (idx[0] & idx[1] or idx[0] & idx[2] or idx[1] & idx[2])
    assert set().union(*idx) == set(df.index)


def test_deterministic_and_seed_dependent(df):
    a = split_dataset(df, "70_15_15", seed=42)
    b = split_dataset(df, "70_15_15", seed=42)
    c = split_dataset(df, "70_15_15", seed=43)
    assert a["val"].index.equals(b["val"].index)
    assert not a["val"].index.equals(c["val"].index)


def test_shuffled(df):
    s = split_dataset(df, "70_15_15", seed=42)
    assert not s["train"].index.equals(pd.RangeIndex(7000))


def test_unknown_scheme(df):
    with pytest.raises(ValueError):
        split_dataset(df, "50_25_25", seed=42)  # type: ignore[arg-type]


def test_save_load_roundtrip(df, tmp_path):
    s = split_dataset(df, "70_15_15", seed=42)
    path = tmp_path / "splits" / "nmos_off.npz"
    save_split_indices(s, path)
    loaded = load_split_indices(path)
    for k in ("train", "val", "test"):
        assert loaded[k] == s[k].index.tolist()
