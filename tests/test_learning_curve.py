"""Tests for leakml.data.learning_curve (brief Part A.1)."""

from __future__ import annotations

import numpy as np
import pytest

from leakml.constants import Case
from leakml.data.learning_curve import learning_curve_subsets

TRAIN = np.arange(100, 7100)  # stand-in training indices (not starting at 0, not all rows)


def test_sizes_and_cap():
    s = learning_curve_subsets(TRAIN, [1000, 2000, 5000, 10_000], seed=42, case=Case.NMOS_OFF)
    assert list(s) == [1000, 2000, 5000, 7000]
    assert all(len(v) == n for n, v in s.items())


def test_nested():
    s = learning_curve_subsets(TRAIN, [1000, 2000, 5000, 7000], seed=42, case="nmos_off")
    ns = sorted(s)
    for small, big in zip(ns, ns[1:], strict=False):
        assert np.array_equal(s[big][:small], s[small])


def test_only_training_rows_no_duplicates():
    s = learning_curve_subsets(TRAIN, [7000], seed=42, case="pmos_on")
    assert set(s[7000]) == set(TRAIN)
    assert len(np.unique(s[7000])) == 7000


def test_deterministic_and_case_dependent():
    a = learning_curve_subsets(TRAIN, [1000], seed=42, case="nmos_on")
    b = learning_curve_subsets(TRAIN, [1000], seed=42, case="nmos_on")
    c = learning_curve_subsets(TRAIN, [1000], seed=42, case="pmos_on")
    assert np.array_equal(a[1000], b[1000])
    assert not np.array_equal(a[1000], c[1000])


def test_rejects_nonpositive():
    with pytest.raises(ValueError):
        learning_curve_subsets(TRAIN, [0, 1000], seed=42, case="nmos_on")
