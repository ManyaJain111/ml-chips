"""Tests for leakml.data.checks (Deliverable 3)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from leakml.constants import Case
from leakml.data.checks import (
    kirchhoff_relative_residual,
    kirchhoff_residual,
    kirchhoff_table,
    plot_gate_current_vs_toxe,
    plot_off_current_vs_temperature,
    plot_off_current_vs_vth0,
)


@pytest.fixture
def mock_datasets() -> dict[Case, pd.DataFrame]:
    """Small mock datasets covering all 4 cases."""
    rng = np.random.default_rng(42)
    n = 20
    out = {}
    for case in Case:
        vdd = np.full(n, 0.9)
        temp = rng.uniform(-40, 125, n)
        w = np.full(n, 64e-9)
        vd = rng.uniform(0, 0.9, n)
        vs = rng.uniform(0, 0.9, n)
        vg = np.full(n, 0.9 if "on" in case.value else 0.0)
        vb = np.full(n, 0.9 if "pmos" in case.value else 0.0)
        vth0 = np.full(n, -0.58 if "pmos" in case.value else 0.63)
        toxe = rng.uniform(1.4e-9, 1.8e-9, n)
        ndep = np.full(n, 3e18)
        u0 = np.full(n, 0.04)
        xj = np.full(n, 1e-8)

        # Synthetic currents satisfying KCL exactly
        id_ = rng.uniform(1e-12, 1e-9, n)
        ig_ = rng.uniform(1e-14, 1e-11, n)
        is_ = -(id_ + ig_) * 0.99
        ib_ = -(id_ + ig_ + is_)

        df = pd.DataFrame(
            {
                "vdd": vdd,
                "temp": temp,
                "w": w,
                "vd": vd,
                "vs": vs,
                "vg": vg,
                "vb": vb,
                "vth0": vth0,
                "toxe": toxe,
                "ndep": ndep,
                "u0": u0,
                "xj": xj,
                "id": id_,
                "ig": ig_,
                "is": is_,
                "ib": ib_,
            }
        )
        out[case] = df
    return out


def test_kirchhoff_residual(mock_datasets):
    for df in mock_datasets.values():
        r = kirchhoff_residual(df)
        assert isinstance(r, pd.Series)
        assert np.allclose(r, 0.0, atol=1e-18)


def test_kirchhoff_relative_residual(mock_datasets):
    for df in mock_datasets.values():
        rel = kirchhoff_relative_residual(df)
        assert isinstance(rel, pd.Series)
        assert np.all(rel >= 0)
        assert np.all(rel < 1e-10)


def test_kirchhoff_table(mock_datasets):
    table = kirchhoff_table(mock_datasets)
    assert len(table) == 4
    assert set(table["case"]) == {c.value for c in Case}
    assert "median_abs_r_a" in table.columns
    assert "p95_rel_r" in table.columns


def test_trend_plots(mock_datasets, tmp_path):
    plot_off_current_vs_temperature(mock_datasets, tmp_path)
    assert (tmp_path / "trend_ioff_vs_temp.png").exists()
    assert (tmp_path / "trend_ioff_vs_temp.pdf").exists()

    plot_off_current_vs_vth0(mock_datasets, tmp_path)
    assert (tmp_path / "trend_ioff_vs_vth0.png").exists()
    assert (tmp_path / "trend_ioff_vs_vth0.pdf").exists()

    plot_gate_current_vs_toxe(mock_datasets, tmp_path)
    assert (tmp_path / "trend_ig_vs_toxe.png").exists()
    assert (tmp_path / "trend_ig_vs_toxe.pdf").exists()

