"""leakml.metrics (implemented foundation)."""

import numpy as np
import pandas as pd
import pytest

from leakml.metrics import METRIC_KEYS, r2_log, relative_error, results_table, summarize


def test_relative_error_sign_and_floor():
    e = relative_error(
        np.array([1.1, 0.9, 1.0, 5.0]), np.array([1.0, 1.0, 0.0, 1e-20]), i_floor=1e-15
    )
    assert e[0] == pytest.approx(0.1) and e[1] == pytest.approx(-0.1)
    assert np.isnan(e[2]) and np.isnan(e[3])


def test_negative_true_uses_abs():
    assert relative_error(np.array([-1.1]), np.array([-1.0]))[0] == pytest.approx(-0.1)


def test_summarize_perfect_and_keys():
    t = np.logspace(-12, -6, 50)
    s = summarize(t, t)
    assert tuple(s) == METRIC_KEYS
    assert s["median_err_pct"] == 0 and s["p95_err_pct"] == 0 and s["mean_acc_pct"] == 100
    assert s["r2_log"] == pytest.approx(1.0)


def test_summarize_values():
    true = np.ones(100)
    pred = true * (1 + np.linspace(-0.02, 0.02, 100))
    s = summarize(pred, true)
    assert s["median_err_pct"] == pytest.approx(1.0, abs=0.05)
    assert s["mean_acc_pct"] == pytest.approx(99.0, abs=0.05)
    assert abs(s["bias_pct"]) < 1e-9


def test_r2_log_ignores_zeros():
    assert r2_log(np.array([1e-9, 0.0, 1e-6]), np.array([1e-9, 1e-7, 1e-6])) == pytest.approx(1.0)


def test_results_table_groups_and_all_row():
    df = pd.DataFrame(
        {
            "gate": ["INV", "INV", "NAND2", "NAND2"],
            "inputs": ["0", "0", "01", "01"],
            "i_spice": [1.0, 2.0, 3.0, 4.0],
            "i_ml": [1.0, 2.2, 3.0, 4.0],
        }  # fmt: skip
    )
    t = results_table(df, ["gate", "inputs"])
    assert list(t.gate) == ["INV", "NAND2", "ALL"]
    assert t.loc[t.gate == "NAND2", "median_err_pct"].item() == 0
    assert t.loc[t.gate == "ALL", "n"].item() == 4
