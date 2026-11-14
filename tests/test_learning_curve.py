"""Tests for leakml.data.learning_curve (brief Part A.1)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from leakml.constants import INPUT_COLUMNS, TARGET_COLUMNS, Case
from leakml.data.learning_curve import fit_fixed_on_subsets, learning_curve_subsets
from leakml.ml import train, zoo
from leakml.ml.split import split_dataset

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


# ---- fit_fixed_on_subsets ---------------------------------------------------------------------

NOMINAL = {"vth0": 0.63, "toxe": 1.6e-9, "ndep": 4.12e18, "u0": 0.042, "xj": 5e-8}


def _dataset(n: int = 300, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    vdd = rng.uniform(0.81, 0.99, n)
    df = pd.DataFrame(
        {
            "vdd": vdd,
            "temp": rng.uniform(-40, 125, n),
            "w": rng.uniform(64e-9, 320e-9, n),
            "vd": rng.uniform(0, 1, n) * vdd,
            "vs": rng.uniform(0, 1, n) * vdd,
            "vg": np.zeros(n),
            "vb": np.zeros(n),
            **{p: v * rng.uniform(0.9, 1.1, n) for p, v in NOMINAL.items()},
        }
    )
    vt = 8.617e-5 * (df.temp + 273.15)
    df["id"] = df.w / 1e-7 * 1e-6 * np.exp(-df.vth0 / (1.4 * vt)) * np.tanh(df.vd / vt) + 1e-14
    df["ig"] = -1e-21 * (1.6e-9 / df.toxe) ** 8
    df["ib"] = -1e-14 * (1 + df.vd)
    df["is"] = -(df["id"] + df["ig"] + df["ib"])
    return df.loc[:, list(INPUT_COLUMNS) + list(TARGET_COLUMNS)]


FIXED = dict(
    model_name="random_forest", target_kind="asinh", shape_channel=False, is_kcl=False,
    nominal=NOMINAL, split_seed=42, train_seed=7, i_floor=1e-15,
)  # fmt: skip


@pytest.fixture(scope="module")
def lc_splits():
    return split_dataset(_dataset(), "70_15_15", seed=42)


def test_fixed_curve_shape_and_test_rows_untouched(lc_splits):
    df = pd.concat(lc_splits.values())
    df.loc[lc_splits["test"].index, list(TARGET_COLUMNS)] = np.nan  # would poison any fit/score
    hp = zoo.default_hyperparams("random_forest") | {"n_estimators": 20}
    t = fit_fixed_on_subsets(
        "nmos_off", df, lc_splits["train"].index, lc_splits["val"].index, [50, 100, 10_000],
        hyperparams=hp, **FIXED,
    )  # fmt: skip
    n_tr = len(lc_splits["train"])
    assert sorted(t["n_train"].unique()) == [50, 100, n_tr]
    assert len(t) == 3 * len(TARGET_COLUMNS)
    assert (t["n"] == len(lc_splits["val"])).all()
    assert np.isfinite(t.loc[t.target == "id", "p95_err_pct"]).all()


def test_full_size_matches_fit_one(lc_splits):
    """At n = whole training split the curve reproduces step 04's validation score."""
    hp = zoo.default_hyperparams("random_forest")
    df = pd.concat(lc_splits.values())
    t = fit_fixed_on_subsets(
        "nmos_off", df, lc_splits["train"].index, lc_splits["val"].index, [10_000],
        hyperparams=hp, **FIXED,
    )  # fmt: skip
    r = train.fit_one(Case.NMOS_OFF, "random_forest", lc_splits, seed=7, nominal=NOMINAL,
                      i_floor=1e-15, tune=False, shape_channel=False, is_kcl=False)  # fmt: skip
    np.testing.assert_allclose(
        t["p95_err_pct"].to_numpy(dtype=float), r.val_metrics["p95_err_pct"].to_numpy(dtype=float)
    )
