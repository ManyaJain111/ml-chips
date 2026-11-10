"""Tests for leakml.ml.train and leakml.ml.regressor (P2)."""

from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
import pytest

from leakml.constants import CASES, INPUT_COLUMNS, TARGET_COLUMNS, Case
from leakml.ml import train
from leakml.ml.predictor import TransistorModelBank
from leakml.ml.regressor import TransistorRegressor
from leakml.ml.split import split_dataset

NOMINAL = {"vth0": 0.63, "toxe": 1.6e-9, "ndep": 4.12e18, "u0": 0.042, "xj": 5e-8}


def _dataset(n: int = 400, seed: int = 0) -> pd.DataFrame:
    """Synthetic single-FET data with physics-like currents (exponential in T and vth0)."""
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
    i_sub = df.w / 1e-7 * 1e-12 * np.exp(-df.vth0 / (1.4 * vt)) * 1e6
    df["id"] = i_sub * np.tanh((df.vd - df.vs) / vt) + 1e-14
    df["ig"] = -1e-21 * (1.6e-9 / df.toxe) ** 8
    df["ib"] = -1e-14 * (1 + df.vd)
    df["is"] = -(df["id"] + df["ig"] + df["ib"])
    return df.loc[:, list(INPUT_COLUMNS) + list(TARGET_COLUMNS)]


@pytest.fixture(scope="module")
def splits():
    return split_dataset(_dataset(), "70_15_15", seed=42)


FAST = dict(nominal=NOMINAL, i_floor=1e-15, tune=False)


def test_fit_one_scores_on_val(splits):
    r = train.fit_one(Case.NMOS_OFF, "random_forest", splits, seed=7, **FAST)
    assert list(r.val_metrics["target"]) == list(TARGET_COLUMNS)
    assert r.val_metrics.loc[r.val_metrics.target == "id", "n"].item() == len(splits["val"])
    assert r.n_train == len(splits["train"]) and set(r.train_ranges) == set(INPUT_COLUMNS)
    assert np.isfinite(r.score())
    assert r.hyperparams["n_estimators"] == 300  # defaults recorded


def test_fit_one_never_reads_test(splits):
    """Hard rule 2: corrupting the test split must not change anything fit_one does."""
    bad = dict(splits)
    bad["test"] = splits["test"].assign(id=np.nan)
    a = train.fit_one(Case.NMOS_OFF, "poly_ridge", splits, seed=7, **FAST)
    b = train.fit_one(Case.NMOS_OFF, "poly_ridge", bad, seed=7, **FAST)
    pd.testing.assert_frame_equal(a.val_metrics, b.val_metrics)


def test_tuning_picks_best_grid_point(splits):
    r = train.fit_one(Case.NMOS_OFF, "poly_ridge", splits, seed=7, nominal=NOMINAL, tune=True)
    assert train.n_grid_points(r.search) == 6  # 2 degrees x 3 alphas
    assert r.score() == pytest.approx(r.search["p95_err_pct"].min())


def test_compare_select_and_tables(splits):
    res = train.compare_models(
        Case.NMOS_OFF, splits, ("poly_ridge", "random_forest"), seed=7, **FAST
    )
    best = train.select_best(res)
    assert best.score() == min(r.score() for r in res)
    long = train.comparison_table({Case.NMOS_OFF: res})
    assert len(long) == 2 * 4
    assert list(long.columns[:6]) == [
        "case", "model", "target", "r2_log", "median_err_pct", "p95_err_pct"
    ]  # fmt: skip
    summ = train.comparison_summary(long)
    assert len(summ) == 2 and {"case", "model", "p95_err_pct"} <= set(summ.columns)
    with pytest.raises(ValueError):
        train.select_best([])


def test_select_best_tie_breaks_on_median():
    def fake(name, p95, med, secs):
        m = pd.DataFrame({"target": ["id"], "p95_err_pct": [p95], "median_err_pct": [med]})
        return train.FitResult(Case.NMOS_ON, name, None, None, [], {}, m, secs)

    pick = train.select_best([fake("a", 5, 2, 1), fake("b", 5, 1, 9), fake("c", 6, 0, 0)])
    assert pick.model_name == "b"


def test_evaluate_on_test_and_plot(splits, tmp_path):
    r = train.fit_one(Case.NMOS_OFF, "random_forest", splits, seed=7, **FAST)
    m = train.evaluate_on_test(r, splits["test"], plot_path=tmp_path / "pv")
    assert list(m["target"]) == list(TARGET_COLUMNS)
    assert (m["case"] == "nmos_off").all() and (m["n"] == len(splits["test"])).all()
    assert (tmp_path / "pv.png").is_file() and (tmp_path / "pv.pdf").is_file()


@pytest.mark.filterwarnings("ignore:.*extrapolation:UserWarning")  # tiny synthetic split
def test_saved_models_load_through_the_bank(splits, tmp_path):
    """The P2 -> P3/P4 contract (SPEC §4.6): save_model -> TransistorModelBank.load_all."""
    r = train.fit_one(Case.NMOS_OFF, "random_forest", splits, seed=7, **FAST)
    for case in CASES:  # the same fitted model stands in for all four files
        r.case = case
        train.save_model(r, tmp_path / f"{case.value}.joblib", extra_meta={"note": "test"})
    bundle = joblib.load(tmp_path / "nmos_off.joblib")
    assert isinstance(bundle["regressor"], TransistorRegressor)
    assert bundle["meta"]["note"] == "test" and bundle["meta"]["model_name"] == "random_forest"

    bank = TransistorModelBank.load_all(tmp_path)
    X = splits["test"].loc[:, list(INPUT_COLUMNS)]
    out = bank.predict(X, [Case.NMOS_OFF] * len(X))
    assert list(out.columns) == list(TARGET_COLUMNS) and out.index.equals(X.index)
    r.case = Case.NMOS_OFF
    np.testing.assert_allclose(out.to_numpy(), r.regressor().predict(X), rtol=1e-12)


def test_regressor_rejects_stale_feature_list(splits):
    r = train.fit_one(Case.NMOS_OFF, "poly_ridge", splits, seed=7, **FAST)
    reg = TransistorRegressor(Case.NMOS_OFF, r.estimator, r.target_transform, ["old"], NOMINAL)
    with pytest.raises(ValueError, match="Retrain"):
        reg.predict(splits["val"])


def test_shape_channel_default_is_on_cases_only(splits):
    on = train.fit_one(Case.NMOS_ON, "poly_ridge", splits, seed=7, **FAST)
    off = train.fit_one(Case.NMOS_OFF, "poly_ridge", splits, seed=7, **FAST)
    forced = train.fit_one(Case.NMOS_OFF, "poly_ridge", splits, seed=7, shape_channel=True, **FAST)
    assert on.target_transform.shape_channel and not off.target_transform.shape_channel
    assert forced.target_transform.shape_channel
    X = splits["val"]
    assert np.isfinite(on.regressor().predict(X)).all()


def test_is_from_kcl_enforces_kirchhoff():
    from leakml.ml.regressor import is_from_kcl

    Y = np.array([[1e-12, 2e-15, 5e-13, -3e-14], [-4e-12, 1e-16, 0.0, 2e-14]])
    out = is_from_kcl(Y)
    scale = np.abs(Y).max(axis=1)
    assert (np.abs(out.sum(axis=1)) <= 1e-12 * scale).all()  # zero up to rounding
    np.testing.assert_array_equal(out[:, [0, 1, 3]], Y[:, [0, 1, 3]])  # only is changes
    assert Y[0, 2] == 5e-13  # input untouched


def test_is_kcl_chosen_on_val_and_saved(splits, tmp_path):
    auto = train.fit_one(Case.NMOS_OFF, "poly_ridge", splits, seed=7, **FAST)
    assert set(auto.search["is_kcl"]) == {False, True}  # both variants scored
    assert train.n_grid_points(auto.search) == 1  # untuned: still one grid point
    best_row = auto.search.loc[auto.search["p95_err_pct"].idxmin()]
    assert auto.is_kcl == bool(best_row["is_kcl"])
    forced = train.fit_one(Case.NMOS_OFF, "poly_ridge", splits, seed=7, is_kcl=True, **FAST)
    assert forced.is_kcl and set(forced.search["is_kcl"]) == {True}
    P = forced.regressor().predict(splits["val"])
    assert (np.abs(P.sum(axis=1)) <= 1e-12 * np.abs(P).max(axis=1)).all()  # KCL holds
    train.save_model(forced, tmp_path / "m.joblib")
    assert joblib.load(tmp_path / "m.joblib")["meta"]["is_kcl"] is True


def test_n_grid_points_counts_hyperparams_not_variants(splits):
    r = train.fit_one(Case.NMOS_OFF, "poly_ridge", splits, seed=7, nominal=NOMINAL, tune=True)
    assert train.n_grid_points(r.search) == 6 and len(r.search) == 12
