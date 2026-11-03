"""Guards the Part A -> Part B/C interface (SPEC §4.6). Uses tiny stand-in models on synthetic
currents, so it runs without ngspice or any trained artifacts."""

import joblib
import numpy as np
import pandas as pd
import pytest

from leakml.constants import CASES, INPUT_COLUMNS, TARGET_COLUMNS, Case
from leakml.ml.baseline import AsinhExtraTrees
from leakml.ml.predictor import (
    TransistorModel,
    TransistorModelBank,
    input_ranges,
    save_bundle,
    snapped_gate_voltage,
)
from leakml.sampling import draw_transistor_samples, make_rng


def _fake_currents(X: pd.DataFrame) -> pd.DataFrame:
    idr = 1e-9 * (X.vd - X.vs) * np.exp(X.temp / 50)
    ig = 1e-12 * (X.vg - 0.5 * (X.vd + X.vs))
    ib = -1e-14 * (1 + X.vd)
    return pd.DataFrame({"id": idr, "ig": ig, "is": -(idr + ig + ib), "ib": ib})


@pytest.fixture(scope="module")
def bank_dir(tmp_path_factory, dev_cfg, card):
    d = tmp_path_factory.mktemp("models")
    for case in CASES:
        X = draw_transistor_samples(case, 300, dev_cfg, card, make_rng(3, case.value))
        reg = AsinhExtraTrees(n_estimators=10, seed=0).fit(X, _fake_currents(X))
        save_bundle(d / f"{case.value}.joblib", case, reg,
                    {"model_name": "test", "train_ranges": input_ranges(X)})  # fmt: skip
    return d


def test_predict_output_contract(bank_dir, dev_cfg, card):
    bank = TransistorModelBank.load_all(bank_dir)
    X = pd.concat(
        [draw_transistor_samples(c, 5, dev_cfg, card, make_rng(9, c.value)) for c in CASES],
        ignore_index=True,
    )
    X.index = X.index + 100  # non-default index must be preserved
    cases = [c for c in CASES for _ in range(5)]
    out = bank.predict(X[list(INPUT_COLUMNS)], cases)
    assert list(out.columns) == list(TARGET_COLUMNS)
    assert out.index.equals(X.index)
    assert np.isfinite(out.to_numpy()).all()


def test_select_case():
    sel = TransistorModelBank.select_case
    assert sel("nmos", 1.0, 1.0) == Case.NMOS_ON
    assert sel("nmos", 0.0, 1.0) == Case.NMOS_OFF
    assert sel("pmos", 0.0, 1.0) == Case.PMOS_ON
    assert sel("pmos", 1.0, 1.0) == Case.PMOS_OFF
    assert sel("nmos", 0.97, 1.0) == Case.NMOS_ON  # near-rail recorded voltage snaps correctly
    assert sel("pmos", 0.03, 1.0) == Case.PMOS_ON
    assert snapped_gate_voltage(Case.PMOS_OFF, 0.95) == 0.95
    assert snapped_gate_voltage(Case.NMOS_OFF, 0.95) == 0.0


def test_saved_model_roundtrip(bank_dir, dev_cfg, card):
    X = draw_transistor_samples("nmos_off", 7, dev_cfg, card, make_rng(4, "r"))
    m1 = TransistorModel.load(bank_dir / "nmos_off.joblib")
    m2 = TransistorModel.load(bank_dir / "nmos_off.joblib")
    pd.testing.assert_frame_equal(m1.predict(X), m2.predict(X))
    assert m1.case == Case.NMOS_OFF


def test_missing_and_invalid_files(tmp_path):
    with pytest.raises(FileNotFoundError, match="dev_train_baseline"):
        TransistorModelBank.load_all(tmp_path)
    joblib.dump({"not": "a bundle"}, tmp_path / "x.joblib")
    with pytest.raises(ValueError, match="bundle"):
        TransistorModel.load(tmp_path / "x.joblib")


def test_extrapolation_warns(bank_dir, dev_cfg, card):
    X = draw_transistor_samples("nmos_on", 3, dev_cfg, card, make_rng(4, "w"))
    X["temp"] = 200.0
    with pytest.warns(UserWarning, match="temp"):
        TransistorModel.load(bank_dir / "nmos_on.joblib").predict(X)
