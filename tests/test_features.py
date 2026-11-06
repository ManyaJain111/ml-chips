"""Tests for leakml.ml.features (P2)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from leakml.constants import INPUT_COLUMNS, TARGET_COLUMNS, Case
from leakml.ml.features import TargetTransform, build_features

NOMINAL_N = {"vth0": 0.63, "toxe": 1.6e-9, "ndep": 4.12e18, "u0": 0.042, "xj": 5e-8}
NOMINAL_P = {"vth0": -0.5808, "toxe": 1.62e-9, "ndep": 3.07e18, "u0": 0.016, "xj": 1e-8}


def _inputs(n: int = 50, device: str = "nmos", seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    nom = NOMINAL_N if device == "nmos" else NOMINAL_P
    vdd = rng.uniform(0.81, 0.99, n)
    df = pd.DataFrame(
        {
            "vdd": vdd,
            "temp": rng.uniform(-40, 125, n),
            "w": rng.uniform(64e-9, 320e-9, n),
            "vd": rng.uniform(0, 1, n) * vdd,
            "vs": rng.uniform(0, 1, n) * vdd,
            "vg": np.zeros(n),
            "vb": np.zeros(n) if device == "nmos" else vdd,
            **{p: v * rng.uniform(0.9, 1.1, n) for p, v in nom.items()},
        },
        index=np.arange(100, 100 + n),
    )
    return df.loc[:, list(INPUT_COLUMNS)]


def _currents(n: int, seed: int = 1) -> pd.DataFrame:
    """Mixed-sign currents over ~10 decades, like the real ON-case id/is."""
    rng = np.random.default_rng(seed)
    mag = 10.0 ** rng.uniform(-15, -4, (n, 4))
    sign = rng.choice([-1.0, 1.0], (n, 4))
    return pd.DataFrame(mag * sign, columns=list(TARGET_COLUMNS))


def test_build_features_deterministic_and_finite():
    X = _inputs()
    F1 = build_features(X, Case.NMOS_OFF, NOMINAL_N)
    F2 = build_features(X.copy(), Case.NMOS_OFF, NOMINAL_N)
    pd.testing.assert_frame_equal(F1, F2)
    assert F1.index.equals(X.index)
    assert np.isfinite(F1.to_numpy()).all()
    assert len(F1.columns) == len(set(F1.columns))


def test_build_features_normalises_process_params():
    X = _inputs(device="pmos")
    F = build_features(X, Case.PMOS_OFF, NOMINAL_P)
    for p in ("vth", "toxe", "ndep", "u0", "xj"):
        assert F[p].between(0.89, 1.11).all(), p  # PMOS vth0 < 0, but the ratio is positive
    raw = build_features(X, Case.PMOS_OFF, None)
    assert (raw["vth"] > 0).all()


def test_pmos_features_are_nmos_like():
    """A PMOS bias is the NMOS bias mirrored around the rails, so the features must match."""
    Xp = _inputs(device="pmos")
    Xn = Xp.copy()
    for c in ("vd", "vs", "vg", "vb"):
        Xn[c] = Xp["vdd"] - Xp[c]
    Fp = build_features(Xp, Case.PMOS_ON, None)
    Fn = build_features(Xn, Case.NMOS_ON, None)
    bias = ["vgs", "vds", "vbs", "vgd", "vbd", "abs_vds", "drain_factor"]
    np.testing.assert_allclose(Fp[bias].to_numpy(), Fn[bias].to_numpy(), atol=1e-12)


def test_build_features_missing_column():
    with pytest.raises(KeyError):
        build_features(_inputs().drop(columns="vd"), Case.NMOS_OFF)


def test_asinh_round_trip_mixed_signs_and_zero():
    Y = _currents(500)
    Y.iloc[0, 0] = 0.0
    tt = TargetTransform(kind="asinh").fit(Y)
    Z = tt.transform(Y)
    assert Z.shape == (500, 4) and np.isfinite(Z).all()
    back = tt.inverse_transform(Z)
    np.testing.assert_allclose(back.to_numpy(), Y.to_numpy(), rtol=1e-10, atol=0)
    assert list(back.columns) == list(TARGET_COLUMNS)


def test_asinh_is_log_like_per_target():
    """A shift dz in target space must be ~a relative error dz in amperes (above i0)."""
    Y = pd.DataFrame(
        {
            "id": 10.0 ** np.linspace(-14, -4, 200),
            "ig": 10.0 ** np.linspace(-24, -20, 200),
            "is": -(10.0 ** np.linspace(-14, -4, 200)),
            "ib": -(10.0 ** np.linspace(-15, -11, 200)),
        }
    )
    tt = TargetTransform(kind="asinh").fit(Y)
    assert tt.i0_[1] < 1e-23  # the tiny OFF-state gate current gets its own scale
    big = Y.iloc[20:]  # rows well above the 1 % quantile
    Z = tt.transform(big)
    rel = tt.inverse_transform(Z + 0.01 * np.sign(Z)).to_numpy() / big.to_numpy() - 1
    np.testing.assert_allclose(rel, 0.01, atol=2e-3)


def test_log10_abs_constant_sign_and_mixed_sign_guard():
    Y = _currents(100).abs()
    Y["ib"] *= -1
    tt = TargetTransform(kind="log10_abs").fit(Y)
    back = tt.inverse_transform(tt.transform(Y), X_raw=Y.set_axis(range(7, 107)))
    np.testing.assert_allclose(back.to_numpy(), Y.to_numpy(), rtol=1e-10)
    assert list(back.index) == list(range(7, 107))
    with pytest.raises(ValueError, match="asinh"):
        TargetTransform(kind="log10_abs").fit(_currents(100))


def test_unfitted_transform_raises():
    with pytest.raises(RuntimeError):
        TargetTransform(kind="asinh").transform(_currents(3))


def test_shape_channel_round_trip_and_removes_sign():
    X = _inputs(300, device="nmos")
    Y = _currents(300)
    sgn = np.sign(X["vd"] - X["vs"]).to_numpy()
    Y["id"] = np.abs(Y["id"]) * sgn  # channel-like: id follows sign(vd - vs), is = -id
    Y["is"] = -Y["id"]
    tt = TargetTransform(kind="asinh", shape_channel=True).fit(Y, X)
    Z = tt.transform(Y, X)
    assert (Z[:, 0] > 0).all() and (Z[:, 2] < 0).all()  # one-signed after dividing by tanh
    back = tt.inverse_transform(Z, X)
    np.testing.assert_allclose(back.to_numpy(), Y.to_numpy(), rtol=1e-9)
    assert back.index.equals(X.index)
    with pytest.raises(ValueError, match="X_raw"):
        tt.inverse_transform(Z)


def test_models_pickled_before_shape_channel_still_work():
    """Bundles saved before the field existed unpickle without it: must behave as unshaped."""
    Y = _currents(50)
    tt = TargetTransform(kind="asinh").fit(Y)
    del tt.__dict__["shape_channel"]
    np.testing.assert_allclose(tt.inverse_transform(tt.transform(Y)).to_numpy(), Y.to_numpy())
