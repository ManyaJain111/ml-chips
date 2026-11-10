"""leakml.sampling (implemented foundation)."""

import numpy as np
import pandas as pd
import pytest

from leakml.constants import (
    CASE_BIAS,
    CASE_DEVICE,
    CASES,
    CORNER_COLUMNS,
    INPUT_COLUMNS,
    PROCESS_PARAMS,
)
from leakml.pdk import format_number
from leakml.sampling import (
    LOW_VDS_CASES,
    LOW_VDS_FRAC,
    LOW_VDS_RANGE,
    SOURCE_PIN_CASES,
    SOURCE_PIN_FRAC,
    SOURCE_PIN_RANGE,
    corner_process_params,
    draw_corners,
    draw_transistor_samples,
    make_rng,
)


@pytest.mark.parametrize("case", CASES)
def test_ranges(case, dev_cfg, card):
    df = draw_transistor_samples(case, 2000, dev_cfg, card, make_rng(0, case.value))
    a, s = dev_cfg.assignment, dev_cfg.sampling
    assert list(df.columns) == ["sample_id", *INPUT_COLUMNS]
    assert df.vdd.between(0.9 * a.vdd_nominal, 1.1 * a.vdd_nominal).all()
    assert (df.vd >= 0).all() and (df.vd <= df.vdd).all()
    assert (df.vs >= 0).all() and (df.vs <= df.vdd).all()
    assert df.w.between(a.lmin, 8 * a.lmin).all()
    assert df.temp.between(*s.temp_range_c).all()
    nom = card.nominal[CASE_DEVICE[case]]
    for p in PROCESS_PARAMS:
        ratio = df[p] / nom[p]  # sign-safe: PMOS vth0 stays negative
        assert ratio.between(0.9, 1.1).all(), p
    for term in ("vg", "vb"):
        expected = df.vdd if CASE_BIAS[case][term] == "vdd" else 0.0
        assert np.array_equal(df[term], np.broadcast_to(expected, len(df)))


def test_reproducible_and_case_independent(dev_cfg, card):
    a = draw_transistor_samples("nmos_off", 50, dev_cfg, card, make_rng(1, "nmos_off"))
    b = draw_transistor_samples("nmos_off", 50, dev_cfg, card, make_rng(1, "nmos_off"))
    c = draw_transistor_samples("nmos_off", 50, dev_cfg, card, make_rng(1, "pmos_off"))
    pd.testing.assert_frame_equal(a, b)
    assert not np.allclose(a.vdd, c.vdd)


def test_continuing_stream_gives_new_ids(dev_cfg, card):
    rng = make_rng(1, "x")
    first = draw_transistor_samples("nmos_on", 10, dev_cfg, card, rng)
    more = draw_transistor_samples("nmos_on", 5, dev_cfg, card, rng, start_id=10)
    assert list(more.sample_id) == list(range(10, 15))
    assert not np.isin(more.vdd, first.vdd).any()


def test_corners(dev_cfg, card):
    c = draw_corners(30, dev_cfg, card, make_rng(dev_cfg.seeds.corners, "corners"))
    assert tuple(c.columns) == CORNER_COLUMNS
    assert list(c.corner_id) == list(range(30))
    assert (c.p_vth0 < 0).all() and (c.n_vth0 > 0).all()
    assert not np.allclose(
        c.n_toxe / card.nominal["nmos"]["toxe"], c.p_toxe / card.nominal["pmos"]["toxe"]
    )
    assert corner_process_params(c.iloc[0], "pmos")["vth0"] == c.p_vth0.iloc[0]


@pytest.mark.parametrize("case", CASES)
def test_low_vds_oversampling(case, dev_cfg, card):
    df = draw_transistor_samples(case, 4000, dev_cfg, card, make_rng(2, case.value))
    vds = df.vd - df.vs

    def exact(col):  # low rows are rounded to exactly what the deck prints; uniform rows are not
        return np.array([float(format_number(v)) == v for v in df[col]])

    low = exact("vd") & exact("vs")
    if case in LOW_VDS_CASES:
        assert low.mean() == pytest.approx(LOW_VDS_FRAC, abs=0.04)
        mag = vds[low].abs()
        assert (mag >= 0.9 * LOW_VDS_RANGE[0]).all() and (mag <= LOW_VDS_RANGE[1] * 1.001).all()
        assert mag.min() < 1e-8 and mag.max() > 1e-3  # spans the decades
        assert (vds[low] > 0).mean() == pytest.approx(0.5, abs=0.06)  # both signs
    else:
        assert low.sum() == 0 and (vds.abs() < 1e-3).mean() < 0.01  # OFF cases stay uniform


def test_low_vds_keeps_uniform_rows(dev_cfg, card):
    """The override is drawn after everything else: non-low ON rows equal a plain uniform draw."""
    on = draw_transistor_samples("nmos_on", 500, dev_cfg, card, make_rng(3, "same"))
    off = draw_transistor_samples("nmos_off", 500, dev_cfg, card, make_rng(3, "same"))
    keep = (on.vd - on.vs).abs() > LOW_VDS_RANGE[1]
    for col in ("vdd", "vd", "vs", "w", "temp", *PROCESS_PARAMS):
        assert np.array_equal(on.loc[keep, col], off.loc[keep, col]), col


@pytest.mark.parametrize("case", CASES)
def test_off_source_pinning(case, dev_cfg, card):
    df = draw_transistor_samples(case, 4000, dev_cfg, card, make_rng(5, case.value))
    nmos = case.value.startswith("nmos")
    offset = df.vs if nmos else df.vdd - df.vs  # distance of the source from its own rail
    exact_vs = np.array([float(format_number(v)) == v for v in df.vs])
    pinned = exact_vs & (offset >= 0.9 * SOURCE_PIN_RANGE[0]) & (offset <= SOURCE_PIN_RANGE[1])
    if case in SOURCE_PIN_CASES:
        assert pinned.mean() == pytest.approx(SOURCE_PIN_FRAC, abs=0.04)
        assert offset[pinned].min() < 1e-8 and offset[pinned].max() > 1e-2  # spans the decades
        assert ((df.vs >= 0) & (df.vs <= df.vdd)).all()
        assert np.allclose(df.vb, 0.0 if nmos else df.vdd)  # pinned source sits at the body
        vds = (df.vd - df.vs).abs()[pinned]
        assert vds.max() > 0.8 * df.vdd.max()  # vd stays uniform: full |vds| is covered
    else:
        assert (offset <= SOURCE_PIN_RANGE[1]).mean() < 0.2  # ON cases: no pinning


def test_source_pinning_keeps_other_rows(dev_cfg, card):
    """Pinning is drawn after everything else: only vs of the pinned rows changes."""
    ref = draw_transistor_samples("nmos_on", 500, dev_cfg, card, make_rng(4, "x"))
    off = draw_transistor_samples("nmos_off", 500, dev_cfg, card, make_rng(4, "x"))
    for col in ("vdd", "vd", "w", "temp", *PROCESS_PARAMS):
        same = (ref.vd - ref.vs).abs() > LOW_VDS_RANGE[1]  # ON rows the low-Vds draw left alone
        assert np.array_equal(ref.loc[same, col], off.loc[same, col]), col
