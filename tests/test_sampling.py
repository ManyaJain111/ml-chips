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
from leakml.sampling import corner_process_params, draw_corners, draw_transistor_samples, make_rng


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
