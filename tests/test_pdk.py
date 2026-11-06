"""leakml.pdk (implemented foundation)."""

import re
from pathlib import Path

import pytest

from leakml.constants import PROCESS_PARAMS
from leakml.pdk import load_model_card, parse_spice_number, render_model_card


def test_fixture_nominals(card):
    assert card.nominal["nmos"]["vth0"] == pytest.approx(0.46893)
    assert card.nominal["pmos"]["vth0"] == pytest.approx(-0.49158)  # PMOS vth0 is negative
    assert card.nominal["nmos"]["toxe"] == pytest.approx(1.25e-9)
    assert set(card.nominal["pmos"]) == set(PROCESS_PARAMS)
    assert card.model_names == {"nmos": "nmos", "pmos": "pmos"}


@pytest.mark.parametrize(
    "tok, val",
    [
        ("1.05e-009", 1.05e-9),
        ("5.5e+18", 5.5e18),
        ("45n", 45e-9),
        ("1meg", 1e6),
        ("-0.4", -0.4),
        (".5", 0.5),
    ],
)
def test_parse_spice_number(tok, val):
    assert parse_spice_number(tok) == pytest.approx(val)


def test_render_substitutes_exactly_and_roundtrips(card, tmp_path):
    new = {"vth0": 0.5, "toxe": 1.2e-9, "ndep": 3e18, "u0": 0.05, "xj": 1.5e-8}
    text = render_model_card(card, "nmos", new)
    f = tmp_path / "m.pm"
    f.write_text(text + "\n" + card.pmos_text + "\n")
    again = load_model_card(f)
    assert again.nominal["nmos"] == pytest.approx(new)
    assert again.nominal["pmos"] == pytest.approx(card.nominal["pmos"])
    assert "k1" in text and "dvt0" in text  # other parameters untouched


def test_render_keeps_negative_pmos_vth0(card):
    assert "vth0    = -0.45" in render_model_card(card, "pmos", {"vth0": -0.45})


def test_comments_inside_block_are_ignored(tmp_path):
    f = tmp_path / "c.pm"
    f.write_text(
        ".model nmos nmos level=54\n* vth0 = 9.9 (comment, must be ignored)\n"
        "+vth0 = 0.4 toxe = 1e-9 ndep = 1e18 u0 = 0.03 xj = 1e-8\n"
        ".model pmos pmos level=54\n+vth0 = -0.4 toxe = 1e-9 ndep = 1e18 u0 = 0.01 xj = 1e-8\n"
    )
    assert load_model_card(f).nominal["nmos"]["vth0"] == pytest.approx(0.4)


def test_missing_param_raises(tmp_path):
    f = tmp_path / "bad.pm"
    f.write_text(
        ".model nmos nmos level=54\n+vth0 = 0.4\n.model pmos pmos level=54\n+vth0 = -0.4\n"
    )
    with pytest.raises(ValueError, match="toxe"):
        load_model_card(f)


def test_rename_model(card):
    assert render_model_card(card, "nmos", {}, model_name="nfix").lower().startswith(".model  nfix")


def _flag(text: str, name: str) -> list[str]:
    return re.findall(rf"(?<![\w.]){name}\s*=\s*(\S+)", text, re.IGNORECASE)


def test_rgatemod_forced_off(tmp_path):
    f = tmp_path / "r.pm"
    f.write_text(
        ".model nmos nmos level=54\n+rgatemod= 1 vth0 = 0.4 toxe = 1e-9 ndep = 1e18 u0 = 0.03"
        " xj = 1e-8\n"
        ".model pmos pmos level=54\n+vth0 = -0.4 toxe = 1e-9 ndep = 1e18 u0 = 0.01 xj = 1e-8\n"
    )
    card = load_model_card(f)
    for dev in ("nmos", "pmos"):  # present (set to 0) and absent (appended as 0)
        assert _flag(
            render_model_card(card, dev, {"vth0": 0.3 if dev == "nmos" else -0.3}), "rgatemod"
        ) == ["0"]
    assert _flag(f.read_text(), "rgatemod") == ["1"]


def test_real_card_rgatemod_off():
    pm = Path(__file__).resolve().parents[1] / "pdk" / "32nm_LP.pm"
    if not pm.is_file():
        pytest.skip("assigned PDK not present")
    card = load_model_card(pm)
    for dev in ("nmos", "pmos"):
        assert _flag(card.block(dev), "rgatemod") == ["0"]
