"""Tests for gates/report.py and scripts/09_gate_report.py (Deliverables 6, 7)."""

from __future__ import annotations

import importlib.util
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from leakml import io
from leakml.config import REPO_ROOT
from leakml.constants import GATE_INPUT_COMBOS, GATES
from leakml.gates.report import (
    N_GATE_CASES,
    TABLE_COLUMNS,
    gate_parity_plot,
    gate_results_table,
    merge_spice_ml,
    stand_in_sources,
    worst_cases,
)

N_CORNERS = 3


def _spice_ml(n_corners: int = N_CORNERS, seed: int = 0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Hand-made gate_spice / gate_ml with every (gate, inputs, corner) once."""
    rng = np.random.default_rng(seed)
    keys = pd.DataFrame(
        [(g, b, c) for g in GATES for b in GATE_INPUT_COMBOS[g] for c in range(n_corners)],
        columns=["gate", "inputs", "corner_id"],
    )
    n = len(keys)
    i_sp = 10.0 ** rng.uniform(-12, -9, n)
    spice = keys.assign(vdd=1.0, temp=27.0, i_vdd_spice=i_sp, i_in_spice=1e-3 * i_sp, logic_ok=True)
    ml = keys.assign(
        i_vdd_ml=i_sp * 1.05,
        i_in_ml=1e-3 * i_sp,
        max_vg_snap_err=0.0,
        i_vdd_ml_direct=i_sp * 1.5,
    )
    return spice, ml


# ------------------------------------------------------------------ merge


def test_merge_ok_and_ordered():
    spice, ml = _spice_ml()
    m = merge_spice_ml(spice.sample(frac=1, random_state=1), ml.sample(frac=1, random_state=2))
    assert len(m) == N_GATE_CASES * N_CORNERS
    assert not m.duplicated(["gate", "inputs", "corner_id"]).any()
    assert m["gate"].iloc[0] == "INV" and m["inputs"].iloc[0] == "0"
    assert list(dict.fromkeys(m["gate"])) == list(GATES)
    assert {"i_vdd_spice", "i_vdd_ml", "i_vdd_ml_direct", "temp", "vdd"} <= set(m.columns)


def test_merge_n_corners_mismatch():
    spice, ml = _spice_ml()
    with pytest.raises(ValueError, match=r"expected 18 x 5"):
        merge_spice_ml(spice, ml, n_corners=5)


def test_merge_duplicate_key():
    spice, ml = _spice_ml()
    ml = pd.concat([ml, ml.iloc[[4]]], ignore_index=True)
    with pytest.raises(ValueError, match=r"gate_ml has 2 rows with duplicated keys.*INV"):
        merge_spice_ml(spice, ml)


@pytest.mark.parametrize("side", ["ml", "spice"])
def test_merge_missing_row(side):
    spice, ml = _spice_ml()
    drop = (ml["gate"] == "NOR2") & (ml["inputs"] == "01") & (ml["corner_id"] == 2)
    if side == "ml":
        ml = ml[~drop]
        missing_from = "gate_ml"
    else:
        spice = spice[~drop]
        missing_from = "gate_spice"
    with pytest.raises(ValueError, match=rf"1 keys missing from {missing_from}.*'NOR2', '01', 2"):
        merge_spice_ml(spice, ml)


def test_merge_int_inputs_rejected():
    spice, ml = _spice_ml()
    ml["inputs"] = ml["inputs"].astype(int)  # what plain pd.read_csv does to "01"
    with pytest.raises(TypeError, match=r"gate_ml.inputs must be str"):
        merge_spice_ml(spice, ml)


def test_merge_through_csv_roundtrip(tmp_path):
    spice, ml = _spice_ml()
    io.write_csv(spice, tmp_path / "s.csv")
    io.write_csv(ml, tmp_path / "m.csv")
    m = merge_spice_ml(io.read_csv(tmp_path / "s.csv"), io.read_csv(tmp_path / "m.csv"))
    assert set(m.loc[m["gate"] == "NAND2", "inputs"]) == {"00", "01", "10", "11"}
    with pytest.raises(TypeError):  # plain pandas turns "01" into 1
        merge_spice_ml(pd.read_csv(tmp_path / "s.csv"), pd.read_csv(tmp_path / "m.csv"))


# ------------------------------------------------------------------ table


def test_table_shape_and_columns():
    spice, ml = _spice_ml()
    t = gate_results_table(merge_spice_ml(spice, ml), i_floor=1e-15)
    assert list(t.columns) == TABLE_COLUMNS
    assert len(t) == N_GATE_CASES + 1
    assert t.iloc[-1][["gate", "inputs"]].tolist() == ["ALL", "ALL"]
    assert t["n"].iloc[:-1].eq(N_CORNERS).all() and t["n"].iloc[-1] == N_GATE_CASES * N_CORNERS


def test_table_values_by_hand():
    spice, ml = _spice_ml()
    # INV "0": errors +8 %, -30 %, +0.5 %; direct +50 % everywhere; i_in exact
    inv0 = (ml["gate"] == "INV") & (ml["inputs"] == "0")
    ml.loc[inv0, "i_vdd_ml"] = spice.loc[inv0, "i_vdd_spice"].to_numpy() * [1.08, 0.70, 1.005]
    # INV "1", corner 0: SPICE below the floor -> excluded
    low = (spice["gate"] == "INV") & (spice["inputs"] == "1") & (spice["corner_id"] == 0)
    spice.loc[low, "i_vdd_spice"] = 1e-16
    t = gate_results_table(merge_spice_ml(spice, ml), i_floor=1e-15).set_index(["gate", "inputs"])

    r = t.loc[("INV", "0")]
    assert r["n"] == 3 and r["n_excluded"] == 0
    assert r["median_err_pct"] == pytest.approx(8.0)
    assert r["max_err_pct"] == pytest.approx(30.0)
    assert r["bias_pct"] == pytest.approx(0.5)
    assert r["mean_acc_pct"] == pytest.approx(100 - (8 + 30 + 0.5) / 3)
    assert r["within_1pct"] == pytest.approx(100 / 3)
    assert r["within_10pct"] == pytest.approx(200 / 3)
    assert r["median_err_direct_pct"] == pytest.approx(50.0)
    assert r["median_err_i_in_pct"] == pytest.approx(0.0)
    assert r["spice_median"] == pytest.approx(np.median(spice.loc[inv0, "i_vdd_spice"]))

    assert t.loc[("INV", "1"), "n_excluded"] == 1
    other = t.loc[("NAND2", "11")]
    assert other["median_err_pct"] == pytest.approx(5.0)
    assert other["within_1pct"] == 0.0 and other["within_10pct"] == 100.0
    assert t.loc[("ALL", "ALL"), "n_excluded"] == 1


def test_table_i_in_below_floor_is_nan():
    spice, ml = _spice_ml()
    spice["i_in_spice"] = 1e-20
    t = gate_results_table(merge_spice_ml(spice, ml), i_floor=1e-15)
    assert t["median_err_i_in_pct"].isna().all()


# ------------------------------------------------------------------ worst cases, plot


def test_worst_cases_sorted_20():
    spice, ml = _spice_ml()
    ml["i_vdd_ml"] = ml["i_vdd_ml"] * np.linspace(0.2, 3.0, len(ml))
    m = merge_spice_ml(spice, ml)
    w = worst_cases(m)
    assert len(w) == 20
    assert w["abs_err_pct"].is_monotonic_decreasing
    cols = ["gate", "inputs", "corner_id", "temp", "vdd", "i_vdd_spice", "i_vdd_ml", "err_pct"]
    assert set(cols) <= set(w.columns)
    e_all = 100 * np.abs(m["i_vdd_ml"] / m["i_vdd_spice"] - 1)
    assert w["abs_err_pct"].iloc[0] == pytest.approx(e_all.max())
    assert np.allclose(w["abs_err_pct"], np.abs(w["err_pct"]))


def test_parity_plot_writes_png_pdf(tmp_path):
    spice, ml = _spice_ml()
    m = merge_spice_ml(spice, ml)
    for col in ("i_vdd_ml", "i_vdd_ml_direct"):
        stats = gate_parity_plot(m, tmp_path / col, col, i_floor=1e-15)
        assert (tmp_path / f"{col}.png").is_file() and (tmp_path / f"{col}.pdf").is_file()
        assert stats["n"] == len(m)


def test_stand_in_sources_placeholder(tmp_path):
    _, ml = _spice_ml()
    assert stand_in_sources(ml, tmp_path) == []
    assert stand_in_sources(ml.assign(source="placeholder"), tmp_path)
    assert stand_in_sources(ml.drop(columns="i_vdd_ml_direct"), tmp_path)


# ------------------------------------------------------------------ dev end-to-end (08 -> 09)

DEV_PILOT_INPUTS = {
    "data": ["corners.csv", "gates/gate_spice.csv"],
    "artifacts": [f"models/{c}.joblib" for c in ("nmos_on", "nmos_off", "pmos_on", "pmos_off")],
}


def _load_script(name: str):
    path = REPO_ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"script_{name}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _tmp_dev_config(tmp_path: Path) -> Path:
    """configs/dev.yaml with the pilot paths moved into tmp_path, seeded with the dev pilot inputs
    of steps 06/07 and the bootstrap stand-in models (skip if `make bootstrap` / step 07 never ran)."""
    raw = yaml.safe_load((REPO_ROOT / "configs" / "dev.yaml").read_text())
    for kind, files in DEV_PILOT_INPUTS.items():
        src_root = REPO_ROOT / raw["paths"][f"pilot_{kind}"]
        dst_root = tmp_path / kind
        for f in files:
            if not (src_root / f).is_file():
                pytest.skip(f"{src_root / f} missing: run make bootstrap + step 07 (dev, --quick)")
            (dst_root / f).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(src_root / f, dst_root / f)
    for kind in ("data", "artifacts", "results"):
        raw["paths"][f"pilot_{kind}"] = str(tmp_path / kind)
    cfg_path = tmp_path / "dev_tmp.yaml"
    cfg_path.write_text(yaml.safe_dump(raw))
    return cfg_path


def test_dev_end_to_end_08_09(tmp_path, capsys):
    cfg_path = _tmp_dev_config(tmp_path)
    args = ["--config", str(cfg_path), "--quick"]
    _load_script("08_gate_ml").main(args)
    _load_script("09_gate_report").main(args)

    out = capsys.readouterr().out
    assert "STAND-IN" in out and "overall mean accuracy" in out
    spice = io.read_csv(tmp_path / "data" / "gates" / "gate_spice.csv")
    res = io.read_csv(tmp_path / "data" / "gates" / "gate_results.csv")
    assert len(res) == len(spice) == N_GATE_CASES * spice["corner_id"].nunique()
    t = io.read_csv(tmp_path / "results" / "tables" / "gate_results.csv")
    assert len(t) == N_GATE_CASES + 1 and list(t.columns) == TABLE_COLUMNS
    assert (tmp_path / "results" / "tables" / "gate_results.md").is_file()
    w = io.read_csv(tmp_path / "results" / "tables" / "gate_worst_cases.csv")
    assert len(w) == 20
    for stem in ("gate_parity", "gate_parity_direct"):
        for ext in ("png", "pdf"):
            assert (tmp_path / "results" / "plots" / f"{stem}.{ext}").is_file()
