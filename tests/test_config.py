"""leakml.config (implemented foundation)."""

from pathlib import Path

import pytest
import yaml

from leakml.config import DEV_CONFIG, REPO_ROOT, ConfigError, load_config


def _write_variant(tmp_path: Path, **assignment) -> Path:
    raw = yaml.safe_load(DEV_CONFIG.read_text())
    raw["assignment"].update(assignment)
    p = tmp_path / "cfg.yaml"
    p.write_text(yaml.safe_dump(raw))
    return p


def test_dev_config_loads():
    cfg = load_config(DEV_CONFIG)
    assert cfg.assignment.lmin == pytest.approx(45e-9)
    assert cfg.assignment.model_file.is_file()
    assert "_dev" in str(cfg.paths.data)
    assert not cfg.paths.is_deliverable_location()


def test_quick_mode_redirects_everything():
    full, quick = load_config(DEV_CONFIG), load_config(DEV_CONFIG, quick=True)
    assert quick.sampling.n_samples_per_case < full.sampling.n_samples_per_case
    assert quick.gates.n_corners < full.gates.n_corners
    for p in (
        quick.paths.data,
        quick.paths.artifacts,
        quick.paths.results,
        quick.paths.netlist_examples,
    ):
        assert "_pilot" in str(p)


def test_todo_assignment_rejected(tmp_path):
    with pytest.raises(ConfigError, match="TODO"):
        load_config(_write_variant(tmp_path, node="TODO"))


@pytest.mark.parametrize(
    "override, msg",
    [
        ({"lmin": 22}, "METRES"),
        ({"vdd_nominal": 1.2}, "vdd_nominal"),
        ({"test_circuit": "adder"}, "test_circuit"),
        ({"model_file": "pdk/does_not_exist.pm"}, "does not exist"),
    ],
)
def test_invalid_values_rejected(tmp_path, override, msg):
    with pytest.raises(ConfigError, match=msg):
        load_config(_write_variant(tmp_path, **override))


def test_relative_paths_resolve_against_repo_root(tmp_path):
    cfg = load_config(_write_variant(tmp_path))  # config lives in tmp, paths still repo-relative
    assert cfg.assignment.model_file == REPO_ROOT / "tests/fixtures/fixture_45nm.pm"


def test_path_helpers(dev_cfg):
    p = dev_cfg.paths
    assert p.transistor_csv("nmos_off").name == "nmos_off.csv"
    assert p.model_file("pmos_on").name == "pmos_on.joblib"
    assert p.corners_csv.parent == p.data
    assert p.gate_ml_csv.parent == p.gates_dir
