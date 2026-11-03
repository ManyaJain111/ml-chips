"""Typed access to configs/*.yaml.  IMPLEMENTED (shared foundation, maintained by P4).

Every script starts with `cfg = load_config(args.config, quick=args.quick)`.
No module may hard-code a path, seed, VDD, Lmin or sample count. Read them from `cfg`.

    cfg.assignment.vdd_nominal, cfg.assignment.lmin, cfg.seeds.corners, ...
    cfg.paths.transistor_csv("nmos_off"), cfg.paths.model_file(case), cfg.paths.corners_csv, ...

`quick=True` (scripts' --quick flag) shrinks sample/corner counts and redirects data/, results/
and artifacts/ to the pilot_* paths, so pilot runs can never overwrite deliverables.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from leakml.constants import PROCESS_PARAMS, TEST_CIRCUITS, Case

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "project.yaml"
DEV_CONFIG = REPO_ROOT / "configs" / "dev.yaml"

SPLIT_SCHEMES = ("70_15_15", "80_10_10")
NODE_VOLTAGE_SOURCES = ("spice", "ml_solve")


class ConfigError(ValueError):
    """The YAML config is incomplete or inconsistent."""


@dataclass(frozen=True)
class Assignment:
    model_file: Path
    node: str
    lmin: float  # metres
    vdd_nominal: float  # volts
    test_circuit: str  # "cla3" | "parity8"


@dataclass(frozen=True)
class SimulatorCfg:
    binary: str
    n_workers: int
    timeout_s: float
    work_dir: Path
    keep_failed: bool


@dataclass(frozen=True)
class Seeds:
    transistor_data: int
    split: int
    training: int
    corners: int


@dataclass(frozen=True)
class SamplingCfg:
    n_samples_per_case: int
    vdd_tol: float
    temp_range_c: tuple[float, float]
    w_mult_range: tuple[float, float]
    process_tol: float
    process_params: tuple[str, ...]


@dataclass(frozen=True)
class MLCfg:
    model_types: tuple[str, ...]
    selection_metric: str
    i_floor_a: float


@dataclass(frozen=True)
class GatesCfg:
    n_corners: int
    nmos_w_mult: float
    pmos_w_mult: float
    node_voltage_source: str  # "spice" | "ml_solve"


@dataclass(frozen=True)
class Paths:
    """Every file location in SPEC §8. Use these helpers, never string-build paths."""

    data: Path
    artifacts: Path
    results: Path
    netlist_examples: Path

    # ---- data/
    @property
    def transistor_dir(self) -> Path:
        return self.data / "transistor"

    def transistor_csv(self, case: Case | str) -> Path:
        return self.transistor_dir / f"{Case(case).value}.csv"

    @property
    def corners_csv(self) -> Path:
        return self.data / "corners.csv"

    @property
    def gates_dir(self) -> Path:
        return self.data / "gates"

    @property
    def gate_spice_csv(self) -> Path:
        return self.gates_dir / "gate_spice.csv"

    @property
    def gate_ml_csv(self) -> Path:
        return self.gates_dir / "gate_ml.csv"

    @property
    def gate_results_csv(self) -> Path:
        return self.gates_dir / "gate_results.csv"

    @property
    def circuit_dir(self) -> Path:
        return self.data / "circuit"

    @property
    def circuit_spice_csv(self) -> Path:
        return self.circuit_dir / "circuit_spice.csv"

    @property
    def circuit_ml_csv(self) -> Path:
        return self.circuit_dir / "circuit_ml.csv"

    @property
    def circuit_results_csv(self) -> Path:
        return self.circuit_dir / "circuit_results.csv"

    # ---- artifacts/
    @property
    def models_dir(self) -> Path:
        return self.artifacts / "models"

    def model_file(self, case: Case | str) -> Path:
        return self.models_dir / f"{Case(case).value}.joblib"

    @property
    def candidates_dir(self) -> Path:
        return self.artifacts / "candidates"

    @property
    def splits_dir(self) -> Path:
        return self.artifacts / "splits"

    def split_file(self, case: Case | str) -> Path:
        return self.splits_dir / f"{Case(case).value}.npz"

    @property
    def env_json(self) -> Path:
        return self.artifacts / "env.json"

    @property
    def nominals_json(self) -> Path:
        return self.artifacts / "nominals.json"

    # ---- results/
    @property
    def tables_dir(self) -> Path:
        return self.results / "tables"

    @property
    def plots_dir(self) -> Path:
        return self.results / "plots"

    def all_dirs(self) -> list[Path]:
        return [
            self.transistor_dir,
            self.gates_dir,
            self.circuit_dir,
            self.models_dir,
            self.candidates_dir,
            self.splits_dir,
            self.tables_dir,
            self.plots_dir,
            self.netlist_examples,
        ]

    def is_deliverable_location(self) -> bool:
        """True when these paths point at the real deliverable folders (not pilot/dev ones)."""
        return self.artifacts.resolve() == (REPO_ROOT / "artifacts").resolve()


@dataclass(frozen=True)
class ProjectConfig:
    assignment: Assignment
    simulator: SimulatorCfg
    seeds: Seeds
    sampling: SamplingCfg
    split_scheme: str
    ml: MLCfg
    gates: GatesCfg
    paths: Paths
    quick: bool = False
    source: Path = DEFAULT_CONFIG
    team: dict[str, Any] = field(default_factory=dict, compare=False, repr=False)
    raw: dict[str, Any] = field(default_factory=dict, compare=False, repr=False)

    def snapshot(self) -> dict[str, Any]:
        """Plain-dict copy of the config (stored in model files and env.json for provenance)."""
        return {"source": str(self.source), "quick": self.quick, **self.raw}


# ----------------------------------------------------------------------------- loading


def _resolve(p: str | Path) -> Path:
    p = Path(p)
    return p if p.is_absolute() else REPO_ROOT / p


def _find_todo(obj: Any, prefix: str) -> list[str]:
    if isinstance(obj, dict):
        return [hit for k, v in obj.items() for hit in _find_todo(v, f"{prefix}.{k}")]
    if isinstance(obj, list):
        return [hit for i, v in enumerate(obj) for hit in _find_todo(v, f"{prefix}[{i}]")]
    if isinstance(obj, str) and "TODO" in obj:
        return [prefix]
    return []


def _get(raw: dict[str, Any], dotted: str) -> Any:
    cur: Any = raw
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            raise ConfigError(f"missing key '{dotted}' in config")
        cur = cur[part]
    return cur


def load_config(path: str | Path = DEFAULT_CONFIG, *, quick: bool = False) -> ProjectConfig:
    """Load and validate a project config. Raises ConfigError with every problem listed."""
    path = Path(path)
    if not path.is_absolute() and not path.exists():
        path = REPO_ROOT / path
    if not path.is_file():
        raise ConfigError(f"config file not found: {path}")
    raw: dict[str, Any] = yaml.safe_load(path.read_text()) or {}

    todo = _find_todo(raw.get("assignment", {}), "assignment")
    if todo:
        raise ConfigError(
            f"{path}: fill in the TA assignment first, still TODO: {', '.join(todo)}.\n"
            "To work before that, use the fixture config: --config configs/dev.yaml"
        )

    s = "sampling"
    g = "gates"
    assignment = Assignment(
        model_file=_resolve(_get(raw, "assignment.model_file")),
        node=str(_get(raw, "assignment.node")),
        lmin=float(_get(raw, "assignment.lmin")),
        vdd_nominal=float(_get(raw, "assignment.vdd_nominal")),
        test_circuit=str(_get(raw, "assignment.test_circuit")),
    )
    simulator = SimulatorCfg(
        binary=str(_get(raw, "simulator.binary")),
        n_workers=int(_get(raw, "simulator.n_workers")),
        timeout_s=float(_get(raw, "simulator.timeout_s")),
        work_dir=_resolve(_get(raw, "simulator.work_dir")),
        keep_failed=bool(_get(raw, "simulator.keep_failed")),
    )
    seeds = Seeds(**{k: int(_get(raw, f"seeds.{k}")) for k in Seeds.__dataclass_fields__})
    n_key = "quick_n_samples" if quick else "n_samples_per_case"
    sampling = SamplingCfg(
        n_samples_per_case=int(_get(raw, f"{s}.{n_key}")),
        vdd_tol=float(_get(raw, f"{s}.vdd_tol")),
        temp_range_c=tuple(float(v) for v in _get(raw, f"{s}.temp_range_c")),  # type: ignore[arg-type]
        w_mult_range=tuple(float(v) for v in _get(raw, f"{s}.w_mult_range")),  # type: ignore[arg-type]
        process_tol=float(_get(raw, f"{s}.process_tol")),
        process_params=tuple(str(v) for v in _get(raw, f"{s}.process_params")),
    )
    ml = MLCfg(
        model_types=tuple(str(v) for v in _get(raw, "ml.model_types")),
        selection_metric=str(_get(raw, "ml.selection_metric")),
        i_floor_a=float(_get(raw, "ml.i_floor_a")),
    )
    gates = GatesCfg(
        n_corners=int(_get(raw, f"{g}.quick_n_corners" if quick else f"{g}.n_corners")),
        nmos_w_mult=float(_get(raw, f"{g}.sizing.nmos_w_mult")),
        pmos_w_mult=float(_get(raw, f"{g}.sizing.pmos_w_mult")),
        node_voltage_source=str(_get(raw, f"{g}.node_voltage_source")),
    )
    if quick:
        results = _resolve(_get(raw, "paths.pilot_results"))
        paths = Paths(
            data=_resolve(_get(raw, "paths.pilot_data")),
            artifacts=_resolve(_get(raw, "paths.pilot_artifacts")),
            results=results,
            netlist_examples=results / "netlists",
        )
    else:
        paths = Paths(
            data=_resolve(_get(raw, "paths.data")),
            artifacts=_resolve(_get(raw, "paths.artifacts")),
            results=_resolve(_get(raw, "paths.results")),
            netlist_examples=_resolve(_get(raw, "paths.netlist_examples")),
        )
    split_scheme = str(_get(raw, "split.scheme"))

    errors: list[str] = []
    if not assignment.model_file.is_file():
        errors.append(
            f"assignment.model_file does not exist: {assignment.model_file} "
            "(put the assigned PTM card in pdk/, see pdk/README.md)"
        )
    if not any(math.isclose(assignment.vdd_nominal, v) for v in (0.9, 1.0)):
        errors.append(f"assignment.vdd_nominal must be 0.9 or 1.0 V, got {assignment.vdd_nominal}")
    if not 1e-9 <= assignment.lmin <= 1e-6:
        errors.append(f"assignment.lmin must be in METRES (e.g. 22e-9), got {assignment.lmin}")
    if assignment.test_circuit not in TEST_CIRCUITS:
        errors.append(f"assignment.test_circuit must be one of {TEST_CIRCUITS}")
    lo, hi = sampling.w_mult_range
    for name, mult in (("nmos_w_mult", gates.nmos_w_mult), ("pmos_w_mult", gates.pmos_w_mult)):
        if not lo <= mult <= hi:
            errors.append(f"gates.sizing.{name}={mult} is outside the training range [{lo}, {hi}]")
    if split_scheme not in SPLIT_SCHEMES:
        errors.append(f"split.scheme must be one of {SPLIT_SCHEMES}")
    if sampling.process_params != PROCESS_PARAMS:
        errors.append(f"sampling.process_params must be {list(PROCESS_PARAMS)} (brief §4)")
    if gates.node_voltage_source not in NODE_VOLTAGE_SOURCES:
        errors.append(f"gates.node_voltage_source must be one of {NODE_VOLTAGE_SOURCES}")
    if sampling.n_samples_per_case < 10 or gates.n_corners < 1:
        errors.append("sample / corner counts are too small")
    if simulator.n_workers < 1:
        errors.append("simulator.n_workers must be >= 1")
    if errors:
        raise ConfigError(f"invalid config {path}:\n  - " + "\n  - ".join(errors))

    return ProjectConfig(
        assignment=assignment,
        simulator=simulator,
        seeds=seeds,
        sampling=sampling,
        split_scheme=split_scheme,
        ml=ml,
        gates=gates,
        paths=paths,
        quick=quick,
        source=path,
        team=dict(raw.get("team") or {}),
        raw=raw,
    )


def ensure_dirs(cfg: ProjectConfig) -> None:
    """Create every output directory in SPEC §8 (idempotent)."""
    for d in [*cfg.paths.all_dirs(), cfg.simulator.work_dir]:
        d.mkdir(parents=True, exist_ok=True)
