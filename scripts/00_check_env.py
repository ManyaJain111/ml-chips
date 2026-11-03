"""Step 00: environment sanity check.  IMPLEMENTED.

Checks config, ngspice, model card and one real simulation, then writes artifacts/env.json
(the report's Setup section: simulator + version, library versions, machine). Exit code 0 = all good.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

from leakml.cli import setup_logging
from leakml.config import DEFAULT_CONFIG, REPO_ROOT, ConfigError, ensure_dirs, load_config
from leakml.constants import Case
from leakml.pdk import load_model_card
from leakml.spice.fet_netlist import FET_EXPECT, FET_SOURCES, render_fet_netlist
from leakml.spice.parse import currents_into_device
from leakml.spice.runner import ngspice_version, run_netlist

PACKAGES = ("numpy", "pandas", "scipy", "scikit-learn", "lightgbm", "matplotlib", "joblib")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default=DEFAULT_CONFIG)
    p.add_argument("--quick", action="store_true", help="pilot run with small sample counts")
    return p.parse_args(argv)


def _git_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=REPO_ROOT, timeout=10
        )
        return out.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def main(argv=None) -> None:
    args = parse_args(argv)
    setup_logging()
    try:
        cfg = load_config(args.config, quick=args.quick)
    except ConfigError as e:
        print(f"CONFIG ERROR\n{e}", file=sys.stderr)
        sys.exit(2)
    ensure_dirs(cfg)
    a = cfg.assignment

    version = ngspice_version(cfg.simulator.binary)
    card = load_model_card(a.model_file)

    # One real simulation: NMOS OFF at nominal process, Vds = VDD, 27 C.
    sample = {"sample_id": 0, "vdd": a.vdd_nominal, "temp": 27.0, "w": 2 * a.lmin,
              "vd": a.vdd_nominal, "vs": 0.0, "vg": 0.0, "vb": 0.0, **card.nominal["nmos"]}  # fmt: skip
    deck = render_fet_netlist(sample, Case.NMOS_OFF, card, a.lmin)
    res = run_netlist(deck, binary=cfg.simulator.binary, work_dir=cfg.simulator.work_dir,
                      timeout_s=cfg.simulator.timeout_s, expect=FET_EXPECT)  # fmt: skip
    problems = []
    smoke = {}
    if res.ok:
        smoke = currents_into_device(res.values, FET_SOURCES)
        kcl = abs(sum(smoke.values())) / max(abs(v) for v in smoke.values())
        if kcl > 1e-6:
            problems.append(f"smoke deck violates KCL (relative residual {kcl:.2g})")
    else:
        problems.append(f"smoke deck failed: {res.error} (deck kept at {res.netlist_path})")

    env = {
        "timestamp_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "simulator": version,
        "python": platform.python_version(),
        "packages": {p: importlib.metadata.version(p) for p in PACKAGES},
        "platform": platform.platform(),
        "hostname": platform.node(),
        "inside_docker": Path("/.dockerenv").exists(),
        "git_commit": _git_commit(),
        "config": str(cfg.source),
        "quick": cfg.quick,
        "assignment": {
            "model_file": str(a.model_file),
            "node": a.node,
            "lmin_m": a.lmin,
            "vdd_nominal_v": a.vdd_nominal,
            "test_circuit": a.test_circuit,
        },  # fmt: skip
        "nominal": card.nominal,
        "smoke_nmos_off_currents_a": smoke,
    }
    cfg.paths.env_json.parent.mkdir(parents=True, exist_ok=True)
    cfg.paths.env_json.write_text(json.dumps(env, indent=2) + "\n")

    print(f"config      : {cfg.source} (quick={cfg.quick})")
    print(f"simulator   : {version}  ({'docker' if env['inside_docker'] else 'host'})")
    print(
        f"model file  : {a.model_file.name}  node={a.node}  Lmin={a.lmin:g} m  VDD={a.vdd_nominal} V"
    )
    print(f"smoke Ioff  : id = {smoke.get('id', float('nan')):.4g} A  (NMOS OFF, Vds=VDD, 27 C)")
    print(f"wrote       : {cfg.paths.env_json}")
    if problems:
        print("PROBLEMS:\n  - " + "\n  - ".join(problems), file=sys.stderr)
        sys.exit(1)
    print("OK")


if __name__ == "__main__":
    main()
