"""Run ngspice in batch mode, in parallel.  IMPLEMENTED (shared foundation, maintained by P1).

    res = run_netlist(deck, expect=["i(vvdd)"])            # one deck
    results = run_many_cfg(decks, cfg, expect=[...])        # many decks, order preserved
    if not res.ok: print(res.error, res.netlist_path)       # failed decks are kept for debugging

Runtime budget (SPEC §9): process start-up dominates (~15 ms per deck), so parallelism matters.
"""

from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
import tempfile
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

from tqdm import tqdm

from leakml.config import ProjectConfig
from leakml.spice.parse import parse_print_output

log = logging.getLogger(__name__)

# Lowercased substrings of ngspice output that mean the operating point is not trustworthy.
FAILURE_MARKERS = (
    "error",
    "singular matrix",
    "no convergence",
    "timestep too small",
    "gmin stepping failed",
    "source stepping failed",
    "simulation interrupted",
    "aborted",
    "no simulations run",
    "could not find",
    "unknown subckt",
)


@dataclass
class SpiceResult:
    ok: bool
    values: dict[str, float] = field(default_factory=dict)  # parsed print output, lowercase keys
    stdout: str = ""
    stderr: str = ""
    returncode: int = 0
    error: str = ""  # short reason when ok is False
    netlist_path: Path | None = None  # kept only when the run failed and keep_failed=True


def _default_work_dir() -> Path:
    return Path(tempfile.gettempdir()) / "leakml_spice"


def _check_binary(binary: str) -> None:
    if shutil.which(binary) is None:
        raise FileNotFoundError(
            f"'{binary}' not found on PATH. Install ngspice, or use the container: make docker-shell"
        )


def ngspice_version(binary: str = "ngspice") -> str:
    """e.g. 'ngspice-47'. Needed for the report's Setup section."""
    _check_binary(binary)
    out = subprocess.run([binary, "--version"], capture_output=True, text=True, timeout=30)
    m = re.search(r"ngspice-[0-9][0-9.]*", out.stdout + out.stderr)
    return m.group(0) if m else "unknown"


def _failure_reason(proc_out: str) -> str:
    low = proc_out.lower()
    for marker in FAILURE_MARKERS:
        if marker in low:
            # return the offending line for context
            for line in proc_out.splitlines():
                if marker in line.lower():
                    return line.strip()[:200]
            return marker
    return ""


def run_netlist(
    netlist: str,
    *,
    binary: str = "ngspice",
    timeout_s: float = 60.0,
    work_dir: Path | None = None,
    keep_failed: bool = True,
    expect: Sequence[str] | None = None,
) -> SpiceResult:
    """Run one deck with `ngspice -b` and parse its printed values.

    ok=False when: non-zero exit, timeout, a FAILURE_MARKERS line in the output, or any key in
    `expect` (e.g. ["i(vd)", "i(vg)"]) missing from the parsed values.
    """
    _check_binary(binary)
    work_dir = Path(work_dir) if work_dir else _default_work_dir()
    work_dir.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(suffix=".sp", prefix="deck_", dir=work_dir)
    path = Path(name)
    with os.fdopen(fd, "w") as fh:
        fh.write(netlist if netlist.endswith("\n") else netlist + "\n")

    try:
        proc = subprocess.run(
            [binary, "-b", str(path)],
            capture_output=True,
            text=True,
            timeout=timeout_s,
            cwd=work_dir,
        )
        res = SpiceResult(
            ok=True,
            values=parse_print_output(proc.stdout),
            stdout=proc.stdout,
            stderr=proc.stderr,
            returncode=proc.returncode,
        )
        reason = _failure_reason(proc.stdout + "\n" + proc.stderr)
        if proc.returncode != 0:
            res.ok, res.error = False, reason or f"ngspice exited with {proc.returncode}"
        elif reason:
            res.ok, res.error = False, reason
        else:
            missing = [k.lower() for k in (expect or []) if k.lower() not in res.values]
            if missing:
                res.ok, res.error = False, f"missing outputs: {missing}"
    except subprocess.TimeoutExpired:
        res = SpiceResult(ok=False, error=f"timeout after {timeout_s}s")

    if res.ok or not keep_failed:
        path.unlink(missing_ok=True)
    else:
        res.netlist_path = path
    return res


def run_many(
    netlists: Sequence[str],
    *,
    binary: str = "ngspice",
    n_workers: int = 8,
    timeout_s: float = 60.0,
    work_dir: Path | None = None,
    keep_failed: bool = True,
    expect: Sequence[str] | None = None,
    desc: str = "ngspice",
    progress: bool = True,
) -> list[SpiceResult]:
    """Run many decks in parallel. Results come back in the SAME ORDER as `netlists`."""
    _check_binary(binary)

    def one(deck: str) -> SpiceResult:
        return run_netlist(
            deck,
            binary=binary,
            timeout_s=timeout_s,
            work_dir=work_dir,
            keep_failed=keep_failed,
            expect=expect,
        )

    with ThreadPoolExecutor(max_workers=max(1, n_workers)) as ex:
        it = ex.map(one, netlists)
        results = list(tqdm(it, total=len(netlists), desc=desc, disable=not progress))

    n_bad = sum(not r.ok for r in results)
    if n_bad:
        first = next(r for r in results if not r.ok)
        log.warning(
            "%s: %d/%d decks failed. First failure: %s (%s)",
            desc,
            n_bad,
            len(results),
            first.error,
            first.netlist_path,
        )
    else:
        log.info("%s: %d decks ok", desc, len(results))
    return results


def run_many_cfg(
    netlists: Sequence[str],
    cfg: ProjectConfig,
    *,
    expect: Sequence[str] | None = None,
    desc: str = "ngspice",
    progress: bool = True,
) -> list[SpiceResult]:
    """run_many with the simulator settings from the project config."""
    sim = cfg.simulator
    return run_many(
        netlists,
        binary=sim.binary,
        n_workers=sim.n_workers,
        timeout_s=sim.timeout_s,
        work_dir=sim.work_dir,
        keep_failed=sim.keep_failed,
        expect=expect,
        desc=desc,
        progress=progress,
    )
