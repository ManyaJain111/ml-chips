"""Generate the four Part A datasets (Deliverable 1).
IMPLEMENTED (critical-path foundation, owned and hardened by P1).

    df = generate_case_dataset(Case.NMOS_OFF, cfg, card)   # sample -> simulate -> top up failures
    save_dataset(df, Case.NMOS_OFF, cfg)                   # data/transistor/nmos_off.csv
    df = load_dataset(Case.NMOS_OFF, cfg)                  # P2: validated read
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import pandas as pd

from leakml import io
from leakml.config import ProjectConfig
from leakml.constants import DATASET_COLUMNS, TARGET_COLUMNS, Case
from leakml.pdk import ModelCard
from leakml.sampling import draw_transistor_samples, make_rng
from leakml.spice.fet_netlist import FET_EXPECT, FET_SOURCES, render_fet_netlist
from leakml.spice.parse import currents_into_device
from leakml.spice.runner import run_many_cfg

log = logging.getLogger(__name__)

MAX_FAIL_FRACTION = 0.01  # brief: investigate rather than hide failures
MAX_TOPUP_ROUNDS = 5


def simulate_samples(
    samples: pd.DataFrame, case: Case | str, cfg: ProjectConfig, card: ModelCard
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run SPICE for every sample row.

    Returns (ok, failed): `ok` = samples + id/ig/is/ib for converged rows, `failed` = the input rows
    that did not converge, with an `error` column. Failed decks are kept in cfg.simulator.work_dir.
    """
    case = Case(case)
    rows = samples.to_dict("records")
    decks = [render_fet_netlist(r, case, card, cfg.assignment.lmin) for r in rows]
    results = run_many_cfg(decks, cfg, expect=FET_EXPECT, desc=f"spice {case.value}")

    currents, ok_mask, errors = [], [], []
    for res in results:
        if res.ok:
            currents.append(currents_into_device(res.values, FET_SOURCES))
            ok_mask.append(True)
            errors.append("")
        else:
            ok_mask.append(False)
            errors.append(f"{res.error} [{res.netlist_path}]")

    mask = pd.Series(ok_mask, index=samples.index)
    ok = samples[mask].copy()
    cur = pd.DataFrame(currents, index=ok.index, columns=list(TARGET_COLUMNS))
    ok = pd.concat([ok, cur], axis=1)
    failed = samples[~mask].copy()
    failed["error"] = [e for e, m in zip(errors, ok_mask, strict=True) if not m]
    return ok, failed


def generate_case_dataset(
    case: Case | str, cfg: ProjectConfig, card: ModelCard, *, save_example: bool = True
) -> pd.DataFrame:
    """Sample + simulate exactly cfg.sampling.n_samples_per_case converged rows for one case.

    Deterministic: failed rows are replaced by further draws from the SAME rng stream. Raises if
    more than MAX_FAIL_FRACTION of the rows fail (that is a deck/model problem, not noise).
    The returned frame keeps `sample_id` for traceability. save_dataset drops it.
    """
    case = Case(case)
    n = cfg.sampling.n_samples_per_case
    rng = make_rng(cfg.seeds.transistor_data, case.value)
    t0 = time.perf_counter()

    parts, n_failed, next_id = [], 0, 0
    need = n
    for _ in range(MAX_TOPUP_ROUNDS):
        batch = draw_transistor_samples(case, need, cfg, card, rng, start_id=next_id)
        next_id += need
        ok, failed = simulate_samples(batch, case, cfg, card)
        parts.append(ok)
        n_failed += len(failed)
        if len(failed):
            log.warning(
                "%s: %d rows failed, e.g. %s", case.value, len(failed), failed["error"].iloc[0]
            )
        if n_failed > MAX_FAIL_FRACTION * n + 5:
            raise RuntimeError(
                f"{case.value}: {n_failed} simulations failed (>{MAX_FAIL_FRACTION:.0%}). "
                f"Inspect the kept decks in {cfg.simulator.work_dir}. First error: "
                f"{failed['error'].iloc[0] if len(failed) else '?'}"
            )
        need = n - sum(len(p) for p in parts)
        if need == 0:
            break
    else:
        raise RuntimeError(f"{case.value}: could not reach {n} converged rows")

    df = pd.concat(parts, ignore_index=True)
    log.info(
        "%s: %d rows (%d failed and replaced) in %.1fs",
        case.value,
        len(df),
        n_failed,
        time.perf_counter() - t0,
    )
    if save_example:
        out = cfg.paths.netlist_examples / f"fet_{case.value}.sp"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(render_fet_netlist(df.iloc[0].to_dict(), case, card, cfg.assignment.lmin))
    return df


def save_dataset(df: pd.DataFrame, case: Case | str, cfg: ProjectConfig) -> Path:
    """data/transistor/{case}.csv with EXACTLY constants.DATASET_COLUMNS, in that order."""
    return io.write_csv(df.loc[:, list(DATASET_COLUMNS)], cfg.paths.transistor_csv(case))


def load_dataset(case: Case | str, cfg: ProjectConfig) -> pd.DataFrame:
    """Read data/transistor/{case}.csv and enforce the column contract (SPEC §4.3)."""
    path = cfg.paths.transistor_csv(case)
    df = io.read_csv(path)
    if tuple(df.columns) != DATASET_COLUMNS:
        raise ValueError(f"{path}: columns {list(df.columns)} != {list(DATASET_COLUMNS)}")
    if df.isna().any().any():
        raise ValueError(f"{path}: contains NaN")
    return df
