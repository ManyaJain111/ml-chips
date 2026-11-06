"""Part C step 6: SPICE reference for the complete test circuit.  Owner: P4."""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from leakml import io
from leakml.circuit import logic
from leakml.circuit.definition import Circuit
from leakml.circuit.netlist import circuit_print_exprs, input_source, render_circuit_netlist
from leakml.config import ProjectConfig
from leakml.pdk import ModelCard
from leakml.spice.runner import run_many_cfg

log = logging.getLogger(__name__)

CHUNK = 2000  # decks per checkpoint file
COLUMNS = [
    "circuit", "vector", "corner_id", "vdd", "temp",
    "i_vdd_spice", "i_in_spice", "logic_ok",
]  # fmt: skip


def checkpoint_dir(c: Circuit, cfg: ProjectConfig) -> Path:
    """Per-run checkpoint folder (outside the deliverable folders, one per output location)."""
    tag = str(cfg.paths.circuit_dir.resolve()).strip("/").replace("/", "_")
    return Path(cfg.simulator.work_dir) / "circuit_ckpt" / f"{tag}_{c.name}"


def extract_circuit_result(
    c: Circuit, vector: str, vdd: float, values: dict[str, float]
) -> tuple[float, float, bool]:
    """(i_vdd, i_in, logic_ok) from parsed print values of one deck."""
    i_vdd = -values["i(vvdd)"]
    i_in = sum(
        -values[f"i({input_source(pi)})"]
        for pi, bit in zip(c.primary_inputs, vector, strict=True)
        if bit == "1"
    )
    expect = logic.reference_function(c, vector)
    ok = all(abs(values[f"v({po})"] - expect[po] * vdd) < 0.1 * vdd for po in c.primary_outputs)
    return i_vdd, i_in, bool(ok)


def run_circuit_reference(
    c: Circuit,
    cfg: ProjectConfig,
    corners: pd.DataFrame,
    card: ModelCard,
    *,
    resume: bool = False,
) -> pd.DataFrame:
    """Every input vector x the SAME corners as Part B -> data/circuit/circuit_spice.csv.

    Runs in chunks of CHUNK decks. Each completed chunk (all decks ok) is checkpointed, and with
    resume=True finished chunks are loaded instead of re-simulated. Failed decks yield NaN currents
    and logic_ok=False (the script refuses to save those).
    Columns: circuit, vector, corner_id, vdd, temp, i_vdd_spice, i_in_spice, logic_ok.
    """
    ckpt = checkpoint_dir(c, cfg)
    if not resume and ckpt.exists():
        shutil.rmtree(ckpt)
    ckpt.mkdir(parents=True, exist_ok=True)

    vectors = logic.all_input_vectors(c)
    crows = corners.to_dict("records")
    jobs = [(v, cr) for v in vectors for cr in crows]  # vector-major, deterministic order
    parts: list[pd.DataFrame] = []
    n_chunks = -(-len(jobs) // CHUNK)
    for k in range(n_chunks):
        f = ckpt / f"chunk_{k:05d}_{CHUNK}.csv"
        if f.exists():
            parts.append(io.read_csv(f))
            continue
        batch = jobs[k * CHUNK : (k + 1) * CHUNK]
        decks = [render_circuit_netlist(c, v, cr, card, cfg) for v, cr in batch]
        results = run_many_cfg(
            decks, cfg, expect=circuit_print_exprs(c), desc=f"{c.name} {k + 1}/{n_chunks}"
        )
        rows = []
        for (v, cr), res in zip(batch, results, strict=True):
            vdd = float(cr["vdd"])
            if res.ok:
                i_vdd, i_in, ok = extract_circuit_result(c, v, vdd, res.values)
            else:
                i_vdd, i_in, ok = np.nan, np.nan, False
            rows.append((c.name, v, int(cr["corner_id"]), vdd, float(cr["temp"]), i_vdd, i_in, ok))
        df = pd.DataFrame(rows, columns=COLUMNS)
        if all(r.ok for r in results):
            io.write_csv(df, f)
        else:
            log.warning("chunk %d has failed decks; not checkpointed", k)
        parts.append(df)
    return pd.concat(parts, ignore_index=True)
