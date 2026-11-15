"""Random draws: Part A transistor samples and Part B/C PVT corners.
IMPLEMENTED (shared foundation, maintained by P1).

Brief §4: draw every parameter independently, VDD FIRST (Vd and Vs ranges depend on it).
All draws go through numpy Generators from make_rng(seed, label) (SPEC §3.4).
"""

from __future__ import annotations

import zlib
from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

from leakml.config import ProjectConfig
from leakml.constants import CASE_BIAS, CASE_DEVICE, CORNER_COLUMNS, PROCESS_PARAMS, Case
from leakml.pdk import ModelCard, format_number

# Low-|Vds| oversampling for the ON cases (SPEC §4.1). Inside a gate every ON FET sits at
# |vds| ~ 1e-7 V (drain and source pulled to the same rail), but uniform vd, vs give only ~0.2 % of
# rows with |vds| < 1 mV. A fraction LOW_VDS_FRAC of ON rows therefore get
# vd = vs ± |vds|, |vds| log-uniform in LOW_VDS_RANGE (random sign, reflected to stay in [0, vdd]).
# TODO(P4): move these two knobs into configs/*.yaml + SamplingCfg if the team wants them tunable.
LOW_VDS_CASES: frozenset[Case] = frozenset({Case.NMOS_ON, Case.PMOS_ON})
LOW_VDS_FRAC = 0.4
LOW_VDS_RANGE = (1e-9, 1e-2)  # volts

# Source pinning for the OFF cases (SPEC §4.1). Inside a static gate an OFF FET sits with its
# source on its own rail (NMOS: 0, PMOS: vdd, = its body) and the drain anywhere up to the other
# rail; stacked OFF FETs see a small source offset (internal node). Uniform vd, vs put fewer than
# 1 % of rows near that point. A fraction SOURCE_PIN_FRAC of OFF rows therefore get vs = rail
# ± offset, offset log-uniform in SOURCE_PIN_RANGE (towards the inside of [0, vdd]); vd stays
# uniform. Drawn after everything else, so the other rows are unchanged.
SOURCE_PIN_CASES: frozenset[Case] = frozenset({Case.NMOS_OFF, Case.PMOS_OFF})
SOURCE_PIN_FRAC = 0.4
SOURCE_PIN_RANGE = (1e-9, 1e-1)  # volts

# Drain pinning for the OFF cases (SPEC §4.1). The OFF FET that sets a static gate's leakage has
# its drain on the OTHER rail (NMOS: vdd, PMOS: 0), i.e. |vds| ~ VDD. With only the source pinned,
# ~4 % of OFF rows had |vds| > 0.9 VDD, and those had the largest validation error. An independent
# DRAIN_PIN_FRAC of OFF rows get vd = other rail ∓ offset (offset log-uniform in SOURCE_PIN_RANGE),
# so ~DRAIN_PIN_FRAC * SOURCE_PIN_FRAC of rows sit at the full-VDD operating point. Drawn last.
DRAIN_PIN_FRAC = 0.5

# Stack pinning for the ON cases (SPEC §4.1). An ON FET in series with an OFF one (NAND2 "10":
# MNA, NOR2 "10": MPB) has its drain on the far rail (NMOS: vdd, PMOS: 0) and its source on the
# floating stack node, which settles where the ON FET is only weakly on (NMOS: high, PMOS: low).
# Uniform vd, vs put <1 % of ON rows there, and those had ~10 % median / ~50 % p95 validation error
# on id. A fraction STACK_PIN_FRAC of the ON rows NOT already in the low-|Vds| set get vd = far rail
# ∓ offset (offset log-uniform in SOURCE_PIN_RANGE) and vs uniform in the STACK_VS_BAND of the
# supply, measured from the device's own rail (NMOS: vs/vdd in 1 - band, PMOS: vs/vdd in band).
STACK_PIN_FRAC = 0.25
STACK_VS_BAND = (0.03, 0.5)  # fraction of vdd away from the FAR rail (the drain's rail)

# Boundary oversampling, all cases (SPEC §4.1). Tree models are least accurate near the edges of
# the sampled box (fewer neighbours; on the 10k data the OFF-case validation p95 was ~58 % for rows
# within 5 % of an edge vs ~45 % inside), and the 200 corners reach those edges too. A fraction
# BOUNDARY_FRAC of rows are boundary rows: on each, every input in BOUNDARY_INPUTS is, with
# probability BOUNDARY_PER_INPUT, redrawn uniformly from the outer BOUNDARY_BAND of its range at a
# random end. Nothing leaves the original ranges. vdd is set before vd/vs are drawn, so
# vd, vs <= vdd still holds. The boundary draws use a child stream, so the low-|Vds| and
# source-pinning overrides see the same base draws in every case.
BOUNDARY_FRAC = 0.3
BOUNDARY_PER_INPUT = 0.5
BOUNDARY_BAND = 0.05  # fraction of each input's range width, at each end
BOUNDARY_INPUTS: tuple[str, ...] = ("vdd", "temp", "w", *PROCESS_PARAMS)


def make_rng(seed: int, *stream: int | str) -> np.random.Generator:
    """Deterministic child RNG for (seed, *labels).

    make_rng(cfg.seeds.transistor_data, "nmos_off") always gives the same stream, whatever order
    the cases are generated in. String labels are hashed with CRC32, which is stable across runs and
    machines (unlike Python's hash()).
    """
    keys = [int(s) if isinstance(s, int) else zlib.crc32(str(s).encode()) for s in stream]
    return np.random.default_rng(np.random.SeedSequence([int(seed), *keys]))


def sample_process_params(
    nominal: Mapping[str, float],
    tol: float,
    n: int,
    rng: np.random.Generator,
    params: Sequence[str] = PROCESS_PARAMS,
) -> pd.DataFrame:
    """n draws of each process parameter ~ nominal * (1 + U(-tol, tol)). Negative nominals keep their sign."""
    return pd.DataFrame({p: nominal[p] * (1.0 + rng.uniform(-tol, tol, n)) for p in params})


def _as_in_deck(x: np.ndarray) -> np.ndarray:
    """Round to the precision the netlist prints (format_number), so stored == simulated."""
    return np.array([float(format_number(v)) for v in x])


def _low_vds_vd(
    vd: np.ndarray, vs: np.ndarray, vdd: np.ndarray, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Replace vd (and round vs) on a random LOW_VDS_FRAC of rows so that |vd - vs| is tiny."""
    n = len(vd)
    low = rng.random(n) < LOW_VDS_FRAC
    lo, hi = np.log10(LOW_VDS_RANGE[0]), np.log10(LOW_VDS_RANGE[1])
    mag = 10.0 ** rng.uniform(lo, hi, n)
    sign = np.where(rng.random(n) < 0.5, -1.0, 1.0)
    new_vd = vs + sign * mag
    out = (new_vd < 0.0) | (new_vd > vdd)
    new_vd[out] = vs[out] - sign[out] * mag[out]  # mag << vdd, so the reflection is in range
    vd, vs = vd.copy(), vs.copy()
    vs[low] = np.minimum(_as_in_deck(vs[low]), vdd[low])
    vd[low] = np.minimum(_as_in_deck(new_vd[low]), vdd[low])
    return vd, vs


def _boundary_masks(n: int, rng: np.random.Generator) -> dict[str, np.ndarray]:
    """{input: rows to push to an edge}: BOUNDARY_FRAC rows, each input with BOUNDARY_PER_INPUT."""
    row = rng.random(n) < BOUNDARY_FRAC
    return {c: row & (rng.random(n) < BOUNDARY_PER_INPUT) for c in BOUNDARY_INPUTS}


def _edge_values(lo: float, hi: float, n: int, rng: np.random.Generator) -> np.ndarray:
    """n values uniform in the outer BOUNDARY_BAND of [lo, hi], at a random end."""
    band = BOUNDARY_BAND * (hi - lo)
    u = rng.uniform(0.0, band, n)
    return np.where(rng.random(n) < 0.5, lo + u, hi - u)


def _pin_source(
    vs: np.ndarray, vdd: np.ndarray, device: str, rng: np.random.Generator
) -> np.ndarray:
    """Pin vs to the device's own rail (+ a small offset) on a random SOURCE_PIN_FRAC of rows."""
    n = len(vs)
    pin = rng.random(n) < SOURCE_PIN_FRAC
    lo, hi = np.log10(SOURCE_PIN_RANGE[0]), np.log10(SOURCE_PIN_RANGE[1])
    offset = 10.0 ** rng.uniform(lo, hi, n)
    new_vs = offset if device == "nmos" else vdd - offset
    vs = vs.copy()
    vs[pin] = np.clip(_as_in_deck(new_vs[pin]), 0.0, vdd[pin])
    return vs


def _pin_drain(
    vd: np.ndarray, vdd: np.ndarray, device: str, rng: np.random.Generator
) -> np.ndarray:
    """Pin vd to the rail opposite the device's source (- a small offset) on DRAIN_PIN_FRAC rows."""
    n = len(vd)
    pin = rng.random(n) < DRAIN_PIN_FRAC
    lo, hi = np.log10(SOURCE_PIN_RANGE[0]), np.log10(SOURCE_PIN_RANGE[1])
    offset = 10.0 ** rng.uniform(lo, hi, n)
    new_vd = vdd - offset if device == "nmos" else offset
    vd = vd.copy()
    vd[pin] = np.clip(_as_in_deck(new_vd[pin]), 0.0, vdd[pin])
    return vd


def _pin_stack(
    vd: np.ndarray, vs: np.ndarray, vdd: np.ndarray, device: str, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """ON cases: put STACK_PIN_FRAC of the non-low-|Vds| rows at the stacked operating point."""
    n = len(vd)
    pin = (rng.random(n) < STACK_PIN_FRAC) & (np.abs(vd - vs) > LOW_VDS_RANGE[1])
    lo, hi = np.log10(SOURCE_PIN_RANGE[0]), np.log10(SOURCE_PIN_RANGE[1])
    offset = 10.0 ** rng.uniform(lo, hi, n)
    gap = rng.uniform(*STACK_VS_BAND, n) * vdd  # distance of the source from the drain's rail
    if device == "nmos":
        new_vd, new_vs = vdd - offset, vdd - gap
    else:
        new_vd, new_vs = offset, gap
    vd, vs = vd.copy(), vs.copy()
    vd[pin] = np.clip(_as_in_deck(new_vd[pin]), 0.0, vdd[pin])
    vs[pin] = np.clip(_as_in_deck(new_vs[pin]), 0.0, vdd[pin])
    return vd, vs


def draw_transistor_samples(
    case: Case | str,
    n: int,
    cfg: ProjectConfig,
    card: ModelCard,
    rng: np.random.Generator,
    start_id: int = 0,
) -> pd.DataFrame:
    """Draw n input rows for one Part A case (no currents yet).

    Fixed draw order: vdd, vd, vs, w, temp, then the process params, with the boundary rows
    (BOUNDARY_*) pushed to the range edges from a child stream; ON cases then draw the
    low-|Vds| override (LOW_VDS_*) then the stack pinning (STACK_*), and OFF cases the source and
    drain pinning (SOURCE_PIN_*, DRAIN_PIN_*), so the rows they do not touch are unaffected.
    vg/vb follow CASE_BIAS using each row's own vdd. Returns `sample_id` + constants.INPUT_COLUMNS. Calling it again with the
    same rng continues the stream (used to top up failed simulations deterministically).
    """
    case = Case(case)
    s, a = cfg.sampling, cfg.assignment
    edge_rng = np.random.default_rng(rng.integers(2**63))  # child stream for the boundary draws
    edge = _boundary_masks(n, edge_rng)

    def push(x: np.ndarray, col: str, lo: float, hi: float) -> np.ndarray:
        m = edge[col]
        x = x.copy()
        x[m] = _edge_values(lo, hi, int(m.sum()), edge_rng)
        return x

    vdd_lo, vdd_hi = a.vdd_nominal * (1 - s.vdd_tol), a.vdd_nominal * (1 + s.vdd_tol)
    vdd = push(rng.uniform(vdd_lo, vdd_hi, n), "vdd", vdd_lo, vdd_hi)
    vd = rng.uniform(0.0, vdd)
    vs = rng.uniform(0.0, vdd)
    w_lo, w_hi = s.w_mult_range[0] * a.lmin, s.w_mult_range[1] * a.lmin
    w = push(rng.uniform(*s.w_mult_range, n) * a.lmin, "w", w_lo, w_hi)
    temp = push(rng.uniform(*s.temp_range_c, n), "temp", *s.temp_range_c)
    nominal = card.nominal[CASE_DEVICE[case]]
    proc = sample_process_params(nominal, s.process_tol, n, rng)
    for p in PROCESS_PARAMS:  # in ratio space, so PMOS vth0 keeps its sign
        m = edge[p]
        ratio = _edge_values(1 - s.process_tol, 1 + s.process_tol, int(m.sum()), edge_rng)
        proc.loc[m, p] = nominal[p] * ratio
    if case in LOW_VDS_CASES:
        vd, vs = _low_vds_vd(vd, vs, vdd, rng)
        vd, vs = _pin_stack(vd, vs, vdd, CASE_DEVICE[case], rng)
    if case in SOURCE_PIN_CASES:
        vs = _pin_source(vs, vdd, CASE_DEVICE[case], rng)
        vd = _pin_drain(vd, vdd, CASE_DEVICE[case], rng)
    bias = CASE_BIAS[case]
    vg = vdd if bias["vg"] == "vdd" else np.zeros(n)
    vb = vdd if bias["vb"] == "vdd" else np.zeros(n)
    df = pd.DataFrame(
        {
            "sample_id": np.arange(start_id, start_id + n),
            "vdd": vdd,
            "temp": temp,
            "w": w,
            "vd": vd,
            "vs": vs,
            "vg": vg,
            "vb": vb,
        }
    )
    return pd.concat([df, proc], axis=1)


def draw_corners(
    n: int, cfg: ProjectConfig, card: ModelCard, rng: np.random.Generator
) -> pd.DataFrame:
    """The n PVT corners shared by Part B and Part C (brief §5).

    One row per corner: vdd, temp, one NMOS process draw (n_*), one PMOS draw (p_*).
    Same distributions as Part A. Columns = constants.CORNER_COLUMNS.
    """
    s, a = cfg.sampling, cfg.assignment
    vdd = rng.uniform(a.vdd_nominal * (1 - s.vdd_tol), a.vdd_nominal * (1 + s.vdd_tol), n)
    temp = rng.uniform(*s.temp_range_c, n)
    nm = sample_process_params(card.nominal["nmos"], s.process_tol, n, rng).add_prefix("n_")
    pm = sample_process_params(card.nominal["pmos"], s.process_tol, n, rng).add_prefix("p_")
    df = pd.concat(
        [pd.DataFrame({"corner_id": np.arange(n), "vdd": vdd, "temp": temp}), nm, pm], axis=1
    )
    return df.loc[:, list(CORNER_COLUMNS)]


def corner_process_params(corner: Mapping[str, float], device: str) -> dict[str, float]:
    """{"vth0": ..., ...} for one device type from a corner row (n_* for nmos, p_* for pmos)."""
    prefix = {"nmos": "n_", "pmos": "p_"}[device]
    return {p: float(corner[prefix + p]) for p in PROCESS_PARAMS}
