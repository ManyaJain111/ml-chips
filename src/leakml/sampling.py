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


def draw_transistor_samples(
    case: Case | str,
    n: int,
    cfg: ProjectConfig,
    card: ModelCard,
    rng: np.random.Generator,
    start_id: int = 0,
) -> pd.DataFrame:
    """Draw n input rows for one Part A case (no currents yet).

    Fixed draw order: vdd, vd, vs, w, temp, then the process params; ON cases then draw the
    low-|Vds| override (LOW_VDS_*), so OFF rows and the uniform ON rows are unaffected by it.
    vg/vb follow CASE_BIAS using each row's own vdd. Returns `sample_id` + constants.INPUT_COLUMNS. Calling it again with the
    same rng continues the stream (used to top up failed simulations deterministically).
    """
    case = Case(case)
    s, a = cfg.sampling, cfg.assignment
    vdd = rng.uniform(a.vdd_nominal * (1 - s.vdd_tol), a.vdd_nominal * (1 + s.vdd_tol), n)
    vd = rng.uniform(0.0, vdd)
    vs = rng.uniform(0.0, vdd)
    w = rng.uniform(*s.w_mult_range, n) * a.lmin
    temp = rng.uniform(*s.temp_range_c, n)
    proc = sample_process_params(card.nominal[CASE_DEVICE[case]], s.process_tol, n, rng)
    if case in LOW_VDS_CASES:
        vd, vs = _low_vds_vd(vd, vs, vdd, rng)
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
