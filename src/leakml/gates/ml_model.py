"""Part B step 4: ML gate leakage from the transistor models.  Owner: P3.  Also used by P4.

Procedure (SPEC §5.3):
  1. take the SPICE-recorded node voltages (out + internal nodes) for this gate/input/corner
  2. resolve each FET's (Vd, Vg, Vs, Vb) from the topology
  3. pick each FET's case (device + gate logic level) and build its INPUT_COLUMNS row from the corner
  4. TransistorModelBank.predict -> terminal currents
  5. leakage = sum of currents INTO FET terminals that sit on the vdd net

Two compositions of step 5 are computed (both in gate_ml.csv):
  - i_vdd_ml_direct: the SPEC §3.5 sum of the predicted currents as they come out of the models.
  - i_vdd_ml (KCL):  the same sum after replacing the channel current of every ON FET that sits at
    |vds| < KCL_VDS_MAX by the value KCL at the floating nodes (out, x, y, n1) requires. See
    kcl_corrected_currents. This is the value Part C uses (GateLeakageModel reads i_vdd_ml).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence

import numpy as np
import pandas as pd

from leakml import io, sampling
from leakml.config import ProjectConfig
from leakml.constants import (
    GATE_INPUT_COMBOS,
    INPUT_COLUMNS,
    NODE_GND,
    NODE_VDD,
    TARGET_COLUMNS,
    Case,
)
from leakml.gates.netlist import gate_widths
from leakml.gates.topology import GateTopology, get_gate, input_levels, terminals_on
from leakml.ml import predictor
from leakml.ml.predictor import TransistorModelBank

# Ready-made helpers for P3 (all implemented):
#   topology.input_levels(gate, bits), topology.terminals_on(gate, "vdd")
#   netlist.gate_widths(cfg)                       -> (nmos_w, pmos_w)
#   sampling.corner_process_params(corner, dev)    -> {"vth0": ..., ...} from n_* / p_*
#   TransistorModelBank.select_case(s)             -> Case from (device, vg, vdd)
#   predictor.snapped_gate_voltage(case, vdd)      -> exact 0 / vdd the model was trained with

# KCL composition threshold. An ON transistor inside a static gate conducts between two nodes that
# are (almost) at the same voltage, so it carries only leakage-sized current set by the REST of the
# gate (e.g. the off pull-down it feeds). Its |vds| there is ~1e-9..1e-7 V on 32nm_LP. The Part A
# models were trained on vd, vs ~ U(0, VDD) and cannot resolve those currents: their absolute error
# at vds ~ 0 (~1e-10 A) is 10-1000x the true current. Below this |vds| the channel current is
# therefore taken from KCL instead of the model. 1 mV is far above the in-gate values (largest
# seen: ~1e-4 V incl. AND2/OR2) and far below any genuinely biased ON device (NOR2 "10": MPB at
# |vds| ~ 0.2 V keeps its ML current).
KCL_VDS_MAX = 1e-3  # volts

# SPEC §5.4, in this order, plus i_vdd_ml_direct (extension: the uncorrected SPEC §3.5 sum).
GATE_ML_COLUMNS: tuple[str, ...] = (
    "gate",
    "inputs",
    "corner_id",
    "i_vdd_ml",
    "i_in_ml",
    "v_x_ml",
    "v_y_ml",
    "v_n1_ml",
    "max_vg_snap_err",
    "i_vdd_ml_direct",
)
NODE_ML_COLUMNS: dict[str, str] = {"x": "v_x_ml", "y": "v_y_ml", "n1": "v_n1_ml"}
# The ONLY gate_spice.csv columns step 08 may read (hard rule 1: never the SPICE currents).
_RECORDED_COLUMNS: dict[str, str] = {
    "out": "v_out_spice",
    "x": "v_x_spice",
    "y": "v_y_spice",
    "n1": "v_n1_spice",
}
_KEY = ["gate", "inputs", "corner_id"]
SNAP_ERR_ATTR = (
    "vg_snap_err"  # fet_feature_rows(...)[0].attrs[SNAP_ERR_ATTR]: |vg_raw - vg| per FET
)


# ------------------------------------------------------------------ shared array kernels
# The single-case functions below and the vectorised run_gate_ml both go through these, so the
# two paths cannot disagree. Every array has one entry per (gate case, corner).


def _node_arrays(
    gate: GateTopology,
    inputs: str,
    vdd: np.ndarray,
    recorded: Mapping[str, float | np.ndarray],
) -> dict[str, np.ndarray]:
    vdd = np.asarray(vdd, dtype=float)
    node_v: dict[str, np.ndarray] = {NODE_VDD: vdd, NODE_GND: np.zeros_like(vdd)}
    for pin, level in input_levels(gate, inputs).items():
        node_v[pin] = vdd * float(level)  # exactly 0.0 or vdd
    for node in (gate.output, *gate.internal_nodes):
        if node not in recorded:
            raise KeyError(
                f"{gate.name} {inputs}: no recorded voltage for node {node!r} "
                f"(have {sorted(recorded)})"
            )
        v = np.broadcast_to(np.asarray(recorded[node], dtype=float), vdd.shape)
        if not np.isfinite(v).all():
            raise ValueError(f"{gate.name} {inputs}: recorded voltage of node {node!r} is NaN/inf")
        node_v[node] = v
    return node_v


def _fet_arrays(
    gate: GateTopology,
    node_v: Mapping[str, np.ndarray],
    vdd: np.ndarray,
    temp: np.ndarray,
    params: Mapping[str, pd.DataFrame],
    cfg: ProjectConfig,
) -> list[tuple[pd.DataFrame, list, np.ndarray]]:
    """Per FET (gate.fets order): (X block [INPUT_COLUMNS], cases, |vg_raw - vg| snap error).

    params[device] holds PROCESS_PARAMS columns (one row per entry of vdd)."""
    widths = dict(zip(("nmos", "pmos"), gate_widths(cfg), strict=True))
    vdd = np.asarray(vdd, dtype=float)
    out = []
    for fet in gate.fets:
        try:
            vd, vg_raw, vs, vb = (np.asarray(node_v[fet.node(t)], dtype=float) for t in "dgsb")
        except KeyError as e:
            raise KeyError(f"{gate.name}.{fet.name}: no voltage for node {e.args[0]!r}") from None
        cases = TransistorModelBank.select_cases([fet.device] * len(vdd), vg_raw, vdd)
        vg = np.array(
            [predictor.snapped_gate_voltage(c, v) for c, v in zip(cases, vdd, strict=True)]
        )
        cols = {
            "vdd": vdd,
            "temp": np.asarray(temp, dtype=float),
            "w": np.full(len(vdd), widths[fet.device]),
            "vd": vd,
            "vs": vs,
            "vg": vg,
            "vb": vb,
            **{p: params[fet.device][p].to_numpy(dtype=float) for p in params[fet.device]},
        }
        X = pd.DataFrame({c: cols[c] for c in INPUT_COLUMNS})
        out.append((X, cases, np.abs(vg_raw - vg)))
    return out


def _leakage_arrays(
    gate: GateTopology, inputs: str, current: Callable[[int, str], np.ndarray]
) -> tuple[np.ndarray, np.ndarray]:
    """(i_vdd, i_in) from current(fet_index, "id"|"ig"|"is"|"ib") -> currents INTO that terminal.

    i_vdd: KCL at the vdd node. The supply is the only other element on that net, so the current
    it delivers (-i(vvdd), SPEC §3.5) equals the sum of the currents flowing INTO every FET
    terminal tied to vdd. i_in: likewise, each input source only feeds FET gates.
    """
    index = {f.name: k for k, f in enumerate(gate.fets)}
    i_vdd = sum(current(index[f.name], f"i{t}") for f, t in terminals_on(gate, NODE_VDD))
    high = {pin for pin, lv in input_levels(gate, inputs).items() if lv}
    i_in = sum((current(k, "ig") for k, f in enumerate(gate.fets) if f.g in high), 0.0)
    i_vdd = np.asarray(i_vdd, dtype=float)
    return i_vdd, np.broadcast_to(np.asarray(i_in, dtype=float), i_vdd.shape).copy()


def _kcl_arrays(
    gate: GateTopology,
    current: Callable[[int, str], np.ndarray],
    free: np.ndarray,
) -> dict[int, dict[str, np.ndarray]]:
    """Per-FET currents {fet_index: {"id"|"ig"|"is"|"ib": array}} after the KCL correction.

    free[k, r] marks FET k in row r as an ON device at |vds| < KCL_VDS_MAX. For those FETs the
    model's ig and ib are kept, id is the unknown, and is = -(id + ig + ib) (KCL of the FET
    itself). The unknowns are fixed by KCL at every floating node (out + internal nodes): the
    currents into all FET terminals on that node sum to zero. All other FETs keep their model
    currents unchanged.

    Rows with the same `free` pattern share one matrix A (nodes x free FETs) and are solved
    together with the Moore-Penrose pseudo-inverse: the least-squares, minimum-norm solution.
      - More equations than unknowns (a node with no free FET): that node's ML residual stays,
        the other nodes are solved in the least-squares sense.
      - More unknowns than equations (ON FETs in parallel, e.g. both PMOS of NAND2 "00" between
        vdd and out): only their SUM is determined, and the minimum-norm choice splits it equally.
        Any other split differs by a null-space vector of A, i.e. a current circulating round the
        loop the parallel FETs form. Such a loop enters and leaves vdd equally, so i_vdd does not
        depend on the split. This is CHECKED for every pattern (ValueError otherwise), which can
        only fail for a gate with an all-ON path from vdd to ground, i.e. not static CMOS.
    """
    nfet = len(gate.fets)
    out = {
        k: {f"i{t}": np.array(current(k, f"i{t}"), dtype=float) for t in "dgsb"}
        for k in range(nfet)
    }
    if not free.any():
        return out
    nodes = [gate.output, *gate.internal_nodes]
    index = {f.name: k for k, f in enumerate(gate.fets)}
    patterns, inverse = np.unique(free.T, axis=0, return_inverse=True)
    for p, pattern in enumerate(patterns):
        fk = [int(k) for k in np.flatnonzero(pattern)]
        if not fk:
            continue
        rows = np.flatnonzero(inverse.ravel() == p)
        col = {k: j for j, k in enumerate(fk)}
        A = np.zeros((len(nodes), len(fk)))
        R = np.zeros((len(nodes), len(rows)))
        for i, node in enumerate(nodes):
            for f, t in terminals_on(gate, node):
                k = index[f.name]
                if k in col and t == "d":
                    A[i, col[k]] += 1.0
                elif k in col and t == "s":  # is = -(id + ig + ib)
                    A[i, col[k]] -= 1.0
                    R[i] += out[k]["ig"][rows] + out[k]["ib"][rows]
                else:
                    R[i] -= out[k][f"i{t}"][rows]
        # i_vdd sensitivity to each unknown id: +1 if the drain is on vdd, -1 if the source is
        c_vdd = np.array([(gate.fets[k].d == NODE_VDD) - (gate.fets[k].s == NODE_VDD) for k in fk])
        _u, sv, vt = np.linalg.svd(A)
        rank = int((sv > 1e-9 * max(sv.max(initial=0.0), 1.0)).sum())
        null = vt[rank:]
        if null.size and np.abs(null @ c_vdd).max() > 1e-9:
            raise ValueError(
                f"{gate.name}: ON FETs {[gate.fets[k].name for k in fk]} form a vdd-to-ground "
                "path; the KCL split would change i_vdd"
            )
        D = np.linalg.pinv(A) @ R
        for j, k in enumerate(fk):
            out[k]["id"][rows] = D[j]
            out[k]["is"][rows] = -(D[j] + out[k]["ig"][rows] + out[k]["ib"][rows])
    return out


def _free_mask(cases: Sequence, vd: np.ndarray, vs: np.ndarray) -> np.ndarray:
    on = np.array([Case(c) in (Case.NMOS_ON, Case.PMOS_ON) for c in cases])
    return on & (np.abs(np.asarray(vd, dtype=float) - np.asarray(vs, dtype=float)) < KCL_VDS_MAX)


# ------------------------------------------------------------------ single case (SPEC §5.3)


def resolve_node_voltages(
    gate: GateTopology, inputs: str, vdd: float, recorded: Mapping[str, float]
) -> dict[str, float]:
    """Voltage of every node in the gate.

    "vdd" -> vdd, "0" -> 0.0, input pins -> exactly 0.0 or vdd from the bit string (bit i drives
    gate.inputs[i]), out and the internal nodes -> `recorded` (the SPICE values, or the ML-solved
    ones in ml_solve mode). Extra keys in `recorded` are ignored. Raises KeyError if a node the
    gate uses is missing, ValueError if it is NaN.
    """
    node_v = _node_arrays(gate, inputs, np.array([float(vdd)]), recorded)
    return {n: float(v[0]) for n, v in node_v.items()}


def _params_frame(corner: Mapping[str, float], device: str, n: int = 1) -> pd.DataFrame:
    return pd.DataFrame([sampling.corner_process_params(corner, device)] * n)


def fet_feature_rows(
    gate: GateTopology,
    inputs: str,
    corner: Mapping[str, float],
    node_v: Mapping[str, float],
    cfg: ProjectConfig,
) -> tuple[pd.DataFrame, list]:
    """One constants.INPUT_COLUMNS row per FET, plus the list of Cases (same order as gate.fets).

    - vd, vs, vb from node_v[fet.d / fet.s / fet.b]
    - case = TransistorModelBank.select_case(fet.device, vg_raw, vdd), vg_raw = node_v[fet.g]
    - vg = predictor.snapped_gate_voltage(case, vdd): exactly 0 or vdd, as in training. The snap
      error |vg_raw - vg| per FET is in X.attrs["vg_snap_err"] (SNAP_ERR_ATTR). It is non-zero
      only where a FET gate is driven by a recorded node (n1, the INV stage of AND2/OR2)
    - w from netlist.gate_widths(cfg), temp/vdd from the corner, process params from
      sampling.corner_process_params(corner, device) (n_* for NMOS, p_* for PMOS)
    The row index is the FET name. `inputs` is accepted for the contract; the input-pin voltages
    already sit in node_v.
    """
    vdd = np.array([float(corner["vdd"])])
    temp = np.array([float(corner["temp"])])
    params = {d: _params_frame(corner, d) for d in ("nmos", "pmos")}
    node_arr = {n: np.array([float(v)]) for n, v in node_v.items()}
    blocks = _fet_arrays(gate, node_arr, vdd, temp, params, cfg)
    X = pd.concat([b[0] for b in blocks], ignore_index=True)
    X.index = pd.Index([f.name for f in gate.fets], name="fet")
    X.attrs[SNAP_ERR_ATTR] = [float(b[2][0]) for b in blocks]
    return X, [b[1][0] for b in blocks]


def leakage_from_currents(
    gate: GateTopology, fet_currents: pd.DataFrame, inputs: str
) -> dict[str, float]:
    """Combine per-FET predicted currents into gate leakage (SPEC §3.5).

    fet_currents: one row per FET with columns id, ig, is, ib (into-device sign, amperes), indexed
    by FET name, or positionally in gate.fets order.
      - i_vdd_ml = sum of the currents INTO every FET terminal on net "vdd"
        (e.g. INV: is + ib of MP). Same meaning as i_vdd_spice = -i(vvdd)
      - i_in_ml  = sum of ig over FETs whose gate sits on an input at logic 1 (0.0 if none)
    """
    names = [f.name for f in gate.fets]
    missing = [c for c in TARGET_COLUMNS if c not in fet_currents.columns]
    if missing:
        raise KeyError(f"fet_currents lacks columns {missing}")
    if set(fet_currents.index) >= set(names):
        cur = fet_currents.loc[names]
    elif len(fet_currents) == len(names):
        cur = fet_currents
    else:
        raise ValueError(
            f"{gate.name}: expected {len(names)} FET rows ({names}), got {len(fet_currents)}"
        )
    i_vdd, i_in = _leakage_arrays(gate, inputs, lambda k, t: np.array([float(cur[t].iloc[k])]))
    return {"i_vdd_ml": float(i_vdd[0]), "i_in_ml": float(i_in[0])}


def kcl_corrected_currents(
    gate: GateTopology, fet_currents: pd.DataFrame, X: pd.DataFrame, cases: Sequence
) -> pd.DataFrame:
    """Per-FET currents after the KCL correction (see _kcl_arrays and KCL_VDS_MAX).

    fet_currents: id, ig, is, ib per FET (as from bank.predict on fet_feature_rows output);
    X, cases: that fet_feature_rows output (vd, vs pick the ON devices at |vds| < KCL_VDS_MAX).
    Both in gate.fets order or indexed by FET name. leakage_from_currents(gate, <result>, inputs)
    then gives the KCL i_vdd_ml.
    """
    names = [f.name for f in gate.fets]
    cur = fet_currents.loc[names] if set(fet_currents.index) >= set(names) else fet_currents
    Xn = X.loc[names] if set(X.index) >= set(names) else X
    if len(cur) != len(names) or len(Xn) != len(names) or len(cases) != len(names):
        raise ValueError(f"{gate.name}: expected {len(names)} FET rows ({names})")
    free = _free_mask(cases, Xn["vd"].to_numpy(), Xn["vs"].to_numpy())[:, None]
    fixed = _kcl_arrays(gate, lambda k, t: np.array([float(cur[t].iloc[k])]), free)
    return pd.DataFrame(
        [[fixed[k][t][0] for t in TARGET_COLUMNS] for k in range(len(names))],
        index=pd.Index(names, name="fet"),
        columns=list(TARGET_COLUMNS),
    )


def predict_gate_leakage(
    gate: GateTopology,
    inputs: str,
    corner: Mapping[str, float],
    recorded_node_v: Mapping[str, float],
    bank: TransistorModelBank,
    cfg: ProjectConfig,
) -> dict[str, float]:
    """One (gate, inputs, corner): resolve_node_voltages -> fet_feature_rows -> bank.predict ->
    leakage_from_currents. Returns {"i_vdd_ml" (KCL), "i_in_ml", "max_vg_snap_err",
    "i_vdd_ml_direct" (SPEC §3.5 sum of the raw predictions)}."""
    node_v = resolve_node_voltages(gate, inputs, float(corner["vdd"]), recorded_node_v)
    X, cases = fet_feature_rows(gate, inputs, corner, node_v, cfg)
    currents = bank.predict(X, cases)
    direct = leakage_from_currents(gate, currents, inputs)
    kcl = leakage_from_currents(gate, kcl_corrected_currents(gate, currents, X, cases), inputs)
    return {
        "i_vdd_ml": kcl["i_vdd_ml"],
        "i_in_ml": direct["i_in_ml"],  # ig is never corrected, so both methods agree
        "max_vg_snap_err": float(max(X.attrs[SNAP_ERR_ATTR])),
        "i_vdd_ml_direct": direct["i_vdd_ml"],
    }


# ------------------------------------------------------------------ all cases, vectorised


def _check_spice_mode(cfg: ProjectConfig) -> None:
    if cfg.gates.node_voltage_source != "spice":
        raise NotImplementedError(
            f"gates.node_voltage_source={cfg.gates.node_voltage_source!r}: only 'spice' is "
            "implemented (ml_solve is the optional stretch goal, SPEC §5.5)"
        )


def _normalise_bits(gates: Sequence[str], bits: Sequence[object]) -> list[str]:
    out = []
    for g, b in zip(gates, bits, strict=True):
        s = str(b)
        if s not in GATE_INPUT_COMBOS[g]:
            raise ValueError(
                f"gate_spice row {g}/{b!r}: inputs must be a bit string from {GATE_INPUT_COMBOS[g]}"
                " (read CSVs with leakml.io.read_csv)"
            )
        out.append(s)
    return out


def run_gate_ml(
    cfg: ProjectConfig, gate_spice: pd.DataFrame, corners: pd.DataFrame, bank: TransistorModelBank
) -> pd.DataFrame:
    """ML prediction for every row of gate_spice.csv -> data/gates/gate_ml.csv (SPEC §5.3/§5.4).

    Only the KEY and recorded node-voltage columns of gate_spice are read (v_out/x/y/n1_spice),
    never its currents (hard rule 1). VDD, temperature and process parameters come from
    `corners`, by corner_id. All FET rows of all cases are built first, bank.predict is called
    ONCE, and the currents are summed back per case. Output: GATE_ML_COLUMNS, one row per
    gate_spice row, same order; i_vdd_ml is the KCL composition, i_vdd_ml_direct the plain
    SPEC §3.5 sum; v_<node>_ml equal the SPICE values in "spice" mode (NaN where the gate lacks
    the node).
    """
    _check_spice_mode(cfg)
    gs = gate_spice.loc[:, _KEY + list(_RECORDED_COLUMNS.values())].reset_index(drop=True)
    gs["inputs"] = _normalise_bits(gs["gate"], gs["inputs"])
    gs["corner_id"] = gs["corner_id"].astype(int)
    if gs.duplicated(_KEY).any():
        raise ValueError("gate_spice has duplicate (gate, inputs, corner_id) rows")

    cn = corners.set_index(corners["corner_id"].astype(int))
    if cn.index.duplicated().any():
        raise ValueError("corners has duplicate corner_id values")
    absent = sorted(set(gs["corner_id"]) - set(cn.index))
    if absent:
        raise KeyError(f"gate_spice uses corner_ids missing from corners: {absent[:10]}")
    params = {
        d: pd.DataFrame(
            [sampling.corner_process_params(r, d) for r in cn.to_dict("records")], index=cn.index
        )
        for d in ("nmos", "pmos")
    }

    X_parts: list[pd.DataFrame] = []
    case_parts: list = []
    groups = []  # (gate, bits, row positions, [(start, stop) per FET], snap errors, free mask)
    offset = 0
    for (g, bits), pos in gs.groupby(["gate", "inputs"], sort=False).indices.items():
        gate = get_gate(g)
        sub = gs.iloc[pos]
        cid = sub["corner_id"].to_numpy()
        vdd = cn.loc[cid, "vdd"].to_numpy(dtype=float)
        temp = cn.loc[cid, "temp"].to_numpy(dtype=float)
        recorded = {
            n: sub[c].to_numpy(dtype=float)
            for n, c in _RECORDED_COLUMNS.items()
            if n == gate.output or n in gate.internal_nodes
        }
        node_v = _node_arrays(gate, bits, vdd, recorded)
        blocks = _fet_arrays(gate, node_v, vdd, temp, {d: params[d].loc[cid] for d in params}, cfg)
        spans = []
        for X, cases, _ in blocks:
            X_parts.append(X)
            case_parts.extend(cases)
            spans.append((offset, offset + len(X)))
            offset += len(X)
        snap = np.max(np.vstack([b[2] for b in blocks]), axis=0)
        free = np.vstack([_free_mask(c, X["vd"], X["vs"]) for X, c, _ in blocks])
        groups.append((gate, bits, pos, spans, snap, free))

    X_all = pd.concat(X_parts, ignore_index=True)
    I_all = bank.predict(X_all, case_parts)  # ONE batched call for every FET of every case
    I_np = {t: I_all[t].to_numpy(dtype=float) for t in TARGET_COLUMNS}

    out = pd.DataFrame(index=gs.index, columns=list(GATE_ML_COLUMNS[3:]), dtype=float)
    for gate, bits, pos, spans, snap, free in groups:

        def raw(k: int, t: str, spans=spans) -> np.ndarray:
            return I_np[t][spans[k][0] : spans[k][1]]

        i_vdd_direct, i_in = _leakage_arrays(gate, bits, raw)
        fixed = _kcl_arrays(gate, raw, free)
        i_vdd, _ = _leakage_arrays(gate, bits, lambda k, t, fixed=fixed: fixed[k][t])
        out.iloc[pos, out.columns.get_loc("i_vdd_ml")] = i_vdd
        out.iloc[pos, out.columns.get_loc("i_vdd_ml_direct")] = i_vdd_direct
        out.iloc[pos, out.columns.get_loc("i_in_ml")] = i_in
        out.iloc[pos, out.columns.get_loc("max_vg_snap_err")] = snap
        for node, col in NODE_ML_COLUMNS.items():
            if node in gate.internal_nodes:
                out.iloc[pos, out.columns.get_loc(col)] = gs[_RECORDED_COLUMNS[node]].to_numpy()[
                    pos
                ]
    result = pd.concat([gs[_KEY], out], axis=1)
    return result.loc[:, list(GATE_ML_COLUMNS)]


def solve_internal_nodes_ml(
    gate: GateTopology,
    inputs: str,
    corner: Mapping[str, float],
    v_out: float,
    bank: TransistorModelBank,
    cfg: ProjectConfig,
) -> dict[str, float]:
    """STRETCH GOAL (cfg.gates.node_voltage_source == "ml_solve"): find internal node voltages
    without SPICE.

    TODO(P3, optional): for each internal node (x, y, n1), solve KCL (sum of ML-predicted currents
    into the node = 0) with scipy.optimize.brentq on [0, vdd]. Coupled nodes (AND2: x and n1)
    need scipy.optimize.root. Compare the ML-solved voltages against SPICE in the report.
    """
    raise NotImplementedError


class GateLeakageModel:
    """Gate-level model that P4 calls when composing the test circuit (SPEC §6.3).  IMPLEMENTED.

    predict(gate, inputs, corner_id) -> ML leakage of that gate at that corner, using the Part B
    result for the same (gate, inputs, corner) (the corners are shared on purpose).

        gm = GateLeakageModel.from_csv(cfg.paths.gate_ml_csv)
        gm.predict("NAND2", "01", corner_id=17)              # -> float, amperes
    """

    KEY = ("gate", "inputs", "corner_id")

    def __init__(self, lut: dict[str, dict[tuple[str, str, int], float]]):
        self._lut = lut

    @staticmethod
    def _bits(gate: str, value: object) -> str:
        # CSV round-trips can turn "01" into 1. Restore the zero padding from the gate's arity.
        width = len(GATE_INPUT_COMBOS[gate][0])
        s = str(value)
        if s.endswith(".0"):
            s = s[:-2]
        return s.zfill(width)

    @classmethod
    def from_results(cls, gate_ml: pd.DataFrame) -> GateLeakageModel:
        missing = [c for c in cls.KEY if c not in gate_ml.columns]
        if missing:
            raise KeyError(f"gate_ml table lacks key columns {missing}")
        keys = [
            (str(g), cls._bits(str(g), b), int(c))
            for g, b, c in zip(
                gate_ml["gate"], gate_ml["inputs"], gate_ml["corner_id"], strict=True
            )
        ]
        if len(set(keys)) != len(keys):
            raise ValueError("gate_ml table has duplicate (gate, inputs, corner_id) rows")
        metrics = [c for c in gate_ml.columns if c.startswith("i_")]
        lut = {m: dict(zip(keys, gate_ml[m].astype(float), strict=True)) for m in metrics}
        return cls(lut)

    @classmethod
    def from_csv(cls, path) -> GateLeakageModel:
        return cls.from_results(io.read_csv(path))

    @property
    def metrics(self) -> list[str]:
        return list(self._lut)

    def corner_ids(self) -> list[int]:
        first = next(iter(self._lut.values()), {})
        return sorted({k[2] for k in first})

    def predict(self, gate: str, inputs: str, corner_id: int, metric: str = "i_vdd_ml") -> float:
        if metric not in self._lut:
            raise KeyError(f"metric {metric!r} not in gate_ml table (have {self.metrics})")
        key = (gate, inputs, int(corner_id))
        try:
            return self._lut[metric][key]
        except KeyError:
            raise KeyError(
                f"no gate-ML result for gate={gate} inputs={inputs} corner_id={corner_id}. "
                "Run scripts/08_gate_ml.py with the same --config/--quick as this run."
            ) from None
