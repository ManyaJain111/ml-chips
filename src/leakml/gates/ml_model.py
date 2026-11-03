"""Part B step 4: ML gate leakage from the transistor models.  Owner: P3.  Also used by P4.

Procedure (SPEC §5.3):
  1. take the SPICE-recorded node voltages (out + internal nodes) for this gate/input/corner
  2. resolve each FET's (Vd, Vg, Vs, Vb) from the topology
  3. pick each FET's case (device + gate logic level) and build its INPUT_COLUMNS row from the corner
  4. TransistorModelBank.predict -> terminal currents
  5. leakage = sum of currents INTO FET terminals that sit on the vdd net
"""

from __future__ import annotations

from collections.abc import Mapping

import pandas as pd

from leakml import io
from leakml.config import ProjectConfig
from leakml.constants import GATE_INPUT_COMBOS
from leakml.gates.topology import GateTopology
from leakml.ml.predictor import TransistorModelBank

# Ready-made helpers for P3 (all implemented):
#   topology.input_levels(gate, bits), topology.terminals_on(gate, "vdd")
#   netlist.gate_widths(cfg)                       -> (nmos_w, pmos_w)
#   sampling.corner_process_params(corner, dev)    -> {"vth0": ..., ...} from n_* / p_*
#   TransistorModelBank.select_case(s)             -> Case from (device, vg, vdd)
#   predictor.snapped_gate_voltage(case, vdd)      -> exact 0 / vdd the model was trained with


def resolve_node_voltages(
    gate: GateTopology, inputs: str, vdd: float, recorded: Mapping[str, float]
) -> dict[str, float]:
    """Voltage of every node in the gate.

    TODO(P3): vdd -> vdd, "0" -> 0.0, input pins -> 0.0 or vdd from the bit string, out and
    internal nodes -> `recorded` (the SPICE values, or the ML-solved ones in ml_solve mode).
    Raise if a node is missing.
    """
    raise NotImplementedError


def fet_feature_rows(
    gate: GateTopology,
    inputs: str,
    corner: Mapping[str, float],
    node_v: Mapping[str, float],
    cfg: ProjectConfig,
) -> tuple[pd.DataFrame, list]:
    """One constants.INPUT_COLUMNS row per FET, plus the list of Cases (same order as gate.fets).

    TODO(P3):
      - vd, vs, vb from node_v[fet.d / fet.s / fet.b]
      - vg: SNAP to exactly 0 or vdd (the models were trained with vg in {0, VDD}). Record the
        snap error |vg_raw - vg_snapped|, which is non-trivial only for the INV stage of AND2/OR2
        (its gate is n1) and is worth discussing in the error analysis
      - case = TransistorModelBank.select_case(fet.device, vg_raw, vdd)
      - w = nmos_w_mult*lmin or pmos_w_mult*lmin, temp/vdd from the corner,
        process params from corner n_* (NMOS) or p_* (PMOS)
    """
    raise NotImplementedError


def leakage_from_currents(
    gate: GateTopology, fet_currents: pd.DataFrame, inputs: str
) -> dict[str, float]:
    """Combine per-FET predicted currents into gate leakage.

    TODO(P3):
      - i_vdd_ml = sum over FET terminals connected to "vdd" of the current INTO that terminal
        (e.g. INV: is + ib of MP). This matches i_vdd_spice = -i(vvdd)
      - i_in_ml  = sum over FET gate terminals on input nets at logic 1 of ig (secondary metric)
    """
    raise NotImplementedError


def predict_gate_leakage(
    gate: GateTopology,
    inputs: str,
    corner: Mapping[str, float],
    recorded_node_v: Mapping[str, float],
    bank: TransistorModelBank,
    cfg: ProjectConfig,
) -> dict[str, float]:
    """TODO(P3): resolve_node_voltages -> fet_feature_rows -> bank.predict -> leakage_from_currents."""
    raise NotImplementedError


def run_gate_ml(
    cfg: ProjectConfig, gate_spice: pd.DataFrame, corners: pd.DataFrame, bank: TransistorModelBank
) -> pd.DataFrame:
    """ML prediction for every row of gate_spice.csv -> data/gates/gate_ml.csv.

    TODO(P3): vectorise. Build ALL FET rows for all 3600 cases, call bank.predict ONCE, then
    group back. Output columns: gate, inputs, corner_id, i_vdd_ml, i_in_ml, v_<node>_ml (equal
    to SPICE in "spice" mode), max_vg_snap_err.
    """
    raise NotImplementedError


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
