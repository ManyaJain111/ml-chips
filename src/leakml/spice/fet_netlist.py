"""Single-transistor decks for Part A.  IMPLEMENTED (shared foundation, maintained by P1).

Brief §4: every terminal of the single-FET netlist has ITS OWN voltage source.
"""

from __future__ import annotations

from collections.abc import Mapping

from leakml.constants import CASE_DEVICE, PROCESS_PARAMS, Case
from leakml.pdk import ModelCard
from leakml.spice.common import END, fmt, model_block, netlist_header, op_control_block

# Source name per terminal. Keep in sync with parse.currents_into_device.
FET_SOURCES = {"id": "vd", "ig": "vg", "is": "vs", "ib": "vb"}
FET_EXPECT = [f"i({s})" for s in FET_SOURCES.values()]


def render_fet_netlist(
    sample: Mapping[str, float], case: Case | str, card: ModelCard, lmin: float
) -> str:
    """Full deck for one sample row (constants.INPUT_COLUMNS, optionally `sample_id`) of one case."""
    case = Case(case)
    device = CASE_DEVICE[case]
    params = {p: float(sample[p]) for p in PROCESS_PARAMS}
    sid = sample.get("sample_id", -1)
    return "\n".join(
        [
            netlist_header(f"{case.value} sample {int(sid)}", float(sample["temp"])),
            model_block(
                card,
                params if device == "nmos" else None,
                params if device == "pmos" else None,
            ),
            f"Vd d 0 dc {fmt(sample['vd'])}",
            f"Vg g 0 dc {fmt(sample['vg'])}",
            f"Vs s 0 dc {fmt(sample['vs'])}",
            f"Vb b 0 dc {fmt(sample['vb'])}",
            f"M1 d g s b {card.model_names[device]} W={fmt(sample['w'])} L={fmt(lmin)}",
            op_control_block(FET_EXPECT),
            END,
        ]
    )
