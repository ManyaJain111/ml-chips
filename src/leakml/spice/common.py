"""Netlist building blocks shared by FET, gate and circuit decks.
IMPLEMENTED (shared foundation, maintained by P1).

A complete deck in this project looks like (SPEC §4.2):

    * <title: the first line of a SPICE deck is always ignored>
    .options gmin=1e-18 abstol=1e-18 reltol=1e-6 vntol=1e-9      <- constants.SPICE_OPTIONS
    .temp <T in degC>
    .model nmos nmos level=54 ... (corner/sample-specific vth0, toxe, ndep, u0, xj)
    .model pmos pmos level=54 ...
    <sources + devices>
    .control
    set numdgt=12
    op
    print <exprs>
    .endc
    .end

Typical use (see spice/fet_netlist.py for a complete example):

    deck = "\n".join([
        netlist_header("NAND2 01 corner 3", corner["temp"]),
        model_block(card, nmos_params, pmos_params),
        ...sources and devices...,
        op_control_block(["i(vvdd)", "v(out)", "v(x1.x)"]),
        END,
    ])
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from leakml.constants import SPICE_OPTIONS
from leakml.pdk import ModelCard, format_number, render_model_card

END = ".end\n"
PRINT_CHUNK = 8  # expressions per `print` line (keeps lines short; output format is identical)

fmt = format_number  # re-export: one numeric formatter for every deck


def netlist_header(title: str, temp_c: float) -> str:
    """Title line, the mandatory .options line, and .temp."""
    title = " ".join(str(title).split())  # single line
    if not title.startswith("*"):
        title = "* " + title
    return f"{title}\n{SPICE_OPTIONS}\n.temp {fmt(temp_c)}"


def model_block(
    card: ModelCard,
    nmos_params: Mapping[str, float] | None,
    pmos_params: Mapping[str, float] | None,
) -> str:
    """`.model` blocks for every device whose params are given (one draw per device type)."""
    parts = []
    if nmos_params is not None:
        parts.append(render_model_card(card, "nmos", nmos_params))
    if pmos_params is not None:
        parts.append(render_model_card(card, "pmos", pmos_params))
    if not parts:
        raise ValueError("model_block needs at least one of nmos_params / pmos_params")
    return "\n".join(parts)


def op_control_block(print_exprs: Sequence[str]) -> str:
    """`.control` block: one DC operating point, then print each expression as `name = value`.

    Keep in sync with parse.parse_print_output (verified on ngspice 44.2 and 47).
    """
    if not print_exprs:
        raise ValueError("nothing to print")
    exprs = [e.lower() for e in print_exprs]
    prints = [
        "print " + " ".join(exprs[i : i + PRINT_CHUNK]) for i in range(0, len(exprs), PRINT_CHUNK)
    ]
    return "\n".join([".control", "set numdgt=12", "op", *prints, ".endc"])
