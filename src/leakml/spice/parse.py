"""Parse ngspice batch-mode stdout.  IMPLEMENTED (shared foundation, maintained by P1)."""

from __future__ import annotations

import re
from collections.abc import Mapping

# `i(vd) = -6.75118037774e-09`, `v(x1.x) = 6.719982934379e-01`. Anything with trailing text
# ("Doing analysis at TEMP = 87.3 and TNOM = 27") deliberately does not match.
_LINE = re.compile(r"^\s*([^\s=]+)\s*=\s*([-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)\s*$")


def parse_print_output(stdout: str) -> dict[str, float]:
    """{expr: value} for every scalar printed by `print` after `op`. Keys are lowercase.

    Only the scalar `name = value` format is supported (that is what `op` + `print` gives).
    Vectors (e.g. from a DC sweep) print as a table and are NOT parsed here.
    """
    values: dict[str, float] = {}
    for line in stdout.splitlines():
        m = _LINE.match(line)
        if m:
            values[m.group(1).lower()] = float(m.group(2))
    return values


def currents_into_device(
    values: Mapping[str, float], source_names: Mapping[str, str]
) -> dict[str, float]:
    """Source currents -> currents flowing INTO the device terminals (SPEC §3.2).

    ngspice reports i(Vx) as current flowing from the + node through the source to the - node, so
    with `Vd d 0 dc ...` the current into the drain is I_d = -i(vd).
    source_names maps terminal -> source, e.g. {"id": "vd", "ig": "vg", "is": "vs", "ib": "vb"}.
    """
    out = {}
    for term, src in source_names.items():
        key = f"i({src.lower()})"
        if key not in values:
            raise KeyError(f"{key} missing from ngspice output (have: {sorted(values)})")
        out[term] = -values[key]
    return out
