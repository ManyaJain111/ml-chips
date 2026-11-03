"""PTM / BSIM4 model-card handling.  IMPLEMENTED (shared foundation, maintained by P1).

The assigned model file (e.g. pdk/22nm_HP.pm) is a Predictive Technology Model card with two
`.model` statements (`nmos` and `pmos`, level=54 = BSIM4). Continuation lines start with `+`.

    card = load_model_card(cfg.assignment.model_file)
    card.nominal["pmos"]["vth0"]                          # -> negative for PMOS
    text = render_model_card(card, "nmos", {"vth0": 0.45, "toxe": 1.2e-9, ...})
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from leakml.constants import PROCESS_PARAMS

Device = Literal["nmos", "pmos"]
DEVICES: tuple[Device, ...] = ("nmos", "pmos")

_NUM = r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?[a-zA-Z]*"
_MODEL_RE = re.compile(r"^\s*\.model\s+(\S+)\s+(nmos|pmos)\b", re.IGNORECASE)
_SCALE = {"t": 1e12, "g": 1e9, "k": 1e3, "m": 1e-3, "u": 1e-6, "n": 1e-9, "p": 1e-12, "f": 1e-15}


def _param_re(name: str) -> re.Pattern[str]:
    # (?<![\w.]) so that e.g. "vth0" never matches inside "dvth0" and "u0" not inside "pu0".
    return re.compile(rf"((?<![\w.]){re.escape(name)}\s*=\s*)({_NUM})", re.IGNORECASE)


def parse_spice_number(token: str) -> float:
    """'1.05e-009' -> 1.05e-9, '45n' -> 4.5e-8, '1meg' -> 1e6. Unknown unit letters are ignored (SPICE rule)."""
    m = re.fullmatch(r"([-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)([a-zA-Z]*)", token.strip())
    if not m:
        raise ValueError(f"not a SPICE number: {token!r}")
    value, suffix = float(m.group(1)), m.group(2).lower()
    if suffix.startswith("meg"):
        return value * 1e6
    if suffix and suffix[0] in _SCALE:
        return value * _SCALE[suffix[0]]
    return value


def format_number(x: float) -> str:
    """Full-precision number for netlists (10 significant digits)."""
    return f"{float(x):.10g}"


@dataclass(frozen=True)
class ModelCard:
    path: Path
    model_names: dict[str, str]  # {"nmos": "nmos", "pmos": "pmos"}, as named in the file
    nmos_text: str  # `.model` line + its '+' continuation lines (comments/blank lines dropped)
    pmos_text: str
    nominal: dict[str, dict[str, float]]  # {"nmos": {"vth0": ..., ...}, "pmos": {...}}

    def block(self, device: Device) -> str:
        if device not in DEVICES:
            raise ValueError(f"device must be one of {DEVICES}, got {device!r}")
        return self.nmos_text if device == "nmos" else self.pmos_text


def _split_blocks(text: str) -> dict[str, tuple[str, str]]:
    """{device: (model_name, block_text)} for every `.model <name> nmos|pmos` statement."""
    lines = text.splitlines()
    blocks: dict[str, tuple[str, str]] = {}
    i = 0
    while i < len(lines):
        m = _MODEL_RE.match(lines[i])
        if not m:
            i += 1
            continue
        name, device = m.group(1), m.group(2).lower()
        kept = [lines[i].rstrip()]
        j = i + 1
        while j < len(lines):
            s = lines[j].strip()
            if s.startswith("+"):
                kept.append(lines[j].rstrip())
            elif s and not s.startswith("*"):
                break  # next statement
            j += 1
        if device in blocks:
            raise ValueError(f"more than one .model block of type {device}")
        blocks[device] = (name, "\n".join(kept))
        i = j
    return blocks


def _extract(block: str, device: str, param: str) -> float:
    hits = _param_re(param).findall(block)
    if len(hits) != 1:
        raise ValueError(f"{device} model: expected exactly one '{param} = ...', found {len(hits)}")
    return parse_spice_number(hits[0][1])


def load_model_card(path: Path | str) -> ModelCard:
    """Parse the PTM file and pull the nominal values of constants.PROCESS_PARAMS for both devices."""
    path = Path(path)
    blocks = _split_blocks(path.read_text(errors="replace"))
    missing = [d for d in DEVICES if d not in blocks]
    if missing:
        raise ValueError(f"{path}: no .model block for {missing} (expected one nmos and one pmos)")
    nominal = {d: {p: _extract(blocks[d][1], d, p) for p in PROCESS_PARAMS} for d in DEVICES}
    return ModelCard(
        path=path,
        model_names={d: blocks[d][0] for d in DEVICES},
        nmos_text=blocks["nmos"][1],
        pmos_text=blocks["pmos"][1],
        nominal=nominal,
    )


def render_model_card(
    card: ModelCard,
    device: Device,
    params: Mapping[str, float],
    model_name: str | None = None,
) -> str:
    """The `.model` block for `device` with the given process parameters substituted.

    `params` may hold any subset of PROCESS_PARAMS (raw values: PMOS vth0 stays negative).
    Each given parameter must occur exactly once in the block.
    """
    text = card.block(device)
    for p, v in params.items():
        if p not in PROCESS_PARAMS:
            raise KeyError(f"{p!r} is not a process parameter {PROCESS_PARAMS}")
        text, n = _param_re(p).subn(lambda m, v=v: m.group(1) + format_number(v), text)
        if n != 1:
            raise ValueError(f"{device} model: substituted '{p}' {n} times (expected 1)")
    if model_name is not None:
        text = _MODEL_RE.sub(lambda m: m.group(0).replace(m.group(1), model_name, 1), text, count=1)
    return text


def dump_nominals(card: ModelCard, out_path: Path) -> None:
    """Write artifacts/nominals.json (used by the report's Setup section)."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_file": str(card.path),
        "model_names": card.model_names,
        "nominal": card.nominal,
    }
    out_path.write_text(json.dumps(payload, indent=2) + "\n")
