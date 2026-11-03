"""CSV and table I/O shared by every stage.  IMPLEMENTED (shared foundation).

Use these instead of bare `pd.read_csv` / `to_csv`:
  - read_csv keeps bit-string columns ("inputs", "vector") as strings. Plain pandas would turn
    "01" into the integer 1 and "0010110" into 10110, which silently breaks lookups.
  - write_csv writes floats at full precision. Leakage currents are ~1e-12 A, and the default
    float format loses digits.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

# Columns that hold identifiers or bit strings and must never be parsed as numbers.
STRING_COLUMNS = {"inputs": str, "vector": str, "gate": str, "circuit": str, "case": str}

FLOAT_FORMAT = "%.10e"


def read_csv(path: str | Path) -> pd.DataFrame:
    """Read a project CSV, keeping bit-string/identifier columns as str."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"{path} does not exist. Did the step that produces it run?")
    header = pd.read_csv(path, nrows=0).columns
    dtypes = {c: t for c, t in STRING_COLUMNS.items() if c in header}
    return pd.read_csv(path, dtype=dtypes)


def write_csv(df: pd.DataFrame, path: str | Path) -> Path:
    """Write `df` (no index) at full float precision, creating parent dirs."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, float_format=FLOAT_FORMAT)
    return path


def save_table(df: pd.DataFrame, out_stem: str | Path, floatfmt: str = ".4g") -> tuple[Path, Path]:
    """Write a results table as `<out_stem>.csv` (full precision) and `<out_stem>.md` (for the report)."""
    out_stem = Path(out_stem)
    out_stem.parent.mkdir(parents=True, exist_ok=True)
    csv_path = Path(f"{out_stem}.csv")
    md_path = Path(f"{out_stem}.md")
    df.to_csv(csv_path, index=False, float_format=FLOAT_FORMAT)
    md_path.write_text(df.to_markdown(index=False, floatfmt=floatfmt) + "\n")
    return csv_path, md_path
