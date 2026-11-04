"""Train / validation / test split (brief Part A step 2).  Owner: P2 (written by P1, agreed with P2)."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from leakml.sampling import make_rng

SPLIT_FRACTIONS = {"70_15_15": (0.70, 0.15, 0.15), "80_10_10": (0.80, 0.10, 0.10)}
SPLIT_NAMES = ("train", "val", "test")


def split_dataset(
    df: pd.DataFrame, scheme: Literal["70_15_15", "80_10_10"], seed: int
) -> dict[str, pd.DataFrame]:
    """Shuffle with a fixed seed and cut into {"train", "val", "test"}.

    10k rows -> 7000/1500/1500 (70_15_15) or 8000/1000/1000 (80_10_10). Every case uses the same
    seed, which is fine because the datasets are independent. The original row index of df is
    kept, so save_split_indices records which CSV rows went where.
    """
    if scheme not in SPLIT_FRACTIONS:
        raise ValueError(
            f"unknown split scheme {scheme!r}, expected one of {list(SPLIT_FRACTIONS)}"
        )
    f_train, f_val, _ = SPLIT_FRACTIONS[scheme]
    n = len(df)
    perm = make_rng(seed, "split").permutation(n)
    n_train = round(f_train * n)
    n_val = round(f_val * n)
    cuts = np.split(perm, [n_train, n_train + n_val])
    return {name: df.iloc[idx] for name, idx in zip(SPLIT_NAMES, cuts, strict=True)}


def save_split_indices(splits: dict[str, pd.DataFrame], out_path: Path) -> None:
    """Save the row indices of each split to artifacts/splits/{case}.npz (reproducible, auditable)."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_path, **{name: splits[name].index.to_numpy(dtype=np.int64) for name in SPLIT_NAMES}
    )


def load_split_indices(path: Path) -> dict[str, list[int]]:
    """Inverse of save_split_indices."""
    with np.load(Path(path)) as z:
        return {name: z[name].tolist() for name in SPLIT_NAMES}
