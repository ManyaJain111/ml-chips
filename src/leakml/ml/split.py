"""Train / validation / test split (brief Part A step 2).  Owner: P2."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import pandas as pd

SPLIT_FRACTIONS = {"70_15_15": (0.70, 0.15, 0.15), "80_10_10": (0.80, 0.10, 0.10)}


def split_dataset(
    df: pd.DataFrame, scheme: Literal["70_15_15", "80_10_10"], seed: int
) -> dict[str, pd.DataFrame]:
    """Shuffle with a fixed seed and cut into {"train", "val", "test"}.

    TODO(P2):
      - rng = np.random.default_rng(seed); perm = rng.permutation(len(df))
      - cut at the fractions in SPLIT_FRACTIONS (10k rows -> 7000/1500/1500 or 8000/1000/1000)
      - every case uses the same seed, which is fine because the datasets are independent
    """
    raise NotImplementedError


def save_split_indices(splits: dict[str, pd.DataFrame], out_path: Path) -> None:
    """Save the row indices of each split to artifacts/splits/{case}.npz (reproducible, auditable).

    TODO(P2)
    """
    raise NotImplementedError


def load_split_indices(path: Path) -> dict[str, list[int]]:
    """TODO(P2): inverse of save_split_indices."""
    raise NotImplementedError
