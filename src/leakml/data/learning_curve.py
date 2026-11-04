"""Dataset-size study ("smallest dataset possible", brief Part A.1).  Owner: P1.

The training sizes are nested subsets of ONE seeded shuffle of the training split, so a bigger n
always contains every row of a smaller n and the curve shows only the effect of adding data.
Validation and test rows never enter a subset (hard rules 1 and 2).
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from leakml.constants import Case
from leakml.sampling import make_rng


def learning_curve_subsets(
    train_idx: Sequence[int], sizes: Sequence[int], seed: int, case: Case | str
) -> dict[int, np.ndarray]:
    """{n: row indices} for each n in sizes, all drawn from train_idx and nested.

    Sizes larger than the training split are capped at len(train_idx), so the last point of the
    curve is always "the whole training set" (7000 rows for 10k samples and a 70_15_15 split).
    """
    train_idx = np.asarray(train_idx, dtype=np.int64)
    order = make_rng(seed, Case(case).value, "learning_curve").permutation(train_idx)
    sizes = sorted({min(int(n), len(order)) for n in sizes})
    if sizes and sizes[0] <= 0:
        raise ValueError(f"training sizes must be positive, got {sizes}")
    return {n: order[:n] for n in sizes}
