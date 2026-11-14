"""Dataset-size study ("smallest dataset possible", brief Part A.1).  Owner: P1.

The training sizes are nested subsets of ONE seeded shuffle of the training split, so a bigger n
always contains every row of a smaller n and the curve shows only the effect of adding data.
Validation and test rows never enter a subset (hard rules 1 and 2).
"""

from __future__ import annotations

import logging
import time
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

from leakml.constants import TARGET_COLUMNS, Case
from leakml.sampling import make_rng

log = logging.getLogger(__name__)


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


def fit_fixed_on_subsets(
    case: Case | str,
    df: pd.DataFrame,
    train_idx: Sequence[int],
    val_idx: Sequence[int],
    sizes: Sequence[int],
    *,
    model_name: str,
    hyperparams: Mapping[str, Any],
    target_kind: str,
    shape_channel: bool,
    is_kcl: bool,
    nominal: Mapping[str, float] | None,
    split_seed: int,
    train_seed: int,
    i_floor: float,
) -> pd.DataFrame:
    """Learning curve with the CHOSEN model held fixed: one row per (n_train, target).

    The model type, hyperparameters, channel shaping and KCL choice are the ones step 05 saved, so
    only n changes along the curve (no re-tuning per n). The target transform is re-fitted on each
    subset, as step 04 does on the full training split. Every n is scored on the same validation
    rows; the test split is never read (hard rule 2).
    """
    from leakml.metrics import summarize
    from leakml.ml import zoo
    from leakml.ml.features import TargetTransform, build_features
    from leakml.ml.regressor import is_from_kcl

    case = Case(case)
    targets = list(TARGET_COLUMNS)
    val = df.loc[list(val_idx)]
    X_va = build_features(val, case, nominal)
    hp = {k: v for k, v in hyperparams.items() if k in zoo.default_hyperparams(model_name)}

    rows = []
    train_idx = np.asarray(train_idx, dtype=np.int64)
    for n, rows_n in learning_curve_subsets(train_idx, sizes, split_seed, case).items():
        # Fit in split order (the subset decides WHICH rows, not their order): seeded bootstrap /
        # subsampling depend on row order, and this way n = all reproduces step 04 exactly.
        train = df.loc[train_idx[np.isin(train_idx, rows_n)]]
        tt = TargetTransform(kind=target_kind, shape_channel=shape_channel)
        tt.fit(train[targets], train)
        t0 = time.perf_counter()
        est = zoo.make_model(model_name, train_seed, **hp)
        est.fit(build_features(train, case, nominal), tt.transform(train[targets], train))
        dt = time.perf_counter() - t0
        pred = tt.inverse_transform(est.predict(X_va), val).to_numpy()
        if is_kcl:
            pred = is_from_kcl(pred)
        for i, t in enumerate(targets):
            s = summarize(pred[:, i], val[t].to_numpy(), i_floor=i_floor)
            rows.append({"case": case.value, "n_train": n, "target": t, **s, "fit_s": dt})
        log.info("%s n=%d fitted in %.1fs", case.value, n, dt)
    return pd.DataFrame(rows)
