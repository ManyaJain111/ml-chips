"""Fit, compare, select and save the transistor models (Deliverables 4 and 5).  Owner: P2."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from leakml.constants import Case


@dataclass
class FitResult:
    case: Case
    model_name: str
    estimator: Any  # fitted Pipeline
    target_transform: Any  # fitted features.TargetTransform
    feature_names: list[str]
    hyperparams: dict[str, Any]
    val_metrics: pd.DataFrame = field(default_factory=pd.DataFrame)  # one row per target
    fit_seconds: float = 0.0


def fit_one(case: Case, model_name: str, splits: dict[str, pd.DataFrame], seed: int) -> FitResult:
    """Fit one model type on one case and score it on the validation split.

    TODO(P2):
      - X = features.build_features(train), Z = target_transform.fit(...).transform(train[TARGETS])
      - optional hyperparameter search on val (zoo.search_space)
      - predict on val, inverse_transform back to amperes, then for each target:
        metrics.summarize(pred, true, i_floor=cfg.ml.i_floor_a) -> R2 (log), median err, p95 err
    """
    raise NotImplementedError


def compare_models(
    case: Case, splits: dict[str, pd.DataFrame], model_names: tuple[str, ...], seed: int
) -> list[FitResult]:
    """TODO(P2): fit_one for every model type and return all results (needed for the table)."""
    raise NotImplementedError


def comparison_table(results: dict[Case, list[FitResult]]) -> pd.DataFrame:
    """Deliverable 4: 4 model types x 4 cases, with R2, median error and p95 error on VALIDATION.

    TODO(P2):
      - long format: case, model, target, r2_log, median_err_pct, p95_err_pct
      - plus a compact pivot for the report (rows = case x model, metrics averaged over targets or
        shown for the leakage-dominant target). Say in the report which one is shown
      - write results/tables/model_comparison.{csv,md}
    """
    raise NotImplementedError


def select_best(results: list[FitResult], metric: str = "p95_err_pct") -> FitResult:
    """Pick the best model for one case on the VALIDATION metric (cfg.ml.selection_metric).

    TODO(P2): break ties by median error, then by speed or simplicity. Write the reason down for
    the report.
    """
    raise NotImplementedError


def evaluate_on_test(best: FitResult, test_df: pd.DataFrame) -> pd.DataFrame:
    """Deliverable 5: test-set median and p95 error per target for the CHOSEN model.

    TODO(P2): the test split is touched here and nowhere else. Also produce one predicted-vs-SPICE
    log-log parity plot per case (plotting.parity_plot_log, all 4 targets in subplots or 4 files).
    """
    raise NotImplementedError


def save_model(best: FitResult, out_path: Path) -> None:
    """Save artifacts/models/{case}.joblib (Deliverable 5: "the four saved model files").

    TODO(P2): the file format is FIXED by predictor.py (SPEC §4.6). Use
        predictor.save_bundle(out_path, best.case, regressor, meta)
    where `regressor` is ONE picklable object whose .predict(X_raw: DataFrame[INPUT_COLUMNS])
    returns an (n, 4) array of currents in A (id, ig, is, ib, into-device sign). Write a small
    wrapper class in leakml/ml/ (NOT in a script) that holds build_features + the fitted
    estimator + TargetTransform and does the inverse transform inside predict().
    meta: model_name, hyperparams, feature_names, train_ranges=predictor.input_ranges(X_train),
    library versions, cfg.snapshot(), val/test metrics.
    See ml/baseline.py + scripts/dev_train_baseline.py for a working minimal example.
    """
    raise NotImplementedError
