"""Fit, compare, select and save the transistor models (Deliverables 4 and 5).  Owner: P2.

Flow (scripts 04 and 05):
    splits = split.split_dataset(df, scheme, seeds.split)
    results[case] = compare_models(case, splits, MODEL_TYPES, seeds.training, nominal=..., i_floor=...)
    comparison_table(results)                    -> D4 (validation only)
    best = select_best(results[case], metric)    -> validation only
    evaluate_on_test(best, splits["test"], ...)  -> D5, the ONLY place the test split is used
    save_model(best, models_dir / f"{case}.joblib", extra_meta=...)

The keyword-only arguments (nominal, i_floor, tune, plot_path, extra_meta) carry config values;
the positional signatures are unchanged from the scaffold.
"""

from __future__ import annotations

import itertools
import logging
import platform
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from leakml import plotting
from leakml.constants import TARGET_COLUMNS, Case
from leakml.metrics import summarize
from leakml.ml import zoo
from leakml.ml.features import TargetTransform, build_features
from leakml.ml.predictor import input_ranges, save_bundle
from leakml.ml.regressor import TransistorRegressor, is_from_kcl

log = logging.getLogger(__name__)

DEFAULT_I_FLOOR = 1e-15  # fallback only; scripts pass cfg.ml.i_floor_a
TARGET_KIND = "asinh"  # see features.TargetTransform for the justification
# Cases whose id/is are divided by tanh(vds/2vT) (features.TargetTransform.shape_channel)
SHAPED_CASES: frozenset[Case] = frozenset({Case.NMOS_ON, Case.PMOS_ON})


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
    nominal: dict[str, float] | None = None  # passed to build_features (saved with the model)
    train_ranges: dict[str, list[float]] = field(default_factory=dict)  # raw-input ranges
    n_train: int = 0
    search: pd.DataFrame = field(default_factory=pd.DataFrame)  # one row per grid point
    is_kcl: bool = False  # is = -(id + ig + ib) instead of the direct prediction

    def regressor(self) -> TransistorRegressor:
        return TransistorRegressor(
            self.case, self.estimator, self.target_transform, self.feature_names, self.nominal,
            is_kcl=getattr(self, "is_kcl", False),
        )  # fmt: skip

    def score(self, metric: str = "p95_err_pct") -> float:
        return _score(self.val_metrics, metric)


_SEARCH_META = ("is_kcl", "median_err_pct", "fit_s")


def n_grid_points(search: pd.DataFrame) -> int:
    """Number of distinct hyperparameter settings in a FitResult.search table (1 = untuned).

    The table has one row per (grid point, is_kcl variant) plus score columns, so count the
    distinct hyperparameter combinations only."""
    hp = [c for c in search.columns if c not in _SEARCH_META and not c.endswith("_pct")]
    return len(search[hp].astype(str).drop_duplicates()) if hp else 1


def _score(metrics: pd.DataFrame, metric: str) -> float:
    """Mean of `metric` over the targets that have one (e.g. OFF-state ig is all below i_floor,
    so it has no relative error and is skipped). Lower is better."""
    vals = metrics[metric].to_numpy(dtype=float)
    vals = vals[np.isfinite(vals)]
    return float(vals.mean()) if vals.size else float("inf")


def _target_metrics(pred: pd.DataFrame, true: pd.DataFrame, i_floor: float) -> pd.DataFrame:
    rows = [
        {"target": t, **summarize(pred[t].to_numpy(), true[t].to_numpy(), i_floor=i_floor)}
        for t in TARGET_COLUMNS
    ]
    return pd.DataFrame(rows)


def fit_one(
    case: Case,
    model_name: str,
    splits: dict[str, pd.DataFrame],
    seed: int,
    *,
    nominal: dict[str, float] | None = None,
    i_floor: float = DEFAULT_I_FLOOR,
    tune: bool = True,
    selection_metric: str = "p95_err_pct",
    shape_channel: bool | None = None,
    is_kcl: bool | None = None,
) -> FitResult:
    """Fit one model type on one case and score it on the validation split.

    With tune=True every point of zoo.search_space(model_name) is fitted on TRAIN and scored on
    VAL (selection_metric averaged over targets); the best point is kept. The test split is never
    read here. shape_channel=None means "on for the ON cases" (see features.TargetTransform).
    is_kcl=None scores every grid point twice on VAL, with is predicted directly and with
    is = -(id + ig + ib) (regressor.is_from_kcl), and keeps the better variant; True/False force it.
    """
    case = Case(case)
    train, val = splits["train"], splits["val"]
    targets = list(TARGET_COLUMNS)

    X_tr = build_features(train, case, nominal)
    X_va = build_features(val, case, nominal)
    if shape_channel is None:
        shape_channel = case in SHAPED_CASES
    tt = TargetTransform(kind=TARGET_KIND, shape_channel=shape_channel).fit(train[targets], train)
    Z_tr = tt.transform(train[targets], train)

    space = zoo.search_space(model_name) if tune else {}
    grid = [dict(zip(space, combo, strict=True)) for combo in itertools.product(*space.values())]
    grid = grid or [{}]

    best: tuple[float, float, dict, Any, pd.DataFrame, float] | None = None
    rows = []
    for overrides in grid:
        t0 = time.perf_counter()
        est = zoo.make_model(model_name, seed, **overrides).fit(X_tr, Z_tr)
        dt = time.perf_counter() - t0
        direct = tt.inverse_transform(est.predict(X_va), val)
        variants = [False, True] if is_kcl is None else [bool(is_kcl)]
        for kcl in variants:
            pred = direct
            if kcl:
                pred = pd.DataFrame(is_from_kcl(direct.to_numpy()), index=val.index,
                                    columns=targets)  # fmt: skip
            m = _target_metrics(pred, val[targets], i_floor)
            score = _score(m, selection_metric)
            med = _score(m, "median_err_pct")
            rows.append({**overrides, "is_kcl": kcl, selection_metric: score,
                         "median_err_pct": med, "fit_s": dt})  # fmt: skip
            log.info("%s %s %s is_kcl=%s: val %s %.4g (%.1fs)", case.value, model_name,
                     overrides, kcl, selection_metric, score, dt)  # fmt: skip
            if best is None or (score, med) < (best[0], best[1]):
                best = (score, med, overrides, est, m, dt, kcl)

    assert best is not None
    _, _, overrides, est, m, dt, kcl = best
    hp = zoo.default_hyperparams(model_name)
    hp.update(overrides)
    return FitResult(
        case=case,
        model_name=model_name,
        estimator=est,
        target_transform=tt,
        feature_names=list(X_tr.columns),
        hyperparams=hp,
        val_metrics=m,
        fit_seconds=dt,
        nominal=dict(nominal) if nominal is not None else None,
        train_ranges=input_ranges(train),
        n_train=len(train),
        search=pd.DataFrame(rows),
        is_kcl=kcl,
    )


def compare_models(
    case: Case,
    splits: dict[str, pd.DataFrame],
    model_names: tuple[str, ...],
    seed: int,
    **fit_kwargs: Any,
) -> list[FitResult]:
    """fit_one for every model type; fit_kwargs (nominal, i_floor, tune, ...) are passed through."""
    return [fit_one(case, name, splits, seed, **fit_kwargs) for name in model_names]


def comparison_table(results: dict[Case, list[FitResult]]) -> pd.DataFrame:
    """Deliverable 4 (long format): one row per case x model x target, VALIDATION metrics.

    Columns: case, model, target, r2_log, median_err_pct, p95_err_pct, plus the other
    metrics.METRIC_KEYS and fit_seconds. For the compact report table use comparison_summary().
    """
    rows = []
    for case, fits in results.items():
        for r in fits:
            for rec in r.val_metrics.to_dict("records"):
                rows.append(
                    {"case": Case(case).value, "model": r.model_name, **rec,
                     "fit_seconds": r.fit_seconds}
                )  # fmt: skip
    df = pd.DataFrame(rows)
    first = ["case", "model", "target", "r2_log", "median_err_pct", "p95_err_pct"]
    return df[first + [c for c in df.columns if c not in first]]


def comparison_summary(long: pd.DataFrame) -> pd.DataFrame:
    """Compact D4 table: rows = case x model, metrics AVERAGED over the targets that have a
    relative error (OFF-state ig is below i_floor everywhere and is skipped)."""
    g = long.groupby(["case", "model"], sort=False)
    return (
        g[["r2_log", "median_err_pct", "p95_err_pct", "mean_acc_pct"]]
        .mean()
        .join(g["fit_seconds"].first())
        .reset_index()
    )


def select_best(results: list[FitResult], metric: str = "p95_err_pct") -> FitResult:
    """Best model for one case on the VALIDATION metric (mean over targets, lower is better).

    Ties (within 1e-9) are broken by median error, then by fit time (faster/simpler wins).
    """
    if not results:
        raise ValueError("no fitted models to choose from")
    return min(
        results,
        key=lambda r: (round(r.score(metric), 9), round(r.score("median_err_pct"), 9),
                       r.fit_seconds),
    )  # fmt: skip


def evaluate_on_test(
    best: FitResult,
    test_df: pd.DataFrame,
    *,
    i_floor: float = DEFAULT_I_FLOOR,
    plot_path: Path | None = None,
) -> pd.DataFrame:
    """Deliverable 5: TEST median and p95 error per target for the CHOSEN model.

    The test split is touched here and nowhere else (hard rule 2). With plot_path, also writes a
    2x2 predicted-vs-SPICE log-log parity plot (one panel per target) via plotting.save_fig.
    """
    targets = list(TARGET_COLUMNS)
    pred = pd.DataFrame(best.regressor().predict(test_df), index=test_df.index, columns=targets)
    m = _target_metrics(pred, test_df[targets], i_floor)
    m.insert(0, "model", best.model_name)
    m.insert(0, "case", best.case.value)

    if plot_path is not None:
        fig, axes = plotting.plt.subplots(2, 2, figsize=(10, 10))
        for ax, t in zip(axes.ravel(), targets, strict=True):
            plotting.parity_axes(
                ax, test_df[t].to_numpy(), pred[t].to_numpy(),
                title=f"{best.case.value}: {t} ({best.model_name}, test)", i_floor=i_floor,
                quantity=f"|{t}| (A)",
            )  # fmt: skip
        fig.tight_layout()
        plotting.save_fig(fig, plot_path)
    return m


def _library_versions() -> dict[str, str]:
    import joblib
    import lightgbm
    import sklearn

    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit-learn": sklearn.__version__,
        "lightgbm": lightgbm.__version__,
        "joblib": joblib.__version__,
    }


def save_model(
    best: FitResult, out_path: Path, *, extra_meta: dict[str, Any] | None = None
) -> None:
    """Save artifacts/models/{case}.joblib (Deliverable 5: "the four saved model files").

    Bundle format fixed by predictor.py (SPEC §4.6): the regressor is a TransistorRegressor,
    whose predict(X_raw) returns (n, 4) currents in A. extra_meta (config snapshot, test metrics,
    selection reason, ...) is merged into meta.
    """
    meta: dict[str, Any] = {
        "model_name": best.model_name,
        "hyperparams": best.hyperparams,
        "feature_names": best.feature_names,
        "target_transform": {"kind": best.target_transform.kind,
                             "shape_channel": bool(getattr(best.target_transform,
                                                           "shape_channel", False)),
                             "i0": [float(x) for x in (best.target_transform.i0_
                                                       if best.target_transform.i0_ is not None
                                                       else [])]},
        "is_kcl": bool(getattr(best, "is_kcl", False)),
        "train_ranges": best.train_ranges,
        "n_train": best.n_train,
        "val_metrics": best.val_metrics.to_dict("records"),
        "fit_seconds": best.fit_seconds,
        "versions": _library_versions(),
    }  # fmt: skip
    meta.update(extra_meta or {})
    save_bundle(out_path, best.case, best.regressor(), meta)
