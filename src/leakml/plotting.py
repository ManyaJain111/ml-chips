"""Shared matplotlib helpers, so every figure in the report looks the same.
IMPLEMENTED (shared foundation, maintained by P1).

    parity_plot_log(spice, ml, plots_dir / "gate_parity", title="Gates", hue=df["gate"])
    fig, axes = plt.subplots(2, 2); parity_axes(axes[0, 0], true, pred, title="id"); save_fig(fig, out)
    trend_scatter(df.temp, df.id, plots_dir / "trend_ioff_vs_temp", xlabel=..., ylabel=..., title=...)

Headless (Agg backend). Every figure is saved as PNG (300 dpi) and PDF.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from leakml.metrics import summarize  # noqa: E402

MAX_CATEGORIES = 12


def save_fig(fig, out_path: str | Path) -> tuple[Path, Path]:
    """Save `fig` as <out_path>.png and <out_path>.pdf (any .png/.pdf suffix is ignored), then close it."""
    out_path = Path(out_path)
    stem = out_path.with_suffix("") if out_path.suffix in (".png", ".pdf") else out_path
    stem.parent.mkdir(parents=True, exist_ok=True)
    png, pdf = Path(f"{stem}.png"), Path(f"{stem}.pdf")
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return png, pdf


def _is_categorical(hue: np.ndarray) -> bool:
    if hue.dtype.kind in "OSUb":
        return True
    return hue.dtype.kind in "iu" and len(np.unique(hue)) <= MAX_CATEGORIES


def _scatter(ax, x, y, hue, hue_label, **kw):
    if hue is None:
        ax.scatter(x, y, **kw)
        return
    hue = np.asarray(hue)
    if _is_categorical(hue):
        for cat in dict.fromkeys(hue.tolist()):  # first-appearance order
            m = hue == cat
            ax.scatter(x[m], y[m], label=str(cat), **kw)
        ax.legend(title=hue_label, fontsize=7, markerscale=2, loc="lower right")
    else:
        sc = ax.scatter(x, y, c=hue, cmap="viridis", **kw)
        ax.figure.colorbar(sc, ax=ax, label=hue_label)


def parity_axes(
    ax,
    true: np.ndarray,
    pred: np.ndarray,
    *,
    title: str = "",
    hue: np.ndarray | None = None,
    hue_label: str | None = None,
    bands_pct: tuple[float, ...] = (1.0, 5.0),
    i_floor: float | None = None,
    quantity: str = "|I| (A)",
) -> dict[str, float]:
    """Draw a log-log ML-vs-SPICE parity plot on `ax`. Returns the metrics shown in the corner box."""
    true = np.asarray(true, dtype=float)
    pred = np.asarray(pred, dtype=float)
    stats = summarize(pred, true, i_floor=i_floor)
    sign_mismatch = int(np.sum(np.sign(true) != np.sign(pred)))

    at, ap = np.abs(true), np.abs(pred)
    ok = (at > 0) & (ap > 0) & np.isfinite(at) & np.isfinite(ap)
    hue_ok = None if hue is None else np.asarray(hue)[ok]
    _scatter(ax, at[ok], ap[ok], hue_ok, hue_label, s=6, alpha=0.5, linewidths=0)

    if ok.any():
        lo, hi = min(at[ok].min(), ap[ok].min()) / 1.5, max(at[ok].max(), ap[ok].max()) * 1.5
        line = np.array([lo, hi])
        ax.plot(line, line, color="black", lw=0.8)
        for b, ls in zip(bands_pct, ("--", ":", "-."), strict=False):
            ax.plot(line, line * (1 + b / 100), color="grey", lw=0.6, ls=ls, label=None)
            ax.plot(line, line * (1 - b / 100), color="grey", lw=0.6, ls=ls)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(f"SPICE {quantity}")
    ax.set_ylabel(f"ML {quantity}")
    ax.set_title(title)
    txt = (
        f"n = {stats['n']}\n"
        f"median err = {stats['median_err_pct']:.3g} %\n"
        f"p95 err = {stats['p95_err_pct']:.3g} %\n"
        f"mean acc = {stats['mean_acc_pct']:.4g} %"
    )
    if sign_mismatch:
        txt += f"\nsign mismatch = {sign_mismatch}"
    ax.text(0.03, 0.97, txt, transform=ax.transAxes, va="top", fontsize=7,
            bbox={"boxstyle": "round", "fc": "white", "alpha": 0.8})  # fmt: skip
    ax.grid(True, which="major", lw=0.3, alpha=0.5)
    return stats


def parity_plot_log(
    true: np.ndarray,
    pred: np.ndarray,
    out_path: str | Path,
    *,
    title: str,
    hue: np.ndarray | None = None,
    hue_label: str | None = None,
    bands_pct: tuple[float, ...] = (1.0, 5.0),
    i_floor: float | None = None,
    quantity: str = "|I| (A)",
) -> dict[str, float]:
    """ML-vs-SPICE parity plot on log-log axes (Deliverables 5, 7, 8). Returns the metrics shown."""
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    stats = parity_axes(
        ax, true, pred, title=title, hue=hue, hue_label=hue_label,
        bands_pct=bands_pct, i_floor=i_floor, quantity=quantity,
    )  # fmt: skip
    save_fig(fig, out_path)
    return stats


def trend_scatter(
    x: np.ndarray,
    y: np.ndarray,
    out_path: str | Path,
    *,
    xlabel: str,
    ylabel: str,
    title: str,
    logy: bool = True,
    hue: np.ndarray | None = None,
    hue_label: str | None = None,
    n_bins: int = 20,
) -> None:
    """Scatter + binned-median line (Deliverable 3 trend plots). With logy, |y| is plotted."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if logy:
        y = np.abs(y)
    ok = np.isfinite(x) & np.isfinite(y) & ((y > 0) if logy else True)
    x, y = x[ok], y[ok]
    hue = None if hue is None else np.asarray(hue)[ok]

    fig, ax = plt.subplots(figsize=(6, 4.2))
    _scatter(ax, x, y, hue, hue_label, s=4, alpha=0.25, linewidths=0)
    if x.size:
        edges = np.linspace(x.min(), x.max(), n_bins + 1)
        idx = np.clip(np.digitize(x, edges) - 1, 0, n_bins - 1)
        centers, medians = [], []
        for b in range(n_bins):
            m = idx == b
            if m.sum() >= 3:
                centers.append(0.5 * (edges[b] + edges[b + 1]))
                medians.append(np.median(y[m]))
        ax.plot(centers, medians, color="crimson", lw=1.5, marker="o", ms=3, label="binned median")
        ax.legend(fontsize=7, loc="best")
    if logy:
        ax.set_yscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, lw=0.3, alpha=0.5)
    save_fig(fig, out_path)
