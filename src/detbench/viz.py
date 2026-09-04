"""Plots and annotated images. Every figure in this repo is produced here.

Matplotlib is optional: import it lazily so the library, the metric and the
tests all work on a machine that does not have it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .analysis.curves import OperatingPoint
from .analysis.per_class import ClassReport
from .coco_classes import CATEGORY_NAMES
from .metrics.coco_map import COCOResults

__all__ = [
    "MATPLOTLIB_AVAILABLE",
    "plot_accuracy_vs_latency",
    "plot_per_class_ap",
    "plot_pr_curves",
    "plot_error_breakdown",
    "plot_threshold_sweep",
    "draw_detections",
]

try:  # pragma: no cover - environment dependent
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    MATPLOTLIB_AVAILABLE = True
except ImportError:  # pragma: no cover
    plt = None  # type: ignore[assignment]
    MATPLOTLIB_AVAILABLE = False

_INK = "#1b1f23"
_MUTED = "#6a737d"
_GRID = "#dfe2e5"
_SERIES = ["#1f6feb", "#d1493a", "#2f9e44", "#8250df", "#c98a00", "#0b7285"]


def _require_matplotlib() -> None:
    if not MATPLOTLIB_AVAILABLE:
        raise RuntimeError("matplotlib is required to draw figures")


def _style(ax) -> None:
    ax.set_facecolor("white")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(_GRID)
    ax.tick_params(colors=_MUTED, labelsize=8, length=3)
    ax.grid(True, color=_GRID, linewidth=0.6, alpha=0.9)
    ax.set_axisbelow(True)
    ax.xaxis.label.set_color(_INK)
    ax.yaxis.label.set_color(_INK)


def _save(fig, path: Path, dpi: int = 110) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def _spread_labels(values: Sequence[float], minimum_gap: float) -> List[float]:
    """Push overlapping label positions apart while keeping their order.

    Quantised variants land within a couple of milliseconds and a point of mAP
    of each other, so their labels collide. This nudges them apart along one
    axis, which keeps the chart readable without moving the markers. Positions
    are recentred afterwards so the group does not drift off the plot.
    """
    if not values:
        return []
    order = sorted(range(len(values)), key=lambda i: values[i])
    placed = list(values)
    for step, i in enumerate(order[1:], start=1):
        previous = placed[order[step - 1]]
        if placed[i] - previous < minimum_gap:
            placed[i] = previous + minimum_gap
    drift = (
        (max(placed) + min(placed)) - (max(values) + min(values))
    ) / 2.0
    return [v - drift for v in placed]


def plot_accuracy_vs_latency(
    variants: Sequence[Mapping[str, object]],
    path: Path,
    latency_key: str = "p50_ms",
    title: str = "Accuracy versus latency",
) -> Path:
    """Scatter mAP against end-to-end latency for each variant.

    Marker area encodes the model file size, so the three axes a deployment
    decision actually turns on - accuracy, latency and footprint - are all
    visible at once.

    Args:
        variants: Dicts with ``name``, ``map``, ``latency`` and ``size_mb``.
        path: Output PNG path.
        latency_key: Label for the latency axis.
        title: Figure title.
    """
    _require_matplotlib()
    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    _style(ax)

    xs = [float(v["latency"]) for v in variants]
    ys = [float(v["map"]) * 100.0 for v in variants]
    for i, v in enumerate(variants):
        ax.scatter(
            xs[i],
            ys[i],
            s=40.0 + 18.0 * float(v.get("size_mb", 1.0)),
            color=_SERIES[i % len(_SERIES)],
            edgecolor="white",
            linewidth=1.2,
            zorder=3,
        )

    ax.set_xlabel(f"end-to-end latency per image, {latency_key} (ms)")
    ax.set_ylabel("COCO mAP @ 0.50:0.95 (%)")
    ax.set_title(title, color=_INK, fontsize=11, loc="left")
    ax.margins(x=0.22, y=0.16)

    # Label placement happens after the limits are settled, so collisions can
    # be resolved in pixels and the leader lines stay short.
    fig.canvas.draw()
    pixels = ax.transData.transform(list(zip(xs, ys)))
    label_py = _spread_labels([float(p[1]) for p in pixels], minimum_gap=13.0)
    x_mid = 0.5 * (min(xs) + max(xs))
    for i, v in enumerate(variants):
        right = xs[i] <= x_mid
        dx = 14.0 if right else -14.0
        ax.annotate(
            str(v["name"]),
            xy=(xs[i], ys[i]),
            xycoords="data",
            xytext=(dx, label_py[i] - float(pixels[i][1])),
            textcoords="offset points",
            fontsize=8.5,
            color=_INK,
            ha="left" if right else "right",
            va="center",
            arrowprops={
                "arrowstyle": "-",
                "color": _GRID,
                "linewidth": 0.9,
                "shrinkA": 0,
                "shrinkB": 4,
            },
        )
    return _save(fig, path)


def plot_per_class_ap(
    rows: Sequence[ClassReport],
    path: Path,
    limit: Optional[int] = None,
    title: str = "Per-class AP, sorted",
) -> Path:
    """Horizontal bar chart of per-class AP, best at the top."""
    _require_matplotlib()
    data = list(rows) if limit is None else list(rows)[:limit]
    height = max(3.0, 0.16 * len(data) + 0.9)
    fig, ax = plt.subplots(figsize=(6.6, height))
    _style(ax)

    names = [r.name for r in data][::-1]
    values = [r.ap * 100.0 for r in data][::-1]
    mean_ap = float(np.mean([r.ap for r in rows])) * 100.0

    colours = [
        _SERIES[0] if v >= mean_ap else _SERIES[1] for v in values
    ]
    ax.barh(range(len(values)), values, color=colours, height=0.72)
    ax.set_yticks(range(len(values)))
    ax.set_yticklabels(names, fontsize=6.5)
    ax.axvline(mean_ap, color=_MUTED, linewidth=1.0, linestyle="--")
    ax.annotate(
        f"mean {mean_ap:.1f}",
        (mean_ap, len(values) - 0.4),
        textcoords="offset points",
        xytext=(4, 0),
        fontsize=7.5,
        color=_MUTED,
    )
    ax.set_xlabel("AP @ 0.50:0.95 (%)")
    ax.set_title(title, color=_INK, fontsize=11, loc="left")
    ax.set_ylim(-0.8, len(values) - 0.2)
    return _save(fig, path)


def plot_pr_curves(
    results: COCOResults,
    category_ids: Sequence[int],
    path: Path,
    iou: float = 0.5,
    title: Optional[str] = None,
) -> Path:
    """Interpolated precision-recall curves for a few classes."""
    _require_matplotlib()
    fig, ax = plt.subplots(figsize=(5.6, 4.2))
    _style(ax)

    for i, cid in enumerate(category_ids):
        recalls, precisions = results.pr_curve(cid, iou=iou)
        ap = float(np.mean(precisions))
        ax.plot(
            recalls,
            precisions,
            color=_SERIES[i % len(_SERIES)],
            linewidth=1.8,
            label=f"{CATEGORY_NAMES.get(cid, cid)}  AP{int(iou * 100)}={ap:.3f}",
        )

    ax.set_xlabel("recall")
    ax.set_ylabel("precision")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_title(
        title or f"Precision-recall at IoU {iou:.2f}",
        color=_INK,
        fontsize=11,
        loc="left",
    )
    ax.legend(
        fontsize=7.5,
        loc="lower left",
        frameon=True,
        framealpha=0.92,
        facecolor="white",
        edgecolor=_GRID,
    )
    return _save(fig, path)


def plot_error_breakdown(
    counts: Mapping[str, int],
    delta_ap: Mapping[str, float],
    path: Path,
    title: str = "Error taxonomy",
) -> Path:
    """Paired chart: how many errors of each type, and what each one costs."""
    _require_matplotlib()
    types = [t for t in delta_ap if t in counts]
    types.sort(key=lambda t: -delta_ap[t])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.4, 3.6))
    for ax in (ax1, ax2):
        _style(ax)

    ax1.bar(
        range(len(types)),
        [counts[t] for t in types],
        color=_SERIES[0],
        width=0.68,
    )
    ax1.set_xticks(range(len(types)))
    ax1.set_xticklabels(types, rotation=30, ha="right", fontsize=7.5)
    ax1.set_ylabel("count")
    ax1.set_title("how many", color=_INK, fontsize=10, loc="left")

    ax2.bar(
        range(len(types)),
        [delta_ap[t] * 100.0 for t in types],
        color=_SERIES[1],
        width=0.68,
    )
    ax2.set_xticks(range(len(types)))
    ax2.set_xticklabels(types, rotation=30, ha="right", fontsize=7.5)
    ax2.set_ylabel("AP50 recovered if fixed (pts)")
    ax2.set_title("what it costs", color=_INK, fontsize=10, loc="left")

    fig.suptitle(title, color=_INK, fontsize=11, x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return _save(fig, path)


def plot_threshold_sweep(
    points: Sequence[OperatingPoint],
    path: Path,
    title: str = "Confidence threshold sweep",
) -> Path:
    """Precision, recall and F1 against the confidence threshold."""
    _require_matplotlib()
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    _style(ax)

    thr = [p.threshold for p in points]
    ax.plot(thr, [p.precision for p in points], color=_SERIES[0],
            linewidth=1.8, label="precision")
    ax.plot(thr, [p.recall for p in points], color=_SERIES[1],
            linewidth=1.8, label="recall")
    ax.plot(thr, [p.f1 for p in points], color=_SERIES[2],
            linewidth=1.8, linestyle="--", label="F1")

    best = max(points, key=lambda p: p.f1)
    ax.axvline(best.threshold, color=_MUTED, linewidth=1.0, linestyle=":")
    ax.annotate(
        f"best F1 {best.f1:.3f} @ {best.threshold:.2f}",
        (best.threshold, best.f1),
        textcoords="offset points",
        xytext=(8, -12),
        fontsize=8,
        color=_MUTED,
    )

    ax.set_xlabel("confidence threshold")
    ax.set_ylabel("value")
    ax.set_ylim(0, 1.02)
    ax.set_title(title, color=_INK, fontsize=11, loc="left")
    ax.legend(fontsize=8, frameon=False)
    return _save(fig, path)


def draw_detections(
    image: np.ndarray,
    detections: Sequence[Mapping[str, object]],
    path: Path,
    ground_truth: Sequence[Mapping[str, object]] = (),
    score_threshold: float = 0.25,
    caption: str = "",
    jpeg_quality: int = 80,
) -> Path:
    """Draw predicted boxes (and optionally ground truth) onto an image.

    Ground truth is drawn as thin dashed-looking outlines, predictions as solid
    boxes with a class label and score, so a failure case is readable without a
    legend.
    """
    try:
        import cv2  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("opencv is required to draw detections") from exc

    canvas = np.ascontiguousarray(image.copy())
    for gt in ground_truth:
        x, y, w, h = (float(v) for v in gt["bbox"])  # type: ignore[index]
        cv2.rectangle(
            canvas,
            (int(x), int(y)),
            (int(x + w), int(y + h)),
            (90, 200, 90),
            1,
        )

    for det in detections:
        score = float(det["score"])  # type: ignore[index]
        if score < score_threshold:
            continue
        x, y, w, h = (float(v) for v in det["bbox"])  # type: ignore[index]
        name = CATEGORY_NAMES.get(int(det["category_id"]), "?")  # type: ignore[index]
        cv2.rectangle(
            canvas, (int(x), int(y)), (int(x + w), int(y + h)), (40, 90, 235), 2
        )
        label = f"{name} {score:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.44, 1)
        cv2.rectangle(
            canvas,
            (int(x), max(int(y) - th - 5, 0)),
            (int(x) + tw + 4, max(int(y), th + 5)),
            (40, 90, 235),
            -1,
        )
        cv2.putText(
            canvas,
            label,
            (int(x) + 2, max(int(y) - 4, th)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.44,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    if caption:
        # Wrap rather than truncate: a clipped caption on a failure case hides
        # the one number that explains why the image is there.
        scale, thickness = 0.44, 1
        max_width = canvas.shape[1] - 12
        lines: list[str] = []
        current = ""
        for word in caption.split():
            trial = f"{current} {word}".strip()
            (width, _), _ = cv2.getTextSize(
                trial, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness
            )
            if width > max_width and current:
                lines.append(current)
                current = word
            else:
                current = trial
        if current:
            lines.append(current)

        bar = np.full(
            (10 + 17 * len(lines), canvas.shape[1], 3), 245, dtype=np.uint8
        )
        for i, line in enumerate(lines):
            cv2.putText(
                bar,
                line,
                (6, 18 + 17 * i),
                cv2.FONT_HERSHEY_SIMPLEX,
                scale,
                (40, 40, 40),
                thickness,
                cv2.LINE_AA,
            )
        canvas = np.vstack([canvas, bar])

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(
        str(path), canvas, [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)]
    )
    if not ok:
        raise OSError(f"failed to write {path}")
    return path
