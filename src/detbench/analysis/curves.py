"""Operating-point curves: what a confidence threshold actually buys you.

mAP integrates over every confidence threshold at once, which is the right way
to compare models and the wrong way to configure one. A deployed system picks a
single threshold, and that choice trades precision against recall along a curve
that mAP never shows. This module produces that curve from real detections.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Sequence

import numpy as np

from ..metrics.box_ops import iou_matrix
from ..metrics.coco_map import GroundTruth

__all__ = ["OperatingPoint", "score_threshold_sweep", "format_sweep_table"]


@dataclass(frozen=True)
class OperatingPoint:
    """Dataset-wide precision and recall at one confidence threshold."""

    threshold: float
    true_positives: int
    false_positives: int
    false_negatives: int
    precision: float
    recall: float
    f1: float
    detections_per_image: float


def _match_at_iou(
    ground_truth: GroundTruth,
    detections: Sequence[Mapping[str, object]],
    iou_threshold: float,
    max_dets: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Greedy class-aware matching over the whole dataset.

    Returns ``(scores, is_true_positive, n_ground_truths)`` with detections in
    descending score order. Crowd regions are excluded from the ground-truth
    count and detections landing in them are dropped, mirroring the metric.
    """
    by_image_gt: Dict[int, List[dict]] = {}
    n_gt = 0
    for ann in ground_truth.annotations:
        by_image_gt.setdefault(int(ann["image_id"]), []).append(ann)
        if not int(ann.get("iscrowd", 0)):
            n_gt += 1

    by_image_det: Dict[int, List[dict]] = {}
    for det in detections:
        by_image_det.setdefault(int(det["image_id"]), []).append(dict(det))

    scores: List[float] = []
    is_tp: List[bool] = []

    for image_id, dets in by_image_det.items():
        dets.sort(key=lambda d: -float(d["score"]))
        dets = dets[:max_dets]
        gts = by_image_gt.get(image_id, [])
        if not gts:
            for det in dets:
                scores.append(float(det["score"]))
                is_tp.append(False)
            continue

        gt_boxes = np.array([g["bbox"] for g in gts], dtype=np.float64)
        gt_classes = np.array([int(g["category_id"]) for g in gts])
        gt_crowd = np.array([int(g.get("iscrowd", 0)) for g in gts], dtype=bool)
        claimed = np.zeros(len(gts), dtype=bool)

        det_boxes = np.array([d["bbox"] for d in dets], dtype=np.float64)
        det_classes = np.array([int(d["category_id"]) for d in dets])
        ious = iou_matrix(det_boxes, gt_boxes, gt_crowd)

        for row, det in enumerate(dets):
            eligible = (gt_classes == det_classes[row]) & (~claimed | gt_crowd)
            masked = np.where(eligible, ious[row], -1.0)
            best = int(np.argmax(masked)) if masked.size else -1
            if best >= 0 and masked[best] >= iou_threshold:
                if gt_crowd[best]:
                    continue  # ignored by the metric, so ignored here
                claimed[best] = True
                scores.append(float(det["score"]))
                is_tp.append(True)
            else:
                scores.append(float(det["score"]))
                is_tp.append(False)

    order = np.argsort(-np.asarray(scores), kind="mergesort")
    return np.asarray(scores)[order], np.asarray(is_tp)[order], n_gt


def score_threshold_sweep(
    ground_truth: GroundTruth,
    detections: Sequence[Mapping[str, object]],
    thresholds: Optional[Sequence[float]] = None,
    iou_threshold: float = 0.5,
    max_dets: int = 100,
    n_images: Optional[int] = None,
) -> List[OperatingPoint]:
    """Precision and recall as a function of the confidence threshold.

    Args:
        ground_truth: Parsed COCO ground truth.
        detections: COCO-format detection records.
        thresholds: Confidence thresholds to report. Defaults to 0.01 to 0.95.
        iou_threshold: IoU required to count a detection as correct.
        max_dets: Per-image detection cap.
        n_images: Image count, used for detections-per-image. Defaults to the
            ground truth's image count.

    Returns:
        One :class:`OperatingPoint` per threshold.
    """
    if thresholds is None:
        thresholds = [0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45,
                      0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
    scores, is_tp, n_gt = _match_at_iou(
        ground_truth, detections, iou_threshold, max_dets
    )
    images = n_images if n_images is not None else len(ground_truth.image_ids)
    images = max(images, 1)

    points: List[OperatingPoint] = []
    for thr in thresholds:
        keep = scores >= thr
        tp = int(np.count_nonzero(is_tp[keep]))
        fp = int(np.count_nonzero(keep) - tp)
        fn = max(n_gt - tp, 0)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / n_gt if n_gt else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )
        points.append(
            OperatingPoint(
                threshold=float(thr),
                true_positives=tp,
                false_positives=fp,
                false_negatives=fn,
                precision=precision,
                recall=recall,
                f1=f1,
                detections_per_image=int(np.count_nonzero(keep)) / images,
            )
        )
    return points


def format_sweep_table(points: Sequence[OperatingPoint]) -> str:
    """Render a threshold sweep as a fixed-width table."""
    header = (
        f"{'conf':>6}{'TP':>9}{'FP':>9}{'FN':>9}"
        f"{'precision':>11}{'recall':>9}{'F1':>8}{'det/img':>9}"
    )
    lines = [header, "-" * len(header)]
    for p in points:
        lines.append(
            f"{p.threshold:>6.2f}{p.true_positives:>9}{p.false_positives:>9}"
            f"{p.false_negatives:>9}{p.precision:>11.4f}{p.recall:>9.4f}"
            f"{p.f1:>8.4f}{p.detections_per_image:>9.2f}"
        )
    return "\n".join(lines)
