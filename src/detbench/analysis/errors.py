"""Error taxonomy: where the detector actually loses accuracy.

The decomposition follows TIDE (Bolya et al., 2020). Every false positive is
assigned to exactly one cause, every unmatched ground truth is a miss, and each
cause is then given a magnitude by fixing it with an oracle and re-scoring:

======================  ====================================================
Type                    Definition
======================  ====================================================
``correct``             IoU >= 0.5 with an unclaimed ground truth of the same
                        class. Not an error.
``localisation``        Right class, 0.1 <= IoU < 0.5. The object was found
                        but the box is loose.
``classification``      Wrong class, but IoU >= 0.5 with a ground truth of
                        some other class. The box is right, the label is not.
``both``                Wrong class *and* a loose box (0.1 <= IoU < 0.5).
``duplicate``           IoU >= 0.5 with a ground truth of the same class that
                        a higher-scoring detection already claimed. This is
                        what a badly tuned NMS threshold produces.
``background``          IoU < 0.1 with everything. A hallucination.
``missed``              A ground truth no detection accounted for.
======================  ====================================================

Counts alone are misleading: a thousand background false positives scored 0.002
cost almost no AP, while fifty missed people cost a great deal. So each type
also gets a ``delta_ap``: the AP gained if that error class were fixed
perfectly and nothing else changed. That is the number that says what to work
on next.
"""

from __future__ import annotations

import copy
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from ..coco_classes import CATEGORY_NAMES
from ..metrics.box_ops import iou_matrix
from ..metrics.coco_map import (
    AreaRange,
    COCOMeanAP,
    EvalParams,
    GroundTruth,
)

__all__ = [
    "ERROR_TYPES",
    "ErrorBreakdown",
    "classify_errors",
    "tide_analysis",
    "confusion_pairs",
]

ERROR_TYPES = (
    "localisation",
    "classification",
    "both",
    "duplicate",
    "background",
    "missed",
)

#: Foreground IoU: at or above this, a box is considered to be on the object.
IOU_FOREGROUND = 0.5
#: Background IoU: below this, a box is not on any object at all.
IOU_BACKGROUND = 0.1


@dataclass
class ErrorBreakdown:
    """Per-detection error labels plus the oracle-corrected detection sets."""

    labels: List[str]
    matched_gt: List[Optional[int]]
    counts: Dict[str, int]
    n_correct: int
    missed_ann_ids: List[int]
    detections: List[dict]
    ground_truth: GroundTruth
    confusions: Counter = field(default_factory=Counter)

    def fixed_detections(self, error_type: str) -> List[dict]:
        """Detections with one error type corrected by an oracle.

        ``localisation`` snaps the box onto the ground truth it was closest to;
        ``classification`` relabels to the correct class; ``both`` does both;
        ``duplicate`` and ``background`` delete the offending detections.
        """
        if error_type not in ERROR_TYPES:
            raise ValueError(f"unknown error type: {error_type!r}")
        if error_type == "missed":
            return [dict(d) for d in self.detections]

        gt_by_id = {int(a["id"]): a for a in self.ground_truth.annotations}
        out: List[dict] = []
        for det, label, gt_id in zip(self.detections, self.labels, self.matched_gt):
            if label != error_type:
                out.append(dict(det))
                continue
            if error_type in ("duplicate", "background"):
                continue
            fixed = dict(det)
            gt = gt_by_id.get(int(gt_id)) if gt_id is not None else None
            if gt is None:
                continue
            if error_type in ("localisation", "both"):
                fixed["bbox"] = [float(v) for v in gt["bbox"]]
            if error_type in ("classification", "both"):
                fixed["category_id"] = int(gt["category_id"])
            out.append(fixed)
        return out

    def fixed_ground_truth(self, error_type: str) -> GroundTruth:
        """Ground truth with missed objects removed, for the ``missed`` oracle."""
        if error_type != "missed":
            return self.ground_truth
        drop = set(self.missed_ann_ids)
        return GroundTruth(
            image_ids=list(self.ground_truth.image_ids),
            category_ids=list(self.ground_truth.category_ids),
            annotations=[
                a for a in self.ground_truth.annotations if int(a["id"]) not in drop
            ],
        )


def classify_errors(
    ground_truth: GroundTruth,
    detections: Sequence[Mapping[str, object]],
    score_threshold: float = 0.0,
    max_dets: int = 100,
) -> ErrorBreakdown:
    """Assign every detection an error type and find the missed objects.

    Args:
        ground_truth: Parsed COCO ground truth.
        detections: COCO-format detection records.
        score_threshold: Ignore detections below this score. TIDE's own
            analysis keeps everything; raising the threshold answers the
            different question of what a deployed confidence filter would see.
        max_dets: Per-image detection cap, matching the metric.

    Returns:
        An :class:`ErrorBreakdown`.
    """
    dets = [dict(d) for d in detections if float(d["score"]) >= score_threshold]

    by_image_det: Dict[int, List[int]] = {}
    for i, det in enumerate(dets):
        by_image_det.setdefault(int(det["image_id"]), []).append(i)
    by_image_gt: Dict[int, List[dict]] = {}
    for ann in ground_truth.annotations:
        by_image_gt.setdefault(int(ann["image_id"]), []).append(ann)

    labels: List[str] = ["background"] * len(dets)
    matched_gt: List[Optional[int]] = [None] * len(dets)
    missed: List[int] = []
    confusions: Counter = Counter()
    n_correct = 0

    image_ids = set(by_image_det) | set(by_image_gt)
    for image_id in sorted(image_ids):
        idxs = by_image_det.get(image_id, [])
        idxs.sort(key=lambda i: -float(dets[i]["score"]))
        idxs = idxs[:max_dets]
        gts = by_image_gt.get(image_id, [])

        if not gts:
            for i in idxs:
                labels[i] = "background"
            continue

        gt_boxes = np.array([g["bbox"] for g in gts], dtype=np.float64)
        gt_classes = np.array([int(g["category_id"]) for g in gts])
        gt_crowd = np.array([int(g.get("iscrowd", 0)) for g in gts], dtype=bool)
        claimed = np.zeros(len(gts), dtype=bool)

        if idxs:
            det_boxes = np.array([dets[i]["bbox"] for i in idxs], dtype=np.float64)
            det_classes = np.array([int(dets[i]["category_id"]) for i in idxs])
            ious = iou_matrix(det_boxes, gt_boxes, gt_crowd)
        else:
            ious = np.zeros((0, len(gts)))
            det_classes = np.zeros((0,), dtype=np.int64)

        touched_by_error = np.zeros(len(gts), dtype=bool)

        for row, det_idx in enumerate(idxs):
            same = gt_classes == det_classes[row]
            other = ~same

            best_same, best_same_i = _best(ious[row], same)
            best_other, best_other_i = _best(ious[row], other)

            if best_same >= IOU_FOREGROUND:
                if gt_crowd[best_same_i]:
                    # Detections inside a crowd region are neither right nor
                    # wrong; the metric ignores them, so the taxonomy does too.
                    labels[det_idx] = "correct"
                    n_correct += 1
                    continue
                if not claimed[best_same_i]:
                    claimed[best_same_i] = True
                    labels[det_idx] = "correct"
                    matched_gt[det_idx] = int(gts[best_same_i]["id"])
                    n_correct += 1
                else:
                    labels[det_idx] = "duplicate"
                    matched_gt[det_idx] = int(gts[best_same_i]["id"])
                continue

            if best_same >= IOU_BACKGROUND and best_other < IOU_FOREGROUND:
                labels[det_idx] = "localisation"
                matched_gt[det_idx] = int(gts[best_same_i]["id"])
                touched_by_error[best_same_i] = True
                continue

            if best_other >= IOU_FOREGROUND:
                labels[det_idx] = "classification"
                matched_gt[det_idx] = int(gts[best_other_i]["id"])
                touched_by_error[best_other_i] = True
                confusions[
                    (int(gt_classes[best_other_i]), int(det_classes[row]))
                ] += 1
                continue

            if best_other >= IOU_BACKGROUND:
                labels[det_idx] = "both"
                matched_gt[det_idx] = int(gts[best_other_i]["id"])
                touched_by_error[best_other_i] = True
                continue

            labels[det_idx] = "background"

        for g_i, gt in enumerate(gts):
            if gt_crowd[g_i] or claimed[g_i] or touched_by_error[g_i]:
                continue
            missed.append(int(gt["id"]))

    counts = {t: 0 for t in ERROR_TYPES}
    for label in labels:
        if label in counts:
            counts[label] += 1
    counts["missed"] = len(missed)

    return ErrorBreakdown(
        labels=labels,
        matched_gt=matched_gt,
        counts=counts,
        n_correct=n_correct,
        missed_ann_ids=missed,
        detections=dets,
        ground_truth=ground_truth,
        confusions=confusions,
    )


def _best(row: np.ndarray, mask: np.ndarray) -> Tuple[float, int]:
    """Highest IoU in ``row`` restricted to ``mask``, with its index."""
    if row.size == 0 or not mask.any():
        return 0.0, -1
    masked = np.where(mask, row, -1.0)
    idx = int(np.argmax(masked))
    return float(max(masked[idx], 0.0)), idx


def tide_analysis(
    ground_truth: GroundTruth,
    detections: Sequence[Mapping[str, object]],
    image_ids: Optional[Sequence[int]] = None,
    iou_threshold: float = 0.5,
    max_dets: int = 100,
) -> Dict[str, object]:
    """Count errors and measure what fixing each type would be worth.

    The oracle AP is computed at a single IoU threshold with one area range,
    not the full 10x4x3 sweep, because the question is "how much AP does this
    error cost" rather than "what is the headline number", and the reduced
    sweep is about forty times cheaper.

    Args:
        ground_truth: Parsed COCO ground truth.
        detections: COCO-format detection records.
        image_ids: Images to score over. Defaults to the ground truth's.
        iou_threshold: IoU at which the oracle APs are measured.
        max_dets: Per-image detection cap.

    Returns:
        A dictionary with ``counts``, ``baseline_ap``, ``delta_ap`` per error
        type and the top class confusions.
    """
    params = EvalParams(
        iou_thresholds=np.array([iou_threshold]),
        area_ranges=(AreaRange("all", 0.0, 1e10),),
        max_dets=(max_dets,),
    )
    breakdown = classify_errors(ground_truth, detections, max_dets=max_dets)

    baseline = COCOMeanAP(ground_truth, copy.deepcopy(params)).evaluate(
        breakdown.detections, image_ids
    )
    baseline_ap = baseline.ap(area="all", max_dets=max_dets)

    deltas: Dict[str, float] = {}
    for error_type in ERROR_TYPES:
        fixed_gt = breakdown.fixed_ground_truth(error_type)
        fixed_dets = breakdown.fixed_detections(error_type)
        oracle = COCOMeanAP(fixed_gt, copy.deepcopy(params)).evaluate(
            fixed_dets, image_ids
        )
        deltas[error_type] = (
            oracle.ap(area="all", max_dets=max_dets) - baseline_ap
        )

    return {
        "iou_threshold": iou_threshold,
        "counts": dict(breakdown.counts),
        "n_correct": breakdown.n_correct,
        "n_detections": len(breakdown.detections),
        "baseline_ap": baseline_ap,
        "delta_ap": deltas,
        "confusions": confusion_pairs(breakdown, limit=20),
    }


def confusion_pairs(
    breakdown: ErrorBreakdown, limit: int = 20
) -> List[Dict[str, object]]:
    """Most frequent (true class -> predicted class) confusions."""
    out: List[Dict[str, object]] = []
    for (true_id, pred_id), count in breakdown.confusions.most_common(limit):
        out.append(
            {
                "true_id": true_id,
                "true_name": CATEGORY_NAMES.get(true_id, str(true_id)),
                "pred_id": pred_id,
                "pred_name": CATEGORY_NAMES.get(pred_id, str(pred_id)),
                "count": count,
            }
        )
    return out
