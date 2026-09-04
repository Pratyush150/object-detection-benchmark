"""Non-maximum suppression.

Detection heads emit thousands of overlapping candidates. NMS is what turns
them into one box per object, and it is where two mistakes are common:

* Running it class-agnostically, so a dog standing in front of a sofa loses one
  of the two boxes because they overlap. COCO scores classes independently, so
  suppression must be class-aware.
* Sorting by the wrong score. The whole point of the greedy rule is that the
  most confident box survives; if the sort is unstable or reversed, the metric
  quietly drops.
"""

from __future__ import annotations

import numpy as np

__all__ = ["nms", "batched_nms"]


def nms(boxes_xyxy: np.ndarray, scores: np.ndarray, iou_threshold: float) -> np.ndarray:
    """Greedy NMS over a single class.

    Args:
        boxes_xyxy: ``(N, 4)`` boxes as ``[x1, y1, x2, y2]``.
        scores: ``(N,)`` confidence scores.
        iou_threshold: Boxes overlapping a kept box by more than this are
            discarded.

    Returns:
        Indices of the kept boxes, in descending score order.
    """
    boxes = np.asarray(boxes_xyxy, dtype=np.float64).reshape(-1, 4)
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    if boxes.shape[0] == 0:
        return np.empty((0,), dtype=np.int64)
    if boxes.shape[0] != scores.shape[0]:
        raise ValueError("boxes and scores must have the same length")

    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = np.maximum(x2 - x1, 0.0) * np.maximum(y2 - y1, 0.0)
    order = np.argsort(-scores, kind="mergesort")

    keep: list[int] = []
    while order.size:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break
        rest = order[1:]
        xx1 = np.maximum(x1[i], x1[rest])
        yy1 = np.maximum(y1[i], y1[rest])
        xx2 = np.minimum(x2[i], x2[rest])
        yy2 = np.minimum(y2[i], y2[rest])
        inter = np.maximum(xx2 - xx1, 0.0) * np.maximum(yy2 - yy1, 0.0)
        union = areas[i] + areas[rest] - inter
        with np.errstate(divide="ignore", invalid="ignore"):
            iou = np.where(union > 0.0, inter / union, 0.0)
        order = rest[iou <= iou_threshold]
    return np.asarray(keep, dtype=np.int64)


def batched_nms(
    boxes_xyxy: np.ndarray,
    scores: np.ndarray,
    class_ids: np.ndarray,
    iou_threshold: float,
    max_dets: int = 300,
) -> np.ndarray:
    """Class-aware NMS via the coordinate-offset trick.

    Each class is shifted into its own region of an enormous virtual canvas, so
    boxes of different classes can never overlap and a single NMS pass behaves
    exactly like one pass per class - at a fraction of the Python overhead.

    Args:
        boxes_xyxy: ``(N, 4)`` boxes as ``[x1, y1, x2, y2]``.
        scores: ``(N,)`` confidence scores.
        class_ids: ``(N,)`` integer class ids.
        iou_threshold: IoU above which a lower-scoring box is suppressed.
        max_dets: Cap on the number of surviving detections. COCO scores at
            most 100 per image, so 300 leaves headroom without unbounded cost.

    Returns:
        Indices of the kept boxes, in descending score order.
    """
    boxes = np.asarray(boxes_xyxy, dtype=np.float64).reshape(-1, 4)
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    class_ids = np.asarray(class_ids).astype(np.int64).reshape(-1)
    if boxes.shape[0] == 0:
        return np.empty((0,), dtype=np.int64)

    span = float(boxes.max()) - float(boxes.min()) + 1.0
    offsets = class_ids.astype(np.float64) * span
    shifted = boxes + offsets[:, None]
    keep = nms(shifted, scores, iou_threshold)
    return keep[:max_dets]
