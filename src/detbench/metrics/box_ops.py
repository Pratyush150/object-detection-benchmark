"""Bounding-box geometry used by the COCO evaluator.

Boxes are stored in COCO's ``[x, y, w, h]`` convention (top-left corner plus
width and height, in absolute pixels) unless a function name says otherwise.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "xywh_to_xyxy",
    "xyxy_to_xywh",
    "box_areas",
    "iou_matrix",
]


def xywh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
    """Convert ``[x, y, w, h]`` boxes to ``[x1, y1, x2, y2]``."""
    boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    out = np.empty_like(boxes)
    out[:, 0] = boxes[:, 0]
    out[:, 1] = boxes[:, 1]
    out[:, 2] = boxes[:, 0] + boxes[:, 2]
    out[:, 3] = boxes[:, 1] + boxes[:, 3]
    return out


def xyxy_to_xywh(boxes: np.ndarray) -> np.ndarray:
    """Convert ``[x1, y1, x2, y2]`` boxes to ``[x, y, w, h]``."""
    boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    out = np.empty_like(boxes)
    out[:, 0] = boxes[:, 0]
    out[:, 1] = boxes[:, 1]
    out[:, 2] = boxes[:, 2] - boxes[:, 0]
    out[:, 3] = boxes[:, 3] - boxes[:, 1]
    return out


def box_areas(boxes_xywh: np.ndarray) -> np.ndarray:
    """Return ``w * h`` for each ``[x, y, w, h]`` box, clamped at zero."""
    boxes = np.asarray(boxes_xywh, dtype=np.float64).reshape(-1, 4)
    return np.maximum(boxes[:, 2], 0.0) * np.maximum(boxes[:, 3], 0.0)


def iou_matrix(
    dets_xywh: np.ndarray,
    gts_xywh: np.ndarray,
    iscrowd: np.ndarray | None = None,
) -> np.ndarray:
    """Pairwise overlap between detections and ground truths.

    For ordinary ground truths this is intersection over union. For ground
    truths flagged ``iscrowd`` it is intersection over the *detection* area
    (intersection-over-area), which is what the COCO protocol requires: a crowd
    region is a blob covering many instances, so a detection that falls inside
    it should score 1.0 regardless of how much larger the blob is.

    Args:
        dets_xywh: ``(D, 4)`` detection boxes in ``[x, y, w, h]``.
        gts_xywh: ``(G, 4)`` ground-truth boxes in ``[x, y, w, h]``.
        iscrowd: Optional ``(G,)`` array of 0/1 crowd flags.

    Returns:
        ``(D, G)`` array of overlaps in ``[0, 1]``. Zero-area boxes give 0.
    """
    dets = np.asarray(dets_xywh, dtype=np.float64).reshape(-1, 4)
    gts = np.asarray(gts_xywh, dtype=np.float64).reshape(-1, 4)
    if dets.size == 0 or gts.size == 0:
        return np.zeros((dets.shape[0], gts.shape[0]), dtype=np.float64)

    crowd = (
        np.zeros(gts.shape[0], dtype=bool)
        if iscrowd is None
        else np.asarray(iscrowd).astype(bool).reshape(-1)
    )
    if crowd.shape[0] != gts.shape[0]:
        raise ValueError("iscrowd length must match the number of ground truths")

    d_x1, d_y1 = dets[:, 0], dets[:, 1]
    d_x2, d_y2 = dets[:, 0] + dets[:, 2], dets[:, 1] + dets[:, 3]
    g_x1, g_y1 = gts[:, 0], gts[:, 1]
    g_x2, g_y2 = gts[:, 0] + gts[:, 2], gts[:, 1] + gts[:, 3]

    inter_w = np.minimum(d_x2[:, None], g_x2[None, :]) - np.maximum(
        d_x1[:, None], g_x1[None, :]
    )
    inter_h = np.minimum(d_y2[:, None], g_y2[None, :]) - np.maximum(
        d_y1[:, None], g_y1[None, :]
    )
    inter = np.clip(inter_w, 0.0, None) * np.clip(inter_h, 0.0, None)

    d_area = box_areas(dets)[:, None]
    g_area = box_areas(gts)[None, :]

    union = d_area + g_area - inter
    union = np.where(crowd[None, :], np.broadcast_to(d_area, inter.shape), union)

    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(union > 0.0, inter / union, 0.0)
    return out
