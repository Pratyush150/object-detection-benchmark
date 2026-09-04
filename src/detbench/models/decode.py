"""Decode a raw YOLO detection head into boxes, scores and class ids.

YOLOv8 exports a single tensor of shape ``(1, 4 + C, A)``: for each of ``A``
anchor points, four box values in network-input pixels (centre x, centre y,
width, height) followed by ``C`` per-class confidences that already have
sigmoid applied. There is no separate objectness term, unlike YOLOv5.

Two decisions here move the final mAP by a measurable amount and are exposed as
arguments rather than buried:

* ``conf_threshold`` - the confidence floor. A high floor looks better in a
  demo and scores worse on COCO, because the metric integrates precision over
  the whole recall range and low-confidence detections still add recall.
* ``multi_label`` - whether an anchor may emit one detection per class above
  threshold, or only its single best class.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

__all__ = ["decode_yolo_head"]


def decode_yolo_head(
    raw: np.ndarray,
    conf_threshold: float = 0.001,
    multi_label: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Turn a raw head tensor into candidate boxes in network coordinates.

    Args:
        raw: ``(1, 4 + C, A)`` or ``(4 + C, A)`` output tensor. The transposed
            ``(1, A, 4 + C)`` layout some exporters emit is detected and
            handled, since ``A`` (8400 at 640x640) is always far larger than
            ``4 + C`` (84 for COCO).
        conf_threshold: Minimum class confidence to keep a candidate.
        multi_label: Emit every class above threshold for an anchor instead of
            only the best one.

    Returns:
        ``(boxes_xyxy, scores, class_ids)`` in network-input pixels, unsorted.
    """
    arr = np.asarray(raw)
    if arr.ndim == 3:
        if arr.shape[0] != 1:
            raise ValueError(f"expected batch size 1, got {arr.shape[0]}")
        arr = arr[0]
    if arr.ndim != 2:
        raise ValueError(f"expected a 2-D head after batch removal, got {arr.shape}")
    # Anchors always outnumber channels; orient the tensor as (channels, anchors).
    if arr.shape[0] > arr.shape[1]:
        arr = arr.T
    arr = arr.astype(np.float32, copy=False)

    box = arr[:4]
    cls = arr[4:]
    if cls.shape[0] == 0:
        raise ValueError("head tensor has no class channels")

    cx, cy, w, h = box[0], box[1], box[2], box[3]
    x1 = cx - w / 2.0
    y1 = cy - h / 2.0
    x2 = cx + w / 2.0
    y2 = cy + h / 2.0
    boxes_all = np.stack([x1, y1, x2, y2], axis=1)

    if multi_label:
        cls_idx, anchor_idx = np.nonzero(cls > conf_threshold)
        boxes = boxes_all[anchor_idx]
        scores = cls[cls_idx, anchor_idx].astype(np.float64)
        class_ids = cls_idx.astype(np.int64)
    else:
        class_ids_all = np.argmax(cls, axis=0)
        scores_all = cls[class_ids_all, np.arange(cls.shape[1])]
        keep = scores_all > conf_threshold
        boxes = boxes_all[keep]
        scores = scores_all[keep].astype(np.float64)
        class_ids = class_ids_all[keep].astype(np.int64)

    return boxes.astype(np.float64), scores, class_ids
