"""A detector backend with no model file behind it.

Every stage of the real pipeline is exercised - letterboxing, raw-head
decoding, class-aware NMS, un-letterboxing - but the ONNX session is replaced
by a tensor synthesised from a registry of known boxes. That makes the whole
test suite and the ``--demo`` path runnable with no weights and no dataset,
while still catching the coordinate bugs that a hand-written stub would miss.
"""

from __future__ import annotations

import hashlib
import time
from typing import Dict, Optional, Tuple

import numpy as np

from .base import Detections, Detector
from .decode import decode_yolo_head
from .letterbox import letterbox
from .nms import batched_nms

__all__ = ["MockYoloDetector", "image_key"]


def image_key(image: np.ndarray) -> str:
    """Stable content hash of an image, used to look up planted detections."""
    arr = np.ascontiguousarray(image)
    digest = hashlib.sha1(arr.tobytes())
    digest.update(str(arr.shape).encode("ascii"))
    return digest.hexdigest()


class MockYoloDetector(Detector):
    """Replay planted boxes through the real decode/NMS/un-letterbox path.

    Args:
        registry: Maps an image content hash to ``(boxes_xyxy, scores,
            class_ids)`` in *original image* coordinates.
        num_classes: Width of the synthetic class block.
        num_anchors: Anchor count of the synthetic head tensor. Kept at the
            real 8400 so array shapes and NMS costs stay representative.
        input_size: ``(width, height)`` of the pretend network input.
        latency_ms: Optional artificial inference delay, for exercising the
            profiling code deterministically.
    """

    def __init__(
        self,
        registry: Optional[
            Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]
        ] = None,
        num_classes: int = 80,
        num_anchors: int = 8400,
        input_size: Tuple[int, int] = (640, 640),
        conf_threshold: float = 0.001,
        iou_threshold: float = 0.7,
        max_dets: int = 300,
        latency_ms: float = 0.0,
        name: str = "mock-yolo",
    ) -> None:
        self.registry = dict(registry or {})
        self.num_classes = int(num_classes)
        self.num_anchors = int(num_anchors)
        self.input_size = (int(input_size[0]), int(input_size[1]))
        self.conf_threshold = float(conf_threshold)
        self.iou_threshold = float(iou_threshold)
        self.max_dets = int(max_dets)
        self.latency_ms = float(latency_ms)
        self.name = name

    def register(
        self,
        image: np.ndarray,
        boxes_xyxy: np.ndarray,
        scores: np.ndarray,
        class_ids: np.ndarray,
    ) -> str:
        """Plant detections for an image and return its key."""
        key = image_key(image)
        self.registry[key] = (
            np.asarray(boxes_xyxy, dtype=np.float64).reshape(-1, 4),
            np.asarray(scores, dtype=np.float64).reshape(-1),
            np.asarray(class_ids, dtype=np.int64).reshape(-1),
        )
        return key

    def _synthesise_head(
        self,
        boxes_xyxy_net: np.ndarray,
        scores: np.ndarray,
        class_ids: np.ndarray,
    ) -> np.ndarray:
        """Build a ``(1, 4 + C, A)`` tensor with the boxes planted in it."""
        head = np.zeros((4 + self.num_classes, self.num_anchors), dtype=np.float32)
        n = min(len(scores), self.num_anchors)
        if n:
            b = boxes_xyxy_net[:n]
            head[0, :n] = (b[:, 0] + b[:, 2]) / 2.0
            head[1, :n] = (b[:, 1] + b[:, 3]) / 2.0
            head[2, :n] = b[:, 2] - b[:, 0]
            head[3, :n] = b[:, 3] - b[:, 1]
            head[4 + class_ids[:n], np.arange(n)] = scores[:n]
        return head[None]

    def predict(self, image: np.ndarray) -> Detections:
        """Run the pipeline on one image using its planted detections."""
        t0 = time.perf_counter()
        _, transform = letterbox(image, self.input_size)
        t1 = time.perf_counter()

        boxes, scores, class_ids = self.registry.get(
            image_key(image),
            (np.zeros((0, 4)), np.zeros((0,)), np.zeros((0,), dtype=np.int64)),
        )
        head = self._synthesise_head(
            transform.forward_xyxy(boxes) if len(scores) else np.zeros((0, 4)),
            scores,
            class_ids,
        )
        if self.latency_ms > 0:
            time.sleep(self.latency_ms / 1e3)
        t2 = time.perf_counter()

        d_boxes, d_scores, d_classes = decode_yolo_head(
            head, self.conf_threshold, multi_label=False
        )
        keep = batched_nms(
            d_boxes, d_scores, d_classes, self.iou_threshold, self.max_dets
        )
        t3 = time.perf_counter()

        out_boxes = (
            transform.inverse_xyxy(d_boxes[keep]) if keep.size else np.zeros((0, 4))
        )
        result = Detections(
            boxes_xyxy=out_boxes,
            scores=d_scores[keep],
            class_ids=d_classes[keep],
        )
        t4 = time.perf_counter()
        result.stage_times_ms = {
            "preprocess": (t1 - t0) * 1e3,
            "inference": (t2 - t1) * 1e3,
            "nms": (t3 - t2) * 1e3,
            "postprocess": (t4 - t3) * 1e3,
            "total": (t4 - t0) * 1e3,
        }
        return result
