"""Detector interface and the detection container passed around the pipeline."""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from ..coco_classes import category_id_for_index

__all__ = ["Detections", "Detector"]


@dataclass
class Detections:
    """Detections for one image, in original-image pixel coordinates.

    Attributes:
        boxes_xyxy: ``(N, 4)`` boxes as ``[x1, y1, x2, y2]``.
        scores: ``(N,)`` confidences in ``[0, 1]``.
        class_ids: ``(N,)`` contiguous class indices 0..79.
        stage_times_ms: Wall-clock milliseconds per pipeline stage.
    """

    boxes_xyxy: np.ndarray
    scores: np.ndarray
    class_ids: np.ndarray
    stage_times_ms: Dict[str, float] = field(default_factory=dict)

    def __len__(self) -> int:
        return int(self.boxes_xyxy.shape[0])

    @classmethod
    def empty(cls) -> "Detections":
        """An empty result, still shaped correctly for downstream code."""
        return cls(
            boxes_xyxy=np.zeros((0, 4), dtype=np.float64),
            scores=np.zeros((0,), dtype=np.float64),
            class_ids=np.zeros((0,), dtype=np.int64),
        )

    def to_coco(self, image_id: int, max_dets: Optional[int] = 100) -> List[dict]:
        """Convert to COCO result records.

        Boxes become ``[x, y, w, h]`` and contiguous class indices become real
        COCO ``category_id`` values. Scores are rounded to five decimals, which
        is well below the resolution the metric can distinguish and keeps the
        results file to a sane size.
        """
        order = np.argsort(-self.scores, kind="mergesort")
        if max_dets is not None:
            order = order[:max_dets]
        out: List[dict] = []
        for i in order:
            x1, y1, x2, y2 = (float(v) for v in self.boxes_xyxy[i])
            out.append(
                {
                    "image_id": int(image_id),
                    "category_id": category_id_for_index(int(self.class_ids[i])),
                    "bbox": [
                        round(x1, 3),
                        round(y1, 3),
                        round(x2 - x1, 3),
                        round(y2 - y1, 3),
                    ],
                    "score": round(float(self.scores[i]), 5),
                }
            )
        return out


class Detector(abc.ABC):
    """Anything that turns an image into :class:`Detections`."""

    name: str = "detector"

    @abc.abstractmethod
    def predict(self, image: np.ndarray) -> Detections:
        """Run the full pipeline on one ``(H, W, 3)`` BGR uint8 image."""

    def warmup(self, runs: int = 1, size: int = 640) -> None:
        """Run a few throwaway inferences so timings exclude lazy setup.

        The first inference of an ONNX Runtime session pays for arena
        allocation and kernel selection; without a warmup it lands in the p99
        and makes the tail look far worse than it is.
        """
        dummy = np.zeros((size, size, 3), dtype=np.uint8)
        for _ in range(max(0, runs)):
            self.predict(dummy)

    def close(self) -> None:
        """Release any backend resources. Default is a no-op."""
