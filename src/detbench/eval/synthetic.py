"""A small synthetic detection dataset, generated deterministically.

This exists so the whole toolchain - evaluation, error analysis, profiling,
plots - can be demonstrated and tested without downloading a gigabyte of COCO
or shipping model weights. Numbers produced from it describe the synthetic
data and nothing else; they are never presented as benchmark results.

The generator draws solid rectangles on a textured background, records them as
ground truth, and produces a matching "prediction" set by applying a controlled
mix of the failure modes a real detector shows: jittered boxes, class swaps,
duplicates, background false positives and dropped objects. That is enough
structure for the error taxonomy to have something real to decompose.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

from ..coco_classes import COCO80_TO_COCO91
from ..metrics.coco_map import GroundTruth
from ..models.mock import MockYoloDetector, image_key

__all__ = ["SyntheticDataset", "make_synthetic_dataset"]


@dataclass
class SyntheticDataset:
    """Images, ground truth and a detector that "predicts" on them."""

    images: Dict[int, np.ndarray]
    ground_truth: GroundTruth
    detector: MockYoloDetector

    @property
    def image_ids(self) -> List[int]:
        """Sorted image ids."""
        return sorted(self.images)


def _draw_rect(
    canvas: np.ndarray, box: Tuple[float, float, float, float], colour: np.ndarray
) -> None:
    x1, y1, x2, y2 = (int(round(v)) for v in box)
    canvas[max(y1, 0) : max(y2, 0), max(x1, 0) : max(x2, 0)] = colour


def make_synthetic_dataset(
    n_images: int = 24,
    width: int = 480,
    height: int = 360,
    n_classes: int = 8,
    seed: int = 20260101,
    drop_rate: float = 0.12,
    swap_rate: float = 0.10,
    duplicate_rate: float = 0.15,
    false_positive_rate: float = 0.8,
    jitter_px: float = 6.0,
) -> SyntheticDataset:
    """Build a deterministic synthetic dataset and a matching mock detector.

    Args:
        n_images: Number of images to generate.
        width: Image width in pixels.
        height: Image height in pixels.
        n_classes: How many of the 80 COCO classes to use.
        seed: RNG seed. The same seed always gives the same dataset, the same
            detections and therefore the same mAP.
        drop_rate: Fraction of objects the detector misses entirely.
        swap_rate: Fraction of detections given the wrong class.
        duplicate_rate: Fraction of objects that get a second, lower-scoring
            box, which the metric must punish as a false positive.
        false_positive_rate: Expected number of pure background detections per
            image.
        jitter_px: Standard deviation of the box-corner noise.

    Returns:
        A :class:`SyntheticDataset`.
    """
    rng = np.random.default_rng(seed)
    images: Dict[int, np.ndarray] = {}
    annotations: List[dict] = []
    registry: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    ann_id = 1

    for i in range(n_images):
        image_id = 1000 + i
        canvas = (
            rng.integers(40, 80, size=(height, width, 3), dtype=np.int16)
            .astype(np.uint8)
        )
        n_obj = int(rng.integers(1, 7))
        boxes: List[Tuple[float, float, float, float]] = []
        classes: List[int] = []
        for _ in range(n_obj):
            w = float(rng.uniform(20, width / 3))
            h = float(rng.uniform(20, height / 3))
            x1 = float(rng.uniform(0, width - w))
            y1 = float(rng.uniform(0, height - h))
            cls = int(rng.integers(0, n_classes))
            _draw_rect(
                canvas,
                (x1, y1, x1 + w, y1 + h),
                np.array(
                    [(cls * 29) % 256, (cls * 71 + 60) % 256, (cls * 113 + 120) % 256],
                    dtype=np.uint8,
                ),
            )
            boxes.append((x1, y1, x1 + w, y1 + h))
            classes.append(cls)
            annotations.append(
                {
                    "id": ann_id,
                    "image_id": image_id,
                    "category_id": COCO80_TO_COCO91[cls],
                    "bbox": [x1, y1, w, h],
                    "area": w * h,
                    "iscrowd": 0,
                }
            )
            ann_id += 1

        det_boxes: List[List[float]] = []
        det_scores: List[float] = []
        det_classes: List[int] = []
        for box, cls in zip(boxes, classes):
            if rng.random() < drop_rate:
                continue
            jitter = rng.normal(0.0, jitter_px, size=4)
            noisy = [float(v + j) for v, j in zip(box, jitter)]
            out_cls = cls
            if rng.random() < swap_rate:
                out_cls = int((cls + 1 + rng.integers(0, n_classes - 1)) % n_classes)
            det_boxes.append(noisy)
            det_scores.append(float(rng.uniform(0.45, 0.99)))
            det_classes.append(out_cls)
            if rng.random() < duplicate_rate:
                jitter2 = rng.normal(0.0, jitter_px * 2.0, size=4)
                det_boxes.append([float(v + j) for v, j in zip(box, jitter2)])
                det_scores.append(float(rng.uniform(0.10, 0.44)))
                det_classes.append(out_cls)

        for _ in range(int(rng.poisson(false_positive_rate))):
            w = float(rng.uniform(20, width / 4))
            h = float(rng.uniform(20, height / 4))
            x1 = float(rng.uniform(0, width - w))
            y1 = float(rng.uniform(0, height - h))
            det_boxes.append([x1, y1, x1 + w, y1 + h])
            det_scores.append(float(rng.uniform(0.05, 0.5)))
            det_classes.append(int(rng.integers(0, n_classes)))

        images[image_id] = canvas
        registry[image_key(canvas)] = (
            np.asarray(det_boxes, dtype=np.float64).reshape(-1, 4),
            np.asarray(det_scores, dtype=np.float64),
            np.asarray(det_classes, dtype=np.int64),
        )

    ground_truth = GroundTruth(
        image_ids=sorted(images),
        category_ids=sorted(COCO80_TO_COCO91),
        annotations=annotations,
    )
    detector = MockYoloDetector(registry=registry, name="synthetic-mock")
    return SyntheticDataset(
        images=images, ground_truth=ground_truth, detector=detector
    )
