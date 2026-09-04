"""Dataset access and the evaluation runner."""

from .dataset import CocoDetectionDataset, ImageRecord, load_image
from .runner import (
    RunResult,
    evaluate_run,
    file_sha256,
    run_detector,
    score_detections,
    write_coco_results,
)
from .synthetic import SyntheticDataset, make_synthetic_dataset

__all__ = [
    "CocoDetectionDataset",
    "ImageRecord",
    "RunResult",
    "SyntheticDataset",
    "evaluate_run",
    "file_sha256",
    "load_image",
    "make_synthetic_dataset",
    "run_detector",
    "score_detections",
    "write_coco_results",
]
