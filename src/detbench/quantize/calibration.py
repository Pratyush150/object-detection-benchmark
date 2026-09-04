"""Calibration data feeding for static post-training quantisation.

Static INT8 needs to know the dynamic range of every activation tensor, which
means running real images through the FP32 graph and recording min/max (or a
percentile / entropy estimate) per tensor. Two things about that matter more
than the code:

* **The calibration images must be preprocessed exactly as at inference time.**
  Same letterbox, same padding colour, same channel order, same scaling. A
  mismatch here produces ranges that are wrong in a way that looks like the
  quantiser is broken.
* **The calibration set must not overlap the evaluation set.** Otherwise the
  quantiser has seen the exact activations it will be scored on, and the
  reported accuracy drop is optimistically small. This module takes an explicit
  list of image ids and the caller is expected to pass a disjoint split; see
  :meth:`detbench.eval.dataset.CocoDetectionDataset.split_ids`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np

from ..models.letterbox import letterbox

__all__ = [
    "CALIBRATION_AVAILABLE",
    "ImageCalibrationReader",
    "preprocess_for_calibration",
]

try:  # pragma: no cover - depends on the installed onnxruntime build
    from onnxruntime.quantization import CalibrationDataReader

    CALIBRATION_AVAILABLE = True
except ImportError:  # pragma: no cover
    CalibrationDataReader = object  # type: ignore[assignment,misc]
    CALIBRATION_AVAILABLE = False


def preprocess_for_calibration(
    image: np.ndarray, input_size: Tuple[int, int] = (640, 640)
) -> np.ndarray:
    """Produce the exact ``(1, 3, H, W)`` float32 blob inference would use."""
    padded, _ = letterbox(image, input_size)
    blob = padded[:, :, ::-1].transpose(2, 0, 1)
    return np.ascontiguousarray(blob, dtype=np.float32)[None] / 255.0


class ImageCalibrationReader(CalibrationDataReader):  # type: ignore[misc]
    """Feed preprocessed images to the ONNX Runtime calibrator.

    Args:
        image_paths: Files to calibrate on.
        input_name: Name of the model's input tensor.
        input_size: ``(width, height)`` network input.
        loader: Callable that reads a path into a BGR uint8 array.
    """

    def __init__(
        self,
        image_paths: Sequence[Path],
        input_name: str,
        input_size: Tuple[int, int] = (640, 640),
        loader=None,
    ) -> None:
        if not image_paths:
            raise ValueError("calibration set is empty")
        self.image_paths: List[Path] = [Path(p) for p in image_paths]
        self.input_name = input_name
        self.input_size = input_size
        if loader is None:
            from ..eval.dataset import load_image as _default_loader

            loader = _default_loader
        self._loader = loader
        self._iter: Optional[Iterator[Path]] = None
        self.consumed = 0

    def rewind(self) -> None:
        """Restart the stream, as multi-pass calibrators require."""
        self._iter = None
        self.consumed = 0

    def get_next(self) -> Optional[Dict[str, np.ndarray]]:
        """Return the next calibration batch, or None when exhausted."""
        if self._iter is None:
            self._iter = iter(self.image_paths)
        try:
            path = next(self._iter)
        except StopIteration:
            return None
        self.consumed += 1
        blob = preprocess_for_calibration(self._loader(path), self.input_size)
        return {self.input_name: blob}
