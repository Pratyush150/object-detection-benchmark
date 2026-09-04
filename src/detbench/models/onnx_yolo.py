"""ONNX Runtime inference wrapper for a YOLO detection head.

The pipeline is deliberately split into four timed stages, because they behave
very differently under quantisation: preprocessing is fixed-cost image work,
inference is the part INT8 speeds up, NMS depends on how many candidates
survive the confidence floor, and postprocessing is cheap array shuffling.
Reporting a single end-to-end number hides which of the four actually moved.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np

from .base import Detections, Detector
from .decode import decode_yolo_head
from .letterbox import letterbox
from .nms import batched_nms

__all__ = ["OnnxYoloDetector", "ONNX_AVAILABLE"]

try:  # pragma: no cover - exercised by whichever branch the environment takes
    import onnxruntime as ort

    ONNX_AVAILABLE = True
except ImportError:  # pragma: no cover
    ort = None  # type: ignore[assignment]
    ONNX_AVAILABLE = False


class OnnxYoloDetector(Detector):
    """Run a YOLO ONNX model end to end on BGR uint8 images.

    Args:
        model_path: Path to the ``.onnx`` file.
        input_size: ``(width, height)`` of the network input. Must match the
            exported model unless it has dynamic axes.
        conf_threshold: Confidence floor before NMS. The COCO default of 0.001
            is intentionally low; see :mod:`detbench.models.decode`.
        iou_threshold: NMS IoU threshold.
        max_dets: Cap on detections kept per image after NMS.
        multi_label: Allow one anchor to emit several classes.
        providers: ONNX Runtime execution providers. Defaults to CPU only, so
            a machine with a half-working GPU stack does not silently change
            the numbers.
        intra_op_threads: Thread cap for the session. Pinned to 1 for latency
            measurements by the profiling entry points, because thread-pool
            contention is the single biggest source of run-to-run variance in
            CPU latency percentiles.
    """

    def __init__(
        self,
        model_path: str | Path,
        input_size: Tuple[int, int] = (640, 640),
        conf_threshold: float = 0.001,
        iou_threshold: float = 0.7,
        max_dets: int = 300,
        multi_label: bool = False,
        providers: Optional[Sequence[str]] = None,
        intra_op_threads: Optional[int] = None,
        name: Optional[str] = None,
    ) -> None:
        if not ONNX_AVAILABLE:
            raise RuntimeError(
                "onnxruntime is not installed; use MockDetector or install it"
            )
        self.model_path = Path(model_path)
        if not self.model_path.is_file():
            raise FileNotFoundError(f"model not found: {self.model_path}")

        self.input_size = (int(input_size[0]), int(input_size[1]))
        self.conf_threshold = float(conf_threshold)
        self.iou_threshold = float(iou_threshold)
        self.max_dets = int(max_dets)
        self.multi_label = bool(multi_label)
        self.name = name or self.model_path.stem

        options = ort.SessionOptions()
        if intra_op_threads is not None:
            options.intra_op_num_threads = int(intra_op_threads)
            options.inter_op_num_threads = 1
        options.graph_optimization_level = (
            ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        )
        self.session = ort.InferenceSession(
            str(self.model_path),
            sess_options=options,
            providers=list(providers) if providers else ["CPUExecutionProvider"],
        )
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [o.name for o in self.session.get_outputs()]

    @property
    def size_bytes(self) -> int:
        """On-disk size of the model file."""
        return self.model_path.stat().st_size

    def predict(self, image: np.ndarray) -> Detections:
        """Detect objects in one ``(H, W, 3)`` BGR uint8 image."""
        t0 = time.perf_counter()
        padded, transform = letterbox(image, self.input_size)
        # BGR -> RGB, HWC -> CHW, uint8 -> float32 in [0, 1].
        blob = padded[:, :, ::-1].transpose(2, 0, 1)
        blob = np.ascontiguousarray(blob, dtype=np.float32) / 255.0
        blob = blob[None]
        t1 = time.perf_counter()

        raw = self.session.run(self.output_names, {self.input_name: blob})[0]
        t2 = time.perf_counter()

        boxes, scores, class_ids = decode_yolo_head(
            raw, self.conf_threshold, self.multi_label
        )
        keep = batched_nms(
            boxes, scores, class_ids, self.iou_threshold, self.max_dets
        )
        t3 = time.perf_counter()

        boxes = transform.inverse_xyxy(boxes[keep]) if keep.size else np.zeros((0, 4))
        result = Detections(
            boxes_xyxy=boxes,
            scores=scores[keep],
            class_ids=class_ids[keep],
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

    def close(self) -> None:
        """Drop the session so its memory arena is released."""
        self.session = None  # type: ignore[assignment]
