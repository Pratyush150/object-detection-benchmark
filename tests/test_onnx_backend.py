"""The ONNX Runtime backend, exercised against the real exported model."""

from __future__ import annotations

import numpy as np
import pytest

from detbench.models.onnx_yolo import ONNX_AVAILABLE, OnnxYoloDetector

from conftest import FP32_MODEL, requires_model

requires_onnx = pytest.mark.skipif(
    not ONNX_AVAILABLE, reason="onnxruntime is not installed"
)


def test_missing_model_file_raises(tmp_path):
    if not ONNX_AVAILABLE:
        pytest.skip("onnxruntime is not installed")
    with pytest.raises(FileNotFoundError):
        OnnxYoloDetector(tmp_path / "nope.onnx")


@requires_onnx
@requires_model
def test_model_input_is_the_expected_shape():
    detector = OnnxYoloDetector(FP32_MODEL)
    shape = detector.session.get_inputs()[0].shape
    assert list(shape) == [1, 3, 640, 640]
    detector.close()


@requires_onnx
@requires_model
def test_blank_image_produces_no_confident_detections():
    detector = OnnxYoloDetector(FP32_MODEL, conf_threshold=0.5)
    result = detector.predict(np.zeros((480, 640, 3), dtype=np.uint8))
    assert len(result) == 0
    detector.close()


@requires_onnx
@requires_model
def test_detections_stay_inside_the_original_image():
    detector = OnnxYoloDetector(FP32_MODEL, conf_threshold=0.05)
    rng = np.random.default_rng(0)
    image = rng.integers(0, 255, (427, 640, 3), dtype=np.uint8)
    result = detector.predict(image)
    if len(result):
        assert result.boxes_xyxy[:, 0].min() >= -1e-6
        assert result.boxes_xyxy[:, 1].min() >= -1e-6
        assert result.boxes_xyxy[:, 2].max() <= 640 + 1e-6
        assert result.boxes_xyxy[:, 3].max() <= 427 + 1e-6
    detector.close()


@requires_onnx
@requires_model
def test_inference_is_deterministic():
    detector = OnnxYoloDetector(FP32_MODEL, conf_threshold=0.05)
    image = np.random.default_rng(7).integers(
        0, 255, (300, 500, 3), dtype=np.uint8
    )
    a = detector.predict(image)
    b = detector.predict(image)
    assert np.array_equal(a.boxes_xyxy, b.boxes_xyxy)
    assert np.array_equal(a.scores, b.scores)
    detector.close()


@requires_onnx
@requires_model
def test_all_four_stages_are_timed():
    detector = OnnxYoloDetector(FP32_MODEL)
    result = detector.predict(np.zeros((320, 480, 3), dtype=np.uint8))
    assert set(result.stage_times_ms) == {
        "preprocess", "inference", "nms", "postprocess", "total"
    }
    assert result.stage_times_ms["inference"] > 0
    detector.close()


@requires_onnx
@requires_model
def test_a_higher_confidence_threshold_never_adds_detections():
    low = OnnxYoloDetector(FP32_MODEL, conf_threshold=0.01)
    high = OnnxYoloDetector(FP32_MODEL, conf_threshold=0.5)
    image = np.random.default_rng(3).integers(
        0, 255, (400, 600, 3), dtype=np.uint8
    )
    assert len(high.predict(image)) <= len(low.predict(image))
    low.close()
    high.close()


@requires_onnx
@requires_model
def test_reported_model_size_matches_the_file():
    detector = OnnxYoloDetector(FP32_MODEL)
    assert detector.size_bytes == FP32_MODEL.stat().st_size
    detector.close()
