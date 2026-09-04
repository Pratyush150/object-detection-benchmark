"""Quantisation plumbing: preprocessing, the decode tail, and the reports."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from detbench.quantize.calibration import (
    CALIBRATION_AVAILABLE,
    ImageCalibrationReader,
    preprocess_for_calibration,
)
from detbench.quantize.ptq import (
    QUANTIZATION_AVAILABLE,
    QuantizationReport,
)

from conftest import FP32_MODEL, requires_model

requires_quantization = pytest.mark.skipif(
    not QUANTIZATION_AVAILABLE,
    reason="onnxruntime.quantization is not available",
)


def test_calibration_blob_matches_the_inference_blob():
    # The calibrated ranges are only valid if calibration preprocesses the
    # image exactly as inference does.
    from detbench.models.letterbox import letterbox

    image = np.random.default_rng(0).integers(
        0, 255, (300, 400, 3), dtype=np.uint8
    )
    blob = preprocess_for_calibration(image, (640, 640))
    padded, _ = letterbox(image, (640, 640))
    expected = np.ascontiguousarray(
        padded[:, :, ::-1].transpose(2, 0, 1), dtype=np.float32
    )[None] / 255.0
    assert np.array_equal(blob, expected)


def test_calibration_blob_shape_and_range():
    image = np.zeros((100, 200, 3), dtype=np.uint8)
    blob = preprocess_for_calibration(image, (320, 320))
    assert blob.shape == (1, 3, 320, 320)
    assert blob.dtype == np.float32
    assert 0.0 <= float(blob.min()) and float(blob.max()) <= 1.0


@pytest.mark.skipif(not CALIBRATION_AVAILABLE, reason="calibrator unavailable")
def test_reader_streams_every_image_once_then_stops():
    seen = []

    def loader(path):
        seen.append(path)
        return np.zeros((64, 64, 3), dtype=np.uint8)

    reader = ImageCalibrationReader(
        [Path("a.jpg"), Path("b.jpg")], "images", (64, 64), loader=loader
    )
    assert reader.get_next() is not None
    assert reader.get_next() is not None
    assert reader.get_next() is None
    assert len(seen) == 2


@pytest.mark.skipif(not CALIBRATION_AVAILABLE, reason="calibrator unavailable")
def test_reader_rewinds_for_multi_pass_calibration():
    reader = ImageCalibrationReader(
        [Path("a.jpg")], "images", (64, 64),
        loader=lambda p: np.zeros((64, 64, 3), dtype=np.uint8),
    )
    assert reader.get_next() is not None
    assert reader.get_next() is None
    reader.rewind()
    assert reader.get_next() is not None


@pytest.mark.skipif(not CALIBRATION_AVAILABLE, reason="calibrator unavailable")
def test_reader_uses_the_declared_input_name():
    reader = ImageCalibrationReader(
        [Path("a.jpg")], "input_tensor", (32, 32),
        loader=lambda p: np.zeros((32, 32, 3), dtype=np.uint8),
    )
    assert list(reader.get_next()) == ["input_tensor"]


def test_empty_calibration_set_raises():
    with pytest.raises(ValueError):
        ImageCalibrationReader([], "images")


def test_report_reports_the_size_ratio():
    report = QuantizationReport(
        variant="int8", model_path=Path("a"), source_path=Path("b"),
        size_bytes=250, source_size_bytes=1000,
    )
    assert report.size_ratio == pytest.approx(0.25)


@requires_quantization
@requires_model
def test_decode_tail_is_found_and_is_small():
    from detbench.quantize.ptq import decode_tail_nodes

    nodes = decode_tail_nodes(FP32_MODEL)
    # The tail is the reshape/concat/softmax/arithmetic block plus the DFL
    # convolution: a couple of dozen nodes, not the whole graph.
    assert 5 < len(nodes) < 60
    assert any("Concat" in n for n in nodes)
    assert any("Softmax" in n for n in nodes)


@requires_quantization
@requires_model
def test_decode_tail_excludes_the_backbone_convolutions():
    import onnx

    from detbench.quantize.ptq import decode_tail_nodes

    excluded = set(decode_tail_nodes(FP32_MODEL))
    model = onnx.load(str(FP32_MODEL))
    convs = [n.name for n in model.graph.node if n.op_type == "Conv"]
    kept = [n for n in convs if n not in excluded]
    # Only the DFL projection is excluded; every other convolution stays
    # available for quantisation, which is where the speedup comes from.
    assert len(kept) >= len(convs) - 1
