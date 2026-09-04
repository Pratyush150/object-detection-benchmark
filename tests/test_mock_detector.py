"""The mock backend, which is also the pipeline's own coordinate test."""

from __future__ import annotations

import numpy as np
import pytest

from detbench.coco_classes import COCO80_TO_COCO91
from detbench.models.base import Detections
from detbench.models.mock import MockYoloDetector, image_key


def _image(h=360, w=480):
    return np.zeros((h, w, 3), dtype=np.uint8)


def test_planted_boxes_come_back_in_original_coordinates():
    img = _image()
    det = MockYoloDetector()
    boxes = np.array([[10.0, 20.0, 110.0, 220.0], [300.0, 100.0, 400.0, 300.0]])
    det.register(img, boxes, np.array([0.9, 0.5]), np.array([0, 7]))
    out = det.predict(img)
    assert np.allclose(out.boxes_xyxy, boxes, atol=1e-4)


def test_unknown_image_produces_no_detections():
    det = MockYoloDetector()
    assert len(det.predict(_image())) == 0


def test_image_key_depends_on_content():
    a, b = _image(), _image()
    b[0, 0, 0] = 1
    assert image_key(a) != image_key(b)


def test_stage_timings_are_reported():
    img = _image()
    det = MockYoloDetector()
    det.register(img, np.array([[1.0, 1.0, 5.0, 5.0]]), np.array([0.9]),
                 np.array([0]))
    out = det.predict(img)
    assert set(out.stage_times_ms) == {
        "preprocess", "inference", "nms", "postprocess", "total"
    }
    assert out.stage_times_ms["total"] > 0


def test_coco_conversion_maps_indices_to_category_ids():
    result = Detections(
        boxes_xyxy=np.array([[10.0, 20.0, 30.0, 50.0]]),
        scores=np.array([0.75]),
        class_ids=np.array([11]),
    )
    record = result.to_coco(image_id=42)[0]
    assert record["image_id"] == 42
    assert record["category_id"] == COCO80_TO_COCO91[11] == 13
    assert record["bbox"] == [10.0, 20.0, 20.0, 30.0]
    assert record["score"] == pytest.approx(0.75)


def test_coco_conversion_sorts_by_score_and_applies_max_dets():
    result = Detections(
        boxes_xyxy=np.array([[0.0, 0.0, 1.0, 1.0], [0.0, 0.0, 2.0, 2.0]]),
        scores=np.array([0.2, 0.8]),
        class_ids=np.array([0, 1]),
    )
    records = result.to_coco(image_id=1, max_dets=1)
    assert len(records) == 1
    assert records[0]["score"] == pytest.approx(0.8)


def test_empty_detections_have_the_right_shapes():
    empty = Detections.empty()
    assert len(empty) == 0
    assert empty.boxes_xyxy.shape == (0, 4)
    assert empty.to_coco(1) == []


def test_duplicate_planted_boxes_are_suppressed_by_nms():
    img = _image()
    det = MockYoloDetector(iou_threshold=0.5)
    boxes = np.array([[10.0, 10.0, 110.0, 110.0], [12.0, 12.0, 112.0, 112.0]])
    det.register(img, boxes, np.array([0.9, 0.8]), np.array([0, 0]))
    assert len(det.predict(img)) == 1


def test_same_box_different_classes_both_survive():
    img = _image()
    det = MockYoloDetector(iou_threshold=0.5)
    boxes = np.array([[10.0, 10.0, 110.0, 110.0], [10.0, 10.0, 110.0, 110.0]])
    det.register(img, boxes, np.array([0.9, 0.8]), np.array([0, 5]))
    assert len(det.predict(img)) == 2


def test_warmup_does_not_raise_without_a_registry():
    MockYoloDetector().warmup(runs=2, size=64)
