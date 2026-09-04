"""IoU geometry, including the degenerate cases that break naive code."""

from __future__ import annotations

import numpy as np
import pytest

from detbench.metrics.box_ops import (
    box_areas,
    iou_matrix,
    xywh_to_xyxy,
    xyxy_to_xywh,
)


def test_identical_boxes_have_iou_one():
    box = np.array([[10.0, 20.0, 30.0, 40.0]])
    assert iou_matrix(box, box)[0, 0] == pytest.approx(1.0)


def test_disjoint_boxes_have_zero_iou():
    a = np.array([[0.0, 0.0, 10.0, 10.0]])
    b = np.array([[50.0, 50.0, 10.0, 10.0]])
    assert iou_matrix(a, b)[0, 0] == pytest.approx(0.0)


def test_touching_edges_do_not_overlap():
    a = np.array([[0.0, 0.0, 10.0, 10.0]])
    b = np.array([[10.0, 0.0, 10.0, 10.0]])
    assert iou_matrix(a, b)[0, 0] == pytest.approx(0.0)


def test_containment_is_ratio_of_areas():
    outer = np.array([[0.0, 0.0, 10.0, 10.0]])
    inner = np.array([[2.0, 2.0, 5.0, 5.0]])
    assert iou_matrix(inner, outer)[0, 0] == pytest.approx(25.0 / 100.0)


def test_half_overlap():
    a = np.array([[0.0, 0.0, 10.0, 10.0]])
    b = np.array([[5.0, 0.0, 10.0, 10.0]])
    # intersection 50, union 150
    assert iou_matrix(a, b)[0, 0] == pytest.approx(50.0 / 150.0)


def test_zero_area_boxes_give_zero_not_nan():
    a = np.array([[5.0, 5.0, 0.0, 0.0]])
    b = np.array([[0.0, 0.0, 10.0, 10.0]])
    value = iou_matrix(a, b)[0, 0]
    assert np.isfinite(value)
    assert value == pytest.approx(0.0)


def test_two_zero_area_boxes_do_not_divide_by_zero():
    a = np.array([[5.0, 5.0, 0.0, 0.0]])
    assert iou_matrix(a, a)[0, 0] == pytest.approx(0.0)


def test_negative_dimensions_clamp_to_zero_area():
    assert box_areas(np.array([[0.0, 0.0, -4.0, 5.0]]))[0] == pytest.approx(0.0)


def test_crowd_uses_intersection_over_detection_area():
    detection = np.array([[0.0, 0.0, 10.0, 10.0]])
    crowd = np.array([[0.0, 0.0, 100.0, 100.0]])
    plain = iou_matrix(detection, crowd)[0, 0]
    ioa = iou_matrix(detection, crowd, np.array([1]))[0, 0]
    assert plain == pytest.approx(100.0 / 10000.0)
    assert ioa == pytest.approx(1.0)


def test_crowd_flag_applies_per_ground_truth():
    detection = np.array([[0.0, 0.0, 10.0, 10.0]])
    gts = np.array([[0.0, 0.0, 100.0, 100.0], [0.0, 0.0, 100.0, 100.0]])
    out = iou_matrix(detection, gts, np.array([1, 0]))
    assert out[0, 0] == pytest.approx(1.0)
    assert out[0, 1] == pytest.approx(0.01)


def test_empty_inputs_give_correctly_shaped_output():
    assert iou_matrix(np.zeros((0, 4)), np.zeros((3, 4))).shape == (0, 3)
    assert iou_matrix(np.zeros((2, 4)), np.zeros((0, 4))).shape == (2, 0)


def test_iscrowd_length_mismatch_raises():
    with pytest.raises(ValueError):
        iou_matrix(np.zeros((1, 4)), np.zeros((2, 4)), np.array([1]))


def test_xywh_xyxy_round_trip():
    boxes = np.array([[3.0, 4.0, 10.0, 20.0], [0.0, 0.0, 1.5, 2.5]])
    assert np.allclose(xyxy_to_xywh(xywh_to_xyxy(boxes)), boxes)


def test_iou_matrix_shape_matches_inputs():
    out = iou_matrix(np.zeros((4, 4)), np.zeros((7, 4)))
    assert out.shape == (4, 7)
