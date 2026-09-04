"""Non-maximum suppression: score order, overlap removal, class awareness."""

from __future__ import annotations

import numpy as np
import pytest

from detbench.models.nms import batched_nms, nms


def test_highest_scoring_box_always_survives():
    boxes = np.array([[0, 0, 10, 10], [1, 1, 11, 11], [2, 2, 12, 12]], dtype=float)
    scores = np.array([0.3, 0.9, 0.5])
    keep = nms(boxes, scores, 0.5)
    assert keep[0] == 1


def test_overlapping_lower_scoring_boxes_are_suppressed():
    boxes = np.array([[0, 0, 10, 10], [0, 0, 10, 10]], dtype=float)
    keep = nms(boxes, np.array([0.9, 0.8]), 0.5)
    assert keep.tolist() == [0]


def test_non_overlapping_boxes_are_all_kept():
    boxes = np.array([[0, 0, 10, 10], [100, 100, 110, 110]], dtype=float)
    keep = nms(boxes, np.array([0.9, 0.8]), 0.5)
    assert sorted(keep.tolist()) == [0, 1]


def test_output_is_in_descending_score_order():
    boxes = np.array(
        [[0, 0, 10, 10], [100, 0, 110, 10], [200, 0, 210, 10]], dtype=float
    )
    scores = np.array([0.1, 0.9, 0.5])
    keep = nms(boxes, scores, 0.5)
    assert scores[keep].tolist() == sorted(scores[keep].tolist(), reverse=True)


def test_threshold_boundary_keeps_boxes_at_exactly_the_threshold():
    # IoU is exactly 1/3; a threshold of 1/3 must keep the second box, since
    # suppression requires strictly greater overlap.
    boxes = np.array([[0, 0, 10, 10], [5, 0, 15, 10]], dtype=float)
    assert nms(boxes, np.array([0.9, 0.8]), 1.0 / 3.0).tolist() == [0, 1]
    assert nms(boxes, np.array([0.9, 0.8]), 0.33).tolist() == [0]


def test_class_aware_nms_keeps_overlapping_boxes_of_different_classes():
    boxes = np.array([[0, 0, 10, 10], [0, 0, 10, 10]], dtype=float)
    scores = np.array([0.9, 0.8])
    assert len(nms(boxes, scores, 0.5)) == 1
    assert len(batched_nms(boxes, scores, np.array([0, 1]), 0.5)) == 2


def test_class_aware_nms_still_suppresses_within_a_class():
    boxes = np.array([[0, 0, 10, 10], [0, 0, 10, 10]], dtype=float)
    keep = batched_nms(boxes, np.array([0.9, 0.8]), np.array([3, 3]), 0.5)
    assert keep.tolist() == [0]


def test_class_offsets_do_not_leak_between_distant_classes():
    # Class 79's shifted boxes must not collide with class 0's.
    boxes = np.array([[0, 0, 640, 640], [0, 0, 640, 640]], dtype=float)
    keep = batched_nms(boxes, np.array([0.9, 0.8]), np.array([0, 79]), 0.5)
    assert len(keep) == 2


def test_max_dets_caps_the_output():
    boxes = np.array([[i * 100, 0, i * 100 + 10, 10] for i in range(10)], dtype=float)
    keep = batched_nms(
        boxes, np.linspace(0.1, 0.9, 10), np.zeros(10, dtype=int), 0.5, max_dets=3
    )
    assert len(keep) == 3


def test_empty_input_returns_empty_output():
    assert nms(np.zeros((0, 4)), np.zeros((0,)), 0.5).shape == (0,)
    assert batched_nms(
        np.zeros((0, 4)), np.zeros((0,)), np.zeros((0,), dtype=int), 0.5
    ).shape == (0,)


def test_mismatched_lengths_raise():
    with pytest.raises(ValueError):
        nms(np.zeros((3, 4)), np.zeros((2,)), 0.5)


def test_single_box_is_kept():
    assert nms(np.array([[0.0, 0.0, 1.0, 1.0]]), np.array([0.5]), 0.5).tolist() == [0]
