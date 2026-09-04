"""Decoding a raw YOLO head tensor."""

from __future__ import annotations

import numpy as np
import pytest

from detbench.models.decode import decode_yolo_head


def _head(n_classes: int = 80, n_anchors: int = 100) -> np.ndarray:
    return np.zeros((1, 4 + n_classes, n_anchors), dtype=np.float32)


def test_centre_width_height_becomes_corner_form():
    head = _head()
    head[0, :4, 0] = [50.0, 60.0, 20.0, 10.0]
    head[0, 4 + 3, 0] = 0.9
    boxes, scores, classes = decode_yolo_head(head, conf_threshold=0.1)
    assert boxes[0].tolist() == [40.0, 55.0, 60.0, 65.0]
    assert scores[0] == pytest.approx(0.9, abs=1e-6)
    assert classes[0] == 3


def test_confidence_threshold_filters_candidates():
    head = _head()
    head[0, 4, 0] = 0.05
    head[0, 4, 1] = 0.5
    _, scores, _ = decode_yolo_head(head, conf_threshold=0.1)
    assert len(scores) == 1


def test_single_label_mode_keeps_only_the_best_class():
    head = _head()
    head[0, 4 + 1, 0] = 0.4
    head[0, 4 + 7, 0] = 0.8
    _, scores, classes = decode_yolo_head(head, conf_threshold=0.1, multi_label=False)
    assert len(scores) == 1
    assert classes[0] == 7


def test_multi_label_mode_emits_every_class_above_threshold():
    head = _head()
    head[0, 4 + 1, 0] = 0.4
    head[0, 4 + 7, 0] = 0.8
    _, scores, classes = decode_yolo_head(head, conf_threshold=0.1, multi_label=True)
    assert len(scores) == 2
    assert sorted(classes.tolist()) == [1, 7]


def test_transposed_layout_is_detected():
    head = _head()
    head[0, :4, 0] = [10.0, 10.0, 4.0, 4.0]
    head[0, 4, 0] = 0.7
    a = decode_yolo_head(head, conf_threshold=0.1)
    b = decode_yolo_head(np.transpose(head, (0, 2, 1)), conf_threshold=0.1)
    assert np.allclose(a[0], b[0])
    assert np.allclose(a[1], b[1])


def test_batch_dimension_is_optional():
    head = _head()
    head[0, 4, 0] = 0.7
    assert len(decode_yolo_head(head[0], conf_threshold=0.1)[1]) == 1


def test_batch_larger_than_one_raises():
    with pytest.raises(ValueError):
        decode_yolo_head(np.zeros((2, 84, 10)))


def test_wrong_rank_raises():
    with pytest.raises(ValueError):
        decode_yolo_head(np.zeros((84,)))


def test_no_candidates_returns_empty_arrays():
    boxes, scores, classes = decode_yolo_head(_head(), conf_threshold=0.5)
    assert boxes.shape == (0, 4)
    assert scores.shape == (0,)
    assert classes.shape == (0,)


def test_number_of_classes_is_taken_from_the_tensor():
    head = np.zeros((1, 4 + 5, 20), dtype=np.float32)
    head[0, 4 + 4, 0] = 0.9
    _, _, classes = decode_yolo_head(head, conf_threshold=0.1)
    assert classes[0] == 4
