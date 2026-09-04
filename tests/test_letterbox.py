"""Letterbox padding and the inverse mapping back to original pixels."""

from __future__ import annotations

import numpy as np
import pytest

from detbench.models.letterbox import LetterboxTransform, letterbox


def _image(h: int, w: int) -> np.ndarray:
    return np.full((h, w, 3), 200, dtype=np.uint8)


def test_output_matches_requested_network_size():
    padded, _ = letterbox(_image(480, 640), (640, 640))
    assert padded.shape == (640, 640, 3)


def test_aspect_ratio_is_preserved():
    _, t = letterbox(_image(480, 640), (640, 640))
    assert t.scale == pytest.approx(1.0)
    assert t.pad_x == pytest.approx(0.0)
    assert t.pad_y == pytest.approx(80.0)


def test_padding_uses_the_yolo_grey():
    padded, t = letterbox(_image(480, 640), (640, 640))
    assert tuple(int(v) for v in padded[0, 0]) == (114, 114, 114)
    assert tuple(int(v) for v in padded[int(t.pad_y) + 5, 5]) == (200, 200, 200)


def test_round_trip_recovers_original_coordinates():
    _, t = letterbox(_image(427, 640), (640, 640))
    boxes = np.array(
        [[0.0, 0.0, 100.0, 80.0], [12.5, 33.25, 400.0, 300.0], [1.0, 1.0, 2.0, 2.0]]
    )
    recovered = t.inverse_xyxy(t.forward_xyxy(boxes))
    assert np.allclose(recovered, boxes, atol=1e-9)


def test_round_trip_holds_for_portrait_images():
    _, t = letterbox(_image(640, 360), (640, 640))
    boxes = np.array([[10.0, 20.0, 300.0, 500.0]])
    assert np.allclose(t.inverse_xyxy(t.forward_xyxy(boxes)), boxes, atol=1e-9)


def test_round_trip_holds_for_tiny_images():
    _, t = letterbox(_image(17, 23), (640, 640))
    boxes = np.array([[1.0, 2.0, 15.0, 16.0]])
    assert np.allclose(t.inverse_xyxy(t.forward_xyxy(boxes)), boxes, atol=1e-9)


def test_inverse_clips_to_the_original_image():
    t = LetterboxTransform(
        scale=1.0, pad_x=0.0, pad_y=0.0, orig_w=100, orig_h=50, net_w=100, net_h=50
    )
    out = t.inverse_xyxy(np.array([[-20.0, -30.0, 500.0, 500.0]]))
    assert out[0].tolist() == [0.0, 0.0, 100.0, 50.0]


def test_inverse_can_leave_boxes_unclipped():
    t = LetterboxTransform(
        scale=1.0, pad_x=0.0, pad_y=0.0, orig_w=100, orig_h=50, net_w=100, net_h=50
    )
    out = t.inverse_xyxy(np.array([[-20.0, -30.0, 500.0, 500.0]]), clip=False)
    assert out[0].tolist() == [-20.0, -30.0, 500.0, 500.0]


def test_scale_up_can_be_disabled():
    _, t = letterbox(_image(100, 100), (640, 640), scale_up=False)
    assert t.scale == pytest.approx(1.0)


def test_padding_is_centred():
    _, t = letterbox(_image(320, 640), (640, 640))
    assert t.pad_x == pytest.approx(0.0)
    assert t.pad_y == pytest.approx(160.0)


def test_non_three_channel_input_raises():
    with pytest.raises(ValueError):
        letterbox(np.zeros((10, 10), dtype=np.uint8))


def test_forward_maps_into_network_bounds():
    _, t = letterbox(_image(480, 640), (640, 640))
    out = t.forward_xyxy(np.array([[0.0, 0.0, 640.0, 480.0]]))
    assert out[0, 0] >= 0 and out[0, 2] <= t.net_w
    assert out[0, 1] >= 0 and out[0, 3] <= t.net_h
