"""The 101-point interpolated AP, checked against a hand-computed value."""

from __future__ import annotations

import numpy as np
import pytest

from detbench.metrics.coco_map import (
    AreaRange,
    COCOMeanAP,
    EvalParams,
    GroundTruth,
)

ONE_BAND = (AreaRange("all", 0.0, 1e10),)


def _params():
    return EvalParams(
        iou_thresholds=np.array([0.5]), area_ranges=ONE_BAND, max_dets=(100,)
    )


def _gt(n_objects: int) -> GroundTruth:
    anns = []
    for i in range(n_objects):
        anns.append(
            {
                "id": i + 1,
                "image_id": 1,
                "category_id": 1,
                "bbox": [100.0 * i, 0.0, 10.0, 10.0],
                "area": 100.0,
                "iscrowd": 0,
            }
        )
    return GroundTruth(image_ids=[1], category_ids=[1], annotations=anns)


def _det(index: int, score: float, offset: float = 0.0) -> dict:
    return {
        "image_id": 1,
        "category_id": 1,
        "bbox": [100.0 * index + offset, 0.0, 10.0, 10.0],
        "score": score,
    }


def test_perfect_detector_scores_one():
    res = COCOMeanAP(_gt(2), _params()).evaluate([_det(0, 0.9), _det(1, 0.8)])
    assert res.ap() == pytest.approx(1.0)


def test_no_detections_scores_zero_not_nan():
    res = COCOMeanAP(_gt(2), _params()).evaluate([])
    assert res.ap() == pytest.approx(0.0)


def test_hand_computed_case_two_of_four_found():
    # Two ground truths, two detections: the first is correct, the second is a
    # background false positive scored lower.
    #
    #   rank 1: TP -> recall 0.5, precision 1.0
    #   rank 2: FP -> recall 0.5, precision 0.5
    #
    # Interpolated precision is 1.0 for recall levels 0.00..0.50 (51 of the 101
    # sample points) and 0.0 above, so AP = 51/101.
    dets = [_det(0, 0.9), {"image_id": 1, "category_id": 1,
                           "bbox": [500.0, 500.0, 10.0, 10.0], "score": 0.4}]
    res = COCOMeanAP(_gt(2), _params()).evaluate(dets)
    assert res.ap() == pytest.approx(51.0 / 101.0, abs=1e-12)


def test_hand_computed_case_one_of_three_found():
    # One correct detection out of three objects: precision 1.0 up to recall
    # 1/3. The 101 sample points at or below 1/3 are 0.00..0.33, i.e. 34 of
    # them, so AP = 34/101.
    res = COCOMeanAP(_gt(3), _params()).evaluate([_det(0, 0.9)])
    assert res.ap() == pytest.approx(34.0 / 101.0, abs=1e-12)


def test_false_positive_ranked_first_lowers_ap():
    # Same two true positives, but a background box outranks them. Precision at
    # recall 0.5 is 1/2 and at recall 1.0 is 2/3, and the monotone envelope
    # lifts the first to 2/3 as well: AP = 2/3 over all 101 points.
    dets = [
        {"image_id": 1, "category_id": 1,
         "bbox": [500.0, 500.0, 10.0, 10.0], "score": 0.99},
        _det(0, 0.9),
        _det(1, 0.8),
    ]
    res = COCOMeanAP(_gt(2), _params()).evaluate(dets)
    assert res.ap() == pytest.approx(2.0 / 3.0, abs=1e-12)


def test_precision_envelope_is_monotonically_non_increasing():
    dets = [_det(0, 0.95),
            {"image_id": 1, "category_id": 1,
             "bbox": [900.0, 900.0, 5.0, 5.0], "score": 0.9},
            _det(1, 0.5)]
    res = COCOMeanAP(_gt(2), _params()).evaluate(dets)
    _, precision = res.pr_curve(1, iou=0.5)
    assert np.all(np.diff(precision) <= 1e-12)


def test_recall_thresholds_are_101_evenly_spaced_points():
    params = EvalParams()
    assert params.recall_thresholds.shape == (101,)
    assert params.recall_thresholds[0] == pytest.approx(0.0)
    assert params.recall_thresholds[-1] == pytest.approx(1.0)
    assert np.allclose(np.diff(params.recall_thresholds), 0.01)


def test_iou_sweep_is_ten_thresholds_from_50_to_95():
    params = EvalParams()
    assert params.iou_thresholds.shape == (10,)
    assert params.iou_thresholds[0] == pytest.approx(0.5)
    assert params.iou_thresholds[-1] == pytest.approx(0.95)


def test_loose_box_passes_at_iou50_and_fails_at_iou75():
    # A box shifted by 3px on a 10px object has IoU 0.7/1.3 ~= 0.538.
    params = EvalParams(area_ranges=ONE_BAND, max_dets=(100,))
    res = COCOMeanAP(_gt(1), params).evaluate([_det(0, 0.9, offset=3.0)])
    assert res.ap(iou=0.5) == pytest.approx(1.0)
    assert res.ap(iou=0.75) == pytest.approx(0.0)
