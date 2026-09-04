"""Greedy matching: score order, single ownership, and duplicate penalties."""

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


def _params(iou=0.5):
    return EvalParams(
        iou_thresholds=np.array([iou]), area_ranges=ONE_BAND, max_dets=(100,)
    )


def _gt(boxes):
    return GroundTruth(
        image_ids=[1],
        category_ids=[1],
        annotations=[
            {
                "id": i + 1,
                "image_id": 1,
                "category_id": 1,
                "bbox": list(b),
                "area": b[2] * b[3],
                "iscrowd": 0,
            }
            for i, b in enumerate(boxes)
        ],
    )


def _d(box, score):
    return {"image_id": 1, "category_id": 1, "bbox": list(box), "score": score}


def test_one_ground_truth_can_only_be_matched_once():
    # Two identical detections on one object: one true positive, one duplicate.
    gt = _gt([[0.0, 0.0, 10.0, 10.0]])
    res = COCOMeanAP(gt, _params()).evaluate(
        [_d([0.0, 0.0, 10.0, 10.0], 0.9), _d([0.0, 0.0, 10.0, 10.0], 0.8)]
    )
    # Precision falls to 0.5 after the duplicate, but the envelope holds the
    # value reached at full recall, so AP is 1.0 only if the duplicate is
    # ranked after full recall - which it is. The recall is still 1.0.
    assert res.ar() == pytest.approx(1.0)
    assert res.ap() == pytest.approx(1.0)


def test_duplicate_ranked_first_costs_precision():
    gt = _gt([[0.0, 0.0, 10.0, 10.0], [200.0, 0.0, 10.0, 10.0]])
    good = [_d([0.0, 0.0, 10.0, 10.0], 0.9), _d([200.0, 0.0, 10.0, 10.0], 0.8)]
    with_dupe = [
        _d([0.0, 0.0, 10.0, 10.0], 0.9),
        _d([0.5, 0.5, 10.0, 10.0], 0.85),  # duplicate of the first object
        _d([200.0, 0.0, 10.0, 10.0], 0.8),
    ]
    clean = COCOMeanAP(gt, _params()).evaluate(good).ap()
    dirty = COCOMeanAP(gt, _params()).evaluate(with_dupe).ap()
    assert clean == pytest.approx(1.0)
    assert dirty < clean


def test_higher_scoring_detection_claims_the_ground_truth_first():
    # Two objects, two detections. The higher-scoring detection overlaps both
    # but fits object A better; greedy matching must give it A, leaving B for
    # the lower-scoring one, so both are true positives.
    gt = _gt([[0.0, 0.0, 10.0, 10.0], [8.0, 0.0, 10.0, 10.0]])
    dets = [_d([0.0, 0.0, 10.0, 10.0], 0.9), _d([8.0, 0.0, 10.0, 10.0], 0.5)]
    res = COCOMeanAP(gt, _params()).evaluate(dets)
    assert res.ap() == pytest.approx(1.0)
    assert res.ar() == pytest.approx(1.0)


def test_score_order_changes_the_result():
    # The same three boxes, only the scores swapped: putting the wrong box
    # first must lower AP. If it does not, the sort is being ignored.
    gt = _gt([[0.0, 0.0, 10.0, 10.0]])
    good_first = [_d([0.0, 0.0, 10.0, 10.0], 0.9),
                  _d([500.0, 500.0, 10.0, 10.0], 0.1)]
    bad_first = [_d([0.0, 0.0, 10.0, 10.0], 0.1),
                 _d([500.0, 500.0, 10.0, 10.0], 0.9)]
    assert COCOMeanAP(gt, _params()).evaluate(good_first).ap() == pytest.approx(1.0)
    assert COCOMeanAP(gt, _params()).evaluate(bad_first).ap() == pytest.approx(0.5)


def test_max_dets_truncates_by_score_not_by_order():
    gt = _gt([[0.0, 0.0, 10.0, 10.0]])
    params = EvalParams(
        iou_thresholds=np.array([0.5]), area_ranges=ONE_BAND, max_dets=(1,)
    )
    # The correct detection is supplied last but scores highest, so a maxDets
    # of 1 must keep it.
    dets = [_d([500.0, 500.0, 10.0, 10.0], 0.2), _d([0.0, 0.0, 10.0, 10.0], 0.9)]
    res = COCOMeanAP(gt, params).evaluate(dets)
    assert res.ap(max_dets=1) == pytest.approx(1.0)


def test_max_dets_limit_drops_low_scoring_true_positives():
    gt = _gt([[0.0, 0.0, 10.0, 10.0], [200.0, 0.0, 10.0, 10.0]])
    params = EvalParams(
        iou_thresholds=np.array([0.5]), area_ranges=ONE_BAND, max_dets=(1, 100)
    )
    dets = [_d([0.0, 0.0, 10.0, 10.0], 0.9), _d([200.0, 0.0, 10.0, 10.0], 0.4)]
    res = COCOMeanAP(gt, params).evaluate(dets)
    assert res.ar(max_dets=1) == pytest.approx(0.5)
    assert res.ar(max_dets=100) == pytest.approx(1.0)


def test_wrong_class_detection_never_matches():
    gt = _gt([[0.0, 0.0, 10.0, 10.0]])
    gt.category_ids = [1, 2]
    det = {"image_id": 1, "category_id": 2,
           "bbox": [0.0, 0.0, 10.0, 10.0], "score": 0.9}
    res = COCOMeanAP(gt, _params()).evaluate([det])
    assert res.ap() == pytest.approx(0.0)


def test_detection_on_an_image_without_ground_truth_is_a_false_positive():
    gt = GroundTruth(
        image_ids=[1, 2],
        category_ids=[1],
        annotations=[
            {"id": 1, "image_id": 1, "category_id": 1,
             "bbox": [0.0, 0.0, 10.0, 10.0], "area": 100.0, "iscrowd": 0}
        ],
    )
    dets = [
        {"image_id": 1, "category_id": 1,
         "bbox": [0.0, 0.0, 10.0, 10.0], "score": 0.5},
        {"image_id": 2, "category_id": 1,
         "bbox": [0.0, 0.0, 10.0, 10.0], "score": 0.9},
    ]
    res = COCOMeanAP(gt, _params()).evaluate(dets)
    assert res.ap() == pytest.approx(0.5)


def test_result_counts_reflect_the_inputs():
    gt = _gt([[0.0, 0.0, 10.0, 10.0]])
    res = COCOMeanAP(gt, _params()).evaluate([_d([0.0, 0.0, 10.0, 10.0], 0.9)])
    assert res.n_images == 1
    assert res.n_detections == 1
    assert res.n_ground_truths == 1
