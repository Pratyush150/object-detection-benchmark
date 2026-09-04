"""Crowd regions must neither reward nor penalise a detector."""

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


def _gt(entries):
    return GroundTruth(
        image_ids=[1],
        category_ids=[1],
        annotations=[
            {
                "id": i + 1,
                "image_id": 1,
                "category_id": 1,
                "bbox": list(box),
                "area": box[2] * box[3],
                "iscrowd": crowd,
            }
            for i, (box, crowd) in enumerate(entries)
        ],
    )


def _d(box, score):
    return {"image_id": 1, "category_id": 1, "bbox": list(box), "score": score}


def test_crowd_only_image_produces_no_score():
    # A crowd region is the only annotation, so there is nothing to recall and
    # AP is undefined rather than zero.
    gt = _gt([([0.0, 0.0, 100.0, 100.0], 1)])
    res = COCOMeanAP(gt, _params()).evaluate([])
    assert np.isnan(res.ap())


def test_detection_inside_a_crowd_is_not_a_false_positive():
    gt = _gt([([0.0, 0.0, 10.0, 10.0], 0), ([200.0, 200.0, 300.0, 300.0], 1)])
    clean = COCOMeanAP(gt, _params()).evaluate([_d([0.0, 0.0, 10.0, 10.0], 0.9)])
    with_crowd_hit = COCOMeanAP(gt, _params()).evaluate(
        [_d([0.0, 0.0, 10.0, 10.0], 0.9), _d([250.0, 250.0, 20.0, 20.0], 0.8)]
    )
    assert clean.ap() == pytest.approx(1.0)
    assert with_crowd_hit.ap() == pytest.approx(1.0)


def test_a_normal_false_positive_still_hurts():
    # The control for the test above: the same extra box, but nowhere near a
    # crowd region, must lower AP.
    gt = _gt([([0.0, 0.0, 10.0, 10.0], 0)])
    res = COCOMeanAP(gt, _params()).evaluate(
        [_d([0.0, 0.0, 10.0, 10.0], 0.9), _d([250.0, 250.0, 20.0, 20.0], 0.8)]
    )
    assert res.ap() < 1.0


def test_many_detections_in_one_crowd_are_all_ignored():
    gt = _gt([([0.0, 0.0, 10.0, 10.0], 0), ([200.0, 200.0, 300.0, 300.0], 1)])
    dets = [_d([0.0, 0.0, 10.0, 10.0], 0.99)]
    for i in range(10):
        dets.append(_d([210.0 + i * 5, 210.0, 20.0, 20.0], 0.9 - i * 0.01))
    res = COCOMeanAP(gt, _params()).evaluate(dets)
    assert res.ap() == pytest.approx(1.0)


def test_crowd_regions_do_not_count_towards_recall():
    gt_with_crowd = _gt([([0.0, 0.0, 10.0, 10.0], 0),
                         ([200.0, 200.0, 300.0, 300.0], 1)])
    gt_without = _gt([([0.0, 0.0, 10.0, 10.0], 0)])
    det = [_d([0.0, 0.0, 10.0, 10.0], 0.9)]
    a = COCOMeanAP(gt_with_crowd, _params()).evaluate(det).ar()
    b = COCOMeanAP(gt_without, _params()).evaluate(det).ar()
    assert a == pytest.approx(b) == pytest.approx(1.0)


def test_partial_overlap_with_crowd_below_threshold_is_a_false_positive():
    # Only 25% of the detection lies inside the crowd region, so the
    # intersection-over-area is 0.25 and the detection is not absorbed.
    gt = _gt([([0.0, 0.0, 10.0, 10.0], 0), ([100.0, 100.0, 100.0, 100.0], 1)])
    res = COCOMeanAP(gt, _params()).evaluate(
        [_d([0.0, 0.0, 10.0, 10.0], 0.9), _d([90.0, 90.0, 20.0, 20.0], 0.8)]
    )
    assert res.ap() < 1.0
