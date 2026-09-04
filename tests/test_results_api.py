"""The COCOResults accessors and the twelve-number summary."""

from __future__ import annotations

import numpy as np
import pytest

from detbench.metrics.coco_map import EvalParams, GroundTruth


def test_summary_has_the_twelve_coco_numbers(synthetic_run):
    _, results = synthetic_run
    summary = results.summary()
    assert len(summary) == 12
    assert set(summary) == {
        "mAP", "mAP50", "mAP75", "mAP_small", "mAP_medium", "mAP_large",
        "AR_1", "AR_10", "AR_100", "AR_small", "AR_medium", "AR_large",
    }


def test_formatted_summary_has_twelve_lines(synthetic_run):
    _, results = synthetic_run
    assert len(results.format_summary().splitlines()) == 12


def test_precision_tensor_has_the_documented_shape(synthetic_run):
    _, results = synthetic_run
    params = results.params
    assert results.precision.shape == (
        len(params.iou_thresholds),
        len(params.recall_thresholds),
        len(results.category_ids),
        len(params.area_ranges),
        len(params.max_dets),
    )
    assert results.recall.shape == (
        len(params.iou_thresholds),
        len(results.category_ids),
        len(params.area_ranges),
        len(params.max_dets),
    )


def test_pr_curve_has_101_points(synthetic_run, synthetic):
    _, results = synthetic_run
    cid = int(synthetic.ground_truth.annotations[0]["category_id"])
    recalls, precisions = results.pr_curve(cid, iou=0.5)
    assert recalls.shape == (101,)
    assert precisions.shape == (101,)
    assert precisions.min() >= 0.0


def test_map_is_the_mean_of_the_per_class_aps(synthetic_run):
    _, results = synthetic_run
    per_class = [v for v in results.per_class_ap().values() if not np.isnan(v)]
    assert results.ap() == pytest.approx(float(np.mean(per_class)), abs=1e-9)


def test_top_classes_are_ordered(synthetic_run):
    _, results = synthetic_run
    best = results.top_classes(n=3, best=True)
    worst = results.top_classes(n=3, best=False)
    assert [b[2] for b in best] == sorted([b[2] for b in best], reverse=True)
    assert [w[2] for w in worst] == sorted([w[2] for w in worst])
    assert best[0][2] >= worst[0][2]


def test_map50_is_at_least_map(synthetic_run):
    _, results = synthetic_run
    assert results.ap(iou=0.5) >= results.ap()


def test_unknown_iou_threshold_raises(synthetic_run):
    _, results = synthetic_run
    with pytest.raises(KeyError):
        results.ap(iou=0.62)


def test_unknown_max_dets_raises():
    with pytest.raises(KeyError):
        EvalParams().max_det_index(7)


def test_ground_truth_subset_filters_annotations(synthetic):
    ids = synthetic.image_ids[:3]
    subset = synthetic.ground_truth.subset(ids)
    assert subset.image_ids == sorted(ids)
    assert all(int(a["image_id"]) in set(ids) for a in subset.annotations)
    assert len(subset.annotations) < len(synthetic.ground_truth.annotations)


def test_ground_truth_from_coco_dict_sorts_ids():
    data = {
        "images": [{"id": 5}, {"id": 2}],
        "categories": [{"id": 9}, {"id": 3}],
        "annotations": [],
    }
    gt = GroundTruth.from_coco_dict(data)
    assert gt.image_ids == [2, 5]
    assert gt.category_ids == [3, 9]
