"""Confidence-threshold sweeps and per-class reports."""

from __future__ import annotations

import pytest

from detbench.analysis.curves import format_sweep_table, score_threshold_sweep
from detbench.analysis.per_class import (
    format_class_table,
    instance_counts,
    per_class_report,
)
from detbench.metrics.coco_map import GroundTruth


def _gt():
    return GroundTruth(
        image_ids=[1],
        category_ids=[1],
        annotations=[
            {"id": 1, "image_id": 1, "category_id": 1,
             "bbox": [0.0, 0.0, 10.0, 10.0], "area": 100.0, "iscrowd": 0},
            {"id": 2, "image_id": 1, "category_id": 1,
             "bbox": [100.0, 0.0, 10.0, 10.0], "area": 100.0, "iscrowd": 0},
        ],
    )


def _d(box, score):
    return {"image_id": 1, "category_id": 1, "bbox": list(box), "score": score}


def test_recall_is_monotonically_non_increasing_in_threshold():
    dets = [_d([0.0, 0.0, 10.0, 10.0], 0.9), _d([100.0, 0.0, 10.0, 10.0], 0.4)]
    points = score_threshold_sweep(_gt(), dets, thresholds=[0.1, 0.5, 0.95])
    recalls = [p.recall for p in points]
    assert recalls == sorted(recalls, reverse=True)


def test_precision_rises_as_the_threshold_rises_when_fps_score_low():
    dets = [
        _d([0.0, 0.0, 10.0, 10.0], 0.9),
        _d([500.0, 500.0, 10.0, 10.0], 0.2),
    ]
    low, high = score_threshold_sweep(_gt(), dets, thresholds=[0.1, 0.5])
    assert low.precision == pytest.approx(0.5)
    assert high.precision == pytest.approx(1.0)


def test_counts_add_up():
    dets = [_d([0.0, 0.0, 10.0, 10.0], 0.9), _d([500.0, 500.0, 10.0, 10.0], 0.8)]
    point = score_threshold_sweep(_gt(), dets, thresholds=[0.1])[0]
    assert point.true_positives == 1
    assert point.false_positives == 1
    assert point.false_negatives == 1


def test_f1_is_the_harmonic_mean():
    dets = [_d([0.0, 0.0, 10.0, 10.0], 0.9)]
    p = score_threshold_sweep(_gt(), dets, thresholds=[0.1])[0]
    expected = 2 * p.precision * p.recall / (p.precision + p.recall)
    assert p.f1 == pytest.approx(expected)


def test_detections_per_image_is_reported():
    dets = [_d([0.0, 0.0, 10.0, 10.0], 0.9), _d([500.0, 500.0, 10.0, 10.0], 0.8)]
    p = score_threshold_sweep(_gt(), dets, thresholds=[0.1], n_images=2)[0]
    assert p.detections_per_image == pytest.approx(1.0)


def test_crowd_ground_truth_is_excluded_from_the_denominator():
    gt = _gt()
    gt.annotations[1]["iscrowd"] = 1
    p = score_threshold_sweep(
        gt, [_d([0.0, 0.0, 10.0, 10.0], 0.9)], thresholds=[0.1]
    )[0]
    assert p.recall == pytest.approx(1.0)


def test_higher_iou_requirement_lowers_recall():
    dets = [_d([3.0, 0.0, 10.0, 10.0], 0.9)]
    loose = score_threshold_sweep(_gt(), dets, thresholds=[0.1], iou_threshold=0.5)[0]
    strict = score_threshold_sweep(_gt(), dets, thresholds=[0.1], iou_threshold=0.9)[0]
    assert loose.recall > strict.recall


def test_sweep_table_renders_a_row_per_threshold():
    points = score_threshold_sweep(
        _gt(), [_d([0.0, 0.0, 10.0, 10.0], 0.9)], thresholds=[0.1, 0.5]
    )
    text = format_sweep_table(points)
    assert len(text.splitlines()) == 4  # header, rule, two rows


def test_per_class_report_is_sorted_best_first(synthetic, synthetic_run):
    _, results = synthetic_run
    rows = per_class_report(results, synthetic.ground_truth)
    assert rows
    assert [r.ap for r in rows] == sorted([r.ap for r in rows], reverse=True)


def test_per_class_report_drops_classes_with_no_ground_truth(
    synthetic, synthetic_run
):
    _, results = synthetic_run
    rows = per_class_report(results, synthetic.ground_truth)
    present = {int(a["category_id"]) for a in synthetic.ground_truth.annotations}
    assert {r.category_id for r in rows} == present


def test_instance_counts_ignore_crowd_regions():
    gt = _gt()
    gt.annotations[1]["iscrowd"] = 1
    assert instance_counts(gt) == {1: 1}


def test_class_table_renders(synthetic, synthetic_run):
    _, results = synthetic_run
    rows = per_class_report(results, synthetic.ground_truth)
    text = format_class_table(rows, limit=3)
    assert len(text.splitlines()) == 5
    assert "AP50" in text


def test_gap_between_ap50_and_ap75_is_reported(synthetic, synthetic_run):
    _, results = synthetic_run
    rows = per_class_report(results, synthetic.ground_truth)
    assert all(r.gap_50_75 == pytest.approx(r.ap50 - r.ap75) for r in rows)
