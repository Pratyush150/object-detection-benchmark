"""The TIDE-style error taxonomy on cases built to trigger one error each."""

from __future__ import annotations

import pytest

from detbench.analysis.errors import ERROR_TYPES, classify_errors, tide_analysis
from detbench.metrics.coco_map import GroundTruth


def _gt(entries):
    return GroundTruth(
        image_ids=[1],
        category_ids=[1, 2],
        annotations=[
            {
                "id": i + 1,
                "image_id": 1,
                "category_id": cat,
                "bbox": list(box),
                "area": box[2] * box[3],
                "iscrowd": crowd,
            }
            for i, (box, cat, crowd) in enumerate(entries)
        ],
    )


def _d(box, cat=1, score=0.9):
    return {"image_id": 1, "category_id": cat, "bbox": list(box), "score": score}


ONE_OBJECT = _gt([([0.0, 0.0, 100.0, 100.0], 1, 0)])


def test_exact_box_is_correct():
    out = classify_errors(ONE_OBJECT, [_d([0.0, 0.0, 100.0, 100.0])])
    assert out.labels == ["correct"]
    assert out.counts["missed"] == 0


def test_loose_box_is_a_localisation_error():
    # IoU about 0.36: on the object, but below the 0.5 foreground threshold.
    out = classify_errors(ONE_OBJECT, [_d([40.0, 40.0, 100.0, 100.0])])
    assert out.labels == ["localisation"]


def test_right_box_wrong_class_is_a_classification_error():
    out = classify_errors(ONE_OBJECT, [_d([0.0, 0.0, 100.0, 100.0], cat=2)])
    assert out.labels == ["classification"]
    assert out.confusions[(1, 2)] == 1


def test_wrong_class_and_loose_box_is_both():
    out = classify_errors(ONE_OBJECT, [_d([40.0, 40.0, 100.0, 100.0], cat=2)])
    assert out.labels == ["both"]


def test_second_box_on_the_same_object_is_a_duplicate():
    out = classify_errors(
        ONE_OBJECT,
        [_d([0.0, 0.0, 100.0, 100.0], score=0.9),
         _d([2.0, 2.0, 100.0, 100.0], score=0.5)],
    )
    assert out.labels == ["correct", "duplicate"]


def test_box_on_nothing_is_a_background_error():
    out = classify_errors(ONE_OBJECT, [_d([500.0, 500.0, 40.0, 40.0])])
    assert out.labels == ["background"]
    assert out.counts["missed"] == 1


def test_object_with_no_detection_is_missed():
    out = classify_errors(ONE_OBJECT, [])
    assert out.counts["missed"] == 1
    assert out.missed_ann_ids == [1]


def test_localisation_error_does_not_also_count_as_missed():
    out = classify_errors(ONE_OBJECT, [_d([40.0, 40.0, 100.0, 100.0])])
    assert out.counts["localisation"] == 1
    assert out.counts["missed"] == 0


def test_detection_inside_a_crowd_is_not_an_error():
    gt = _gt([([0.0, 0.0, 500.0, 500.0], 1, 1)])
    out = classify_errors(gt, [_d([10.0, 10.0, 40.0, 40.0])])
    assert out.labels == ["correct"]
    assert out.counts["missed"] == 0


def test_localisation_oracle_snaps_the_box_onto_the_object():
    out = classify_errors(ONE_OBJECT, [_d([40.0, 40.0, 100.0, 100.0])])
    fixed = out.fixed_detections("localisation")
    assert fixed[0]["bbox"] == [0.0, 0.0, 100.0, 100.0]


def test_classification_oracle_relabels_the_detection():
    out = classify_errors(ONE_OBJECT, [_d([0.0, 0.0, 100.0, 100.0], cat=2)])
    assert out.fixed_detections("classification")[0]["category_id"] == 1


def test_background_oracle_deletes_the_detection():
    out = classify_errors(ONE_OBJECT, [_d([500.0, 500.0, 40.0, 40.0])])
    assert out.fixed_detections("background") == []


def test_missed_oracle_removes_the_unfound_object():
    out = classify_errors(ONE_OBJECT, [])
    assert out.fixed_ground_truth("missed").annotations == []


def test_unknown_error_type_raises():
    out = classify_errors(ONE_OBJECT, [])
    with pytest.raises(ValueError):
        out.fixed_detections("gremlins")


def test_tide_analysis_reports_every_error_type(synthetic, synthetic_run):
    run, _ = synthetic_run
    tide = tide_analysis(
        synthetic.ground_truth, run.detections, synthetic.image_ids
    )
    assert set(tide["delta_ap"]) == set(ERROR_TYPES)
    assert set(tide["counts"]) == set(ERROR_TYPES)
    assert 0.0 <= tide["baseline_ap"] <= 1.0


def test_fixing_an_error_never_lowers_ap(synthetic, synthetic_run):
    run, _ = synthetic_run
    tide = tide_analysis(
        synthetic.ground_truth, run.detections, synthetic.image_ids
    )
    for name, delta in tide["delta_ap"].items():
        assert delta >= -1e-9, f"{name} oracle lowered AP"
