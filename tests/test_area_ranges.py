"""Area-range assignment, especially at the exact 32^2 and 96^2 boundaries."""

from __future__ import annotations

import numpy as np
import pytest

from detbench.metrics.coco_map import (
    DEFAULT_AREA_RANGES,
    AreaRange,
    COCOMeanAP,
    EvalParams,
    GroundTruth,
)

SMALL = DEFAULT_AREA_RANGES[1]
MEDIUM = DEFAULT_AREA_RANGES[2]
LARGE = DEFAULT_AREA_RANGES[3]


def test_default_bands_are_the_coco_bands():
    names = [r.name for r in DEFAULT_AREA_RANGES]
    assert names == ["all", "small", "medium", "large"]
    assert SMALL.hi == pytest.approx(32.0**2)
    assert MEDIUM.lo == pytest.approx(32.0**2)
    assert MEDIUM.hi == pytest.approx(96.0**2)
    assert LARGE.lo == pytest.approx(96.0**2)


def test_just_below_small_boundary_is_small_only():
    area = 32.0**2 - 1e-6
    assert SMALL.contains(area)
    assert not MEDIUM.contains(area)


def test_exactly_at_small_boundary_is_in_both_bands():
    # The reference treats both bounds as inclusive, so 1024 px^2 belongs to
    # small and to medium. The bands slice results, they do not partition them.
    area = 32.0**2
    assert SMALL.contains(area)
    assert MEDIUM.contains(area)


def test_just_above_small_boundary_is_medium_only():
    area = 32.0**2 + 1e-6
    assert not SMALL.contains(area)
    assert MEDIUM.contains(area)


def test_exactly_at_large_boundary_is_in_both_bands():
    area = 96.0**2
    assert MEDIUM.contains(area)
    assert LARGE.contains(area)


def test_just_above_large_boundary_is_large_only():
    area = 96.0**2 + 1e-6
    assert not MEDIUM.contains(area)
    assert LARGE.contains(area)


def _single_object_eval(gt_area, det_box, band):
    gt = GroundTruth(
        image_ids=[1],
        category_ids=[1],
        annotations=[
            {
                "id": 1,
                "image_id": 1,
                "category_id": 1,
                "bbox": [0.0, 0.0, 10.0, 10.0],
                "area": gt_area,
                "iscrowd": 0,
            }
        ],
    )
    params = EvalParams(
        iou_thresholds=np.array([0.5]),
        area_ranges=(AreaRange("all", 0.0, 1e10), band),
        max_dets=(100,),
    )
    dets = [{"image_id": 1, "category_id": 1, "bbox": det_box, "score": 0.9}]
    return COCOMeanAP(gt, params).evaluate(dets)


def test_object_outside_the_band_is_ignored_not_scored_zero():
    # A perfect detection of a large object must not appear in the small band.
    res = _single_object_eval(200.0**2, [0.0, 0.0, 10.0, 10.0], SMALL)
    assert np.isnan(res.ap(area="small"))
    assert res.ap(area="all") == pytest.approx(1.0)


def test_object_inside_the_band_is_scored():
    res = _single_object_eval(20.0**2, [0.0, 0.0, 10.0, 10.0], SMALL)
    assert res.ap(area="small") == pytest.approx(1.0)


def test_annotation_area_field_wins_over_bbox_area():
    # COCO's 'area' is the segmentation area, which is smaller than the box.
    # Using the box area instead silently moves objects between bands.
    res = _single_object_eval(10.0, [0.0, 0.0, 10.0, 10.0], SMALL)
    assert res.ap(area="small") == pytest.approx(1.0)


def test_unknown_band_name_raises():
    params = EvalParams()
    with pytest.raises(KeyError):
        params.area_index("enormous")
