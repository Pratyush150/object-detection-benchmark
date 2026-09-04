"""The 80-class to 91-id mapping that silently zeroes a score file if wrong."""

from __future__ import annotations

import pytest

from detbench.coco_classes import (
    CATEGORY_NAMES,
    COCO80_NAMES,
    COCO80_TO_COCO91,
    COCO91_TO_COCO80,
    category_id_for_index,
    verify_against_categories,
)


def test_there_are_exactly_eighty_classes():
    assert len(COCO80_TO_COCO91) == 80
    assert len(COCO80_NAMES) == 80


def test_ids_are_strictly_increasing():
    assert COCO80_TO_COCO91 == sorted(COCO80_TO_COCO91)
    assert len(set(COCO80_TO_COCO91)) == 80


def test_ids_are_not_contiguous():
    # Eleven category ids were retired after the 2014 release; assuming
    # contiguity is the classic way to score near zero on a valid detector.
    assert COCO80_TO_COCO91[-1] == 90
    assert 12 not in COCO80_TO_COCO91


def test_known_anchor_points_in_the_mapping():
    assert category_id_for_index(0) == 1
    assert COCO80_NAMES[0] == "person"
    assert category_id_for_index(11) == 13
    assert COCO80_NAMES[11] == "stop sign"
    assert category_id_for_index(79) == 90
    assert COCO80_NAMES[79] == "toothbrush"


def test_reverse_mapping_round_trips():
    for index, cat_id in enumerate(COCO80_TO_COCO91):
        assert COCO91_TO_COCO80[cat_id] == index


def test_category_names_are_keyed_by_id():
    assert CATEGORY_NAMES[1] == "person"
    assert CATEGORY_NAMES[90] == "toothbrush"


def test_out_of_range_index_raises():
    with pytest.raises(ValueError):
        category_id_for_index(80)
    with pytest.raises(ValueError):
        category_id_for_index(-1)


def test_verification_accepts_a_matching_category_block():
    cats = [
        {"id": cid, "name": name}
        for cid, name in zip(COCO80_TO_COCO91, COCO80_NAMES)
    ]
    verify_against_categories(cats)


def test_verification_rejects_shifted_ids():
    cats = [
        {"id": cid + 1, "name": name}
        for cid, name in zip(COCO80_TO_COCO91, COCO80_NAMES)
    ]
    with pytest.raises(ValueError):
        verify_against_categories(cats)


def test_verification_rejects_renamed_classes():
    cats = [
        {"id": cid, "name": name}
        for cid, name in zip(COCO80_TO_COCO91, COCO80_NAMES)
    ]
    cats[0]["name"] = "human"
    with pytest.raises(ValueError):
        verify_against_categories(cats)
