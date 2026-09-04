"""Dataset loading and the calibration/evaluation split."""

from __future__ import annotations

import json

import numpy as np
import pytest

from detbench.eval.dataset import CocoDetectionDataset

from conftest import ANNOTATION_FILE, IMAGE_DIR, requires_dataset


@requires_dataset
def test_val2017_has_five_thousand_images():
    dataset = CocoDetectionDataset(ANNOTATION_FILE, IMAGE_DIR)
    assert len(dataset) == 5000
    assert len(dataset.ground_truth.category_ids) == 80


@requires_dataset
def test_every_record_points_at_a_file_that_exists():
    dataset = CocoDetectionDataset(ANNOTATION_FILE, IMAGE_DIR)
    for record in list(dataset)[:50]:
        assert record.path.is_file()


@requires_dataset
def test_images_load_as_three_channel_bgr():
    from detbench.eval.dataset import load_image

    dataset = CocoDetectionDataset(ANNOTATION_FILE, IMAGE_DIR)
    record = next(iter(dataset))
    image = load_image(record.path)
    assert image.ndim == 3
    assert image.shape[2] == 3
    assert image.shape[:2] == (record.height, record.width)
    assert image.dtype == np.uint8


@requires_dataset
def test_calibration_and_evaluation_splits_are_disjoint():
    dataset = CocoDetectionDataset(ANNOTATION_FILE, IMAGE_DIR)
    calib, rest = dataset.split_ids(128, seed=0)
    assert len(calib) == 128
    assert len(rest) == len(dataset) - 128
    assert not set(calib) & set(rest)
    assert sorted(calib + rest) == dataset.image_ids


@requires_dataset
def test_split_is_reproducible_and_seed_dependent():
    dataset = CocoDetectionDataset(ANNOTATION_FILE, IMAGE_DIR)
    a, _ = dataset.split_ids(64, seed=0)
    b, _ = dataset.split_ids(64, seed=0)
    c, _ = dataset.split_ids(64, seed=1)
    assert a == b
    assert a != c


@requires_dataset
def test_subsetting_restricts_both_images_and_annotations():
    dataset = CocoDetectionDataset(ANNOTATION_FILE, IMAGE_DIR)
    ids = dataset.image_ids[:10]
    subset = CocoDetectionDataset(ANNOTATION_FILE, IMAGE_DIR, image_ids=ids)
    assert subset.image_ids == ids
    assert all(
        int(a["image_id"]) in set(ids) for a in subset.ground_truth.annotations
    )


@requires_dataset
def test_asking_for_more_calibration_images_than_exist_raises():
    dataset = CocoDetectionDataset(ANNOTATION_FILE, IMAGE_DIR, 
                                   image_ids=CocoDetectionDataset(
                                       ANNOTATION_FILE, IMAGE_DIR
                                   ).image_ids[:20])
    with pytest.raises(ValueError):
        dataset.split_ids(50)


def test_missing_annotation_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        CocoDetectionDataset(tmp_path / "nope.json", tmp_path)


def test_missing_image_directory_raises(tmp_path):
    path = tmp_path / "ann.json"
    path.write_text(json.dumps({"images": [], "annotations": [],
                                "categories": []}))
    with pytest.raises(FileNotFoundError):
        CocoDetectionDataset(path, tmp_path / "missing")


def test_wrong_category_block_is_rejected(tmp_path):
    path = tmp_path / "ann.json"
    path.write_text(
        json.dumps(
            {
                "images": [],
                "annotations": [],
                "categories": [{"id": 1, "name": "person"}],
            }
        )
    )
    (tmp_path / "imgs").mkdir()
    with pytest.raises(ValueError):
        CocoDetectionDataset(path, tmp_path / "imgs")
