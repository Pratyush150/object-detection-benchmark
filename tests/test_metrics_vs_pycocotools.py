"""Agreement with pycocotools, the reference COCO evaluator.

pycocotools appears only here. The library itself never imports it: the point
of this repo is a metric implemented from the protocol, and the reference is
how that implementation is held honest.
"""

from __future__ import annotations

import contextlib
import io
import random

import numpy as np
import pytest

from detbench.metrics.coco_map import COCOMeanAP, GroundTruth

from conftest import (
    ANNOTATION_FILE,
    FP32_MODEL,
    IMAGE_DIR,
    requires_dataset,
    requires_model,
    requires_pycocotools,
)

SUMMARY_ORDER = [
    "mAP", "mAP50", "mAP75", "mAP_small", "mAP_medium", "mAP_large",
    "AR_1", "AR_10", "AR_100", "AR_small", "AR_medium", "AR_large",
]


def _random_case(seed: int, n_images: int = 30, n_cats: int = 5,
                 crowd_rate: float = 0.12):
    """A random but reproducible ground truth and detection set."""
    rng = random.Random(seed)
    cats = [{"id": c + 1, "name": f"c{c}", "supercategory": "x"}
            for c in range(n_cats)]
    images = [{"id": i + 1, "width": 640, "height": 480,
               "file_name": f"{i}.jpg"} for i in range(n_images)]
    anns = []
    ann_id = 1
    for im in images:
        for _ in range(rng.randint(0, 6)):
            w, h = rng.uniform(5, 200), rng.uniform(5, 200)
            x, y = rng.uniform(0, 640 - w), rng.uniform(0, 480 - h)
            anns.append({
                "id": ann_id, "image_id": im["id"],
                "category_id": rng.randint(1, n_cats),
                "bbox": [x, y, w, h],
                "area": w * h * rng.uniform(0.5, 1.0),
                "iscrowd": 1 if rng.random() < crowd_rate else 0,
            })
            ann_id += 1
    gt = {"images": images, "annotations": anns, "categories": cats}

    dets = []
    for im in images:
        for ann in [a for a in anns if a["image_id"] == im["id"]]:
            if rng.random() < 0.75:
                x, y, w, h = ann["bbox"]
                dets.append({
                    "image_id": im["id"],
                    "category_id": (ann["category_id"] if rng.random() < 0.85
                                    else rng.randint(1, n_cats)),
                    "bbox": [x + rng.gauss(0, 10), y + rng.gauss(0, 10),
                             max(2.0, w + rng.gauss(0, 12)),
                             max(2.0, h + rng.gauss(0, 12))],
                    "score": rng.uniform(0.05, 1.0),
                })
        for _ in range(rng.randint(0, 8)):
            w, h = rng.uniform(5, 200), rng.uniform(5, 200)
            x, y = rng.uniform(0, 640 - w), rng.uniform(0, 480 - h)
            dets.append({"image_id": im["id"],
                         "category_id": rng.randint(1, n_cats),
                         "bbox": [x, y, w, h], "score": rng.uniform(0.01, 1.0)})
    return gt, dets


def _reference_stats(gt: dict, dets: list) -> np.ndarray:
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    with contextlib.redirect_stdout(io.StringIO()):
        coco = COCO()
        coco.dataset = gt
        coco.createIndex()
        coco_dt = coco.loadRes([dict(d) for d in dets])
        ev = COCOeval(coco, coco_dt, "bbox")
        ev.evaluate()
        ev.accumulate()
        ev.summarize()
    return np.asarray(ev.stats, dtype=np.float64)


def _ours(gt: dict, dets: list) -> np.ndarray:
    results = COCOMeanAP(GroundTruth.from_coco_dict(gt)).evaluate(dets)
    summary = results.summary()
    return np.asarray([summary[k] for k in SUMMARY_ORDER], dtype=np.float64)


@requires_pycocotools
@pytest.mark.parametrize("seed", [1, 7, 13, 29])
def test_all_twelve_metrics_match_the_reference(seed):
    gt, dets = _random_case(seed)
    ours, ref = _ours(gt, dets), _reference_stats(gt, dets)
    assert np.allclose(ours, ref, atol=1e-9), dict(
        zip(SUMMARY_ORDER, (ours - ref).tolist())
    )


@requires_pycocotools
def test_match_holds_with_no_crowd_regions():
    gt, dets = _random_case(3, crowd_rate=0.0)
    assert np.allclose(_ours(gt, dets), _reference_stats(gt, dets), atol=1e-9)


@requires_pycocotools
def test_match_holds_when_every_ground_truth_is_a_crowd():
    gt, dets = _random_case(5, crowd_rate=1.0)
    ours, ref = _ours(gt, dets), _reference_stats(gt, dets)
    # Every class is undefined, so ours reports NaN where the reference uses -1.
    assert np.all(np.isnan(ours))
    assert np.allclose(ref, -1.0)


@requires_pycocotools
def test_match_holds_with_a_dense_scene():
    gt, dets = _random_case(11, n_images=8, n_cats=2)
    assert np.allclose(_ours(gt, dets), _reference_stats(gt, dets), atol=1e-9)


@requires_pycocotools
@requires_dataset
@requires_model
def test_real_detections_on_real_images_match_the_reference():
    """The measurement that matters: a real model on real COCO images."""
    import json

    from detbench.eval.dataset import CocoDetectionDataset, load_image
    from detbench.eval.runner import run_detector
    from detbench.models.onnx_yolo import OnnxYoloDetector

    with open(ANNOTATION_FILE, "r", encoding="utf-8") as fh:
        raw = json.load(fh)

    dataset = CocoDetectionDataset(ANNOTATION_FILE, IMAGE_DIR)
    image_ids = dataset.image_ids[:40]
    subset = CocoDetectionDataset(ANNOTATION_FILE, IMAGE_DIR, image_ids=image_ids)

    detector = OnnxYoloDetector(FP32_MODEL, conf_threshold=0.001)
    run = run_detector(
        detector,
        ((r.image_id, load_image(r.path)) for r in subset),
        variant="agreement",
    )

    keep = set(image_ids)
    gt = {
        "images": [im for im in raw["images"] if int(im["id"]) in keep],
        "annotations": [
            a for a in raw["annotations"] if int(a["image_id"]) in keep
        ],
        "categories": raw["categories"],
    }
    ours = _ours(gt, run.detections)
    ref = _reference_stats(gt, run.detections)
    assert np.allclose(ours, ref, atol=1e-9), dict(
        zip(SUMMARY_ORDER, (ours - ref).tolist())
    )
