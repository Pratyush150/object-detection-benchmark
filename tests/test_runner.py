"""The evaluation runner and its cache."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from detbench.eval.runner import (
    RunResult,
    file_sha256,
    fingerprint_run,
    run_detector,
    score_detections,
    write_coco_results,
)


def test_run_produces_one_record_set_per_image(synthetic):
    run = run_detector(
        synthetic.detector,
        ((i, synthetic.images[i]) for i in synthetic.image_ids),
        variant="t",
    )
    assert run.n_images == len(synthetic.images)
    assert {int(d["image_id"]) for d in run.detections} <= set(synthetic.image_ids)


def test_run_is_deterministic(synthetic):
    def go():
        return run_detector(
            synthetic.detector,
            ((i, synthetic.images[i]) for i in synthetic.image_ids),
            variant="t",
        ).detections

    first, second = go(), go()
    assert [d["bbox"] for d in first] == [d["bbox"] for d in second]
    assert [d["score"] for d in first] == [d["score"] for d in second]


def test_stage_percentiles_are_ordered(synthetic_run):
    run, _ = synthetic_run
    stats = run.stage_percentiles()
    total = stats["total"]
    assert total["p50"] <= total["p90"] <= total["p99"] <= total["max"]
    assert total["min"] <= total["p50"]


def test_scoring_the_run_gives_a_sane_map(synthetic_run):
    _, results = synthetic_run
    summary = results.summary()
    assert 0.0 < summary["mAP"] < 1.0
    assert summary["mAP50"] >= summary["mAP"]
    assert summary["mAP50"] >= summary["mAP75"]


def test_progress_callback_is_called(synthetic):
    seen = []
    run_detector(
        synthetic.detector,
        ((i, synthetic.images[i]) for i in synthetic.image_ids),
        variant="t",
        progress=lambda d, t: seen.append((d, t)),
        total=len(synthetic.images),
    )
    assert len(seen) == len(synthetic.images)
    assert seen[-1] == (len(synthetic.images), len(synthetic.images))


def test_fingerprint_changes_with_config():
    a = fingerprint_run(None, [1, 2, 3], {"conf": 0.001})
    b = fingerprint_run(None, [1, 2, 3], {"conf": 0.25})
    assert a != b


def test_fingerprint_changes_with_image_set():
    a = fingerprint_run(None, [1, 2, 3], {"conf": 0.001})
    b = fingerprint_run(None, [1, 2, 4], {"conf": 0.001})
    assert a != b


def test_fingerprint_is_stable_for_the_same_inputs():
    a = fingerprint_run(None, [3, 1, 2], {"conf": 0.001})
    b = fingerprint_run(None, [1, 2, 3], {"conf": 0.001})
    assert a == b


def test_file_sha256_matches_hashlib(tmp_path: Path):
    import hashlib

    path = tmp_path / "blob.bin"
    path.write_bytes(b"detbench" * 1000)
    assert file_sha256(path) == hashlib.sha256(b"detbench" * 1000).hexdigest()


def test_write_coco_results_round_trips(tmp_path: Path, synthetic_run):
    run, _ = synthetic_run
    out = write_coco_results(run.detections, tmp_path / "dets.json")
    with out.open() as fh:
        loaded = json.load(fh)
    assert len(loaded) == len(run.detections)
    assert set(loaded[0]) == {"image_id", "category_id", "bbox", "score"}


def test_empty_detection_set_scores_zero(synthetic):
    results = score_detections(
        synthetic.ground_truth, [], synthetic.image_ids
    )
    assert results.ap() == pytest.approx(0.0)


def test_run_result_reports_no_cache_by_default():
    empty = RunResult(variant="v", detections=[], image_ids=[])
    assert empty.from_cache is False
    assert empty.n_images == 0
