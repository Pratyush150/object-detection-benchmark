"""Figure generation. Every committed image is produced by this code."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

matplotlib = pytest.importorskip("matplotlib")

from detbench import viz  # noqa: E402
from detbench.analysis.curves import score_threshold_sweep  # noqa: E402
from detbench.analysis.errors import tide_analysis  # noqa: E402
from detbench.analysis.per_class import per_class_report  # noqa: E402


def test_accuracy_versus_latency_chart(tmp_path: Path):
    path = viz.plot_accuracy_vs_latency(
        [
            {"name": "fp32", "map": 0.36, "latency": 100.0, "size_mb": 12.8},
            {"name": "int8", "map": 0.34, "latency": 45.0, "size_mb": 3.6},
        ],
        tmp_path / "acc-lat.png",
    )
    assert path.is_file() and path.stat().st_size > 2000


def test_per_class_bar_chart(tmp_path: Path, synthetic, synthetic_run):
    _, results = synthetic_run
    rows = per_class_report(results, synthetic.ground_truth)
    path = viz.plot_per_class_ap(rows, tmp_path / "per-class.png")
    assert path.is_file() and path.stat().st_size > 2000


def test_pr_curve_chart(tmp_path: Path, synthetic, synthetic_run):
    _, results = synthetic_run
    rows = per_class_report(results, synthetic.ground_truth)
    path = viz.plot_pr_curves(
        results, [r.category_id for r in rows[:2]], tmp_path / "pr.png"
    )
    assert path.is_file() and path.stat().st_size > 2000


def test_error_breakdown_chart(tmp_path: Path, synthetic, synthetic_run):
    run, _ = synthetic_run
    tide = tide_analysis(
        synthetic.ground_truth, run.detections, synthetic.image_ids
    )
    path = viz.plot_error_breakdown(
        tide["counts"], tide["delta_ap"], tmp_path / "errors.png"
    )
    assert path.is_file() and path.stat().st_size > 2000


def test_threshold_sweep_chart(tmp_path: Path, synthetic, synthetic_run):
    run, _ = synthetic_run
    points = score_threshold_sweep(
        synthetic.ground_truth, run.detections, thresholds=[0.1, 0.3, 0.5, 0.7]
    )
    path = viz.plot_threshold_sweep(points, tmp_path / "sweep.png")
    assert path.is_file() and path.stat().st_size > 2000


def test_detection_overlay(tmp_path: Path):
    pytest.importorskip("cv2")
    image = np.full((240, 320, 3), 60, dtype=np.uint8)
    dets = [{"image_id": 1, "category_id": 1,
             "bbox": [20.0, 30.0, 80.0, 60.0], "score": 0.87}]
    gts = [{"bbox": [22.0, 32.0, 78.0, 58.0]}]
    path = viz.draw_detections(
        image, dets, tmp_path / "overlay.jpg", ground_truth=gts,
        caption="test overlay",
    )
    assert path.is_file() and path.stat().st_size > 500


def test_overlay_respects_the_score_threshold(tmp_path: Path):
    cv2 = pytest.importorskip("cv2")
    image = np.full((120, 160, 3), 60, dtype=np.uint8)
    low = [{"image_id": 1, "category_id": 1,
            "bbox": [10.0, 10.0, 40.0, 40.0], "score": 0.05}]
    path = viz.draw_detections(
        image, low, tmp_path / "empty.jpg", score_threshold=0.5
    )
    written = cv2.imread(str(path))
    assert written is not None
    assert int(written.max()) - int(written.min()) < 30
