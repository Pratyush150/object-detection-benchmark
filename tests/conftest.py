"""Shared fixtures. Adds ``src`` to the path so no install step is needed."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

#: Where the real model and dataset live, if they have been fetched.
ASSET_ROOT = Path(os.environ.get("DETBENCH_ASSETS", REPO_ROOT / "assets"))

ANNOTATION_FILE = ASSET_ROOT / "annotations" / "instances_val2017.json"
IMAGE_DIR = ASSET_ROOT / "val2017"
FP32_MODEL = ASSET_ROOT / "yolov8n.onnx"


def have_dataset() -> bool:
    """True when the COCO val2017 assets are present."""
    return ANNOTATION_FILE.is_file() and IMAGE_DIR.is_dir()


def have_model() -> bool:
    """True when the exported ONNX model is present."""
    return FP32_MODEL.is_file()


def have_pycocotools() -> bool:
    """True when the reference evaluator is installed."""
    try:
        import pycocotools.cocoeval  # noqa: F401,PLC0415
    except ImportError:
        return False
    return True


requires_dataset = pytest.mark.skipif(
    not have_dataset(),
    reason=f"COCO val2017 not found under {ASSET_ROOT}; run tools/fetch_assets.py",
)
requires_model = pytest.mark.skipif(
    not have_model(),
    reason=f"yolov8n.onnx not found under {ASSET_ROOT}; run tools/fetch_assets.py",
)
requires_pycocotools = pytest.mark.skipif(
    not have_pycocotools(), reason="pycocotools is not installed"
)


@pytest.fixture(scope="session")
def synthetic():
    """A deterministic synthetic dataset shared across tests."""
    from detbench.eval.synthetic import make_synthetic_dataset

    return make_synthetic_dataset(n_images=16, seed=4242)


@pytest.fixture(scope="session")
def synthetic_run(synthetic):
    """Detections and scores for the synthetic dataset."""
    from detbench.eval.runner import run_detector, score_detections

    run = run_detector(
        synthetic.detector,
        ((i, synthetic.images[i]) for i in synthetic.image_ids),
        variant="test",
    )
    results = score_detections(
        synthetic.ground_truth, run.detections, synthetic.image_ids
    )
    return run, results
