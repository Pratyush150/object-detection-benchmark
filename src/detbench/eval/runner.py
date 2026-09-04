"""Run a detector over a dataset, emit COCO results, score them, cache it all.

Re-running an evaluation is the expensive part of this repo (thousands of
forward passes on a CPU), so a run is keyed by a fingerprint of everything that
can change its output: the model file's content hash, the input size, the
confidence and NMS thresholds, the multi-label flag and the exact image list.
Change any of those and the cache misses; change nothing and a re-run is a
JSON load.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from ..metrics.coco_map import COCOMeanAP, COCOResults, EvalParams, GroundTruth
from ..models.base import Detector
from .dataset import CocoDetectionDataset, load_image

__all__ = ["RunResult", "run_detector", "evaluate_run", "score_detections"]

_STAGES = ("preprocess", "inference", "nms", "postprocess", "total")


@dataclass
class RunResult:
    """Everything one inference sweep produced."""

    variant: str
    detections: List[dict]
    image_ids: List[int]
    stage_times_ms: Dict[str, List[float]] = field(default_factory=dict)
    from_cache: bool = False
    wall_seconds: float = 0.0
    fingerprint: str = ""

    @property
    def n_images(self) -> int:
        """How many images were run."""
        return len(self.image_ids)

    def stage_percentiles(
        self, percentiles: Sequence[float] = (50, 90, 99)
    ) -> Dict[str, Dict[str, float]]:
        """Per-stage latency percentiles in milliseconds."""
        out: Dict[str, Dict[str, float]] = {}
        for stage, samples in self.stage_times_ms.items():
            if not samples:
                continue
            arr = np.asarray(samples, dtype=np.float64)
            row = {f"p{int(p)}": float(np.percentile(arr, p)) for p in percentiles}
            row["mean"] = float(arr.mean())
            row["min"] = float(arr.min())
            row["max"] = float(arr.max())
            out[stage] = row
        return out


def fingerprint_run(
    model_path: Optional[Path],
    image_ids: Sequence[int],
    config: Mapping[str, object],
) -> str:
    """Content hash of a run's inputs, used as the cache key."""
    digest = hashlib.sha256()
    if model_path is not None and Path(model_path).is_file():
        digest.update(file_sha256(Path(model_path)).encode("ascii"))
    digest.update(json.dumps(dict(config), sort_keys=True).encode("utf-8"))
    digest.update(np.asarray(sorted(image_ids), dtype=np.int64).tobytes())
    return digest.hexdigest()[:16]


def file_sha256(path: Path) -> str:
    """SHA-256 of a file, streamed so large weights do not blow up memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_detector(
    detector: Detector,
    images: Iterable[Tuple[int, np.ndarray]],
    variant: str,
    max_dets: int = 100,
    progress: Optional[Callable[[int, int], None]] = None,
    total: Optional[int] = None,
) -> RunResult:
    """Run a detector over an iterable of ``(image_id, image)`` pairs.

    Args:
        detector: The backend to run.
        images: Yields ``(image_id, BGR uint8 array)``.
        variant: Label for this run, e.g. ``fp32`` or ``int8-static``.
        max_dets: Detections kept per image in the results file. COCO scores
            at most 100, so keeping more only inflates the file.
        progress: Optional ``(done, total)`` callback.
        total: Total image count, for the progress callback.

    Returns:
        A :class:`RunResult` holding the COCO records and per-stage timings.
    """
    detections: List[dict] = []
    image_ids: List[int] = []
    times: Dict[str, List[float]] = {s: [] for s in _STAGES}

    start = time.perf_counter()
    for i, (image_id, image) in enumerate(images):
        result = detector.predict(image)
        detections.extend(result.to_coco(image_id, max_dets=max_dets))
        image_ids.append(int(image_id))
        for stage in _STAGES:
            if stage in result.stage_times_ms:
                times[stage].append(result.stage_times_ms[stage])
        if progress is not None:
            progress(i + 1, total if total is not None else -1)
    wall = time.perf_counter() - start

    return RunResult(
        variant=variant,
        detections=detections,
        image_ids=image_ids,
        stage_times_ms={k: v for k, v in times.items() if v},
        wall_seconds=wall,
    )


def score_detections(
    ground_truth: GroundTruth,
    detections: Sequence[Mapping[str, object]],
    image_ids: Optional[Sequence[int]] = None,
    params: Optional[EvalParams] = None,
) -> COCOResults:
    """Score COCO-format detections with the from-scratch evaluator."""
    return COCOMeanAP(ground_truth, params).evaluate(detections, image_ids)


def evaluate_run(
    detector: Detector,
    dataset: CocoDetectionDataset,
    variant: str,
    cache_dir: Optional[Path] = None,
    model_path: Optional[Path] = None,
    config: Optional[Mapping[str, object]] = None,
    max_dets: int = 100,
    progress: Optional[Callable[[int, int], None]] = None,
    force: bool = False,
) -> Tuple[RunResult, COCOResults]:
    """Run, cache and score a detector on a COCO dataset.

    A cache hit skips inference entirely but still re-scores, so a change to
    the metric code is picked up without re-running the model. Timings are
    restored from the cache and flagged via ``RunResult.from_cache`` - they are
    still real measurements, just not fresh ones.
    """
    image_ids = dataset.image_ids
    cfg = dict(config or {})
    cfg["variant"] = variant
    cfg["max_dets"] = max_dets
    key = fingerprint_run(model_path, image_ids, cfg)

    cache_path = None
    if cache_dir is not None:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = cache_dir / f"{variant}_{key}.json"

    run: Optional[RunResult] = None
    if cache_path is not None and cache_path.is_file() and not force:
        with cache_path.open("r", encoding="utf-8") as fh:
            blob = json.load(fh)
        run = RunResult(
            variant=blob["variant"],
            detections=blob["detections"],
            image_ids=[int(i) for i in blob["image_ids"]],
            stage_times_ms={k: list(v) for k, v in blob["stage_times_ms"].items()},
            from_cache=True,
            wall_seconds=float(blob.get("wall_seconds", 0.0)),
            fingerprint=key,
        )

    if run is None:
        def _images() -> Iterable[Tuple[int, np.ndarray]]:
            for record in dataset:
                yield record.image_id, load_image(record.path)

        run = run_detector(
            detector,
            _images(),
            variant=variant,
            max_dets=max_dets,
            progress=progress,
            total=len(dataset),
        )
        run.fingerprint = key
        if cache_path is not None:
            with cache_path.open("w", encoding="utf-8") as fh:
                json.dump(
                    {
                        "variant": run.variant,
                        "fingerprint": key,
                        "config": cfg,
                        "image_ids": run.image_ids,
                        "stage_times_ms": run.stage_times_ms,
                        "wall_seconds": run.wall_seconds,
                        "detections": run.detections,
                    },
                    fh,
                )

    results = score_detections(dataset.ground_truth, run.detections, image_ids)
    return run, results


def write_coco_results(detections: Sequence[Mapping[str, object]], path: Path) -> Path:
    """Write a COCO-format detections file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(list(detections), fh)
    return path
