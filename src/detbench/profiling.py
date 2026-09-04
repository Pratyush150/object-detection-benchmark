"""Per-stage latency profiling, reported as percentiles rather than a mean.

Why not mean FPS: a pipeline that averages 20 ms per frame but spends 90 ms on
one frame in fifty still shows visible stutter, and it still misses a 30 Hz
control deadline twenty times a minute. The mean hides that completely - one
90 ms frame among fifty 18 ms frames moves the mean by 1.4 ms. The p99 moves by
70 ms. Anything with a deadline should be specified and measured at a
percentile, and the gap between p50 and p99 is the number that predicts whether
a demo will look smooth.

The stage split matters for the same reason it matters under quantisation: if
inference drops from 70 ms to 29 ms but preprocessing stays at 5 ms and NMS at
4 ms, the end-to-end win is 1.8x, not the 2.4x the inference number suggests.
Amdahl's law applies to detection pipelines like everything else.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Optional, Sequence

import numpy as np

from .models.base import Detector

__all__ = ["LatencyStats", "StageProfile", "summarise_samples", "profile_detector"]

DEFAULT_PERCENTILES = (50.0, 90.0, 99.0)


@dataclass(frozen=True)
class LatencyStats:
    """Latency summary for one stage, in milliseconds."""

    stage: str
    count: int
    mean: float
    p50: float
    p90: float
    p99: float
    minimum: float
    maximum: float

    @property
    def jitter_ratio(self) -> float:
        """p99 divided by p50: how much worse the tail is than the typical frame.

        A value near 1.0 means a predictable pipeline. Above roughly 1.5 there
        is a spike source worth finding - thermal throttling, a thread-pool
        stall, page faults on first touch of a large arena, or an image whose
        detection count blows up the NMS cost.
        """
        return self.p99 / self.p50 if self.p50 > 0 else float("nan")

    def as_dict(self) -> Dict[str, float]:
        """Flat dictionary form, for JSON and tables."""
        return {
            "count": self.count,
            "mean_ms": self.mean,
            "p50_ms": self.p50,
            "p90_ms": self.p90,
            "p99_ms": self.p99,
            "min_ms": self.minimum,
            "max_ms": self.maximum,
            "jitter_p99_over_p50": self.jitter_ratio,
        }


@dataclass
class StageProfile:
    """Per-stage latency statistics for one run."""

    label: str
    stages: Dict[str, LatencyStats] = field(default_factory=dict)
    raw_samples: Dict[str, List[float]] = field(default_factory=dict)

    @property
    def mean_fps(self) -> float:
        """Frames per second implied by the mean total latency."""
        total = self.stages.get("total")
        return 1000.0 / total.mean if total and total.mean > 0 else float("nan")

    @property
    def p99_fps(self) -> float:
        """Frames per second guaranteed 99% of the time.

        This is the number to quote when something downstream has a deadline.
        """
        total = self.stages.get("total")
        return 1000.0 / total.p99 if total and total.p99 > 0 else float("nan")

    def format_table(self) -> str:
        """Render the profile as a fixed-width table."""
        header = (
            f"{'stage':<13}{'n':>6}{'mean':>9}{'p50':>9}{'p90':>9}"
            f"{'p99':>9}{'max':>9}{'p99/p50':>9}"
        )
        lines = [header, "-" * len(header)]
        order = ["preprocess", "inference", "nms", "postprocess", "total"]
        for name in order:
            st = self.stages.get(name)
            if st is None:
                continue
            lines.append(
                f"{st.stage:<13}{st.count:>6}{st.mean:>9.2f}{st.p50:>9.2f}"
                f"{st.p90:>9.2f}{st.p99:>9.2f}{st.maximum:>9.2f}"
                f"{st.jitter_ratio:>9.2f}"
            )
        return "\n".join(lines)


def summarise_samples(
    stage: str,
    samples: Sequence[float],
    percentiles: Sequence[float] = DEFAULT_PERCENTILES,
) -> LatencyStats:
    """Summarise raw millisecond samples for one stage."""
    arr = np.asarray(list(samples), dtype=np.float64)
    if arr.size == 0:
        raise ValueError(f"no samples for stage {stage!r}")
    p50, p90, p99 = (float(np.percentile(arr, p)) for p in percentiles)
    return LatencyStats(
        stage=stage,
        count=int(arr.size),
        mean=float(arr.mean()),
        p50=p50,
        p90=p90,
        p99=p99,
        minimum=float(arr.min()),
        maximum=float(arr.max()),
    )


def profile_detector(
    detector: Detector,
    images: Iterable[np.ndarray],
    label: Optional[str] = None,
    warmup: int = 5,
    repeats: int = 1,
    progress: Optional[Callable[[int], None]] = None,
) -> StageProfile:
    """Time a detector stage by stage over a set of images.

    Args:
        detector: Backend to profile.
        images: Images to run. Materialised into a list so ``repeats`` can
            replay the same frames; keep the set small.
        label: Name for the profile. Defaults to the detector's name.
        warmup: Throwaway iterations before timing starts. The first call pays
            for ONNX Runtime's arena allocation and kernel selection, which
            would otherwise land in the p99 and misrepresent the tail.
        repeats: How many passes over the image set to time.
        progress: Optional callback receiving the running sample count.

    Returns:
        A :class:`StageProfile`.
    """
    frames = list(images)
    if not frames:
        raise ValueError("no images to profile")

    for i in range(max(0, warmup)):
        detector.predict(frames[i % len(frames)])

    samples: Dict[str, List[float]] = {}
    done = 0
    for _ in range(max(1, repeats)):
        for frame in frames:
            result = detector.predict(frame)
            for stage, value in result.stage_times_ms.items():
                samples.setdefault(stage, []).append(value)
            done += 1
            if progress is not None:
                progress(done)

    return StageProfile(
        label=label or getattr(detector, "name", "detector"),
        stages={k: summarise_samples(k, v) for k, v in samples.items()},
        raw_samples=samples,
    )


def timed(fn: Callable[[], object]) -> tuple[object, float]:
    """Call ``fn`` and return its result alongside elapsed milliseconds."""
    start = time.perf_counter()
    value = fn()
    return value, (time.perf_counter() - start) * 1e3
