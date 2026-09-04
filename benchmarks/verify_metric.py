"""Compare the from-scratch COCO metric against pycocotools on real results.

Run this after `benchmarks/run_sweep.py`. It loads a cached detection file,
scores it twice - once with `detbench`, once with the reference - and prints
every one of the twelve COCO numbers side by side with the absolute difference.

`pycocotools` is imported here and in the test suite, and nowhere else.

Usage::

    python3 benchmarks/verify_metric.py \\
        --annotations /path/to/instances_val2017.json \\
        --detections /path/to/cache/fp32_<hash>.json
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import sys
import time
from pathlib import Path
from typing import List, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np  # noqa: E402

from detbench.metrics.coco_map import COCOMeanAP, GroundTruth  # noqa: E402

SUMMARY_ORDER = [
    ("mAP", "AP @ 0.50:0.95, all, maxDets=100"),
    ("mAP50", "AP @ 0.50, all, maxDets=100"),
    ("mAP75", "AP @ 0.75, all, maxDets=100"),
    ("mAP_small", "AP @ 0.50:0.95, small, maxDets=100"),
    ("mAP_medium", "AP @ 0.50:0.95, medium, maxDets=100"),
    ("mAP_large", "AP @ 0.50:0.95, large, maxDets=100"),
    ("AR_1", "AR @ 0.50:0.95, all, maxDets=1"),
    ("AR_10", "AR @ 0.50:0.95, all, maxDets=10"),
    ("AR_100", "AR @ 0.50:0.95, all, maxDets=100"),
    ("AR_small", "AR @ 0.50:0.95, small, maxDets=100"),
    ("AR_medium", "AR @ 0.50:0.95, medium, maxDets=100"),
    ("AR_large", "AR @ 0.50:0.95, large, maxDets=100"),
]


def load_detections(path: Path) -> Tuple[List[dict], List[int]]:
    """Accept a plain COCO results list or a runner cache blob."""
    with path.open("r", encoding="utf-8") as fh:
        blob = json.load(fh)
    if isinstance(blob, list):
        ids = sorted({int(d["image_id"]) for d in blob})
        return blob, ids
    return blob["detections"], [int(i) for i in blob["image_ids"]]


def main() -> int:
    """Entry point."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--annotations", required=True, type=Path)
    ap.add_argument("--detections", required=True, type=Path)
    ap.add_argument("--out", default=None, type=Path)
    args = ap.parse_args()

    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    detections, image_ids = load_detections(args.detections)
    with args.annotations.open("r", encoding="utf-8") as fh:
        raw = json.load(fh)

    keep = set(image_ids)
    subset = {
        "images": [im for im in raw["images"] if int(im["id"]) in keep],
        "annotations": [a for a in raw["annotations"] if int(a["image_id"]) in keep],
        "categories": raw["categories"],
    }
    print(f"{len(detections)} detections over {len(image_ids)} images, "
          f"{len(subset['annotations'])} ground-truth annotations\n")

    t0 = time.time()
    ours = COCOMeanAP(GroundTruth.from_coco_dict(subset)).evaluate(
        detections, image_ids
    ).summary()
    ours_seconds = time.time() - t0

    t0 = time.time()
    with contextlib.redirect_stdout(io.StringIO()):
        coco = COCO()
        coco.dataset = subset
        coco.createIndex()
        coco_dt = coco.loadRes([dict(d) for d in detections])
        ev = COCOeval(coco, coco_dt, "bbox")
        ev.evaluate()
        ev.accumulate()
        ev.summarize()
    ref_seconds = time.time() - t0

    print(f"{'metric':<40}{'detbench':>16}{'pycocotools':>16}{'abs diff':>14}")
    print("-" * 86)
    worst = 0.0
    rows = []
    for i, (key, label) in enumerate(SUMMARY_ORDER):
        mine, theirs = float(ours[key]), float(ev.stats[i])
        diff = abs(mine - theirs)
        worst = max(worst, diff)
        rows.append({"metric": key, "label": label, "detbench": mine,
                     "pycocotools": theirs, "abs_diff": diff})
        print(f"{label:<40}{mine:>16.12f}{theirs:>16.12f}{diff:>14.2e}")
    print("-" * 86)
    print(f"largest absolute difference: {worst:.3e}")
    print(f"detbench {ours_seconds:.1f}s, pycocotools {ref_seconds:.1f}s")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("w", encoding="utf-8") as fh:
            json.dump(
                {
                    "n_images": len(image_ids),
                    "n_detections": len(detections),
                    "largest_absolute_difference": worst,
                    "detbench_seconds": ours_seconds,
                    "pycocotools_seconds": ref_seconds,
                    "rows": rows,
                },
                fh,
                indent=2,
            )
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
