"""Produce every figure the README references, from measured results only.

Reads `benchmarks/results/sweep.json` and the cached detection files written by
`benchmarks/run_sweep.py`, then writes PNGs and JPEGs into
`benchmarks/output/`. Nothing here invents a number: if a variant failed to
run, it is absent from the charts rather than filled in.

Usage::

    python3 benchmarks/make_figures.py --sweep benchmarks/results/sweep.json \\
        --cache /path/to/cache --annotations /path/to/instances_val2017.json \\
        --images /path/to/val2017 --out benchmarks/output
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np  # noqa: E402

from detbench import viz  # noqa: E402
from detbench.analysis.curves import score_threshold_sweep  # noqa: E402
from detbench.analysis.errors import classify_errors, tide_analysis  # noqa: E402
from detbench.analysis.per_class import per_class_report  # noqa: E402
from detbench.coco_classes import CATEGORY_NAMES  # noqa: E402
from detbench.eval.dataset import load_image  # noqa: E402
from detbench.eval.runner import score_detections  # noqa: E402
from detbench.metrics.coco_map import GroundTruth  # noqa: E402


def find_cache(cache_dir: Path, variant: str) -> Optional[Path]:
    """Locate the cached detection file for one variant."""
    matches = sorted(cache_dir.glob(f"{variant}_*.json"))
    return matches[0] if matches else None


def main() -> int:
    """Entry point."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sweep", required=True, type=Path)
    ap.add_argument("--cache", required=True, type=Path)
    ap.add_argument("--annotations", required=True, type=Path)
    ap.add_argument("--images", required=True, type=Path)
    ap.add_argument("--out", default=REPO_ROOT / "benchmarks" / "output", type=Path)
    ap.add_argument("--baseline", default="fp32")
    ap.add_argument(
        "--latency",
        default=None,
        type=Path,
        help="latency.json from run_latency.py; preferred over the sweep's own "
             "timings because it measures every variant on identical frames",
    )
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    with args.sweep.open("r", encoding="utf-8") as fh:
        sweep = json.load(fh)
    ok = [v for v in sweep["variants"] if v.get("status") == "ok"]
    if not ok:
        raise SystemExit("no successful variants in the sweep file")

    # ---------------------------------------------------------------- money chart
    latency_source = "the accuracy sweep"
    dedicated: Dict[str, dict] = {}
    if args.latency and Path(args.latency).is_file():
        with Path(args.latency).open("r", encoding="utf-8") as fh:
            dedicated = json.load(fh)["variants"]
        latency_source = "a dedicated pass on identical frames"
    print(f"latency from {latency_source}")

    points = []
    for v in ok:
        name = v["variant"]
        if name in dedicated:
            p50 = dedicated[name]["stages"]["total"]["p50_ms"]
        else:
            p50 = v["latency_ms"]["total"]["p50_ms"]
        points.append(
            {
                "name": name,
                "map": v["metrics"]["mAP"],
                "latency": p50,
                "size_mb": v["size_bytes"] / 1e6,
            }
        )
    viz.plot_accuracy_vs_latency(
        points,
        out / "accuracy-vs-latency.png",
        latency_key="p50",
        title=(
            f"YOLOv8n on COCO val2017 ({ok[0]['n_images']} images), "
            "ONNX Runtime CPU"
        ),
    )
    print("wrote accuracy-vs-latency.png")

    # ---------------------------------------------------------- baseline analysis
    baseline = next(v for v in ok if v["variant"] == args.baseline)
    cache_path = find_cache(Path(args.cache), args.baseline)
    if cache_path is None:
        raise SystemExit(f"no cached detections for {args.baseline}")
    with cache_path.open("r", encoding="utf-8") as fh:
        blob = json.load(fh)
    detections = blob["detections"]
    image_ids = [int(i) for i in blob["image_ids"]]

    with args.annotations.open("r", encoding="utf-8") as fh:
        raw = json.load(fh)
    ground_truth = GroundTruth.from_coco_dict(raw).subset(image_ids)

    print("scoring the baseline for the per-class and PR figures...")
    results = score_detections(ground_truth, detections, image_ids)
    rows = per_class_report(results, ground_truth)

    viz.plot_per_class_ap(
        rows,
        out / "per-class-ap.png",
        title=f"Per-class AP, {args.baseline}, {len(image_ids)} COCO val2017 images",
    )
    print("wrote per-class-ap.png")

    # Three strong classes and the two weakest, so the spread is visible.
    chosen = [r.category_id for r in rows[:3]] + [r.category_id for r in rows[-2:]]
    viz.plot_pr_curves(
        results,
        chosen,
        out / "pr-curves.png",
        iou=0.5,
        title=f"Precision-recall at IoU 0.50, {args.baseline}",
    )
    print("wrote pr-curves.png")

    print("running the error decomposition...")
    tide = tide_analysis(ground_truth, detections, image_ids)
    viz.plot_error_breakdown(
        tide["counts"],
        tide["delta_ap"],
        out / "error-breakdown.png",
        title=f"Where {args.baseline} loses AP50 on COCO val2017",
    )
    print("wrote error-breakdown.png")

    sweep_points = score_threshold_sweep(
        ground_truth, detections, n_images=len(image_ids)
    )
    viz.plot_threshold_sweep(
        sweep_points,
        out / "threshold-sweep.png",
        title=f"Confidence threshold sweep at IoU 0.50, {args.baseline}",
    )
    print("wrote threshold-sweep.png")

    # -------------------------------------------------------------- qualitative
    breakdown = classify_errors(ground_truth, detections, score_threshold=0.25)
    by_image: Dict[int, Dict[str, int]] = {}
    for det, label in zip(breakdown.detections, breakdown.labels):
        entry = by_image.setdefault(int(det["image_id"]), {})
        entry[label] = entry.get(label, 0) + 1

    gt_by_image: Dict[int, List[dict]] = {}
    for ann in ground_truth.annotations:
        gt_by_image.setdefault(int(ann["image_id"]), []).append(ann)
    det_by_image: Dict[int, List[dict]] = {}
    for det in detections:
        det_by_image.setdefault(int(det["image_id"]), []).append(det)

    path_by_id = {
        int(im["id"]): Path(args.images) / str(im["file_name"])
        for im in raw["images"]
    }

    def score_image(image_id: int) -> tuple:
        stats = by_image.get(image_id, {})
        n_gt = len([g for g in gt_by_image.get(image_id, [])
                    if not int(g.get("iscrowd", 0))])
        confident = [d for d in det_by_image.get(image_id, [])
                     if float(d["score"]) >= 0.25]
        correct = stats.get("correct", 0)
        wrong = sum(stats.get(k, 0) for k in
                    ("background", "classification", "both", "duplicate"))
        return correct, wrong, n_gt, len(confident)

    candidates = [i for i in image_ids if det_by_image.get(i)]
    good = max(
        candidates,
        key=lambda i: (score_image(i)[0] if score_image(i)[1] == 0 else -1,
                       score_image(i)[0]),
    )
    bad = max(candidates, key=lambda i: score_image(i)[1])

    for image_id, name, note in (
        (good, "qualitative-success.jpg", "every confident box is correct"),
        (bad, "qualitative-failure.jpg", "confident boxes that are wrong"),
    ):
        stats = score_image(image_id)
        caption = (
            f"image {image_id}: {stats[3]} detections at conf>=0.25, "
            f"{stats[0]} correct, {stats[1]} wrong, {stats[2]} objects labelled "
            f"- {note}"
        )
        viz.draw_detections(
            load_image(path_by_id[image_id]),
            det_by_image[image_id],
            out / name,
            ground_truth=gt_by_image.get(image_id, []),
            score_threshold=0.25,
            caption=caption,
        )
        print(f"wrote {name}  ({caption})")

    summary = {
        "baseline": args.baseline,
        "n_images": len(image_ids),
        "tide": tide,
        "threshold_sweep": [vars(p) for p in sweep_points],
        "per_class": [
            {
                "category_id": r.category_id,
                "name": r.name,
                "ap": r.ap,
                "ap50": r.ap50,
                "ap75": r.ap75,
                "ap_small": r.ap_small,
                "ap_medium": r.ap_medium,
                "ap_large": r.ap_large,
                "n_instances": r.n_instances,
            }
            for r in rows
        ],
        "qualitative": {"success_image_id": int(good), "failure_image_id": int(bad)},
    }
    results_path = Path(args.sweep).parent / "analysis.json"
    with results_path.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    print(f"wrote {results_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
