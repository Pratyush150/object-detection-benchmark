"""Command-line interface for detbench.

Subcommands:

``evaluate``   run a model over a COCO split and print the twelve COCO numbers
``quantize``   build INT8 variants and compare them against float32
``analyse``    per-class AP, error taxonomy and the confidence-threshold sweep
``profile``    per-stage latency percentiles
``demo``       the whole pipeline on generated data, no downloads, no weights
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np

from . import __version__
from .analysis.curves import format_sweep_table, score_threshold_sweep
from .analysis.errors import tide_analysis
from .analysis.per_class import format_class_table, per_class_report
from .eval.dataset import CocoDetectionDataset, load_image
from .eval.runner import evaluate_run, run_detector, score_detections
from .eval.synthetic import make_synthetic_dataset
from .metrics.coco_map import GroundTruth
from .profiling import profile_detector

__all__ = ["main", "build_parser"]


def _progress(done: int, total: int) -> None:
    if done % 250 == 0 or done == total:
        print(f"    {done}/{total}", flush=True)


def _load_dataset(args: argparse.Namespace) -> CocoDetectionDataset:
    dataset = CocoDetectionDataset(args.annotations, args.images)
    if getattr(args, "limit", 0):
        dataset = CocoDetectionDataset(
            args.annotations, args.images, image_ids=dataset.image_ids[: args.limit]
        )
    return dataset


def _make_detector(args: argparse.Namespace):
    from .models.onnx_yolo import OnnxYoloDetector

    return OnnxYoloDetector(
        args.model,
        input_size=(args.imgsz, args.imgsz),
        conf_threshold=args.conf,
        iou_threshold=args.iou,
        max_dets=args.nms_max_dets,
        multi_label=args.multi_label,
        intra_op_threads=args.threads,
    )


# --------------------------------------------------------------------- demo
def cmd_demo(args: argparse.Namespace) -> int:
    """Run the whole pipeline on generated data."""
    print("detbench demo - synthetic data, no model weights, no dataset")
    print("These numbers describe generated shapes, not COCO. They exist to")
    print("show the pipeline working end to end in one command.\n")

    dataset = make_synthetic_dataset(n_images=args.demo_images)
    run = run_detector(
        dataset.detector,
        ((i, dataset.images[i]) for i in dataset.image_ids),
        variant="synthetic",
    )
    results = score_detections(
        dataset.ground_truth, run.detections, dataset.image_ids
    )

    print(f"images: {run.n_images}   detections: {len(run.detections)}")
    print(f"ground-truth objects: {len(dataset.ground_truth.annotations)}\n")
    print(results.format_summary())

    rows = per_class_report(results, dataset.ground_truth)
    print("\nper-class AP (best to worst)")
    print(format_class_table(rows))

    tide = tide_analysis(dataset.ground_truth, run.detections, dataset.image_ids)
    print(f"\nerror taxonomy at IoU {tide['iou_threshold']:.2f} "
          f"(baseline AP50 {tide['baseline_ap']:.4f})")
    print(f"{'type':<16}{'count':>8}{'AP50 if fixed':>16}")
    print("-" * 40)
    for name, delta in sorted(
        tide["delta_ap"].items(), key=lambda kv: -kv[1]  # type: ignore[index]
    ):
        print(f"{name:<16}{tide['counts'][name]:>8}{delta * 100:>15.2f}pt")

    points = score_threshold_sweep(
        dataset.ground_truth,
        run.detections,
        thresholds=[0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        n_images=len(dataset.images),
    )
    print("\nconfidence threshold sweep at IoU 0.50")
    print(format_sweep_table(points))

    profile = profile_detector(
        dataset.detector,
        [dataset.images[i] for i in dataset.image_ids[:8]],
        label="synthetic-mock",
        warmup=2,
        repeats=3,
    )
    print("\nper-stage latency, milliseconds")
    print(profile.format_table())
    print(
        f"\nmean {profile.mean_fps:.1f} FPS, but only "
        f"{profile.p99_fps:.1f} FPS is achieved 99% of the time."
    )

    if args.figures:
        out = Path(args.figures)
        _write_demo_figures(dataset, results, rows, tide, points, out)
        print(f"\nfigures written to {out}")
    return 0


def _write_demo_figures(dataset, results, rows, tide, points, out: Path) -> None:
    from . import viz

    viz.plot_per_class_ap(rows, out / "demo-per-class-ap.png",
                          title="Per-class AP (synthetic demo)")
    viz.plot_pr_curves(
        results,
        [r.category_id for r in rows[:4]],
        out / "demo-pr-curves.png",
        title="Precision-recall (synthetic demo)",
    )
    viz.plot_error_breakdown(
        tide["counts"], tide["delta_ap"], out / "demo-errors.png",
        title="Error taxonomy (synthetic demo)",
    )
    viz.plot_threshold_sweep(points, out / "demo-threshold-sweep.png",
                             title="Threshold sweep (synthetic demo)")


# ----------------------------------------------------------------- evaluate
def cmd_evaluate(args: argparse.Namespace) -> int:
    """Evaluate an ONNX model on a COCO split."""
    dataset = _load_dataset(args)
    detector = _make_detector(args)
    detector.warmup(3)
    print(f"evaluating {Path(args.model).name} on {len(dataset)} images", flush=True)

    run, results = evaluate_run(
        detector,
        dataset,
        variant=args.variant or Path(args.model).stem,
        cache_dir=Path(args.cache) if args.cache else None,
        model_path=Path(args.model),
        config={
            "conf": args.conf,
            "iou": args.iou,
            "imgsz": args.imgsz,
            "multi_label": args.multi_label,
        },
        progress=_progress,
        force=args.no_cache,
    )
    print(f"\n{run.n_images} images, {len(run.detections)} detections, "
          f"cached={run.from_cache}\n")
    print(results.format_summary())

    stages = run.stage_percentiles()
    if stages:
        print("\nper-stage latency, milliseconds")
        print(f"{'stage':<13}{'mean':>9}{'p50':>9}{'p90':>9}{'p99':>9}")
        for stage in ("preprocess", "inference", "nms", "postprocess", "total"):
            if stage not in stages:
                continue
            s = stages[stage]
            print(f"{stage:<13}{s['mean']:>9.2f}{s['p50']:>9.2f}"
                  f"{s['p90']:>9.2f}{s['p99']:>9.2f}")

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as fh:
            json.dump(
                {
                    "model": str(args.model),
                    "n_images": run.n_images,
                    "metrics": results.summary(),
                    "latency_ms": stages,
                },
                fh,
                indent=2,
            )
        print(f"\nwrote {out}")
    return 0


# ----------------------------------------------------------------- quantize
def cmd_quantize(args: argparse.Namespace) -> int:
    """Build INT8 variants and report size, accuracy and latency."""
    from .quantize import quantize_dynamic_int8, quantize_static_int8

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    full = CocoDetectionDataset(args.annotations, args.images)
    calib_ids, eval_ids = full.split_ids(args.calibration_images, seed=args.seed)
    if args.limit:
        eval_ids = eval_ids[: args.limit]
    by_id = {r.image_id: r.path for r in full}
    print(f"{len(calib_ids)} calibration images, {len(eval_ids)} evaluation "
          f"images, disjoint by construction")

    pre = out_dir / "preprocessed.onnx"
    reports = []
    if args.dynamic:
        reports.append(
            quantize_dynamic_int8(
                args.model, out_dir / "int8-dynamic.onnx", preprocessed_path=pre
            )
        )
    reports.append(
        quantize_static_int8(
            args.model,
            out_dir / "int8-static.onnx",
            [by_id[i] for i in calib_ids],
            input_size=(args.imgsz, args.imgsz),
            calibration_method=args.method,
            per_channel=not args.per_tensor,
            exclude_decode_tail=not args.quantize_head,
            preprocessed_path=pre,
        )
    )

    dataset = CocoDetectionDataset(args.annotations, args.images, image_ids=eval_ids)
    print(f"\n{'variant':<22}{'MB':>8}{'ratio':>8}{'mAP':>8}"
          f"{'mAP50':>8}{'p50 ms':>9}")
    print("-" * 63)

    base = Path(args.model)
    for model_path, label in [(base, "fp32")] + [
        (r.model_path, r.variant) for r in reports
    ]:
        size_mb = Path(model_path).stat().st_size / 1e6
        ratio = Path(model_path).stat().st_size / base.stat().st_size
        try:
            args.model = model_path
            detector = _make_detector(args)
        except Exception as exc:  # noqa: BLE001 - unrunnable is a real result
            print(f"{label:<22}{size_mb:>8.2f}{ratio:>8.3f}"
                  f"   session failed: {str(exc).strip().splitlines()[-1][:40]}")
            continue
        detector.warmup(3)
        run, results = evaluate_run(
            detector,
            dataset,
            variant=label,
            cache_dir=Path(args.cache) if args.cache else None,
            model_path=Path(model_path),
            config={"conf": args.conf, "iou": args.iou, "imgsz": args.imgsz},
            progress=_progress,
        )
        s = results.summary()
        p50 = run.stage_percentiles()["total"]["p50"]
        print(f"{label:<22}{size_mb:>8.2f}{ratio:>8.3f}{s['mAP']:>8.4f}"
              f"{s['mAP50']:>8.4f}{p50:>9.2f}")
        detector.close()
    args.model = base
    return 0


# ------------------------------------------------------------------ analyse
def cmd_analyse(args: argparse.Namespace) -> int:
    """Failure analysis on an existing detections file."""
    with Path(args.annotations).open("r", encoding="utf-8") as fh:
        ground_truth = GroundTruth.from_coco_dict(json.load(fh))

    detections, image_ids = _load_detections(Path(args.detections))
    if image_ids is None:
        image_ids = sorted({int(d["image_id"]) for d in detections})
    ground_truth = ground_truth.subset(image_ids)
    print(f"{len(detections)} detections over {len(image_ids)} images\n")

    results = score_detections(ground_truth, detections, image_ids)
    print(results.format_summary())

    rows = per_class_report(results, ground_truth)
    print(f"\ntop {args.top} classes")
    print(format_class_table(rows, limit=args.top))
    print(f"\nbottom {args.top} classes")
    print(format_class_table(rows[-args.top :]))

    tide = tide_analysis(ground_truth, detections, image_ids)
    print(f"\nerror taxonomy at IoU {tide['iou_threshold']:.2f} "
          f"(baseline AP50 {tide['baseline_ap']:.4f})")
    print(f"{'type':<16}{'count':>10}{'AP50 if fixed':>16}")
    print("-" * 42)
    for name, delta in sorted(
        tide["delta_ap"].items(), key=lambda kv: -kv[1]  # type: ignore[index]
    ):
        print(f"{name:<16}{tide['counts'][name]:>10}{delta * 100:>15.2f}pt")

    print("\nmost frequent class confusions (true -> predicted)")
    for row in tide["confusions"][: args.top]:  # type: ignore[index]
        print(f"  {row['true_name']:<16} -> {row['pred_name']:<16} "
              f"{row['count']:>5}")

    points = score_threshold_sweep(
        ground_truth, detections, n_images=len(image_ids)
    )
    print("\nconfidence threshold sweep at IoU 0.50")
    print(format_sweep_table(points))

    if args.figures:
        from . import viz

        out = Path(args.figures)
        viz.plot_per_class_ap(rows, out / "per-class-ap.png")
        viz.plot_pr_curves(
            results,
            [r.category_id for r in rows[:3]] + [rows[-1].category_id],
            out / "pr-curves.png",
        )
        viz.plot_error_breakdown(
            tide["counts"], tide["delta_ap"], out / "error-breakdown.png"
        )
        viz.plot_threshold_sweep(points, out / "threshold-sweep.png")
        print(f"\nfigures written to {out}")

    if args.out:
        with Path(args.out).open("w", encoding="utf-8") as fh:
            json.dump(
                {
                    "metrics": results.summary(),
                    "per_class": [
                        {
                            "category_id": r.category_id,
                            "name": r.name,
                            "ap": r.ap,
                            "ap50": r.ap50,
                            "ap75": r.ap75,
                            "n_instances": r.n_instances,
                        }
                        for r in rows
                    ],
                    "tide": tide,
                    "threshold_sweep": [vars(p) for p in points],
                },
                fh,
                indent=2,
            )
        print(f"wrote {args.out}")
    return 0


def _load_detections(path: Path) -> tuple[List[dict], Optional[List[int]]]:
    """Accept either a plain COCO results list or a runner cache blob."""
    with path.open("r", encoding="utf-8") as fh:
        blob = json.load(fh)
    if isinstance(blob, list):
        return blob, None
    return blob["detections"], [int(i) for i in blob["image_ids"]]


# ------------------------------------------------------------------ profile
def cmd_profile(args: argparse.Namespace) -> int:
    """Measure per-stage latency percentiles."""
    detector = _make_detector(args)
    if args.images and args.annotations:
        dataset = CocoDetectionDataset(args.annotations, args.images)
        frames = [load_image(r.path) for r in list(dataset)[: args.frames]]
        source = f"{len(frames)} COCO images"
    else:
        dataset = make_synthetic_dataset(n_images=args.frames)
        frames = [dataset.images[i] for i in dataset.image_ids]
        source = f"{len(frames)} synthetic frames"

    profile = profile_detector(
        detector,
        frames,
        label=Path(args.model).stem,
        warmup=args.warmup,
        repeats=args.repeats,
    )
    print(f"{profile.label} on {source}, {args.repeats} repeats, "
          f"{args.warmup} warmup iterations")
    print(f"threads: {args.threads if args.threads else 'runtime default'}\n")
    print(profile.format_table())
    print(f"\nmean {profile.mean_fps:.1f} FPS; p99 frame time implies "
          f"{profile.p99_fps:.1f} FPS in the worst 1% of frames.")
    return 0


# ------------------------------------------------------------------- parser
def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser."""
    parser = argparse.ArgumentParser(
        prog="detbench",
        description="Measure detection accuracy, and what it costs in latency.",
    )
    parser.add_argument(
        "--version", action="version", version=f"detbench {__version__}"
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="run the full pipeline on generated data and exit",
    )
    parser.add_argument("--demo-images", type=int, default=24)
    parser.add_argument("--figures", default=None, help="directory for figures")
    sub = parser.add_subparsers(dest="command")

    def add_model_args(p: argparse.ArgumentParser) -> None:
        p.add_argument("--model", required=True, help="path to the ONNX model")
        p.add_argument("--imgsz", type=int, default=640)
        p.add_argument("--conf", type=float, default=0.001)
        p.add_argument("--iou", type=float, default=0.7)
        p.add_argument("--nms-max-dets", type=int, default=300)
        p.add_argument("--multi-label", action="store_true")
        p.add_argument("--threads", type=int, default=None)

    p_demo = sub.add_parser("demo", help="run on generated data")
    p_demo.add_argument("--demo-images", type=int, default=24)
    p_demo.add_argument("--figures", default=None)
    p_demo.set_defaults(func=cmd_demo)

    p_eval = sub.add_parser("evaluate", help="score a model on a COCO split")
    add_model_args(p_eval)
    p_eval.add_argument("--annotations", required=True)
    p_eval.add_argument("--images", required=True)
    p_eval.add_argument("--limit", type=int, default=0)
    p_eval.add_argument("--cache", default=None)
    p_eval.add_argument("--no-cache", action="store_true")
    p_eval.add_argument("--variant", default=None)
    p_eval.add_argument("--out", default=None)
    p_eval.set_defaults(func=cmd_evaluate)

    p_quant = sub.add_parser("quantize", help="build and score INT8 variants")
    add_model_args(p_quant)
    p_quant.add_argument("--annotations", required=True)
    p_quant.add_argument("--images", required=True)
    p_quant.add_argument("--out-dir", required=True)
    p_quant.add_argument("--calibration-images", type=int, default=128)
    p_quant.add_argument("--method", default="minmax",
                         choices=["minmax", "entropy", "percentile"])
    p_quant.add_argument("--per-tensor", action="store_true")
    p_quant.add_argument("--quantize-head", action="store_true",
                         help="also quantise the decode tail (expect mAP to collapse)")
    p_quant.add_argument("--dynamic", action="store_true")
    p_quant.add_argument("--limit", type=int, default=0)
    p_quant.add_argument("--seed", type=int, default=0)
    p_quant.add_argument("--cache", default=None)
    p_quant.set_defaults(func=cmd_quantize)

    p_an = sub.add_parser("analyse", help="failure analysis on a detections file")
    p_an.add_argument("--annotations", required=True)
    p_an.add_argument("--detections", required=True)
    p_an.add_argument("--top", type=int, default=10)
    p_an.add_argument("--figures", default=None)
    p_an.add_argument("--out", default=None)
    p_an.set_defaults(func=cmd_analyse)

    p_prof = sub.add_parser("profile", help="per-stage latency percentiles")
    add_model_args(p_prof)
    p_prof.add_argument("--annotations", default=None)
    p_prof.add_argument("--images", default=None)
    p_prof.add_argument("--frames", type=int, default=50)
    p_prof.add_argument("--warmup", type=int, default=5)
    p_prof.add_argument("--repeats", type=int, default=1)
    p_prof.set_defaults(func=cmd_profile)

    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Entry point."""
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.demo or args.command is None:
        if args.command is None and not args.demo:
            parser.print_help()
            return 1
        return cmd_demo(args)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
