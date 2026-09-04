"""Build every quantisation variant, evaluate each, and write the results.

This is the script that produces the accuracy-versus-latency table in the
README. It is resumable: quantised models and detection files are cached, so
re-running after a crash costs only the work that was not finished.

Usage::

    python3 benchmarks/run_sweep.py --assets /path/to/assets \\
        --annotations /path/to/instances_val2017.json \\
        --images /path/to/val2017 --out benchmarks/results
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np  # noqa: E402

from detbench.eval.dataset import CocoDetectionDataset  # noqa: E402
from detbench.eval.runner import evaluate_run, file_sha256  # noqa: E402
from detbench.models.onnx_yolo import OnnxYoloDetector  # noqa: E402
from detbench.profiling import summarise_samples  # noqa: E402
from detbench.quantize import (  # noqa: E402
    quantize_dynamic_int8,
    quantize_static_int8,
)

#: Sweep definition, kept in config/ so it can be edited without touching code.
VARIANTS_FILE = REPO_ROOT / "config" / "variants.json"


def load_variants(path: Path = VARIANTS_FILE) -> List[dict]:
    """Read the sweep definition."""
    with Path(path).open("r", encoding="utf-8") as fh:
        return list(json.load(fh)["variants"])


def build_model(
    spec: dict,
    fp32_path: Path,
    pre_path: Path,
    assets: Path,
    calib_paths: List[Path],
) -> Dict[str, object]:
    """Produce the ONNX file for one variant, reusing it if already present."""
    name = spec["name"]
    if spec["kind"] == "fp32":
        return {"path": fp32_path, "size_bytes": fp32_path.stat().st_size, "notes": ""}

    out = assets / f"yolov8n_{name}.onnx"
    if out.is_file():
        return {
            "path": out,
            "size_bytes": out.stat().st_size,
            "notes": "reused cached quantised model",
        }

    if spec["kind"] == "dynamic":
        report = quantize_dynamic_int8(
            fp32_path, out, per_channel=spec["per_channel"], preprocessed_path=pre_path
        )
    else:
        report = quantize_static_int8(
            fp32_path,
            out,
            calib_paths[: spec["calib"]],
            input_name="images",
            per_channel=spec["per_channel"],
            calibration_method=spec["method"],
            quant_format="qdq",
            exclude_decode_tail=spec["exclude_tail"],
            preprocessed_path=pre_path,
        )
    return {"path": out, "size_bytes": report.size_bytes, "notes": report.notes}


def main() -> int:
    """Entry point."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--assets", required=True, type=Path)
    ap.add_argument("--annotations", required=True, type=Path)
    ap.add_argument("--images", required=True, type=Path)
    ap.add_argument("--out", default=REPO_ROOT / "benchmarks" / "results", type=Path)
    ap.add_argument("--cache", default=None, type=Path)
    ap.add_argument("--calibration-images", default=128, type=int)
    ap.add_argument("--limit", default=0, type=int, help="cap eval images (0 = all)")
    ap.add_argument("--seed", default=0, type=int)
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(args.cache) if args.cache else Path(args.assets) / "cache"

    full = CocoDetectionDataset(args.annotations, args.images)
    calib_ids, eval_ids = full.split_ids(args.calibration_images, seed=args.seed)
    if args.limit:
        eval_ids = eval_ids[: args.limit]
    print(
        f"dataset: {len(full)} images total, {len(calib_ids)} calibration, "
        f"{len(eval_ids)} evaluation (disjoint)",
        flush=True,
    )

    split_meta = {
        "dataset": "COCO val2017",
        "total_images": len(full),
        "calibration_images": len(calib_ids),
        "evaluation_images": len(eval_ids),
        "split_call": f"CocoDetectionDataset.split_ids({args.calibration_images}, "
                      f"seed={args.seed})",
        "disjoint": True,
    }
    with (out_dir / "split.json").open("w", encoding="utf-8") as fh:
        json.dump(
            {**split_meta,
             "calibration_image_ids": calib_ids,
             "evaluation_image_ids": eval_ids},
            fh,
        )

    by_id = {r.image_id: r.path for r in full}
    calib_paths = [by_id[i] for i in calib_ids]
    eval_ds = CocoDetectionDataset(args.annotations, args.images, image_ids=eval_ids)

    fp32_path = Path(args.assets) / "yolov8n.onnx"
    pre_path = Path(args.assets) / "yolov8n_preprocessed.onnx"

    rows: List[dict] = []
    for spec in load_variants():
        name = spec["name"]
        print(f"\n=== {name} ===", flush=True)
        if not spec.get("enabled", True):
            reason = spec.get("not_measured_reason", "disabled in config")
            print(f"  not measured: {reason}", flush=True)
            rows.append(
                {"variant": name, "status": "not-measured", "reason": reason}
            )
            continue
        t0 = time.time()
        try:
            built = build_model(
                spec, fp32_path, pre_path, Path(args.assets), calib_paths
            )
        except Exception as exc:  # noqa: BLE001 - a build failure is a result
            print(f"  build failed: {exc}", flush=True)
            rows.append({"variant": name, "status": "build-failed", "error": str(exc)})
            continue
        print(
            f"  model {built['path'].name} "
            f"{built['size_bytes'] / 1e6:.2f} MB "
            f"(built in {time.time() - t0:.1f}s)",
            flush=True,
        )

        try:
            detector = OnnxYoloDetector(built["path"], conf_threshold=0.001,
                                        iou_threshold=0.7, max_dets=300)
        except Exception as exc:  # noqa: BLE001 - an unrunnable model is a result
            print(f"  session failed: {exc}", flush=True)
            rows.append(
                {
                    "variant": name,
                    "status": "session-failed",
                    "error": str(exc).strip().splitlines()[-1],
                    "size_bytes": built["size_bytes"],
                    "sha256": file_sha256(built["path"]),
                }
            )
            continue

        detector.warmup(3)
        t0 = time.time()
        run, results = evaluate_run(
            detector,
            eval_ds,
            variant=name,
            cache_dir=cache_dir,
            model_path=built["path"],
            config={
                "conf": 0.001,
                "iou": 0.7,
                "imgsz": 640,
                "multi_label": False,
                "max_dets_nms": 300,
            },
            progress=lambda d, t: (
                print(f"    {d}/{t}", flush=True) if d % 500 == 0 else None
            ),
        )
        summary = results.summary()
        stage_stats = {
            stage: summarise_samples(stage, samples).as_dict()
            for stage, samples in run.stage_times_ms.items()
        }
        print(
            f"  mAP {summary['mAP']:.4f}  mAP50 {summary['mAP50']:.4f}  "
            f"({time.time() - t0:.0f}s, cached={run.from_cache})",
            flush=True,
        )

        rows.append(
            {
                "variant": name,
                "status": "ok",
                "model_file": built["path"].name,
                "size_bytes": built["size_bytes"],
                "sha256": file_sha256(built["path"]),
                "notes": built["notes"],
                "n_images": run.n_images,
                "n_detections": len(run.detections),
                "from_cache": run.from_cache,
                "metrics": summary,
                "per_class_ap": {
                    str(k): (None if np.isnan(v) else v)
                    for k, v in results.per_class_ap().items()
                },
                "latency_ms": stage_stats,
            }
        )
        detector.close()

        with (out_dir / "sweep.json").open("w", encoding="utf-8") as fh:
            json.dump({"split": split_meta, "variants": rows}, fh, indent=2)

    print(f"\nwrote {out_dir / 'sweep.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
