"""Measure per-stage latency for every runnable variant, on identical frames.

Latency is measured in its own pass rather than reused from the accuracy sweep,
for two reasons. Every variant sees exactly the same images in the same order,
so the comparison is not confounded by image size. And the pass is meant to run
on an otherwise idle machine: percentiles measured while something else is
using the cores describe the contention, not the model.

Usage::

    python3 benchmarks/run_latency.py --assets /path/to/assets \\
        --annotations /path/to/instances_val2017.json \\
        --images /path/to/val2017 --frames 200 --out benchmarks/results
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
from pathlib import Path
from typing import Dict, List

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from detbench.eval.dataset import CocoDetectionDataset, load_image  # noqa: E402
from detbench.models.onnx_yolo import OnnxYoloDetector  # noqa: E402
from detbench.profiling import profile_detector  # noqa: E402


def cpu_name() -> str:
    """Best-effort CPU model string."""
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def main() -> int:
    """Entry point."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--assets", required=True, type=Path)
    ap.add_argument("--annotations", required=True, type=Path)
    ap.add_argument("--images", required=True, type=Path)
    ap.add_argument("--out", default=REPO_ROOT / "benchmarks" / "results", type=Path)
    ap.add_argument("--frames", type=int, default=200)
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--sweep", default=None, type=Path,
                    help="sweep.json, to pick up the variants that ran")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    names: List[str]
    if args.sweep and Path(args.sweep).is_file():
        with Path(args.sweep).open("r", encoding="utf-8") as fh:
            names = [
                v["variant"] for v in json.load(fh)["variants"]
                if v.get("status") == "ok"
            ]
    else:
        names = ["fp32"]

    dataset = CocoDetectionDataset(args.annotations, args.images)
    records = list(dataset)[: args.frames]
    print(f"loading {len(records)} frames into memory", flush=True)
    frames = [load_image(r.path) for r in records]

    machine = {
        "cpu": cpu_name(),
        "logical_cores": os.cpu_count(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "threads": args.threads,
        "frames": len(frames),
        "repeats": args.repeats,
        "warmup": args.warmup,
        "image_ids": [r.image_id for r in records],
    }
    print(f"{machine['cpu']}, {machine['logical_cores']} logical cores\n")

    results: Dict[str, dict] = {}
    for name in names:
        path = (
            Path(args.assets) / "yolov8n.onnx"
            if name == "fp32"
            else Path(args.assets) / f"yolov8n_{name}.onnx"
        )
        if not path.is_file():
            print(f"{name}: model missing at {path}, skipped")
            continue
        detector = OnnxYoloDetector(path, intra_op_threads=args.threads, name=name)
        profile = profile_detector(
            detector, frames, label=name, warmup=args.warmup, repeats=args.repeats
        )
        results[name] = {
            "model_file": path.name,
            "size_bytes": path.stat().st_size,
            "stages": {k: v.as_dict() for k, v in profile.stages.items()},
            "mean_fps": profile.mean_fps,
            "p99_fps": profile.p99_fps,
        }
        print(f"=== {name} ===")
        print(profile.format_table())
        print(f"mean {profile.mean_fps:.2f} FPS, p99 {profile.p99_fps:.2f} FPS\n",
              flush=True)
        detector.close()

    payload = {"machine": machine, "variants": results}
    with (out_dir / "latency.json").open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    print(f"wrote {out_dir / 'latency.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
