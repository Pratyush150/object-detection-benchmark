#!/usr/bin/env python3
"""Download COCO val2017 and the YOLOv8n weights, then export the ONNX model.

Nothing this script fetches is committed to the repository: the images are
about 780 MB, the weights are third-party, and both have their own licences.
Every download is checked against a recorded SHA-256 before it is used, so a
truncated or substituted file fails loudly instead of quietly changing the
numbers in the README.

Usage::

    python3 tools/fetch_assets.py --dest assets
    python3 tools/fetch_assets.py --dest assets --skip-images   # metric only
    python3 tools/fetch_assets.py --dest assets --verify-only

The export step needs ``ultralytics`` installed. If it is missing, the script
says so and stops rather than guessing; an ONNX file exported with different
settings would silently change every number this repo reports.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path
from typing import Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST = REPO_ROOT / "config" / "assets.json"


def sha256(path: Path, chunk: int = 1 << 20) -> str:
    """Stream a file through SHA-256."""
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def human(n: int) -> str:
    """Format a byte count."""
    return f"{n / 1e6:.1f} MB" if n < 1e9 else f"{n / 1e9:.2f} GB"


def download(url: str, dest: Path, expected_size: Optional[int] = None) -> Path:
    """Download a URL to ``dest``, showing coarse progress."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    print(f"  downloading {url}")
    with urllib.request.urlopen(url) as response, tmp.open("wb") as out:
        total = int(response.headers.get("Content-Length") or expected_size or 0)
        done = 0
        step = max(total // 20, 1 << 22)
        next_mark = step
        while True:
            block = response.read(1 << 20)
            if not block:
                break
            out.write(block)
            done += len(block)
            if done >= next_mark:
                pct = f" ({done * 100 // total}%)" if total else ""
                print(f"    {human(done)}{pct}", flush=True)
                next_mark += step
    tmp.replace(dest)
    return dest


def ensure(entry: Dict[str, object], dest_dir: Path, verify_only: bool) -> Path:
    """Download and verify one manifest entry."""
    path = dest_dir / str(entry["filename"])
    name = str(entry["name"])
    print(f"[{name}]")

    if path.is_file():
        print(f"  present: {path} ({human(path.stat().st_size)})")
    elif verify_only:
        print("  missing (verify-only mode, not downloading)")
        return path
    else:
        download(str(entry["url"]), path, int(entry.get("size_bytes") or 0))

    if not path.is_file():
        return path

    print("  hashing")
    digest = sha256(path)
    if digest != entry["sha256"]:
        raise SystemExit(
            f"  SHA-256 mismatch for {path}\n"
            f"    expected {entry['sha256']}\n"
            f"    got      {digest}\n"
            "  Delete the file and retry."
        )
    print(f"  sha256 ok: {digest[:16]}...")
    return path


def extract(entry: Dict[str, object], archive: Path, dest_dir: Path) -> None:
    """Extract an archive entry, skipping work that is already done."""
    produces = entry.get("produces")
    if produces:
        target = dest_dir / str(produces)
        if target.exists():
            print(f"  already extracted: {target}")
            return
    members: Optional[List[str]] = entry.get("extract_members")  # type: ignore
    out = dest_dir / str(entry.get("extract_to", "."))
    print(f"  extracting into {out}")
    with zipfile.ZipFile(archive) as zf:
        bad = zf.testzip()
        if bad is not None:
            raise SystemExit(f"  corrupt archive member: {bad}")
        zf.extractall(out, members=members)


def export_onnx(manifest: Dict[str, object], dest_dir: Path) -> None:
    """Export the checkpoint to ONNX with the exact settings used here."""
    spec = manifest["export"]  # type: ignore[index]
    source = dest_dir / str(spec["source"])
    output = dest_dir / str(spec["output"])
    print("[export]")
    if output.is_file():
        print(f"  present: {output} ({human(output.stat().st_size)})")
        print(f"  sha256: {sha256(output)}")
        return
    if not source.is_file():
        print(f"  skipped: {source} is not present")
        return
    try:
        from ultralytics import YOLO  # noqa: PLC0415
    except ImportError:
        print(
            "  ultralytics is not installed, so the ONNX export cannot run.\n"
            "  Install it (pip install ultralytics) and re-run, or supply your\n"
            f"  own {spec['output']} exported with imgsz={spec['imgsz']}, "
            f"opset={spec['opset']}, dynamic={spec['dynamic']}, nms={spec['nms']}."
        )
        return

    print(f"  exporting {source.name} -> {output.name}")
    model = YOLO(str(source))
    produced = model.export(
        format=str(spec["format"]),
        imgsz=int(spec["imgsz"]),
        opset=int(spec["opset"]),
        dynamic=bool(spec["dynamic"]),
        simplify=bool(spec["simplify"]),
        nms=bool(spec["nms"]),
    )
    produced = Path(produced)
    if produced.resolve() != output.resolve():
        shutil.move(str(produced), str(output))
    print(f"  wrote {output} ({human(output.stat().st_size)})")
    print(f"  sha256: {sha256(output)}")
    print(f"  reference hash from {spec['exported_with']}: {spec['sha256_observed']}")
    print(f"  {spec['sha256_note']}")


def main() -> int:
    """Entry point."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dest", default=REPO_ROOT / "assets", type=Path)
    ap.add_argument("--skip-images", action="store_true",
                    help="skip the 780 MB image download")
    ap.add_argument("--verify-only", action="store_true",
                    help="check what is present without downloading")
    ap.add_argument("--keep-archives", action="store_true")
    args = ap.parse_args()

    with MANIFEST.open("r", encoding="utf-8") as fh:
        manifest = json.load(fh)

    dest = Path(args.dest)
    dest.mkdir(parents=True, exist_ok=True)
    print(f"asset directory: {dest.resolve()}\n")

    for entry in manifest["downloads"]:
        if args.skip_images and entry["name"] == "val2017-images":
            print(f"[{entry['name']}]\n  skipped\n")
            continue
        path = ensure(entry, dest, args.verify_only)
        if path.is_file() and path.suffix == ".zip":
            extract(entry, path, dest)
            if not args.keep_archives and not args.verify_only:
                path.unlink()
                print(f"  removed archive {path.name}")
        print()

    export_onnx(manifest, dest)
    print(
        "\nDone. Point the tools at this directory, for example:\n"
        f"  DETBENCH_ASSETS={dest.resolve()} "
        "env -u PYTHONPATH python3 -m pytest -q\n"
        f"  ./tools/detbench evaluate --model {dest}/yolov8n.onnx \\\n"
        f"      --annotations {dest}/annotations/instances_val2017.json \\\n"
        f"      --images {dest}/val2017"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
