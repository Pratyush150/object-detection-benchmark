"""COCO val2017 access: ground truth, image paths, and reproducible subsets."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Optional, Sequence

import numpy as np

from ..coco_classes import verify_against_categories
from ..metrics.coco_map import GroundTruth

__all__ = ["CocoDetectionDataset", "ImageRecord", "load_image"]


@dataclass(frozen=True)
class ImageRecord:
    """One dataset image."""

    image_id: int
    file_name: str
    width: int
    height: int
    path: Path


def load_image(path: Path) -> np.ndarray:
    """Read an image as ``(H, W, 3)`` BGR uint8.

    Grayscale files (COCO val2017 contains a few) are expanded to three
    channels rather than skipped, because dropping them would quietly change
    the image count the reported mAP is based on.
    """
    try:
        import cv2  # noqa: PLC0415  (optional dependency)
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("opencv is required to read dataset images") from exc

    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise OSError(f"could not decode image: {path}")
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)
    return img


class CocoDetectionDataset:
    """Ground truth plus image paths for a COCO detection split.

    Args:
        annotation_file: Path to ``instances_val2017.json``.
        images_dir: Directory holding the JPEGs.
        image_ids: Restrict to these image ids. ``None`` uses all of them.
    """

    def __init__(
        self,
        annotation_file: str | Path,
        images_dir: str | Path,
        image_ids: Optional[Sequence[int]] = None,
    ) -> None:
        self.annotation_file = Path(annotation_file)
        self.images_dir = Path(images_dir)
        if not self.annotation_file.is_file():
            raise FileNotFoundError(f"annotations not found: {self.annotation_file}")
        if not self.images_dir.is_dir():
            raise FileNotFoundError(f"image directory not found: {self.images_dir}")

        with self.annotation_file.open("r", encoding="utf-8") as fh:
            raw = json.load(fh)
        verify_against_categories(raw["categories"])

        keep = set(int(i) for i in image_ids) if image_ids is not None else None
        self.records: List[ImageRecord] = []
        for im in raw["images"]:
            iid = int(im["id"])
            if keep is not None and iid not in keep:
                continue
            self.records.append(
                ImageRecord(
                    image_id=iid,
                    file_name=str(im["file_name"]),
                    width=int(im["width"]),
                    height=int(im["height"]),
                    path=self.images_dir / str(im["file_name"]),
                )
            )
        self.records.sort(key=lambda r: r.image_id)

        self.ground_truth = GroundTruth.from_coco_dict(raw)
        if keep is not None:
            self.ground_truth = self.ground_truth.subset(keep)

    def __len__(self) -> int:
        return len(self.records)

    def __iter__(self) -> Iterator[ImageRecord]:
        return iter(self.records)

    @property
    def image_ids(self) -> List[int]:
        """Sorted image ids in this dataset."""
        return [r.image_id for r in self.records]

    def split_ids(self, n: int, seed: int = 0) -> tuple[List[int], List[int]]:
        """Deterministically split image ids into ``n`` and the rest.

        Used to carve a calibration set out of val2017 that is disjoint from
        the evaluation set. Calibrating on images you also score on inflates
        the result; the split makes that impossible rather than merely
        unlikely.
        """
        ids = self.image_ids
        if n > len(ids):
            raise ValueError(f"cannot take {n} images from a set of {len(ids)}")
        rng = np.random.default_rng(seed)
        order = rng.permutation(len(ids))
        chosen = sorted(ids[i] for i in order[:n])
        rest = sorted(ids[i] for i in order[n:])
        return chosen, rest
