"""Per-class accuracy: which classes work, which quietly do not.

A single mAP number is an average over eighty classes whose individual APs
span roughly an order of magnitude. Shipping a detector on the strength of the
average, when the class you actually care about sits in the bottom decile, is
one of the more expensive mistakes available in this field.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np

from ..coco_classes import CATEGORY_NAMES
from ..metrics.coco_map import COCOResults, GroundTruth

__all__ = ["ClassReport", "per_class_report", "format_class_table"]


@dataclass(frozen=True)
class ClassReport:
    """Accuracy and support for one class."""

    category_id: int
    name: str
    ap: float
    ap50: float
    ap75: float
    ap_small: float
    ap_medium: float
    ap_large: float
    n_instances: int

    @property
    def gap_50_75(self) -> float:
        """AP50 minus AP75: how much of the accuracy is loose localisation.

        A class with AP50 0.60 and AP75 0.20 is being found but not outlined.
        That is a different engineering problem from a class the model misses.
        """
        return self.ap50 - self.ap75


def instance_counts(ground_truth: GroundTruth) -> Dict[int, int]:
    """Non-crowd annotation count per category id."""
    counts: Dict[int, int] = {}
    for ann in ground_truth.annotations:
        if int(ann.get("iscrowd", 0)):
            continue
        cid = int(ann["category_id"])
        counts[cid] = counts.get(cid, 0) + 1
    return counts


def per_class_report(
    results: COCOResults,
    ground_truth: Optional[GroundTruth] = None,
) -> List[ClassReport]:
    """Build a per-class report, sorted best to worst by AP.

    Classes absent from the evaluated images are dropped rather than reported
    as zero, because "no ground truth" and "found nothing" are different facts.
    """
    counts = instance_counts(ground_truth) if ground_truth is not None else {}
    ap = results.per_class_ap()
    ap50 = results.per_class_ap(iou=0.5)
    ap75 = results.per_class_ap(iou=0.75)
    ap_s = results.per_class_ap(area="small")
    ap_m = results.per_class_ap(area="medium")
    ap_l = results.per_class_ap(area="large")

    rows: List[ClassReport] = []
    for cid in results.category_ids:
        if np.isnan(ap[cid]):
            continue
        rows.append(
            ClassReport(
                category_id=cid,
                name=CATEGORY_NAMES.get(cid, str(cid)),
                ap=ap[cid],
                ap50=ap50[cid],
                ap75=ap75[cid],
                ap_small=ap_s[cid],
                ap_medium=ap_m[cid],
                ap_large=ap_l[cid],
                n_instances=counts.get(cid, 0),
            )
        )
    rows.sort(key=lambda r: r.ap, reverse=True)
    return rows


def format_class_table(
    rows: Sequence[ClassReport], limit: Optional[int] = None
) -> str:
    """Render a per-class report as a fixed-width table."""
    header = (
        f"{'class':<16}{'AP':>8}{'AP50':>8}{'AP75':>8}"
        f"{'AP_s':>8}{'AP_m':>8}{'AP_l':>8}{'n':>8}"
    )
    lines = [header, "-" * len(header)]
    subset = rows if limit is None else list(rows)[:limit]
    for r in subset:
        def fmt(v: float) -> str:
            return "  n/a" if np.isnan(v) else f"{v:.3f}"

        lines.append(
            f"{r.name:<16}{fmt(r.ap):>8}{fmt(r.ap50):>8}{fmt(r.ap75):>8}"
            f"{fmt(r.ap_small):>8}{fmt(r.ap_medium):>8}{fmt(r.ap_large):>8}"
            f"{r.n_instances:>8}"
        )
    return "\n".join(lines)
