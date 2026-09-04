"""COCO mean Average Precision, implemented from the protocol description.

This is a self-contained reimplementation of the COCO detection metric. It has
no dependency on ``pycocotools``; the reference implementation is used only in
the test suite, to prove that this one agrees with it.

The protocol, in the order the code applies it:

1. **Overlap.** Intersection over union between each detection and each ground
   truth of the same class in the same image. Ground truths flagged
   ``iscrowd`` use intersection over the *detection* area instead, so a
   detection landing inside a crowd blob scores 1.0.
2. **Ignore flags.** A ground truth is *ignored* for a given evaluation if it
   is a crowd region or if its area falls outside the area range under test.
   Ignored ground truths are sorted to the back so that greedy matching
   prefers a real ground truth over an ignored one.
3. **Greedy matching.** Detections are processed in descending score order.
   Each takes the highest-overlap unmatched ground truth above the IoU
   threshold. A ground truth can be claimed once (crowd regions are exempt and
   may absorb many detections). Later, lower-scoring duplicates therefore
   become false positives, which is what penalises duplicate detections.
4. **Ignored outcomes.** A detection matched to an ignored ground truth, and an
   unmatched detection whose own area falls outside the area range, are dropped
   from both the true- and false-positive counts. Crowd regions neither reward
   nor penalise.
5. **Precision-recall.** True and false positives are accumulated in score
   order across the whole dataset per class. Precision is made monotonically
   non-increasing from the right, then sampled at 101 evenly spaced recall
   levels 0.00, 0.01, ..., 1.00. Recall levels beyond what the detector
   achieved contribute precision 0.
6. **Averaging.** AP is the mean of those 101 samples. mAP is the mean over
   classes, then over the ten IoU thresholds 0.50:0.05:0.95.

Array shapes follow the reference layout so results are directly comparable:
``precision`` is ``(T, R, K, A, M)`` over IoU thresholds, recall thresholds,
classes, area ranges and max-detection caps; ``recall`` is ``(T, K, A, M)``.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from ..coco_classes import CATEGORY_NAMES
from .box_ops import iou_matrix

__all__ = [
    "AreaRange",
    "EvalParams",
    "GroundTruth",
    "COCOResults",
    "COCOMeanAP",
    "DEFAULT_AREA_RANGES",
]

_EPS = float(np.spacing(1))


@dataclass(frozen=True)
class AreaRange:
    """An inclusive object-area band, in square pixels.

    The COCO bands are ``small`` below 32x32, ``medium`` between 32x32 and
    96x96, and ``large`` above 96x96. The bounds are inclusive at both ends, so
    an object of exactly 1024 px^2 belongs to *both* ``small`` and ``medium``.
    That is the reference behaviour, not an oversight here; the bands are used
    to slice results, not to partition them.
    """

    name: str
    lo: float
    hi: float

    def contains(self, area: float) -> bool:
        """Return True if ``area`` falls inside the band (inclusive)."""
        return self.lo <= area <= self.hi


DEFAULT_AREA_RANGES: Tuple[AreaRange, ...] = (
    AreaRange("all", 0.0, 1e10),
    AreaRange("small", 0.0, 32.0**2),
    AreaRange("medium", 32.0**2, 96.0**2),
    AreaRange("large", 96.0**2, 1e10),
)


@dataclass
class EvalParams:
    """Knobs of the COCO protocol. The defaults are the official settings."""

    iou_thresholds: np.ndarray = field(
        default_factory=lambda: np.linspace(0.5, 0.95, 10)
    )
    recall_thresholds: np.ndarray = field(
        default_factory=lambda: np.linspace(0.0, 1.0, 101)
    )
    area_ranges: Tuple[AreaRange, ...] = DEFAULT_AREA_RANGES
    max_dets: Tuple[int, ...] = (1, 10, 100)

    def area_index(self, name: str) -> int:
        """Return the index of the named area range."""
        for i, rng in enumerate(self.area_ranges):
            if rng.name == name:
                return i
        raise KeyError(f"unknown area range: {name!r}")

    def max_det_index(self, max_dets: int) -> int:
        """Return the index of the given max-detections cap."""
        for i, m in enumerate(self.max_dets):
            if m == max_dets:
                return i
        raise KeyError(f"unknown maxDets value: {max_dets}")

    def iou_index(self, iou: float) -> int:
        """Return the index of the IoU threshold closest to ``iou``."""
        diffs = np.abs(self.iou_thresholds - iou)
        i = int(np.argmin(diffs))
        if diffs[i] > 1e-9:
            raise KeyError(f"IoU threshold {iou} is not in the sweep")
        return i


@dataclass
class GroundTruth:
    """Ground truth in COCO instances format, already parsed."""

    image_ids: List[int]
    category_ids: List[int]
    annotations: List[dict]

    @classmethod
    def from_coco_dict(cls, data: Mapping[str, object]) -> "GroundTruth":
        """Build from a loaded ``instances_*.json`` dictionary."""
        images = list(data["images"])  # type: ignore[index]
        cats = list(data["categories"])  # type: ignore[index]
        anns = list(data["annotations"])  # type: ignore[index]
        return cls(
            image_ids=sorted(int(im["id"]) for im in images),
            category_ids=sorted(int(c["id"]) for c in cats),
            annotations=[dict(a) for a in anns],
        )

    def subset(self, image_ids: Iterable[int]) -> "GroundTruth":
        """Restrict the ground truth to a subset of images."""
        keep = set(int(i) for i in image_ids)
        return GroundTruth(
            image_ids=sorted(keep),
            category_ids=list(self.category_ids),
            annotations=[a for a in self.annotations if int(a["image_id"]) in keep],
        )


@dataclass
class COCOResults:
    """Accumulated precision/recall tensors plus convenience accessors."""

    params: EvalParams
    category_ids: List[int]
    precision: np.ndarray  # (T, R, K, A, M)
    recall: np.ndarray  # (T, K, A, M)
    scores: np.ndarray  # (T, R, K, A, M)
    n_images: int
    n_detections: int
    n_ground_truths: int

    # ---------------------------------------------------------------- lookups
    def _slice_precision(
        self,
        iou: Optional[float],
        area: str,
        max_dets: int,
        class_index: Optional[int] = None,
    ) -> np.ndarray:
        a = self.params.area_index(area)
        m = self.params.max_det_index(max_dets)
        p = self.precision[:, :, :, a, m]
        if iou is not None:
            p = p[self.params.iou_index(iou) : self.params.iou_index(iou) + 1]
        if class_index is not None:
            p = p[:, :, class_index : class_index + 1]
        return p

    def ap(
        self,
        iou: Optional[float] = None,
        area: str = "all",
        max_dets: int = 100,
    ) -> float:
        """Average precision, averaged over classes and (unless given) IoUs.

        Returns ``float('nan')`` when no class had any ground truth, which is
        the honest answer for an empty evaluation. The reference returns -1.
        """
        p = self._slice_precision(iou, area, max_dets)
        valid = p[p > -1]
        return float(np.mean(valid)) if valid.size else float("nan")

    def ar(
        self,
        iou: Optional[float] = None,
        area: str = "all",
        max_dets: int = 100,
    ) -> float:
        """Average recall over classes at the given IoU (or all IoUs)."""
        a = self.params.area_index(area)
        m = self.params.max_det_index(max_dets)
        r = self.recall[:, :, a, m]
        if iou is not None:
            i = self.params.iou_index(iou)
            r = r[i : i + 1]
        valid = r[r > -1]
        return float(np.mean(valid)) if valid.size else float("nan")

    def per_class_ap(
        self,
        iou: Optional[float] = None,
        area: str = "all",
        max_dets: int = 100,
    ) -> Dict[int, float]:
        """AP per COCO ``category_id``. Classes with no ground truth are NaN."""
        out: Dict[int, float] = {}
        for k, cat_id in enumerate(self.category_ids):
            p = self._slice_precision(iou, area, max_dets, class_index=k)
            valid = p[p > -1]
            out[cat_id] = float(np.mean(valid)) if valid.size else float("nan")
        return out

    def pr_curve(
        self,
        category_id: int,
        iou: float = 0.5,
        area: str = "all",
        max_dets: int = 100,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return ``(recall_thresholds, precision)`` for one class.

        This is the interpolated curve the metric actually integrates, sampled
        at the 101 recall levels, not a raw scatter of operating points.
        """
        k = self.category_ids.index(category_id)
        t = self.params.iou_index(iou)
        a = self.params.area_index(area)
        m = self.params.max_det_index(max_dets)
        curve = self.precision[t, :, k, a, m].copy()
        return self.params.recall_thresholds.copy(), curve

    # ---------------------------------------------------------------- summary
    def summary(self) -> Dict[str, float]:
        """The twelve standard COCO numbers, keyed by their usual names."""
        return {
            "mAP": self.ap(),
            "mAP50": self.ap(iou=0.5),
            "mAP75": self.ap(iou=0.75),
            "mAP_small": self.ap(area="small"),
            "mAP_medium": self.ap(area="medium"),
            "mAP_large": self.ap(area="large"),
            "AR_1": self.ar(max_dets=1),
            "AR_10": self.ar(max_dets=10),
            "AR_100": self.ar(max_dets=100),
            "AR_small": self.ar(area="small"),
            "AR_medium": self.ar(area="medium"),
            "AR_large": self.ar(area="large"),
        }

    def format_summary(self) -> str:
        """Render :meth:`summary` as the familiar twelve-line block."""
        s = self.summary()
        rows = [
            ("Average Precision", "(AP)", "0.50:0.95", "all", 100, s["mAP"]),
            ("Average Precision", "(AP)", "0.50", "all", 100, s["mAP50"]),
            ("Average Precision", "(AP)", "0.75", "all", 100, s["mAP75"]),
            ("Average Precision", "(AP)", "0.50:0.95", "small", 100, s["mAP_small"]),
            ("Average Precision", "(AP)", "0.50:0.95", "medium", 100, s["mAP_medium"]),
            ("Average Precision", "(AP)", "0.50:0.95", "large", 100, s["mAP_large"]),
            ("Average Recall", "(AR)", "0.50:0.95", "all", 1, s["AR_1"]),
            ("Average Recall", "(AR)", "0.50:0.95", "all", 10, s["AR_10"]),
            ("Average Recall", "(AR)", "0.50:0.95", "all", 100, s["AR_100"]),
            ("Average Recall", "(AR)", "0.50:0.95", "small", 100, s["AR_small"]),
            ("Average Recall", "(AR)", "0.50:0.95", "medium", 100, s["AR_medium"]),
            ("Average Recall", "(AR)", "0.50:0.95", "large", 100, s["AR_large"]),
        ]
        lines = []
        for name, abbr, iou, area, mdet, val in rows:
            lines.append(
                f" {name:<18}{abbr} @[ IoU={iou:<9} | area={area:>6} | "
                f"maxDets={mdet:>3} ] = {val:.4f}"
            )
        return "\n".join(lines)

    def top_classes(
        self, n: int = 10, best: bool = True, area: str = "all"
    ) -> List[Tuple[int, str, float]]:
        """Return the ``n`` best (or worst) classes as ``(id, name, AP)``."""
        items = [
            (cid, CATEGORY_NAMES.get(cid, str(cid)), ap)
            for cid, ap in self.per_class_ap(area=area).items()
            if not np.isnan(ap)
        ]
        items.sort(key=lambda t: t[2], reverse=best)
        return items[:n]


class COCOMeanAP:
    """Evaluator: hold the ground truth, score any number of detection sets.

    Ground-truth indexing and the per-image ignore logic are the expensive
    part, so the object is reusable across quantisation variants.
    """

    def __init__(
        self,
        ground_truth: GroundTruth,
        params: Optional[EvalParams] = None,
    ) -> None:
        self.gt = ground_truth
        self.params = params or EvalParams()
        self._gt_index: Dict[Tuple[int, int], List[dict]] = {}
        self._index_ground_truth()

    # ------------------------------------------------------------- internals
    def _index_ground_truth(self) -> None:
        for ann in self.gt.annotations:
            ann = dict(ann)
            bbox = [float(v) for v in ann["bbox"]]
            ann["bbox"] = bbox
            ann["iscrowd"] = int(ann.get("iscrowd", 0))
            if "area" not in ann or ann["area"] is None:
                ann["area"] = bbox[2] * bbox[3]
            ann["area"] = float(ann["area"])
            # The reference overwrites any user 'ignore' with the crowd flag.
            ann["_ignore_base"] = int(ann["iscrowd"])
            key = (int(ann["image_id"]), int(ann["category_id"]))
            self._gt_index.setdefault(key, []).append(ann)

    @staticmethod
    def _index_detections(
        detections: Sequence[Mapping[str, object]],
    ) -> Dict[Tuple[int, int], List[dict]]:
        index: Dict[Tuple[int, int], List[dict]] = {}
        for i, det in enumerate(detections):
            bbox = [float(v) for v in det["bbox"]]  # type: ignore[index]
            rec = {
                "id": i + 1,
                "image_id": int(det["image_id"]),  # type: ignore[index]
                "category_id": int(det["category_id"]),  # type: ignore[index]
                "bbox": bbox,
                "score": float(det["score"]),  # type: ignore[index]
                "area": bbox[2] * bbox[3],
            }
            index.setdefault((rec["image_id"], rec["category_id"]), []).append(rec)
        for dets in index.values():
            dets.sort(key=lambda d: -d["score"])
        return index

    def _evaluate_image(
        self,
        gts: List[dict],
        dets: List[dict],
        ious: np.ndarray,
        area_range: AreaRange,
        max_det: int,
    ) -> Optional[dict]:
        """Match one (image, class, area range, maxDets) cell.

        Returns None when the cell is empty, matching the reference so that
        empty cells contribute nothing rather than a spurious zero.
        """
        if not gts and not dets:
            return None

        n_thr = len(self.params.iou_thresholds)

        # Ignore crowd regions and anything outside the area band.
        gt_ignore = np.array(
            [
                1
                if (g["_ignore_base"] or not area_range.contains(g["area"]))
                else 0
                for g in gts
            ],
            dtype=np.int64,
        )
        # Stable sort so non-ignored ground truths come first; greedy matching
        # relies on this to prefer a real ground truth over an ignored one.
        order = np.argsort(gt_ignore, kind="mergesort")
        gts = [gts[i] for i in order]
        gt_ignore = gt_ignore[order]
        crowd = np.array([g["iscrowd"] for g in gts], dtype=bool)

        dets = dets[:max_det]
        n_det, n_gt = len(dets), len(gts)

        if ious.size:
            ious = ious[: len(dets)][:, order]

        gt_matched = np.zeros((n_thr, n_gt), dtype=np.int64)
        dt_matched = np.zeros((n_thr, n_det), dtype=np.int64)
        dt_ignore = np.zeros((n_thr, n_det), dtype=np.int64)

        if ious.size:
            for t_i, thr in enumerate(self.params.iou_thresholds):
                for d_i in range(n_det):
                    # Never accept an overlap of exactly 1.0 as "just below".
                    best_iou = min(thr, 1 - 1e-10)
                    best_g = -1
                    for g_i in range(n_gt):
                        if gt_matched[t_i, g_i] > 0 and not crowd[g_i]:
                            continue
                        # Already matched a real ground truth; the sort means
                        # everything from here on is ignored, so stop.
                        if (
                            best_g > -1
                            and gt_ignore[best_g] == 0
                            and gt_ignore[g_i] == 1
                        ):
                            break
                        if ious[d_i, g_i] < best_iou:
                            continue
                        best_iou = float(ious[d_i, g_i])
                        best_g = g_i
                    if best_g == -1:
                        continue
                    dt_ignore[t_i, d_i] = gt_ignore[best_g]
                    dt_matched[t_i, d_i] = gts[best_g]["id"]
                    gt_matched[t_i, best_g] = dets[d_i]["id"]

        # Unmatched detections outside the area band are ignored too, so a
        # small-object evaluation is not polluted by correct large detections.
        if n_det:
            outside = np.array(
                [not area_range.contains(d["area"]) for d in dets], dtype=bool
            ).reshape(1, n_det)
            dt_ignore = np.logical_or(
                dt_ignore, np.logical_and(dt_matched == 0, np.repeat(outside, n_thr, 0))
            ).astype(np.int64)

        return {
            "dt_scores": np.array([d["score"] for d in dets], dtype=np.float64),
            "dt_matched": dt_matched,
            "dt_ignore": dt_ignore,
            "gt_ignore": gt_ignore,
        }

    # ---------------------------------------------------------------- public
    def evaluate(
        self,
        detections: Sequence[Mapping[str, object]],
        image_ids: Optional[Sequence[int]] = None,
    ) -> COCOResults:
        """Score a set of detections against the ground truth.

        Args:
            detections: COCO-format results: dicts with ``image_id``,
                ``category_id``, ``bbox`` (``[x, y, w, h]``) and ``score``.
            image_ids: Restrict evaluation to these images. Defaults to every
                image in the ground truth, so images with no detections
                correctly count as missed recall.

        Returns:
            A :class:`COCOResults` holding the full precision/recall tensors.
        """
        params = self.params
        img_ids = (
            sorted(set(int(i) for i in image_ids))
            if image_ids is not None
            else list(self.gt.image_ids)
        )
        img_set = set(img_ids)
        cat_ids = list(self.gt.category_ids)

        in_scope = [
            d for d in detections
            if int(d["image_id"]) in img_set  # type: ignore[index]
        ]
        dt_index = self._index_detections(in_scope)
        max_det_cap = max(params.max_dets)

        n_thr = len(params.iou_thresholds)
        n_rec = len(params.recall_thresholds)
        n_cat = len(cat_ids)
        n_area = len(params.area_ranges)
        n_maxdet = len(params.max_dets)

        precision = -np.ones((n_thr, n_rec, n_cat, n_area, n_maxdet))
        recall = -np.ones((n_thr, n_cat, n_area, n_maxdet))
        scores = -np.ones((n_thr, n_rec, n_cat, n_area, n_maxdet))

        n_dets_total = 0
        n_gts_total = 0

        for k, cat_id in enumerate(cat_ids):
            # Cache the IoU matrices for this class once, reuse them across
            # every area range and maxDets cap.
            per_image: List[Tuple[List[dict], List[dict], np.ndarray]] = []
            for img_id in img_ids:
                gts = self._gt_index.get((img_id, cat_id), [])
                dets = dt_index.get((img_id, cat_id), [])[:max_det_cap]
                if not gts and not dets:
                    continue
                n_dets_total += len(dets)
                n_gts_total += len(gts)
                if gts and dets:
                    ious = iou_matrix(
                        np.array([d["bbox"] for d in dets]),
                        np.array([g["bbox"] for g in gts]),
                        np.array([g["iscrowd"] for g in gts]),
                    )
                else:
                    ious = np.zeros((len(dets), len(gts)))
                per_image.append((gts, dets, ious))

            for a, area_range in enumerate(params.area_ranges):
                for m, max_det in enumerate(params.max_dets):
                    cells = [
                        self._evaluate_image(g, d, i, area_range, max_det)
                        for (g, d, i) in per_image
                    ]
                    cells = [c for c in cells if c is not None]
                    if not cells:
                        continue
                    self._accumulate_cell(
                        cells, precision, recall, scores, k, a, m
                    )

        return COCOResults(
            params=copy.deepcopy(params),
            category_ids=cat_ids,
            precision=precision,
            recall=recall,
            scores=scores,
            n_images=len(img_ids),
            n_detections=n_dets_total,
            n_ground_truths=n_gts_total,
        )

    def _accumulate_cell(
        self,
        cells: List[dict],
        precision: np.ndarray,
        recall: np.ndarray,
        scores: np.ndarray,
        k: int,
        a: int,
        m: int,
    ) -> None:
        """Turn per-image matches into an interpolated PR curve for one class."""
        params = self.params
        rec_thrs = params.recall_thresholds
        n_thr = len(params.iou_thresholds)
        n_rec = len(rec_thrs)

        gt_ignore = np.concatenate([c["gt_ignore"] for c in cells])
        n_pos = int(np.count_nonzero(gt_ignore == 0))
        if n_pos == 0:
            return  # class absent from this slice; leave the -1 sentinel

        dt_scores = np.concatenate([c["dt_scores"] for c in cells])
        # Stable sort keeps the reference's tie-breaking behaviour.
        order = np.argsort(-dt_scores, kind="mergesort")
        dt_scores = dt_scores[order]

        dt_matched = np.concatenate([c["dt_matched"] for c in cells], axis=1)[:, order]
        dt_ignore = np.concatenate([c["dt_ignore"] for c in cells], axis=1)[:, order]

        keep = np.logical_not(dt_ignore)
        tps = np.logical_and(dt_matched, keep)
        fps = np.logical_and(np.logical_not(dt_matched), keep)

        tp_cum = np.cumsum(tps, axis=1).astype(np.float64)
        fp_cum = np.cumsum(fps, axis=1).astype(np.float64)

        for t in range(n_thr):
            tp, fp = tp_cum[t], fp_cum[t]
            n_det = len(tp)
            rc = tp / n_pos
            pr = tp / (fp + tp + _EPS)

            recall[t, k, a, m] = rc[-1] if n_det else 0.0

            q = np.zeros(n_rec)
            ss = np.zeros(n_rec)
            pr = pr.tolist()
            # Make precision monotonically non-increasing from the right, so
            # the curve reports the best precision achievable at that recall
            # or beyond. Without this, PR curves are saw-toothed and AP
            # depends on noise in the tail.
            for i in range(n_det - 1, 0, -1):
                if pr[i] > pr[i - 1]:
                    pr[i - 1] = pr[i]

            # Sample at the 101 fixed recall levels. searchsorted on the
            # already-sorted cumulative recall gives the first detection index
            # reaching each level; levels never reached keep precision 0.
            inds = np.searchsorted(rc, rec_thrs, side="left")
            for ri, pi in enumerate(inds):
                if pi >= n_det:
                    break
                q[ri] = pr[pi]
                ss[ri] = dt_scores[pi]

            precision[t, :, k, a, m] = q
            scores[t, :, k, a, m] = ss
