"""COCO-80 class metadata.

YOLO detection heads emit 80 contiguous class indices. COCO annotations use 80
non-contiguous ``category_id`` values in the range 1..90 (eleven ids were
retired between the 2014 and 2017 releases). Submitting a detection file with
contiguous indices instead of real category ids silently scores near zero, so
the mapping lives in one place and is verified against the ground truth file
whenever one is loaded.
"""

from __future__ import annotations

from typing import Dict, List, Sequence

#: COCO ``category_id`` for each contiguous YOLO class index 0..79.
COCO80_TO_COCO91: List[int] = [
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
    11, 13, 14, 15, 16, 17, 18, 19, 20, 21,
    22, 23, 24, 25, 27, 28, 31, 32, 33, 34,
    35, 36, 37, 38, 39, 40, 41, 42, 43, 44,
    46, 47, 48, 49, 50, 51, 52, 53, 54, 55,
    56, 57, 58, 59, 60, 61, 62, 63, 64, 65,
    67, 70, 72, 73, 74, 75, 76, 77, 78, 79,
    80, 81, 82, 84, 85, 86, 87, 88, 89, 90,
]

#: Class name for each contiguous YOLO class index 0..79.
COCO80_NAMES: List[str] = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag",
    "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon",
    "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot",
    "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant",
    "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote",
    "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush",
]

#: Reverse mapping: COCO ``category_id`` -> contiguous index 0..79.
COCO91_TO_COCO80: Dict[int, int] = {
    cat_id: idx for idx, cat_id in enumerate(COCO80_TO_COCO91)
}

#: COCO ``category_id`` -> class name.
CATEGORY_NAMES: Dict[int, str] = {
    cat_id: COCO80_NAMES[idx] for idx, cat_id in enumerate(COCO80_TO_COCO91)
}


def category_id_for_index(class_index: int) -> int:
    """Map a contiguous YOLO class index to its COCO ``category_id``."""
    if not 0 <= class_index < len(COCO80_TO_COCO91):
        raise ValueError(f"class index out of range: {class_index}")
    return COCO80_TO_COCO91[class_index]


def verify_against_categories(categories: Sequence[dict]) -> None:
    """Raise if a ground-truth ``categories`` block disagrees with the table.

    Args:
        categories: The ``categories`` list from a COCO instances JSON file.

    Raises:
        ValueError: If the ids or names do not match :data:`COCO80_TO_COCO91`.
    """
    ordered = sorted(categories, key=lambda c: int(c["id"]))
    ids = [int(c["id"]) for c in ordered]
    names = [str(c["name"]) for c in ordered]
    if ids != COCO80_TO_COCO91:
        raise ValueError("ground-truth category ids do not match the COCO-80 table")
    if names != COCO80_NAMES:
        raise ValueError("ground-truth category names do not match the COCO-80 table")
