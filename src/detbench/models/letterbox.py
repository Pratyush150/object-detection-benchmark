"""Letterbox resize and the inverse mapping back to original pixels.

A detector trained at a fixed square input cannot simply be fed a stretched
image: distorting the aspect ratio moves every box and costs accuracy. The
standard fix is to scale by a single factor and pad the short side. The part
that is easy to get wrong is the inverse: predictions come back in padded
network coordinates and have to be un-padded and un-scaled before they can be
compared against ground truth. An off-by-one in the padding shows up as a
small, uniform mAP loss that is very hard to attribute later, which is why the
round trip is covered by a test.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

__all__ = ["LetterboxTransform", "letterbox"]


@dataclass(frozen=True)
class LetterboxTransform:
    """The scale and padding applied to one image, plus its inverse.

    Attributes:
        scale: Single factor applied to both axes.
        pad_x: Pixels of padding added on the left.
        pad_y: Pixels of padding added on the top.
        orig_w: Width of the source image.
        orig_h: Height of the source image.
        net_w: Width of the network input.
        net_h: Height of the network input.
    """

    scale: float
    pad_x: float
    pad_y: float
    orig_w: int
    orig_h: int
    net_w: int
    net_h: int

    def forward_xyxy(self, boxes: np.ndarray) -> np.ndarray:
        """Map original-image ``[x1, y1, x2, y2]`` boxes into network space."""
        boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
        out = boxes * self.scale
        out[:, [0, 2]] += self.pad_x
        out[:, [1, 3]] += self.pad_y
        return out

    def inverse_xyxy(self, boxes: np.ndarray, clip: bool = True) -> np.ndarray:
        """Map network-space ``[x1, y1, x2, y2]`` boxes back to the original.

        Args:
            boxes: ``(N, 4)`` boxes in padded network coordinates.
            clip: Clamp the result to the original image bounds. COCO ground
                truth is clipped to the image, so leaving boxes hanging off the
                edge only ever loses IoU.
        """
        boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
        out = boxes.copy()
        out[:, [0, 2]] -= self.pad_x
        out[:, [1, 3]] -= self.pad_y
        out /= self.scale
        if clip:
            out[:, [0, 2]] = np.clip(out[:, [0, 2]], 0.0, float(self.orig_w))
            out[:, [1, 3]] = np.clip(out[:, [1, 3]], 0.0, float(self.orig_h))
        return out


def letterbox(
    image: np.ndarray,
    net_size: Tuple[int, int] = (640, 640),
    color: Tuple[int, int, int] = (114, 114, 114),
    scale_up: bool = True,
) -> Tuple[np.ndarray, LetterboxTransform]:
    """Resize preserving aspect ratio and pad to ``net_size``.

    Args:
        image: ``(H, W, 3)`` uint8 image.
        net_size: ``(width, height)`` of the network input.
        color: Padding colour. 114 grey is the YOLO convention; it matters
            because the model saw that value during training.
        scale_up: If False, never enlarge an image smaller than the input.

    Returns:
        The padded image and the :class:`LetterboxTransform` describing it.
    """
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"expected an (H, W, 3) image, got shape {image.shape}")
    orig_h, orig_w = int(image.shape[0]), int(image.shape[1])
    net_w, net_h = int(net_size[0]), int(net_size[1])

    scale = min(net_w / orig_w, net_h / orig_h)
    if not scale_up:
        scale = min(scale, 1.0)

    new_w = int(round(orig_w * scale))
    new_h = int(round(orig_h * scale))
    pad_x = (net_w - new_w) / 2.0
    pad_y = (net_h - new_h) / 2.0

    canvas = np.full((net_h, net_w, 3), color, dtype=image.dtype)
    resized = _resize(image, new_w, new_h)
    top, left = int(round(pad_y - 0.1)), int(round(pad_x - 0.1))
    canvas[top : top + new_h, left : left + new_w] = resized

    transform = LetterboxTransform(
        scale=scale,
        pad_x=float(left),
        pad_y=float(top),
        orig_w=orig_w,
        orig_h=orig_h,
        net_w=net_w,
        net_h=net_h,
    )
    return canvas, transform


def _resize(image: np.ndarray, width: int, height: int) -> np.ndarray:
    """Resize with OpenCV when available, else a nearest-neighbour fallback."""
    if image.shape[1] == width and image.shape[0] == height:
        return image
    try:
        import cv2  # noqa: PLC0415  (optional dependency)
    except ImportError:
        ys = (np.arange(height) * (image.shape[0] / height)).astype(np.int64)
        xs = (np.arange(width) * (image.shape[1] / width)).astype(np.int64)
        return image[ys][:, xs]
    interp = cv2.INTER_AREA if width < image.shape[1] else cv2.INTER_LINEAR
    return cv2.resize(image, (width, height), interpolation=interp)
