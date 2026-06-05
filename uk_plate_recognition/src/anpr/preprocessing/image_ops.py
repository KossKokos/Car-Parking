from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True)
class BoundingBoxXYXY:
    """
    Bounding box in absolute pixel coordinates.

    x1, y1 = top-left
    x2, y2 = bottom-right
    """

    x1: float
    y1: float
    x2: float
    y2: float


def load_image_bgr(image_path: str | Path) -> np.ndarray:
    """
    Load an image from disk using OpenCV BGR format.
    """
    image_path = Path(image_path)

    if not image_path.exists():
        raise FileNotFoundError(f"Image does not exist: {image_path}")

    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)

    if image is None:
        raise ValueError(f"Could not read image: {image_path}")

    return image


def add_padding_to_box(
    box: BoundingBoxXYXY,
    padding: float,
) -> BoundingBoxXYXY:
    """
    Add relative padding around a bounding box.

    padding=0.05 means 5% of box width/height on each side.
    """
    if padding < 0:
        raise ValueError("padding must be >= 0.")

    box_width = box.x2 - box.x1
    box_height = box.y2 - box.y1

    if box_width <= 0 or box_height <= 0:
        raise ValueError(f"Invalid box dimensions: {box}.")

    pad_x = box_width * padding
    pad_y = box_height * padding

    return BoundingBoxXYXY(
        x1=box.x1 - pad_x,
        y1=box.y1 - pad_y,
        x2=box.x2 + pad_x,
        y2=box.y2 + pad_y,
    )


def clip_box_to_image(
    box: BoundingBoxXYXY,
    image_shape: tuple[int, ...],
) -> tuple[int, int, int, int]:
    """
    Clip xyxy box coordinates to image bounds.

    Returns:
        x1, y1, x2, y2 as integer pixel coordinates.
    """
    if len(image_shape) < 2:
        raise ValueError(f"Invalid image shape: {image_shape}.")

    image_height, image_width = image_shape[:2]

    x1 = int(np.floor(box.x1))
    y1 = int(np.floor(box.y1))
    x2 = int(np.ceil(box.x2))
    y2 = int(np.ceil(box.y2))

    x1 = max(0, min(x1, image_width))
    y1 = max(0, min(y1, image_height))
    x2 = max(0, min(x2, image_width))
    y2 = max(0, min(y2, image_height))

    if x2 <= x1 or y2 <= y1:
        raise ValueError(
            f"Invalid clipped crop box: {(x1, y1, x2, y2)} "
            f"for image shape {image_shape}."
        )

    return x1, y1, x2, y2


def crop_image_xyxy(
    image: np.ndarray,
    box: BoundingBoxXYXY,
    padding: float = 0.05,
) -> np.ndarray:
    """
    Crop image using xyxy bounding box with optional padding.
    """
    if image is None:
        raise ValueError("image cannot be None.")

    if image.ndim not in (2, 3):
        raise ValueError(f"Expected 2D or 3D image, got shape {image.shape}.")

    padded_box = add_padding_to_box(box, padding=padding)
    x1, y1, x2, y2 = clip_box_to_image(padded_box, image.shape)

    crop = image[y1:y2, x1:x2]

    if crop.size == 0:
        raise ValueError("Crop is empty.")

    return crop.copy()
