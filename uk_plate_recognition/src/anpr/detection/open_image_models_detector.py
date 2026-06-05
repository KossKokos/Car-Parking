from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
from open_image_models import LicensePlateDetector


@dataclass(frozen=True)
class PlateBox:
    """Normalised plate-detection box used by our ANPR pipeline."""

    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float
    label: str = "License Plate"

    @property
    def width(self) -> int:
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        return self.y2 - self.y1

    @property
    def area(self) -> int:
        return self.width * self.height

    def as_xyxy(self) -> tuple[int, int, int, int]:
        return self.x1, self.y1, self.x2, self.y2


@dataclass(frozen=True)
class PlateDetectionOutput:
    """Detector output containing the original image and normalised detections."""

    image: np.ndarray
    detections: list[PlateBox]

    @property
    def best_detection(self) -> PlateBox | None:
        if not self.detections:
            return None
        return max(self.detections, key=lambda d: d.confidence)


class OpenImageModelsPlateDetector:
    """
    Local license-plate detector using open_image_models.

    """

    def __init__(
        self,
        detection_model: str = "yolo-v9-t-256-license-plate-end2end",
        min_confidence: float = 0.25,
        providers: Sequence[str | tuple[str, dict]] | None = None,
    ) -> None:
        self.detection_model = detection_model
        self.min_confidence = min_confidence

        self.detector = LicensePlateDetector(
            detection_model=detection_model,
            conf_thresh=min_confidence,
            providers=providers,
        )

    def predict(self, image_or_path: str | Path | np.ndarray) -> PlateDetectionOutput:
        image = self._load_image(image_or_path)

        raw_detections = self.detector.predict(image)

        detections: list[PlateBox] = []

        for raw_detection in raw_detections:
            box = raw_detection.bounding_box

            plate_box = PlateBox(
                x1=int(box.x1),
                y1=int(box.y1),
                x2=int(box.x2),
                y2=int(box.y2),
                confidence=float(raw_detection.confidence),
                label=str(raw_detection.label),
            )

            plate_box = self._clamp_box(
                plate_box,
                image_width=image.shape[1],
                image_height=image.shape[0],
            )

            if plate_box.width <= 0 or plate_box.height <= 0:
                continue

            if plate_box.confidence < self.min_confidence:
                continue

            detections.append(plate_box)

        detections.sort(key=lambda d: d.confidence, reverse=True)

        return PlateDetectionOutput(
            image=image,
            detections=detections,
        )

    def crop_best_plate(
        self,
        image_or_path: str | Path | np.ndarray,
        padding_ratio: float = 0.05,
    ) -> tuple[np.ndarray, PlateBox]:
        output = self.predict(image_or_path)
        best_detection = output.best_detection

        if best_detection is None:
            raise ValueError("No license plate detected in image.")

        crop = crop_plate_with_padding(
            image=output.image,
            box=best_detection,
            padding_ratio=padding_ratio,
        )

        return crop, best_detection

    @staticmethod
    def _load_image(image_or_path: str | Path | np.ndarray) -> np.ndarray:
        if isinstance(image_or_path, np.ndarray):
            image = image_or_path

            if image.size == 0:
                raise ValueError("Input image array is empty.")

            return image

        image_path = Path(image_or_path)

        if not image_path.exists():
            raise FileNotFoundError(f"Image path does not exist: {image_path}")

        image = cv2.imread(str(image_path))

        if image is None:
            raise ValueError(f"Could not read image: {image_path}")

        return image

    @staticmethod
    def _clamp_box(
        box: PlateBox,
        image_width: int,
        image_height: int,
    ) -> PlateBox:
        return PlateBox(
            x1=max(0, min(box.x1, image_width)),
            y1=max(0, min(box.y1, image_height)),
            x2=max(0, min(box.x2, image_width)),
            y2=max(0, min(box.y2, image_height)),
            confidence=box.confidence,
            label=box.label,
        )


def crop_plate_with_padding(
    image: np.ndarray,
    box: PlateBox,
    padding_ratio: float = 0.05,
) -> np.ndarray:
    """
    Crop plate from original image with optional padding.

    padding_ratio=0.05 means 5% of box width/height added around the box.
    """

    image_height, image_width = image.shape[:2]

    pad_x = int(box.width * padding_ratio)
    pad_y = int(box.height * padding_ratio)

    x1 = max(0, box.x1 - pad_x)
    y1 = max(0, box.y1 - pad_y)
    x2 = min(image_width, box.x2 + pad_x)
    y2 = min(image_height, box.y2 + pad_y)

    crop = image[y1:y2, x1:x2]

    if crop.size == 0:
        raise ValueError(
            f"Empty crop created from box: {(box.x1, box.y1, box.x2, box.y2)}"
        )

    return crop