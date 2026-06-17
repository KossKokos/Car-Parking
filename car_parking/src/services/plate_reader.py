from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

from car_parking.src.services.anpr_service import (
    get_anpr_pipeline,
    read_license_plate,
    read_license_plate_report,
)


MIN_OVERALL_CONFIDENCE = 0.95
MIN_POSITION_CONFIDENCE = 0.80

logger = logging.getLogger(__name__)


class PlatesReader:
    async def get_prediction(self, img: np.ndarray) -> str | None:
        """Return a cropped-image plate only when confidence thresholds pass."""
        try:
            result = get_anpr_pipeline().recogniser_pipeline.recognizer.predict_array_report(
                image=img,
                min_overall_confidence=MIN_OVERALL_CONFIDENCE,
                min_position_confidence=MIN_POSITION_CONFIDENCE,
            )
        except Exception:
            return None

        if not result.should_accept:
            return None

        return result.plate

    async def get_prediction_report(self, img: np.ndarray) -> dict[str, Any]:
        """Return recognition details for a cropped plate image."""
        try:
            result = get_anpr_pipeline().recogniser_pipeline.recognizer.predict_array_report(
                image=img,
                min_overall_confidence=MIN_OVERALL_CONFIDENCE,
                min_position_confidence=MIN_POSITION_CONFIDENCE,
            )
        except Exception as exc:
            return {
                "plate": None,
                "should_accept": False,
                "error": str(exc),
            }

        return result.to_dict()

    async def get_prediction_from_cropped_image_path(
        self,
        image_path: str | Path,
    ) -> dict[str, Any]:
        """Run recognition on an already cropped plate image path."""
        result = get_anpr_pipeline().recogniser_pipeline.predict_cropped_image(
            image_path
        )
        return result.to_dict()

    async def get_prediction_from_car_image_path(
        self,
        image_path: str | Path,
    ) -> str | None:
        """Detect and read a plate from a full car image path."""
        try:
            return read_license_plate(image_path)
        except Exception as exc:
            logger.warning("Plate detection/recognition failed: %s", exc)
            return None

    async def get_prediction_report_from_car_image_path(
        self,
        image_path: str | Path,
    ) -> dict[str, Any]:
        """Return detection and recognition details for a full car image path."""
        try:
            return read_license_plate_report(image_path)
        except Exception as exc:
            return {
                "plate": None,
                "confidence": None,
                "valid_format": False,
                "detection": None,
                "error": str(exc),
            }


pr = PlatesReader()


def read_plate_from_cropped_image(image_path: str | Path) -> dict[str, Any]:
    """Synchronous helper for recognizing a plate from a cropped image path."""
    result = get_anpr_pipeline().recogniser_pipeline.predict_cropped_image(image_path)
    return result.to_dict()
