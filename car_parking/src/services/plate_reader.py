from __future__ import annotations

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


class PlatesReader:
    async def get_prediction(self, img: np.ndarray) -> str | None:
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
        result = get_anpr_pipeline().recogniser_pipeline.predict_cropped_image(
            image_path
        )
        return result.to_dict()

    async def get_prediction_from_car_image_path(
        self,
        image_path: str | Path,
    ) -> str | None:
        try:
            return read_license_plate(image_path)
        except Exception as exc:
            print(f"Plate detection/recognition failed: {exc}")
            return None

    async def get_prediction_report_from_car_image_path(
        self,
        image_path: str | Path,
    ) -> dict[str, Any]:
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
    result = get_anpr_pipeline().recogniser_pipeline.predict_cropped_image(image_path)
    return result.to_dict()
