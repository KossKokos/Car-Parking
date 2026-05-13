from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

import cv2
from pathlib import Path
from typing import Any

##############################################################################################################################
from anpr.inference.pipeline import (
    extract_roboflow_predictions,
    select_best_roboflow_prediction,
)
from anpr.preprocessing.image_ops import (
    crop_image_from_roboflow_prediction,
    load_image_bgr,
)
from car_parking.src.services.roboflow_service import get_roboflow_plate_detector
##############################################################################################################################

from car_parking.src.services.anpr_service import ANPRService
from car_parking.src.services.roboflow_service import get_roboflow_plate_detector


PROJECT_ROOT = Path(__file__).resolve().parents[3]

DEBUG_ANPR_CROP_DIR = PROJECT_ROOT / "debug" / "anpr_crops"

ALLOWED_PLATE_CLASSES = {
    "number-plates",
    "number_plate",
    "number_plates",
    "license_plate",
    "license-plate",
    "plate",
}


_anpr_service: ANPRService | None = None


def get_anpr_service() -> ANPRService:
    """
    Lazily load the ANPR service once and reuse it.

    This prevents loading the PyTorch checkpoint on every request.
    """
    global _anpr_service

    if _anpr_service is None:
        _anpr_service = ANPRService(device="cpu")

    return _anpr_service


class PlatesReader:
    """
    Compatibility wrapper replacing the old TensorFlow PlatesReader.

    Existing route code can keep using:
        plate = await pr.get_prediction(img)

    Important:
        img should be a cropped plate image or a good plate crop.
        For full car images, use Roboflow detection flow.
    """

    async def get_prediction(self, img: np.ndarray) -> str | None:
        """
        Preserve old contract:
            input: OpenCV image array
            output: plate string or None
        """
        try:
            result = get_anpr_service().predict_cropped_plate_array(img)
        except Exception:
            return None

        if not result.should_accept:
            return None

        return result.plate

    async def get_prediction_report(self, img: np.ndarray) -> dict[str, Any] | None:
        """
        New richer interface for debugging/API response if needed.
        """
        try:
            result = get_anpr_service().predict_cropped_plate_array(img)
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
        result = get_anpr_service().predict_cropped_plate_path(image_path)
        return result.to_dict()

    async def get_prediction_from_roboflow_response(
        self,
        image_path: str | Path,
        roboflow_response: dict[str, Any] | list[dict[str, Any]],
    ) -> dict[str, Any]:
        result = get_anpr_service().predict_from_roboflow_response(
            image_path=image_path,
            roboflow_response=roboflow_response,
            crop_padding=0.10,
        )
        return result.to_dict()
    

    async def get_prediction_from_car_image_path(
        self,
        image_path: str | Path,
    ) -> str | None:
        """
        Full car image path -> Roboflow plate detection -> ANPR recognition.

        Returns:
            plate string or None
        """
        try:
            report = await self.get_multi_padding_report_from_car_image_path(
                image_path=image_path,
                save_debug_crops=True,
            )

            best = report["best"]

        except Exception as exc:
            print(f"Plate detection/recognition failed: {exc}")
            return None

        if not best["should_accept"]:
            print(f"Plate rejected: {best}")
            return None

        return best["plate"]

    async def get_prediction_report_from_car_image_path(
        self,
        image_path: str | Path,
    ) -> dict[str, Any]:
        """
        Full car image path -> Roboflow -> ANPR -> full debug result.
        """
        try:
            return await self.get_multi_padding_report_from_car_image_path(
                image_path=image_path,
                save_debug_crops=True,
            )

        except Exception as exc:
            return {
                "plate": None,
                "should_accept": False,
                "error": str(exc),
            }
        
    async def get_multi_padding_report_from_car_image_path(
        self,
        image_path: str | Path,
        paddings: tuple[float, ...] = (0.0, 0.03, 0.05, 0.10, 0.15),
        save_debug_crops: bool = True,
    ) -> dict[str, Any]:
        """
        Full car image -> Roboflow detection -> try multiple crop paddings
        -> recognise each crop -> return best report.

        This is for debugging and improving real-world robustness.
        """
        image_path = Path(image_path)

        roboflow_response = await get_roboflow_plate_detector().detect_from_image_path(
            image_path
        )

        predictions = extract_roboflow_predictions(roboflow_response)

        selected_detection = select_best_roboflow_prediction(
            predictions=predictions,
            min_detection_confidence=0.50,
            allowed_classes=ALLOWED_PLATE_CLASSES,
        )

        full_image = load_image_bgr(image_path)

        attempts: list[dict[str, Any]] = []

        if save_debug_crops:
            DEBUG_ANPR_CROP_DIR.mkdir(parents=True, exist_ok=True)

        for padding in paddings:
            crop = crop_image_from_roboflow_prediction(
                image=full_image,
                prediction=selected_detection,
                padding=padding,
            )

            recognition_result = get_anpr_service().predict_cropped_plate_array(crop)

            crop_path = None

            if save_debug_crops:
                crop_filename = (
                    f"{image_path.stem}_pad_{str(padding).replace('.', '_')}.jpg"
                )
                crop_path = DEBUG_ANPR_CROP_DIR / crop_filename
                cv2.imwrite(str(crop_path), crop)

            attempt = recognition_result.to_dict()
            attempt["padding"] = padding
            attempt["crop_shape"] = tuple(int(value) for value in crop.shape)
            attempt["debug_crop_path"] = str(crop_path) if crop_path else None

            attempts.append(attempt)

        accepted_attempts = [
            attempt for attempt in attempts
            if attempt["should_accept"] is True
        ]

        if accepted_attempts:
            best_attempt = max(
                accepted_attempts,
                key=lambda item: item["overall_confidence"],
            )
        else:
            best_attempt = max(
                attempts,
                key=lambda item: item["overall_confidence"],
            )

        return {
            "best": best_attempt,
            "attempts": attempts,
            "selected_detection": selected_detection,
            "roboflow_response": roboflow_response,
        }


pr = PlatesReader()


def read_plate_from_cropped_image(image_path: str | Path) -> dict[str, Any]:
    """
    Simple sync helper for API/service code.
    """
    result = get_anpr_service().predict_cropped_plate_path(image_path)
    return result.to_dict()


def read_plate_from_roboflow_response(
    image_path: str | Path,
    roboflow_response: dict[str, Any] | list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Simple sync helper for full image + detector response.
    """
    result = get_anpr_service().predict_from_roboflow_response(
        image_path=image_path,
        roboflow_response=roboflow_response,
        crop_padding=0.10,
    )
    return result.to_dict()