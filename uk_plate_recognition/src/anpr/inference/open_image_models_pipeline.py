from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import tempfile
import cv2

from anpr.detection.open_image_models_detector import (
    OpenImageModelsPlateDetector,
    PlateBox,
)
from anpr.inference.pipeline import ANPRPipeline


class OpenImageModelsANPRPipeline:
    """
    Full ANPR runtime pipeline:

    full car image
    -> local YOLOv9 plate detector
    -> plate crop
    -> existing custom CNN recogniser

    This is the full-image inference path used by the application.
    """

    def __init__(
        self,
        recogniser_pipeline: ANPRPipeline,
        detector: OpenImageModelsPlateDetector,
        crop_padding_ratio: float = 0.05,
    ) -> None:
        self.recogniser_pipeline = recogniser_pipeline
        self.detector = detector
        self.crop_padding_ratio = crop_padding_ratio

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        detection_model: str = "yolo-v9-t-256-license-plate-end2end",
        detector_min_confidence: float = 0.25,
        crop_padding_ratio: float = 0.05,
        **recogniser_kwargs: Any,
    ) -> "OpenImageModelsANPRPipeline":
        """
        Build full local ANPR pipeline from your existing recogniser checkpoint.

        Any extra kwargs are passed to ANPRPipeline.from_checkpoint(...).
        """

        recogniser_pipeline = ANPRPipeline.from_checkpoint(
            checkpoint_path=checkpoint_path,
            **recogniser_kwargs,
        )

        detector = OpenImageModelsPlateDetector(
            detection_model=detection_model,
            min_confidence=detector_min_confidence,
        )

        return cls(
            recogniser_pipeline=recogniser_pipeline,
            detector=detector,
            crop_padding_ratio=crop_padding_ratio,
        )

    def predict_from_image(
        self,
        image_or_path: str | Path | np.ndarray,
        debug_crop_path: str | Path | None = None,
    ) -> dict[str, Any]:
        """
        Predict plate text from a full car image.

        The local detector returns a NumPy crop, but the existing recogniser
        currently expects a file path. To avoid changing old recogniser files,
        this method saves the crop temporarily, runs recognition, then deletes
        the temporary crop.

        If debug_crop_path is provided, it also saves the detected crop there
        so you can visually inspect what the detector sent to the recogniser.
        """

        crop, detection = self.detector.crop_best_plate(
            image_or_path=image_or_path,
            padding_ratio=self.crop_padding_ratio,
        )

        if debug_crop_path is not None:
            debug_crop_path = Path(debug_crop_path)
            debug_crop_path.parent.mkdir(parents=True, exist_ok=True)

            saved = cv2.imwrite(str(debug_crop_path), crop)

            if not saved:
                raise ValueError(f"Failed to save debug crop to: {debug_crop_path}")

        with tempfile.TemporaryDirectory(prefix="anpr_detected_crop_") as temp_dir:
            temp_crop_path = Path(temp_dir) / "detected_plate.jpg"

            saved = cv2.imwrite(str(temp_crop_path), crop)

            if not saved:
                raise ValueError(f"Failed to save temporary crop to: {temp_crop_path}")

            recognition_result = self.recogniser_pipeline.predict_cropped_image(
                temp_crop_path
            )

        confidence = self._get_attr_or_key(recognition_result, "confidence")

        if confidence is None:
            confidence = self._get_attr_or_key(
                recognition_result,
                "overall_confidence",
            )

        return {
            "plate": self._get_attr_or_key(recognition_result, "plate"),
            "confidence": confidence,
            "valid_format": self._get_attr_or_key(recognition_result, "valid_format"),
            "should_accept": self._get_attr_or_key(
                recognition_result,
                "should_accept",
            ),
            "rejection_reasons": self._get_attr_or_key(
                recognition_result,
                "rejection_reasons",
            ),
            "detection": self._detection_to_dict(detection),
            "recognition_result": recognition_result,
        }

    @staticmethod
    def _detection_to_dict(detection: PlateBox) -> dict[str, Any]:
        return {
            "x1": detection.x1,
            "y1": detection.y1,
            "x2": detection.x2,
            "y2": detection.y2,
            "width": detection.width,
            "height": detection.height,
            "confidence": detection.confidence,
            "label": detection.label,
        }

    @staticmethod
    def _get_attr_or_key(obj: Any, name: str) -> Any:
        """
        Supports both dataclass-style result.plate and dict-style result["plate"].
        """

        if isinstance(obj, dict):
            return obj.get(name)

        return getattr(obj, name, None)
