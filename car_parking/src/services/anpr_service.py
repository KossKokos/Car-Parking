from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from anpr.inference.pipeline import ANPRPipeline, ANPRPipelineResult
from anpr.inference.result import PlateRecognitionResult


PROJECT_ROOT = Path(__file__).resolve().parents[3]

DEFAULT_CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "uk_plate_recognition"
    / "checkpoints"
    / "baseline_cnn"
    / "plate_cnn_final.pt"
)


class ANPRService:
    """
    API-side wrapper around the PyTorch ANPR pipeline.

    The model is loaded once when ANPRService is created and reused after that.
    """

    def __init__(
        self,
        checkpoint_path: str | Path = DEFAULT_CHECKPOINT_PATH,
        device: str = "cpu",
        min_overall_confidence: float = 0.80,
        min_position_confidence: float = 0.60,
        min_detection_confidence: float = 0.50,
    ) -> None:
        self.min_overall_confidence = min_overall_confidence
        self.min_position_confidence = min_position_confidence

        self.pipeline = ANPRPipeline.from_checkpoint(
            checkpoint_path=checkpoint_path,
            device=device,
            default_crop_padding=0.10,
            min_overall_confidence=min_overall_confidence,
            min_position_confidence=min_position_confidence,
            min_detection_confidence=min_detection_confidence,
            allowed_detection_classes={
                "number-plates",
                "number_plate",
                "number_plates",
                "license_plate",
                "license-plate",
                "plate",
            },
        )

    def predict_cropped_plate_path(
        self,
        image_path: str | Path,
    ) -> ANPRPipelineResult:
        return self.pipeline.predict_cropped_image(image_path)

    def predict_cropped_plate_array(
        self,
        image: np.ndarray,
    ) -> PlateRecognitionResult:
        """
        Predict from an already loaded cropped plate image array.

        This is useful for preserving the old plate_reader.py interface,
        where get_prediction(img) received an OpenCV image array.
        """
        return self.pipeline.recognizer.predict_array_report(
            image=image,
            min_overall_confidence=self.min_overall_confidence,
            min_position_confidence=self.min_position_confidence,
        )

    def predict_from_roboflow_response(
        self,
        image_path: str | Path,
        roboflow_response: dict[str, Any] | list[dict[str, Any]],
        crop_padding: float = 0.05,
    ) -> ANPRPipelineResult:
        return self.pipeline.predict_full_image_from_roboflow_response(
            image_path=image_path,
            roboflow_response=roboflow_response,
            crop_padding=crop_padding,
        )