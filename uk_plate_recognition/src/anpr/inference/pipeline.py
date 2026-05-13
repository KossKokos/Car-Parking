from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from anpr.inference.predict import PlateRecognizer
from anpr.inference.result import PlateRecognitionResult


REQUIRED_ROBOFLOW_BOX_KEYS = {"x", "y", "width", "height"}


def normalise_detection_class_name(value: str) -> str:
    """
    Normalise detector class names so Roboflow labels like:
        number-plates
        number plates
        number_plates
    can be compared safely.
    """
    return (
        str(value)
        .strip()
        .lower()
        .replace("-", "_")
        .replace(" ", "_")
    )


@dataclass(frozen=True)
class ANPRPipelineResult:
    """
    Final app-friendly ANPR result.

    This combines detection information and plate recognition information.
    """

    plate: str
    valid_format: bool
    overall_confidence: float
    should_accept: bool
    rejection_reasons: list[str]
    detection_confidence: float | None
    crop_padding: float
    recognition: dict[str, Any]
    detection: dict[str, Any] | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def extract_roboflow_predictions(
    roboflow_response: dict[str, Any] | list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Extract prediction dictionaries from common Roboflow response formats.

    Supports:
        {"predictions": [...]}
        [...]
        {"x": ..., "y": ..., "width": ..., "height": ...}
    """
    if isinstance(roboflow_response, list):
        predictions = roboflow_response

    elif isinstance(roboflow_response, dict):
        if "predictions" in roboflow_response:
            predictions = roboflow_response["predictions"]
        elif REQUIRED_ROBOFLOW_BOX_KEYS.issubset(roboflow_response):
            predictions = [roboflow_response]
        else:
            raise ValueError(
                "Roboflow response must contain 'predictions' or one box with "
                "x, y, width, height."
            )
    else:
        raise TypeError(
            f"Expected Roboflow response to be dict or list, got {type(roboflow_response)!r}."
        )

    if not isinstance(predictions, list):
        raise ValueError("'predictions' must be a list.")

    return predictions


def select_best_roboflow_prediction(
    predictions: list[dict[str, Any]],
    min_detection_confidence: float = 0.0,
    allowed_classes: set[str] | None = None,
) -> dict[str, Any]:
    """
    Select the best Roboflow plate detection.

    Selection priority:
        1. Valid box keys
        2. Optional class filter
        3. Optional confidence threshold
        4. Highest confidence if present
        5. Largest area as fallback
    """
    if not 0.0 <= min_detection_confidence <= 1.0:
        raise ValueError("min_detection_confidence must be between 0 and 1.")

    normalised_allowed_classes = None

    if allowed_classes is not None:
        normalised_allowed_classes = {
            normalise_detection_class_name(class_name)
            for class_name in allowed_classes
        }

    candidates: list[dict[str, Any]] = []
    rejected_reasons: list[str] = []

    for prediction in predictions:
        if not isinstance(prediction, dict):
            rejected_reasons.append("prediction_not_dict")
            continue

        if not REQUIRED_ROBOFLOW_BOX_KEYS.issubset(prediction):
            rejected_reasons.append(
                f"missing_box_keys:{set(REQUIRED_ROBOFLOW_BOX_KEYS) - set(prediction)}"
            )
            continue

        if normalised_allowed_classes is not None:
            class_name = normalise_detection_class_name(
                prediction.get("class", "")
            )

            if class_name not in normalised_allowed_classes:
                rejected_reasons.append(
                    f"class_rejected:{class_name}"
                )
                continue

        confidence = prediction.get("confidence")

        if confidence is not None and float(confidence) < min_detection_confidence:
            rejected_reasons.append(
                f"confidence_too_low:{confidence}"
            )
            continue

        width = float(prediction["width"])
        height = float(prediction["height"])

        if width <= 0 or height <= 0:
            rejected_reasons.append(
                f"invalid_box_size:{width}x{height}"
            )
            continue

        candidates.append(prediction)

    if not candidates:
        raise ValueError(
            "No valid Roboflow plate detections found. "
            f"Rejected reasons: {rejected_reasons}"
        )

    def sort_key(prediction: dict[str, Any]) -> tuple[float, float]:
        confidence = float(prediction.get("confidence", 0.0))
        area = float(prediction["width"]) * float(prediction["height"])
        return confidence, area

    return max(candidates, key=sort_key)


class ANPRPipeline:
    """
    End-to-end ML inference pipeline for UK plate recognition.

    This class does not call Roboflow itself.
    It consumes Roboflow-style detection output.
    """

    def __init__(
        self,
        recognizer: PlateRecognizer,
        default_crop_padding: float = 0.05,
        min_overall_confidence: float = 0.80,
        min_position_confidence: float = 0.60,
        min_detection_confidence: float = 0.0,
        allowed_detection_classes: set[str] | None = None,
    ) -> None:
        self.recognizer = recognizer
        self.default_crop_padding = default_crop_padding
        self.min_overall_confidence = min_overall_confidence
        self.min_position_confidence = min_position_confidence
        self.min_detection_confidence = min_detection_confidence
        self.allowed_detection_classes = allowed_detection_classes

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        device=None,
        default_crop_padding: float = 0.05,
        min_overall_confidence: float = 0.80,
        min_position_confidence: float = 0.60,
        min_detection_confidence: float = 0.0,
        allowed_detection_classes: set[str] | None = None,
    ) -> "ANPRPipeline":
        recognizer = PlateRecognizer.from_checkpoint(
            checkpoint_path=checkpoint_path,
            device=device,
        )

        return cls(
            recognizer=recognizer,
            default_crop_padding=default_crop_padding,
            min_overall_confidence=min_overall_confidence,
            min_position_confidence=min_position_confidence,
            min_detection_confidence=min_detection_confidence,
            allowed_detection_classes=allowed_detection_classes,
        )

    def predict_cropped_image(
        self,
        image_path: str | Path,
    ) -> ANPRPipelineResult:
        recognition_result = self.recognizer.predict_image_report(
            image_path=image_path,
            min_overall_confidence=self.min_overall_confidence,
            min_position_confidence=self.min_position_confidence,
        )

        return self._build_pipeline_result(
            recognition_result=recognition_result,
            detection=None,
            detection_confidence=None,
            crop_padding=0.0,
        )

    def predict_full_image_with_detection(
        self,
        image_path: str | Path,
        detection: dict[str, Any],
        crop_padding: float | None = None,
    ) -> ANPRPipelineResult:
        if crop_padding is None:
            crop_padding = self.default_crop_padding

        recognition_result = self.recognizer.predict_from_roboflow_crop_report(
            image_path=image_path,
            prediction=detection,
            padding=crop_padding,
            min_overall_confidence=self.min_overall_confidence,
            min_position_confidence=self.min_position_confidence,
        )

        detection_confidence = detection.get("confidence")
        detection_confidence = (
            float(detection_confidence)
            if detection_confidence is not None
            else None
        )

        return self._build_pipeline_result(
            recognition_result=recognition_result,
            detection=detection,
            detection_confidence=detection_confidence,
            crop_padding=crop_padding,
        )

    def predict_full_image_from_roboflow_response(
        self,
        image_path: str | Path,
        roboflow_response: dict[str, Any] | list[dict[str, Any]],
        crop_padding: float | None = None,
    ) -> ANPRPipelineResult:
        predictions = extract_roboflow_predictions(roboflow_response)

        best_detection = select_best_roboflow_prediction(
            predictions=predictions,
            min_detection_confidence=self.min_detection_confidence,
            allowed_classes=self.allowed_detection_classes,
        )

        return self.predict_full_image_with_detection(
            image_path=image_path,
            detection=best_detection,
            crop_padding=crop_padding,
        )

    def _build_pipeline_result(
        self,
        recognition_result: PlateRecognitionResult,
        detection: dict[str, Any] | None,
        detection_confidence: float | None,
        crop_padding: float,
    ) -> ANPRPipelineResult:
        return ANPRPipelineResult(
            plate=recognition_result.plate,
            valid_format=recognition_result.valid_format,
            overall_confidence=recognition_result.overall_confidence,
            should_accept=recognition_result.should_accept,
            rejection_reasons=recognition_result.rejection_reasons,
            detection_confidence=detection_confidence,
            crop_padding=crop_padding,
            recognition=recognition_result.to_dict(),
            detection=detection,
        )