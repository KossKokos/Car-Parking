from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from anpr.data.encoders import NUM_POSITIONS
from anpr.inference.decode import PlatePrediction


@dataclass(frozen=True)
class PlateRecognitionResult:
    """
    App-friendly plate recognition result.

    This wraps the raw model prediction with simple acceptance logic.
    """

    plate: str
    raw_prediction: str
    valid_format: bool
    overall_confidence: float
    position_confidences: list[float]
    low_confidence_positions: list[int]
    should_accept: bool
    rejection_reasons: list[str]

    def to_dict(self) -> dict[str, Any]:
        """Convert the recognition result to a JSON-friendly dictionary."""
        return asdict(self)


def build_recognition_result(
    prediction: PlatePrediction,
    min_overall_confidence: float = 0.80,
    min_position_confidence: float = 0.60,
) -> PlateRecognitionResult:
    """
    Convert a PlatePrediction into an app-friendly recognition result.

    Acceptance rules:
        - prediction must match UK LLDDLLL format
        - overall confidence must be >= min_overall_confidence
        - every position confidence must be >= min_position_confidence
    """
    if not 0.0 <= min_overall_confidence <= 1.0:
        raise ValueError("min_overall_confidence must be between 0 and 1.")

    if not 0.0 <= min_position_confidence <= 1.0:
        raise ValueError("min_position_confidence must be between 0 and 1.")

    if len(prediction.position_confidences) != NUM_POSITIONS:
        raise ValueError(
            f"Expected {NUM_POSITIONS} position confidences, "
            f"got {len(prediction.position_confidences)}."
        )

    low_confidence_positions = [
        position
        for position, confidence in enumerate(prediction.position_confidences)
        if confidence < min_position_confidence
    ]

    rejection_reasons: list[str] = []

    if not prediction.is_valid_format:
        rejection_reasons.append("invalid_uk_plate_format")

    if prediction.overall_confidence < min_overall_confidence:
        rejection_reasons.append("low_overall_confidence")

    if low_confidence_positions:
        rejection_reasons.append("low_position_confidence")

    should_accept = len(rejection_reasons) == 0

    return PlateRecognitionResult(
        plate=prediction.cleaned_prediction,
        raw_prediction=prediction.raw_prediction,
        valid_format=prediction.is_valid_format,
        overall_confidence=prediction.overall_confidence,
        position_confidences=prediction.position_confidences,
        low_confidence_positions=low_confidence_positions,
        should_accept=should_accept,
        rejection_reasons=rejection_reasons,
    )
