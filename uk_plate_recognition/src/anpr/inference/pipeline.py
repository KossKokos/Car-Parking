from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from anpr.inference.predict import PlateRecognizer
from anpr.inference.result import PlateRecognitionResult


@dataclass(frozen=True)
class ANPRPipelineResult:
    """
    Final app-friendly ANPR result.

    This wraps recognition output in the shape consumed by the app.
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


class ANPRPipeline:
    """
    Cropped-plate recognition pipeline for UK plate recognition.
    """

    def __init__(
        self,
        recognizer: PlateRecognizer,
        min_overall_confidence: float = 0.80,
        min_position_confidence: float = 0.60,
    ) -> None:
        self.recognizer = recognizer
        self.min_overall_confidence = min_overall_confidence
        self.min_position_confidence = min_position_confidence

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        device=None,
        min_overall_confidence: float = 0.80,
        min_position_confidence: float = 0.60,
    ) -> "ANPRPipeline":
        recognizer = PlateRecognizer.from_checkpoint(
            checkpoint_path=checkpoint_path,
            device=device,
        )

        return cls(
            recognizer=recognizer,
            min_overall_confidence=min_overall_confidence,
            min_position_confidence=min_position_confidence,
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
        )

    def _build_pipeline_result(
        self,
        recognition_result: PlateRecognitionResult,
    ) -> ANPRPipelineResult:
        return ANPRPipelineResult(
            plate=recognition_result.plate,
            valid_format=recognition_result.valid_format,
            overall_confidence=recognition_result.overall_confidence,
            should_accept=recognition_result.should_accept,
            rejection_reasons=recognition_result.rejection_reasons,
            detection_confidence=None,
            crop_padding=0.0,
            recognition=recognition_result.to_dict(),
            detection=None,
        )
