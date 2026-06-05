from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

from anpr.inference.open_image_models_pipeline import OpenImageModelsANPRPipeline


REPO_ROOT = Path(__file__).resolve().parents[3]

ANPR_ROOT = REPO_ROOT / "uk_plate_recognition"

CHECKPOINT_PATH = (
    ANPR_ROOT
    / "checkpoints"
    / "custom_data_cnn_v2"
    / "plate_cnn_final.pt"
)

DETECTION_MODEL = "yolo-v9-t-256-license-plate-end2end"
MIN_OVERALL_CONFIDENCE = 0.95
MIN_POSITION_CONFIDENCE = 0.80


@lru_cache(maxsize=1)
def get_anpr_pipeline() -> OpenImageModelsANPRPipeline:
    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(f"ANPR checkpoint not found: {CHECKPOINT_PATH}")

    return OpenImageModelsANPRPipeline.from_checkpoint(
        checkpoint_path=CHECKPOINT_PATH,
        detection_model=DETECTION_MODEL,
        detector_min_confidence=0.25,
        crop_padding_ratio=0.05,
        device="cpu",
        min_overall_confidence=MIN_OVERALL_CONFIDENCE,
        min_position_confidence=MIN_POSITION_CONFIDENCE,
    )


def read_license_plate(image_path: str | Path) -> str:
    result = get_anpr_pipeline().predict_from_image(image_path)
    plate = result.get("plate")

    if not plate:
        raise ValueError(f"ANPR failed to read plate from image: {image_path}")

    return plate


def read_license_plate_report(image_path: str | Path) -> dict[str, Any]:
    result = get_anpr_pipeline().predict_from_image(image_path)

    return {
        "plate": result.get("plate"),
        "confidence": result.get("confidence"),
        "valid_format": result.get("valid_format"),
        "should_accept": result.get("should_accept"),
        "rejection_reasons": result.get("rejection_reasons"),
        "detection": result.get("detection"),
    }
