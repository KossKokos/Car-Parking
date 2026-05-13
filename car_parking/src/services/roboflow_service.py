from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

from car_parking.src.conf.config import settings

class RoboflowDetectionError(Exception):
    """Raised when Roboflow detection fails."""


@dataclass(frozen=True)
class RoboflowConfig:
    api_key: str
    detect_url: str
    confidence: int = 40
    overlap: int = 30
    timeout_seconds: float = 30.0


def get_roboflow_config() -> RoboflowConfig:
    api_key = settings.ROBOFLOW_API_KEY
    detect_url = settings.ROBOFLOW_PLATE_DETECT_URL

    if not api_key:
        raise RoboflowDetectionError("Missing ROBOFLOW_API_KEY environment variable.")

    if not detect_url:
        raise RoboflowDetectionError("Missing ROBOFLOW_PLATE_DETECT_URL environment variable.")

    confidence = settings.ROBOFLOW_CONFIDENCE
    overlap = settings.ROBOFLOW_OVERLAP

    return RoboflowConfig(
        api_key=api_key,
        detect_url=detect_url,
        confidence=confidence,
        overlap=overlap,
    )


class RoboflowPlateDetector:
    """
    API-side Roboflow client.

    Responsibility:
        full car image -> Roboflow API -> detection response

    It does not run the PyTorch recogniser.
    It only returns Roboflow detection data.
    """

    def __init__(self, config: RoboflowConfig | None = None) -> None:
        self.config = config or get_roboflow_config()

    async def detect_from_image_path(
        self,
        image_path: str | Path,
    ) -> dict[str, Any]:
        image_path = Path(image_path)

        if not image_path.exists():
            raise RoboflowDetectionError(f"Image does not exist: {image_path}")

        try:
            image_bytes = image_path.read_bytes()
        except OSError as exc:
            raise RoboflowDetectionError(f"Could not read image: {image_path}") from exc

        return await self.detect_from_bytes(
            image_bytes=image_bytes,
            filename=image_path.name,
        )

    async def detect_from_bytes(
        self,
        image_bytes: bytes,
        filename: str = "image.jpg",
    ) -> dict[str, Any]:
        if not image_bytes:
            raise RoboflowDetectionError("Image bytes are empty.")

        params = {
            "api_key": self.config.api_key,
            "confidence": self.config.confidence,
            "overlap": self.config.overlap,
        }

        files = {
            "file": (
                filename,
                image_bytes,
                "application/octet-stream",
            )
        }

        try:
            async with httpx.AsyncClient(timeout=self.config.timeout_seconds) as client:
                response = await client.post(
                    self.config.detect_url,
                    params=params,
                    files=files,
                )

            response.raise_for_status()

        except httpx.HTTPStatusError as exc:
            raise RoboflowDetectionError(
                f"Roboflow returned HTTP {exc.response.status_code}: {exc.response.text}"
            ) from exc

        except httpx.RequestError as exc:
            raise RoboflowDetectionError(
                f"Roboflow request failed: {exc}"
            ) from exc

        try:
            payload = response.json()
        except ValueError as exc:
            raise RoboflowDetectionError("Roboflow response was not valid JSON.") from exc

        self._validate_response(payload)
        return payload

    def _validate_response(self, payload: dict[str, Any]) -> None:
        if not isinstance(payload, dict):
            raise RoboflowDetectionError("Roboflow response must be a dictionary.")

        predictions = payload.get("predictions")

        if predictions is None:
            raise RoboflowDetectionError("Roboflow response missing 'predictions' field.")

        if not isinstance(predictions, list):
            raise RoboflowDetectionError("Roboflow 'predictions' field must be a list.")

        if not predictions:
            raise RoboflowDetectionError("Roboflow did not detect any license plate.")


_roboflow_plate_detector: RoboflowPlateDetector | None = None


def get_roboflow_plate_detector() -> RoboflowPlateDetector:
    """
    Singleton-style Roboflow detector client.
    """
    global _roboflow_plate_detector

    if _roboflow_plate_detector is None:
        _roboflow_plate_detector = RoboflowPlateDetector()

    return _roboflow_plate_detector