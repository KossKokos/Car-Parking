from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

from anpr.inference.preprocess import prepare_plate_tensor
from anpr.inference.decode import PlatePrediction, decode_model_outputs
from anpr.inference.result import PlateRecognitionResult, build_recognition_result
from anpr.models.plate_cnn import PlateCNN
from anpr.preprocessing.image_ops import (
    crop_image_from_roboflow_prediction,
    load_image_bgr,
)


class PlateRecognizer:
    """
    Single-image inference wrapper for the UK plate recogniser.

    Expected input:
        cropped plate image

    Not expected input:
        full car image

    Flow:
        image path -> OpenCV load -> transform -> model -> decode -> PlatePrediction
    """

    def __init__(
        self,
        model: PlateCNN,
        image_height: int,
        image_width: int,
        grayscale: bool = True,
        device: str | torch.device = "cpu",
    ) -> None:
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.model.eval()

        self.image_height = image_height
        self.image_width = image_width
        self.grayscale = grayscale


    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        device: str | torch.device | None = None,
    ) -> "PlateRecognizer":
        checkpoint_path = Path(checkpoint_path)

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device(device)

        checkpoint = torch.load(
            checkpoint_path,
            map_location=device,
        )

        model_config = checkpoint["model_config"]

        model = PlateCNN(
            in_channels=model_config.get("in_channels", 1),
            dropout=model_config.get("dropout", 0.1),
            pooled_width=model_config.get("pooled_width", 10),
        )

        model.load_state_dict(checkpoint["model_state_dict"])

        return cls(
            model=model,
            image_height=model_config.get("image_height", 48),
            image_width=model_config.get("image_width", 160),
            grayscale=model_config.get("grayscale", True),
            device=device,
        )

    def predict_image(
        self,
        image_path: str | Path,
    ) -> PlatePrediction:

        image = self._load_image(image_path)
        return self.predict_array(image)


    def predict_image_as_dict(
        self,
        image_path: str | Path,
    ) -> dict[str, Any]:
        prediction = self.predict_image(image_path)
        return asdict(prediction)

    def _load_image(
        self,
        image_path: str | Path,
    ) -> np.ndarray:
        image_path = Path(image_path)

        if not image_path.exists():
            raise FileNotFoundError(f"Image does not exist: {image_path}")

        if self.grayscale:
            image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)

            if image is None:
                raise ValueError(f"Could not read image: {image_path}")

            image = image[..., None]
            return image

        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)

        if image is None:
            raise ValueError(f"Could not read image: {image_path}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

    def _prepare_image_tensor(
        self,
        image: np.ndarray,
    ) -> torch.Tensor:
        return prepare_plate_tensor(
            image=image,
            image_height=self.image_height,
            image_width=self.image_width,
            grayscale=self.grayscale,
            device=self.device,
        )
    

    def predict_array(
        self,
        image: np.ndarray,
    ) -> PlatePrediction:
        """
        Predict plate text from an already loaded cropped plate image.

        The image should already be cropped around the plate.
        """
        image = self._normalise_input_image(image)
        image_tensor = self._prepare_image_tensor(image)

        with torch.no_grad():
            outputs = self.model(image_tensor)
            prediction = decode_model_outputs(outputs)[0]

        return prediction

    def predict_array_as_dict(
        self,
        image: np.ndarray,
    ) -> dict[str, Any]:
        prediction = self.predict_array(image)
        return asdict(prediction)

    def predict_from_roboflow_crop(
        self,
        image_path: str | Path,
        prediction: dict,
        padding: float = 0.05,
    ) -> PlatePrediction:
        """
        Predict plate text from a full image and one Roboflow detection box.

        Args:
            image_path:
                Path to full car/barrier image.
            prediction:
                Roboflow-style prediction dict with x, y, width, height.
            padding:
                Relative crop padding around the detected plate.

        Returns:
            PlatePrediction
        """
        image = load_image_bgr(image_path)

        crop = crop_image_from_roboflow_prediction(
            image=image,
            prediction=prediction,
            padding=padding,
        )

        return self.predict_array(crop)

    def predict_from_roboflow_crop_as_dict(
        self,
        image_path: str | Path,
        prediction: dict,
        padding: float = 0.05,
    ) -> dict[str, Any]:
        plate_prediction = self.predict_from_roboflow_crop(
            image_path=image_path,
            prediction=prediction,
            padding=padding,
        )

        return asdict(plate_prediction)

    def _normalise_input_image(
        self,
        image: np.ndarray,
    ) -> np.ndarray:
        """
        Convert input image into the channel format expected by the recogniser.
        """
        if image is None:
            raise ValueError("image cannot be None.")

        if self.grayscale:
            if image.ndim == 2:
                return image[..., None]

            if image.ndim == 3 and image.shape[2] == 1:
                return image

            if image.ndim == 3 and image.shape[2] == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                return gray[..., None]

            raise ValueError(f"Unsupported image shape for grayscale mode: {image.shape}")

        if image.ndim == 2:
            return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        if image.ndim == 3 and image.shape[2] == 3:
            return image

        raise ValueError(f"Unsupported image shape: {image.shape}")
    

    def predict_image_report(
        self,
        image_path: str | Path,
        min_overall_confidence: float = 0.80,
        min_position_confidence: float = 0.60,
    ) -> PlateRecognitionResult:
        prediction = self.predict_image(image_path)

        return build_recognition_result(
            prediction=prediction,
            min_overall_confidence=min_overall_confidence,
            min_position_confidence=min_position_confidence,
        )

    def predict_array_report(
        self,
        image: np.ndarray,
        min_overall_confidence: float = 0.80,
        min_position_confidence: float = 0.60,
    ) -> PlateRecognitionResult:
        prediction = self.predict_array(image)

        return build_recognition_result(
            prediction=prediction,
            min_overall_confidence=min_overall_confidence,
            min_position_confidence=min_position_confidence,
        )

    def predict_from_roboflow_crop_report(
        self,
        image_path: str | Path,
        prediction: dict,
        padding: float = 0.05,
        min_overall_confidence: float = 0.80,
        min_position_confidence: float = 0.60,
    ) -> PlateRecognitionResult:
        plate_prediction = self.predict_from_roboflow_crop(
            image_path=image_path,
            prediction=prediction,
            padding=padding,
        )

        return build_recognition_result(
            prediction=plate_prediction,
            min_overall_confidence=min_overall_confidence,
            min_position_confidence=min_position_confidence,
        )