from __future__ import annotations

from dataclasses import dataclass

import torch

from anpr.data.encoders import (
    CLASS_COUNTS,
    INDEX_TO_CHAR,
    NUM_POSITIONS,
)
from anpr.validation.uk_plate import clean_plate_text, is_valid_uk_plate


@dataclass(frozen=True)
class PlatePrediction:
    """
    Decoded prediction for one plate image.
    """

    raw_prediction: str
    cleaned_prediction: str
    is_valid_format: bool
    indices: list[int]
    position_confidences: list[float]
    overall_confidence: float


def validate_model_outputs(outputs: list[torch.Tensor]) -> None:
    """
    Validate raw model outputs before decoding.

    Expected:
        list of 7 tensors:
            [B, 26], [B, 26], [B, 10], [B, 10], [B, 26], [B, 26], [B, 26]
    """
    if not isinstance(outputs, list):
        raise TypeError(f"Expected outputs to be a list, got {type(outputs)!r}.")

    if len(outputs) != NUM_POSITIONS:
        raise ValueError(
            f"Expected {NUM_POSITIONS} output heads, got {len(outputs)}."
        )

    batch_size = outputs[0].shape[0]

    for position, output in enumerate(outputs):
        expected_num_classes = CLASS_COUNTS[position]

        if output.ndim != 2:
            raise ValueError(
                f"Output head {position} should have shape [B, C], "
                f"got {tuple(output.shape)}."
            )

        if output.shape[0] != batch_size:
            raise ValueError(
                f"Batch size mismatch at head {position}: "
                f"expected {batch_size}, got {output.shape[0]}."
            )

        if output.shape[1] != expected_num_classes:
            raise ValueError(
                f"Class count mismatch at head {position}: "
                f"expected {expected_num_classes}, got {output.shape[1]}."
            )


def decode_prediction_indices(indices: list[int]) -> str:
    """
    Convert one list of 7 predicted indices into a plate string.
    """
    if len(indices) != NUM_POSITIONS:
        raise ValueError(
            f"Expected {NUM_POSITIONS} indices, got {len(indices)}."
        )

    chars: list[str] = []

    for position, index in enumerate(indices):
        try:
            chars.append(INDEX_TO_CHAR[position][index])
        except KeyError as exc:
            raise ValueError(
                f"Invalid class index {index} at position {position}."
            ) from exc

    return "".join(chars)


def decode_model_outputs(outputs: list[torch.Tensor]) -> list[PlatePrediction]:
    """
    Decode model logits into plate predictions.

    Args:
        outputs:
            List of 7 raw logit tensors from PlateCNN.

    Returns:
        List of PlatePrediction objects, one per image in the batch.
    """
    validate_model_outputs(outputs)

    with torch.no_grad():
        probabilities = [
            torch.softmax(output, dim=1)
            for output in outputs
        ]

        predicted_indices_per_head: list[torch.Tensor] = []
        confidences_per_head: list[torch.Tensor] = []

        for position_probs in probabilities:
            confidences, indices = torch.max(position_probs, dim=1)
            predicted_indices_per_head.append(indices)
            confidences_per_head.append(confidences)

        batch_size = outputs[0].shape[0]
        predictions: list[PlatePrediction] = []

        for sample_index in range(batch_size):
            indices = [
                int(head_indices[sample_index].item())
                for head_indices in predicted_indices_per_head
            ]

            position_confidences = [
                float(head_confidences[sample_index].item())
                for head_confidences in confidences_per_head
            ]

            raw_prediction = decode_prediction_indices(indices)
            cleaned_prediction = clean_plate_text(raw_prediction)
            is_valid_format = is_valid_uk_plate(cleaned_prediction)

            overall_confidence = float(
                sum(position_confidences) / len(position_confidences)
            )

            predictions.append(
                PlatePrediction(
                    raw_prediction=raw_prediction,
                    cleaned_prediction=cleaned_prediction,
                    is_valid_format=is_valid_format,
                    indices=indices,
                    position_confidences=position_confidences,
                    overall_confidence=overall_confidence,
                )
            )

    return predictions