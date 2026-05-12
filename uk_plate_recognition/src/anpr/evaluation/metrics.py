from __future__ import annotations

from dataclasses import dataclass

from anpr.data.encoders import NUM_POSITIONS
from anpr.inference.decode import PlatePrediction
from anpr.validation.uk_plate import validate_uk_plate


@dataclass(frozen=True)
class PlateMetrics:
    """
    Evaluation metrics for fixed-format UK plate recognition.
    """

    num_samples: int
    full_plate_accuracy: float
    per_position_accuracy: list[float]
    regex_valid_rate: float
    average_confidence: float


def calculate_plate_metrics(
    predictions: list[PlatePrediction],
    labels: list[str],
) -> PlateMetrics:
    """
    Calculate evaluation metrics from decoded predictions and true labels.

    Args:
        predictions:
            Decoded model predictions.
        labels:
            Ground-truth UK plate labels.

    Returns:
        PlateMetrics object.
    """
    if len(predictions) != len(labels):
        raise ValueError(
            f"Number of predictions and labels must match. "
            f"Got {len(predictions)} predictions and {len(labels)} labels."
        )

    if not predictions:
        raise ValueError("Cannot calculate metrics for an empty prediction list.")

    cleaned_labels = [validate_uk_plate(label) for label in labels]

    num_samples = len(predictions)

    exact_matches = 0
    regex_valid_count = 0
    confidence_total = 0.0

    position_correct_counts = [0 for _ in range(NUM_POSITIONS)]

    for prediction, true_label in zip(predictions, cleaned_labels):
        predicted_label = prediction.cleaned_prediction

        if predicted_label == true_label:
            exact_matches += 1

        if prediction.is_valid_format:
            regex_valid_count += 1

        confidence_total += prediction.overall_confidence

        for position in range(NUM_POSITIONS):
            if predicted_label[position] == true_label[position]:
                position_correct_counts[position] += 1

    full_plate_accuracy = exact_matches / num_samples

    per_position_accuracy = [
        correct_count / num_samples
        for correct_count in position_correct_counts
    ]

    regex_valid_rate = regex_valid_count / num_samples
    average_confidence = confidence_total / num_samples

    return PlateMetrics(
        num_samples=num_samples,
        full_plate_accuracy=full_plate_accuracy,
        per_position_accuracy=per_position_accuracy,
        regex_valid_rate=regex_valid_rate,
        average_confidence=average_confidence,
    )


def format_plate_metrics(metrics: PlateMetrics) -> str:
    """
    Format metrics as a readable text summary.
    """
    lines = [
        f"Samples: {metrics.num_samples}",
        f"Full-plate accuracy: {metrics.full_plate_accuracy:.4f}",
        f"Regex-valid rate: {metrics.regex_valid_rate:.4f}",
        f"Average confidence: {metrics.average_confidence:.4f}",
        "Per-position accuracy:",
    ]

    for position, accuracy in enumerate(metrics.per_position_accuracy):
        lines.append(f"  Position {position}: {accuracy:.4f}")

    return "\n".join(lines)