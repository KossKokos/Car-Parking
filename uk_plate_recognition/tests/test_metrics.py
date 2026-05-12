import pytest

from anpr.evaluation.metrics import (
    PlateMetrics,
    calculate_plate_metrics,
    format_plate_metrics,
)
from anpr.inference.decode import PlatePrediction


def make_prediction(
    text: str,
    is_valid_format: bool = True,
    confidence: float = 0.9,
) -> PlatePrediction:
    return PlatePrediction(
        raw_prediction=text,
        cleaned_prediction=text,
        is_valid_format=is_valid_format,
        indices=[0, 0, 0, 0, 0, 0, 0],
        position_confidences=[confidence] * 7,
        overall_confidence=confidence,
    )


def test_calculate_plate_metrics_perfect_predictions():
    predictions = [
        make_prediction("AA04QZH", confidence=0.9),
        make_prediction("AB12CDE", confidence=0.8),
    ]

    labels = ["AA04QZH", "AB12CDE"]

    metrics = calculate_plate_metrics(predictions, labels)

    assert isinstance(metrics, PlateMetrics)
    assert metrics.num_samples == 2
    assert metrics.full_plate_accuracy == 1.0
    assert metrics.per_position_accuracy == [1.0] * 7
    assert metrics.regex_valid_rate == 1.0
    assert metrics.average_confidence == pytest.approx(0.85)


def test_calculate_plate_metrics_partial_prediction():
    predictions = [
        make_prediction("AB04QZH", confidence=0.7),
    ]

    labels = ["AA04QZH"]

    metrics = calculate_plate_metrics(predictions, labels)

    assert metrics.num_samples == 1
    assert metrics.full_plate_accuracy == 0.0
    assert metrics.per_position_accuracy == [
        1.0,  # A == A
        0.0,  # B != A
        1.0,  # 0 == 0
        1.0,  # 4 == 4
        1.0,  # Q == Q
        1.0,  # Z == Z
        1.0,  # H == H
    ]
    assert metrics.regex_valid_rate == 1.0
    assert metrics.average_confidence == pytest.approx(0.7)


def test_calculate_plate_metrics_handles_invalid_prediction_format():
    predictions = [
        make_prediction("AA04QZH", is_valid_format=True, confidence=0.9),
        make_prediction("INVALID", is_valid_format=False, confidence=0.3),
    ]

    labels = ["AA04QZH", "AB12CDE"]

    metrics = calculate_plate_metrics(predictions, labels)

    assert metrics.num_samples == 2
    assert metrics.full_plate_accuracy == 0.5
    assert metrics.regex_valid_rate == 0.5
    assert metrics.average_confidence == pytest.approx(0.6)


def test_calculate_plate_metrics_rejects_length_mismatch():
    predictions = [
        make_prediction("AA04QZH"),
    ]

    labels = ["AA04QZH", "AB12CDE"]

    with pytest.raises(ValueError):
        calculate_plate_metrics(predictions, labels)


def test_calculate_plate_metrics_rejects_empty_predictions():
    with pytest.raises(ValueError):
        calculate_plate_metrics([], [])


def test_format_plate_metrics_returns_readable_string():
    predictions = [
        make_prediction("AA04QZH", confidence=0.9),
    ]

    labels = ["AA04QZH"]

    metrics = calculate_plate_metrics(predictions, labels)
    text = format_plate_metrics(metrics)

    assert "Samples: 1" in text
    assert "Full-plate accuracy: 1.0000" in text
    assert "Regex-valid rate: 1.0000" in text
    assert "Average confidence: 0.9000" in text
    assert "Position 0" in text