import pytest

from anpr.inference.decode import PlatePrediction
from anpr.inference.result import (
    PlateRecognitionResult,
    build_recognition_result,
)


def make_prediction(
    text: str = "AB12CDE",
    valid: bool = True,
    overall_confidence: float = 0.95,
    position_confidences: list[float] | None = None,
) -> PlatePrediction:
    if position_confidences is None:
        position_confidences = [0.95] * 7

    return PlatePrediction(
        raw_prediction=text,
        cleaned_prediction=text,
        is_valid_format=valid,
        indices=[0, 1, 1, 2, 2, 3, 4],
        position_confidences=position_confidences,
        overall_confidence=overall_confidence,
    )


def test_build_recognition_result_accepts_good_prediction():
    prediction = make_prediction()

    result = build_recognition_result(prediction)

    assert isinstance(result, PlateRecognitionResult)
    assert result.plate == "AB12CDE"
    assert result.raw_prediction == "AB12CDE"
    assert result.valid_format is True
    assert result.should_accept is True
    assert result.low_confidence_positions == []
    assert result.rejection_reasons == []


def test_build_recognition_result_rejects_invalid_format():
    prediction = make_prediction(
        text="INVALID",
        valid=False,
        overall_confidence=0.95,
    )

    result = build_recognition_result(prediction)

    assert result.should_accept is False
    assert "invalid_uk_plate_format" in result.rejection_reasons


def test_build_recognition_result_rejects_low_overall_confidence():
    prediction = make_prediction(
        overall_confidence=0.50,
        position_confidences=[0.90] * 7,
    )

    result = build_recognition_result(
        prediction,
        min_overall_confidence=0.80,
        min_position_confidence=0.60,
    )

    assert result.should_accept is False
    assert "low_overall_confidence" in result.rejection_reasons


def test_build_recognition_result_rejects_low_position_confidence():
    prediction = make_prediction(
        overall_confidence=0.90,
        position_confidences=[0.95, 0.95, 0.95, 0.40, 0.95, 0.95, 0.95],
    )

    result = build_recognition_result(
        prediction,
        min_overall_confidence=0.80,
        min_position_confidence=0.60,
    )

    assert result.should_accept is False
    assert result.low_confidence_positions == [3]
    assert "low_position_confidence" in result.rejection_reasons


def test_build_recognition_result_to_dict():
    prediction = make_prediction()

    result = build_recognition_result(prediction)
    result_dict = result.to_dict()

    assert result_dict["plate"] == "AB12CDE"
    assert result_dict["valid_format"] is True
    assert result_dict["should_accept"] is True


def test_build_recognition_result_rejects_invalid_thresholds():
    prediction = make_prediction()

    with pytest.raises(ValueError):
        build_recognition_result(
            prediction,
            min_overall_confidence=-0.1,
        )

    with pytest.raises(ValueError):
        build_recognition_result(
            prediction,
            min_position_confidence=1.5,
        )


def test_build_recognition_result_rejects_wrong_confidence_length():
    prediction = make_prediction(
        position_confidences=[0.95, 0.95],
    )

    with pytest.raises(ValueError):
        build_recognition_result(prediction)