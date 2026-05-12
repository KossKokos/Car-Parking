import pytest

from anpr.inference.pipeline import (
    extract_roboflow_predictions,
    select_best_roboflow_prediction,
)


def test_extract_roboflow_predictions_from_response_dict():
    response = {
        "predictions": [
            {
                "x": 100,
                "y": 50,
                "width": 40,
                "height": 20,
                "confidence": 0.9,
            }
        ]
    }

    predictions = extract_roboflow_predictions(response)

    assert len(predictions) == 1
    assert predictions[0]["confidence"] == 0.9


def test_extract_roboflow_predictions_from_list():
    response = [
        {
            "x": 100,
            "y": 50,
            "width": 40,
            "height": 20,
            "confidence": 0.9,
        }
    ]

    predictions = extract_roboflow_predictions(response)

    assert len(predictions) == 1


def test_extract_roboflow_predictions_from_single_prediction_dict():
    response = {
        "x": 100,
        "y": 50,
        "width": 40,
        "height": 20,
        "confidence": 0.9,
    }

    predictions = extract_roboflow_predictions(response)

    assert len(predictions) == 1
    assert predictions[0]["x"] == 100


def test_extract_roboflow_predictions_rejects_invalid_response():
    with pytest.raises(ValueError):
        extract_roboflow_predictions({"bad": "response"})


def test_select_best_roboflow_prediction_by_confidence():
    predictions = [
        {
            "x": 100,
            "y": 50,
            "width": 40,
            "height": 20,
            "confidence": 0.7,
        },
        {
            "x": 200,
            "y": 80,
            "width": 50,
            "height": 25,
            "confidence": 0.95,
        },
    ]

    best = select_best_roboflow_prediction(predictions)

    assert best["x"] == 200
    assert best["confidence"] == 0.95


def test_select_best_roboflow_prediction_filters_by_min_confidence():
    predictions = [
        {
            "x": 100,
            "y": 50,
            "width": 40,
            "height": 20,
            "confidence": 0.4,
        },
        {
            "x": 200,
            "y": 80,
            "width": 50,
            "height": 25,
            "confidence": 0.9,
        },
    ]

    best = select_best_roboflow_prediction(
        predictions,
        min_detection_confidence=0.8,
    )

    assert best["x"] == 200


def test_select_best_roboflow_prediction_filters_by_class():
    predictions = [
        {
            "x": 100,
            "y": 50,
            "width": 40,
            "height": 20,
            "confidence": 0.95,
            "class": "car",
        },
        {
            "x": 200,
            "y": 80,
            "width": 50,
            "height": 25,
            "confidence": 0.8,
            "class": "license_plate",
        },
    ]

    best = select_best_roboflow_prediction(
        predictions,
        allowed_classes={"license_plate"},
    )

    assert best["class"] == "license_plate"


def test_select_best_roboflow_prediction_uses_area_as_fallback():
    predictions = [
        {
            "x": 100,
            "y": 50,
            "width": 40,
            "height": 20,
        },
        {
            "x": 200,
            "y": 80,
            "width": 80,
            "height": 30,
        },
    ]

    best = select_best_roboflow_prediction(predictions)

    assert best["x"] == 200


def test_select_best_roboflow_prediction_rejects_empty_candidates():
    predictions = [
        {
            "bad": "box",
        }
    ]

    with pytest.raises(ValueError):
        select_best_roboflow_prediction(predictions)


def test_select_best_roboflow_prediction_rejects_invalid_threshold():
    with pytest.raises(ValueError):
        select_best_roboflow_prediction(
            [],
            min_detection_confidence=1.5,
        )