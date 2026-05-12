import pytest
import torch

from anpr.data.encoders import CLASS_COUNTS
from anpr.inference.decode import (
    PlatePrediction,
    decode_model_outputs,
    decode_prediction_indices,
    validate_model_outputs,
)


def make_fake_outputs(batch_size: int = 4):
    return [
        torch.randn(batch_size, num_classes)
        for num_classes in CLASS_COUNTS
    ]


def test_decode_prediction_indices():
    indices = [0, 0, 0, 4, 16, 25, 7]

    assert decode_prediction_indices(indices) == "AA04QZH"


def test_decode_prediction_indices_rejects_wrong_length():
    with pytest.raises(ValueError):
        decode_prediction_indices([0, 1, 2])


def test_decode_prediction_indices_rejects_invalid_index():
    with pytest.raises(ValueError):
        decode_prediction_indices([0, 0, 99, 4, 16, 25, 7])


def test_validate_model_outputs_accepts_valid_outputs():
    outputs = make_fake_outputs(batch_size=4)

    validate_model_outputs(outputs)


def test_validate_model_outputs_rejects_wrong_number_of_heads():
    outputs = make_fake_outputs(batch_size=4)[:6]

    with pytest.raises(ValueError):
        validate_model_outputs(outputs)


def test_validate_model_outputs_rejects_wrong_class_count():
    outputs = make_fake_outputs(batch_size=4)
    outputs[2] = torch.randn(4, 26)

    with pytest.raises(ValueError):
        validate_model_outputs(outputs)


def test_decode_model_outputs_returns_predictions_for_batch():
    outputs = make_fake_outputs(batch_size=4)

    predictions = decode_model_outputs(outputs)

    assert len(predictions) == 4

    for prediction in predictions:
        assert isinstance(prediction, PlatePrediction)
        assert len(prediction.raw_prediction) == 7
        assert len(prediction.cleaned_prediction) == 7
        assert isinstance(prediction.is_valid_format, bool)
        assert len(prediction.indices) == 7
        assert len(prediction.position_confidences) == 7
        assert 0.0 <= prediction.overall_confidence <= 1.0


def test_decode_model_outputs_can_decode_known_logits():
    batch_size = 1

    target_indices = [0, 0, 0, 4, 16, 25, 7]

    outputs = []

    for position, num_classes in enumerate(CLASS_COUNTS):
        logits = torch.zeros(batch_size, num_classes)
        logits[0, target_indices[position]] = 10.0
        outputs.append(logits)

    predictions = decode_model_outputs(outputs)

    assert predictions[0].raw_prediction == "AA04QZH"
    assert predictions[0].cleaned_prediction == "AA04QZH"
    assert predictions[0].is_valid_format is True
    assert predictions[0].indices == target_indices
    assert predictions[0].overall_confidence > 0.9