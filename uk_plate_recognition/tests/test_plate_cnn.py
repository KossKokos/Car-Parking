import torch

from anpr.data.encoders import CLASS_COUNTS
from anpr.models.plate_cnn import PlateCNN, count_trainable_parameters


def test_plate_cnn_outputs_seven_heads():
    model = PlateCNN(in_channels=1)

    x = torch.randn(4, 1, 48, 160)

    outputs = model(x)

    assert isinstance(outputs, list)
    assert len(outputs) == 7


def test_plate_cnn_output_shapes_match_class_counts():
    model = PlateCNN(in_channels=1)

    batch_size = 4
    x = torch.randn(batch_size, 1, 48, 160)

    outputs = model(x)

    for output, num_classes in zip(outputs, CLASS_COUNTS):
        assert output.shape == (batch_size, num_classes)


def test_plate_cnn_has_trainable_parameters():
    model = PlateCNN(in_channels=1)

    assert count_trainable_parameters(model) > 0