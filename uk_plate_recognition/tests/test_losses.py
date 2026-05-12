import pytest
import torch

from anpr.data.encoders import CLASS_COUNTS
from anpr.training.losses import PlateMultiHeadLoss, validate_multi_head_outputs


def make_fake_outputs(batch_size: int = 4):
    return [
        torch.randn(batch_size, num_classes, requires_grad=True)
        for num_classes in CLASS_COUNTS
    ]


def make_fake_targets(batch_size: int = 4):
    columns = []

    for num_classes in CLASS_COUNTS:
        columns.append(torch.randint(0, num_classes, size=(batch_size,)))

    return torch.stack(columns, dim=1)


def test_validate_multi_head_outputs_accepts_valid_shapes():
    outputs = make_fake_outputs(batch_size=4)
    targets = make_fake_targets(batch_size=4)

    validate_multi_head_outputs(outputs, targets)


def test_plate_multi_head_loss_returns_scalar_and_position_losses():
    outputs = make_fake_outputs(batch_size=4)
    targets = make_fake_targets(batch_size=4)

    loss_fn = PlateMultiHeadLoss()

    total_loss, position_losses = loss_fn(outputs, targets)

    assert total_loss.ndim == 0
    assert len(position_losses) == 7

    for position_loss in position_losses:
        assert position_loss.ndim == 0


def test_plate_multi_head_loss_supports_backward():
    outputs = make_fake_outputs(batch_size=4)
    targets = make_fake_targets(batch_size=4)

    loss_fn = PlateMultiHeadLoss()

    total_loss, _ = loss_fn(outputs, targets)
    total_loss.backward()

    for output in outputs:
        assert output.grad is not None


def test_rejects_wrong_number_of_heads():
    outputs = make_fake_outputs(batch_size=4)[:6]
    targets = make_fake_targets(batch_size=4)

    loss_fn = PlateMultiHeadLoss()

    with pytest.raises(ValueError):
        loss_fn(outputs, targets)


def test_rejects_wrong_target_shape():
    outputs = make_fake_outputs(batch_size=4)
    targets = torch.randint(0, 10, size=(4,))

    loss_fn = PlateMultiHeadLoss()

    with pytest.raises(ValueError):
        loss_fn(outputs, targets)


def test_rejects_wrong_class_count():
    outputs = make_fake_outputs(batch_size=4)
    outputs[2] = torch.randn(4, 26, requires_grad=True)  # position 2 should be digit head: 10

    targets = make_fake_targets(batch_size=4)

    loss_fn = PlateMultiHeadLoss()

    with pytest.raises(ValueError):
        loss_fn(outputs, targets)