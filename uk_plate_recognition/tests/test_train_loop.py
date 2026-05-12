import torch

from anpr.data.encoders import CLASS_COUNTS
from anpr.models.plate_cnn import PlateCNN
from anpr.training.losses import PlateMultiHeadLoss
from anpr.training.train_loop import (
    EpochResult,
    evaluate_one_epoch,
    train_one_epoch,
)


def make_fake_targets(batch_size: int = 4) -> torch.Tensor:
    columns = []

    for num_classes in CLASS_COUNTS:
        columns.append(torch.randint(0, num_classes, size=(batch_size,)))

    return torch.stack(columns, dim=1)


def make_fake_batch(batch_size: int = 4) -> dict:
    return {
        "image": torch.randn(batch_size, 1, 48, 160),
        "target": make_fake_targets(batch_size),
        "label": ["AA04QZH" for _ in range(batch_size)],
        "image_path": [f"fake_{index}.png" for index in range(batch_size)],
    }


def make_fake_dataloader(num_batches: int = 2, batch_size: int = 4):
    return [
        make_fake_batch(batch_size=batch_size)
        for _ in range(num_batches)
    ]


def test_train_one_epoch_returns_epoch_result():
    model = PlateCNN(in_channels=1)
    loss_fn = PlateMultiHeadLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    dataloader = make_fake_dataloader(num_batches=2, batch_size=4)

    result = train_one_epoch(
        model=model,
        dataloader=dataloader,
        loss_fn=loss_fn,
        optimizer=optimizer,
        device="cpu",
        show_progress=False,
    )

    assert isinstance(result, EpochResult)
    assert result.loss > 0
    assert len(result.position_losses) == 7
    assert result.metrics is None


def test_evaluate_one_epoch_returns_epoch_result_with_metrics():
    model = PlateCNN(in_channels=1)
    loss_fn = PlateMultiHeadLoss()

    dataloader = make_fake_dataloader(num_batches=2, batch_size=4)

    result = evaluate_one_epoch(
        model=model,
        dataloader=dataloader,
        loss_fn=loss_fn,
        device="cpu",
        show_progress=False,
    )

    assert isinstance(result, EpochResult)
    assert result.loss > 0
    assert len(result.position_losses) == 7
    assert result.metrics is not None
    assert result.metrics.num_samples == 8
    assert 0.0 <= result.metrics.full_plate_accuracy <= 1.0
    assert 0.0 <= result.metrics.regex_valid_rate <= 1.0
    assert 0.0 <= result.metrics.average_confidence <= 1.0


def test_train_one_epoch_respects_max_batches():
    model = PlateCNN(in_channels=1)
    loss_fn = PlateMultiHeadLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    dataloader = make_fake_dataloader(num_batches=5, batch_size=4)

    result = train_one_epoch(
        model=model,
        dataloader=dataloader,
        loss_fn=loss_fn,
        optimizer=optimizer,
        device="cpu",
        max_batches=1,
        show_progress=False,
    )

    assert isinstance(result, EpochResult)
    assert result.loss > 0


def test_evaluate_one_epoch_respects_max_batches():
    model = PlateCNN(in_channels=1)
    loss_fn = PlateMultiHeadLoss()

    dataloader = make_fake_dataloader(num_batches=5, batch_size=4)

    result = evaluate_one_epoch(
        model=model,
        dataloader=dataloader,
        loss_fn=loss_fn,
        device="cpu",
        max_batches=1,
        show_progress=False,
    )

    assert isinstance(result, EpochResult)
    assert result.metrics is not None
    assert result.metrics.num_samples == 4