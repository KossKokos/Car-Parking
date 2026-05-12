from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
from torch import nn
from tqdm.auto import tqdm

from anpr.data.encoders import NUM_POSITIONS
from anpr.evaluation.metrics import PlateMetrics, calculate_plate_metrics
from anpr.inference.decode import decode_model_outputs
from anpr.training.losses import PlateMultiHeadLoss


@dataclass(frozen=True)
class EpochResult:
    """
    Summary of one training or evaluation epoch.
    """

    loss: float
    position_losses: list[float]
    metrics: PlateMetrics | None = None


def _move_batch_to_device(
    batch: dict,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    images = batch["image"].to(device)
    targets = batch["target"].to(device)

    return images, targets


def _average_position_losses(
    position_loss_totals: list[float],
    num_samples: int,
) -> list[float]:
    return [
        position_loss_total / num_samples
        for position_loss_total in position_loss_totals
    ]


def train_one_epoch(
    model: nn.Module,
    dataloader: Iterable[dict],
    loss_fn: PlateMultiHeadLoss,
    optimizer: torch.optim.Optimizer,
    device: str | torch.device,
    max_batches: int | None = None,
    show_progress: bool = True,
) -> EpochResult:
    """
    Train model for one epoch.

    Returns:
        EpochResult with average total loss and average per-position losses.
    """
    device = torch.device(device)
    model.train()

    total_loss = 0.0
    position_loss_totals = [0.0 for _ in range(NUM_POSITIONS)]
    num_samples = 0

    iterator = tqdm(
        dataloader,
        desc="Training",
        leave=False,
    ) if show_progress else dataloader

    for batch_index, batch in enumerate(iterator):
        if max_batches is not None and batch_index >= max_batches:
            break

        images, targets = _move_batch_to_device(batch, device)
        batch_size = images.shape[0]

        optimizer.zero_grad(set_to_none=True)

        outputs = model(images)
        loss, position_losses = loss_fn(outputs, targets)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * batch_size

        for position, position_loss in enumerate(position_losses):
            position_loss_totals[position] += position_loss.item() * batch_size

        num_samples += batch_size

    if num_samples == 0:
        raise ValueError("No samples were processed during training.")

    return EpochResult(
        loss=total_loss / num_samples,
        position_losses=_average_position_losses(
            position_loss_totals,
            num_samples,
        ),
        metrics=None,
    )


def evaluate_one_epoch(
    model: nn.Module,
    dataloader: Iterable[dict],
    loss_fn: PlateMultiHeadLoss,
    device: str | torch.device,
    max_batches: int | None = None,
    show_progress: bool = True,
) -> EpochResult:
    """
    Evaluate model for one epoch.

    Returns:
        EpochResult with loss, per-position losses, and plate metrics.
    """
    device = torch.device(device)
    model.eval()

    total_loss = 0.0
    position_loss_totals = [0.0 for _ in range(NUM_POSITIONS)]
    num_samples = 0

    all_predictions = []
    all_labels: list[str] = []

    iterator = tqdm(
        dataloader,
        desc="Evaluating",
        leave=False,
    ) if show_progress else dataloader

    with torch.no_grad():
        for batch_index, batch in enumerate(iterator):
            if max_batches is not None and batch_index >= max_batches:
                break

            images, targets = _move_batch_to_device(batch, device)
            batch_size = images.shape[0]

            outputs = model(images)
            loss, position_losses = loss_fn(outputs, targets)

            predictions = decode_model_outputs(outputs)

            labels = [str(label) for label in batch["label"]]

            all_predictions.extend(predictions)
            all_labels.extend(labels)

            total_loss += loss.item() * batch_size

            for position, position_loss in enumerate(position_losses):
                position_loss_totals[position] += position_loss.item() * batch_size

            num_samples += batch_size

    if num_samples == 0:
        raise ValueError("No samples were processed during evaluation.")

    metrics = calculate_plate_metrics(
        predictions=all_predictions,
        labels=all_labels,
    )

    return EpochResult(
        loss=total_loss / num_samples,
        position_losses=_average_position_losses(
            position_loss_totals,
            num_samples,
        ),
        metrics=metrics,
    )