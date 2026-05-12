from __future__ import annotations

import torch
from torch import nn

from anpr.data.encoders import CLASS_COUNTS, NUM_POSITIONS


def validate_multi_head_outputs(
    outputs: list[torch.Tensor],
    targets: torch.Tensor,
) -> None:
    """
    Validate that model outputs and encoded targets match the expected
    LLDDLLL multi-head structure.
    """
    if not isinstance(outputs, list):
        raise TypeError(f"Expected outputs to be a list, got {type(outputs)!r}.")

    if len(outputs) != NUM_POSITIONS:
        raise ValueError(
            f"Expected {NUM_POSITIONS} output heads, got {len(outputs)}."
        )

    if targets.ndim != 2:
        raise ValueError(
            f"Expected targets shape [batch_size, {NUM_POSITIONS}], "
            f"got {tuple(targets.shape)}."
        )

    if targets.shape[1] != NUM_POSITIONS:
        raise ValueError(
            f"Expected targets to have {NUM_POSITIONS} positions, "
            f"got {targets.shape[1]}."
        )

    batch_size = targets.shape[0]

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
                f"output batch={output.shape[0]}, target batch={batch_size}."
            )

        if output.shape[1] != expected_num_classes:
            raise ValueError(
                f"Class count mismatch at head {position}: "
                f"expected {expected_num_classes}, got {output.shape[1]}."
            )


class PlateMultiHeadLoss(nn.Module):
    """
    CrossEntropyLoss across 7 fixed UK plate character positions.

    Each output head predicts one position of LLDDLLL.

    Returns:
        total_loss: scalar tensor
        position_losses: list of 7 scalar tensors
    """

    def __init__(self) -> None:
        super().__init__()
        self.criterion = nn.CrossEntropyLoss()

    def forward(
        self,
        outputs: list[torch.Tensor],
        targets: torch.Tensor,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        validate_multi_head_outputs(outputs, targets)

        targets = targets.long()

        position_losses: list[torch.Tensor] = []

        for position, output in enumerate(outputs):
            position_target = targets[:, position]
            position_loss = self.criterion(output, position_target)
            position_losses.append(position_loss)

        total_loss = torch.stack(position_losses).mean()

        return total_loss, position_losses