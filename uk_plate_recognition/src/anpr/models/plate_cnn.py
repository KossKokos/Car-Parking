from __future__ import annotations

import torch
from torch import nn

from anpr.data.encoders import CLASS_COUNTS


class ConvBlock(nn.Module):
    """
    Simple convolutional block for the baseline CNN.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        layers: list[nn.Module] = [
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]

        if dropout > 0:
            layers.append(nn.Dropout2d(dropout))

        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class PlateCNN(nn.Module):
    """
    Baseline CNN recogniser for fixed-format UK plates: LLDDLLL.

    This version keeps horizontal spatial information before the final heads.
    That is important because plate recognition is position-sensitive.

    Input:
        Tensor[B, C, H, W]

    Output:
        list of 7 tensors:
            position 0 -> Tensor[B, 26]
            position 1 -> Tensor[B, 26]
            position 2 -> Tensor[B, 10]
            position 3 -> Tensor[B, 10]
            position 4 -> Tensor[B, 26]
            position 5 -> Tensor[B, 26]
            position 6 -> Tensor[B, 26]
    """

    def __init__(
        self,
        in_channels: int = 1,
        class_counts: tuple[int, ...] = CLASS_COUNTS,
        dropout: float = 0.2,
        pooled_width: int = 10,
    ) -> None:
        super().__init__()

        if len(class_counts) != 7:
            raise ValueError(
                f"Expected 7 output heads for LLDDLLL format, got {len(class_counts)}."
            )

        if pooled_width < 1:
            raise ValueError("pooled_width must be >= 1.")

        self.class_counts = class_counts
        self.pooled_width = pooled_width
        feature_dim = 256 * pooled_width

        self.features = nn.Sequential(
            ConvBlock(in_channels, 32),
            nn.MaxPool2d(kernel_size=2),  # 48x160 -> 24x80

            ConvBlock(32, 64),
            nn.MaxPool2d(kernel_size=2),  # 24x80 -> 12x40

            ConvBlock(64, 128, dropout=0.05),
            nn.MaxPool2d(kernel_size=2),  # 12x40 -> 6x20

            ConvBlock(128, 256, dropout=0.05),

            # Keep horizontal structure instead of collapsing everything to 1x1.
            nn.AdaptiveAvgPool2d((1, pooled_width)),
            nn.Flatten(),
            nn.Dropout(dropout),
        )

        self.heads = nn.ModuleList(
            [nn.Linear(feature_dim, num_classes) for num_classes in class_counts]
        )

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        features = self.features(x)

        return [head(features) for head in self.heads]


def count_trainable_parameters(model: nn.Module) -> int:
    """
    Count trainable model parameters.
    """
    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )