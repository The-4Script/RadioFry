"""VT-CNN2-inspired modulation classifier."""

from __future__ import annotations

import torch
from torch import nn


class ModulationCNN(nn.Module):
    """Classify fixed-length I/Q or engineered signal channels."""

    def __init__(self, input_channels: int = 2, num_classes: int = 11) -> None:
        super().__init__()
        if input_channels not in {2, 4}:
            raise ValueError("input_channels must be 2 or 4")
        self.features = nn.Sequential(
            nn.Conv1d(input_channels, 64, kernel_size=8, padding="same"),
            nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(0.3),
            nn.Conv1d(64, 128, kernel_size=4, padding="same"),
            nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.3),
            nn.Conv1d(128, 128, kernel_size=4, padding="same"),
            nn.BatchNorm1d(128), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(128, 256), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim != 3:
            raise ValueError("inputs must have shape (batch, channels, samples)")
        return self.classifier(self.features(inputs))
