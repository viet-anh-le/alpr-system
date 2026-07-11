from __future__ import annotations

import torch
import torch.nn as nn


class YOLOCustomHead(nn.Module):
    """Custom detection head for Vietnamese license plates."""

    def __init__(self, in_channels: int, num_classes: int = 1) -> None:
        super().__init__()
        self.num_classes = num_classes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
