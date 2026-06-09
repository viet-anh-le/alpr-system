from __future__ import annotations

import torch
import torch.nn as nn


class CTCLoss(nn.Module):
    """CTC loss wrapper with blank token handling."""

    def __init__(self, blank: int = 0) -> None:
        super().__init__()
        self.criterion = nn.CTCLoss(blank=blank, reduction="mean", zero_infinity=True)

    def forward(self, log_probs: torch.Tensor, targets: torch.Tensor, input_lengths: torch.Tensor, target_lengths: torch.Tensor) -> torch.Tensor:
        return self.criterion(log_probs, targets, input_lengths, target_lengths)
