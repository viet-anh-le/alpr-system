"""
tracking/train/loss.py — Losses for vehicle Re-ID metric learning.

BatchHardTripletLoss performs online hard mining: for each anchor in the batch
it selects the hardest positive (same class, max distance) and hardest negative
(different class, min distance), which produces more informative gradients than
offline triplet sampling.

CombinedReIDLoss adds a classification CE head alongside triplet loss, which
acts as a regulariser that prevents feature collapse and speeds up convergence.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def pairwise_squared_distance(embs: torch.Tensor) -> torch.Tensor:
    """Compute all-pairs squared Euclidean distances.

    Args:
        embs: (B, D) L2-normalised embeddings
    Returns:
        (B, B) squared distances, clamped to >= 0
    """
    dot = embs @ embs.T
    sq = dot.diagonal()
    return (sq.unsqueeze(1) + sq.unsqueeze(0) - 2.0 * dot).clamp(min=0.0)


class BatchHardTripletLoss(nn.Module):
    """Online batch-hard triplet loss.

    For each anchor:
      hardest positive  = same-class sample with maximum distance
      hardest negative  = different-class sample with minimum distance

    Args:
        margin:  triplet margin (default 0.3)
        squared: operate on squared distances if True (faster, gradient differs)
    Returns:
        (loss, frac_active) where frac_active is the fraction of non-zero triplets,
        useful as a training health metric (should stay > 0 through training).
    """

    def __init__(self, margin: float = 0.3, squared: bool = False) -> None:
        super().__init__()
        self.margin = margin
        self.squared = squared

    def forward(
        self, embs: torch.Tensor, labels: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        dist2 = pairwise_squared_distance(embs)
        B = embs.size(0)
        eye = torch.eye(B, dtype=torch.bool, device=embs.device)

        if self.squared:
            dist = dist2
        else:
            # Zero the diagonal after sqrt so self-distance never enters mining.
            # clamp(min=1e-12) prevents NaN on exact zeros, then we explicitly
            # zero the diagonal to avoid near-inf gradients at sqrt(0).
            dist = dist2.clamp(min=1e-12).sqrt().masked_fill(eye, 0.0)

        same = labels.unsqueeze(0) == labels.unsqueeze(1)  # (B, B) bool
        same_no_diag = same & ~eye  # exclude anchor comparing to itself

        # Hardest positive: max distance among same-class, diagonal excluded
        dist_ap = dist.masked_fill(~same_no_diag, 0.0).max(dim=1)[0]  # (B,)

        # Hardest negative: min distance among different-class
        dist_an = dist.masked_fill(same, 1e9).min(dim=1)[0]  # (B,)

        loss = F.relu(dist_ap - dist_an + self.margin).mean()
        frac_active = (dist_ap - dist_an + self.margin > 0).float().mean()
        return loss, frac_active


class CombinedReIDLoss(nn.Module):
    """BatchHardTripletLoss + weighted CrossEntropy with label smoothing.

    The classification branch (CE loss) shares the embedding weights and acts
    as a discriminative regulariser that prevents features from collapsing and
    speeds up convergence in the early epochs.

    Args:
        margin:          triplet margin
        ce_weight:       weight for CE term (total = triplet + ce_weight * CE)
        label_smoothing: label smoothing for CE (reduces overconfidence)
    """

    def __init__(
        self,
        margin: float = 0.3,
        ce_weight: float = 0.5,
        label_smoothing: float = 0.1,
    ) -> None:
        super().__init__()
        self.triplet = BatchHardTripletLoss(margin=margin)
        self.ce = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        self.ce_weight = ce_weight

    def forward(
        self,
        embs: torch.Tensor,
        logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        loss_tri, frac = self.triplet(embs, labels)
        loss_ce = self.ce(logits, labels)
        total = loss_tri + self.ce_weight * loss_ce
        stats: dict[str, float] = {
            "loss_triplet": loss_tri.item(),
            "loss_ce": loss_ce.item(),
            "frac_active": frac.item(),
        }
        return total, stats
