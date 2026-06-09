"""
SmallLPR-CTC Lightning Module

Loss:     nn.CTCLoss(blank=0, zero_infinity=True)
Accuracy: Exact-match (toàn bộ chuỗi decode khớp 100% ground truth)
Decode:   Greedy CTC (argmax → bỏ blank + lặp liên tiếp)
"""

from __future__ import annotations

from typing import List, Tuple

import lightning as L
import torch
import torch.nn as nn

from lprnet.small_lpr_ctc import SmallLPRCTC, ctc_greedy_decode, _T_STEPS


class SmallLPRCTCLightning(L.LightningModule):
    """
    Lightning wrapper cho SmallLPR-CTC.

    Batch format (từ collate_fn_ctc):
        images:         (B, 3, H, W)
        targets_1d:     (sum_lengths,)   — labels nối liên tiếp, không padding
        input_lengths:  (B,)             — đều = 72
        target_lengths: (B,)             — độ dài thật của mỗi label
    """

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.save_hyperparameters()

        vocab_size = len(args.chars)   # bao gồm <blank> ở index 0
        d_model = getattr(args, "d_model", 256)
        backbone_ch = getattr(args, "backbone_ch", 256)

        self.model = SmallLPRCTC(
            vocab_size=vocab_size,
            d_model=d_model,
            backbone_ch=backbone_ch,
        )

        # blank=0 bắt buộc khớp với index 0 trong charset
        self.criterion = nn.CTCLoss(blank=0, zero_infinity=True)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.model(images)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def training_step(self, batch, batch_idx: int) -> torch.Tensor:
        images, targets_1d, input_lengths, target_lengths = batch
        B = images.size(0)

        logits = self.model(images)                          # (B, T, C)
        log_probs = logits.log_softmax(2).permute(1, 0, 2)  # (T, B, C) cho CTCLoss

        loss = self.criterion(log_probs, targets_1d, input_lengths, target_lengths)

        acc = self._ctc_accuracy(logits, targets_1d, target_lengths)

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("train_acc", acc, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)

        if batch_idx == 0 and self.current_epoch % 5 == 0:
            self._log_sample(logits[0:1], targets_1d[: target_lengths[0]], prefix="TRAIN")

        return loss

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validation_step(self, batch, batch_idx: int) -> torch.Tensor:
        images, targets_1d, input_lengths, target_lengths = batch

        logits = self.model(images)
        log_probs = logits.log_softmax(2).permute(1, 0, 2)

        loss = self.criterion(log_probs, targets_1d, input_lengths, target_lengths)
        acc = self._ctc_accuracy(logits, targets_1d, target_lengths)

        self.log("val_loss", loss, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("val_acc", acc, on_epoch=True, prog_bar=True, sync_dist=True)

        if batch_idx == 0:
            self._log_sample(logits[0:1], targets_1d[: target_lengths[0]], prefix="VAL")

        return acc

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _ctc_accuracy(
        self,
        logits: torch.Tensor,           # (B, T, C)
        targets_1d: torch.Tensor,       # (sum_lengths,)
        target_lengths: torch.Tensor,   # (B,)
    ) -> torch.Tensor:
        """Exact-match accuracy: chuỗi decode hoàn toàn khớp GT."""
        preds = ctc_greedy_decode(logits, self.args.chars)   # list[str]
        correct = 0
        offset = 0
        for i, pred_str in enumerate(preds):
            length = int(target_lengths[i].item())
            gt_indices = targets_1d[offset : offset + length].tolist()
            gt_str = "".join(self.args.chars[idx] for idx in gt_indices)
            offset += length
            if pred_str == gt_str:
                correct += 1
        return torch.tensor(correct / len(preds), dtype=torch.float32)

    def _log_sample(
        self,
        logits_one: torch.Tensor,   # (1, T, C)
        gt_indices: torch.Tensor,   # (label_len,)
        prefix: str = "SAMPLE",
    ) -> None:
        pred_str = ctc_greedy_decode(logits_one, self.args.chars)[0]
        gt_str = "".join(self.args.chars[int(idx)] for idx in gt_indices)
        print(f"\r[{prefix} Ep{self.current_epoch}] GT: {gt_str:14} | Pred: {pred_str:14}", end="")

    # ------------------------------------------------------------------
    # Optimizer
    # ------------------------------------------------------------------

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.args.lr,
            weight_decay=self.args.weight_decay,
        )
        scheduler_name = getattr(self.args, "scheduler", "cosine")
        min_lr = getattr(self.args, "min_lr", 1e-6)

        if scheduler_name == "cosine_warm_restarts":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer,
                T_0=getattr(self.args, "scheduler_t0", 20),
                T_mult=getattr(self.args, "scheduler_t_mult", 2),
                eta_min=min_lr,
            )
        elif scheduler_name == "constant":
            return optimizer
        else:
            t_max = max(1, int(getattr(self.args, "scheduler_t_max", self.args.max_epochs)))
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=t_max,
                eta_min=min_lr,
            )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch", "frequency": 1},
        }
