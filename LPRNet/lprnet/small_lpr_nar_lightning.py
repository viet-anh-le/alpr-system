"""
SmallLPR-NAR Lightning Module

Loss:     CrossEntropy với ignore_index=0 (<pad>) + label smoothing.
Accuracy: Exact-match — chuỗi decode (bỏ pad) khớp 100% ground truth.
Decode:   argmax theo dim class → bỏ pad token (index 0).
"""

from __future__ import annotations

import math
from typing import List, Tuple

import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F

from lprnet.small_lpr_nar import SmallLPRNAR

PAD_IDX: int = 0


class SmallLPRNARLightning(L.LightningModule):
    """
    Lightning wrapper cho SmallLPR-NAR.

    Batch format (từ collate_fn_nar):
        images:          (B, 3, H, W)
        targets_padded:  (B, max_len)   — pad bằng 0
        target_lengths:  (B,)           — độ dài thật của mỗi label
    """

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.save_hyperparameters()

        vocab_size = len(args.chars)          # bao gồm <pad> ở index 0
        d_model    = getattr(args, "d_model", 256)
        backbone_ch = getattr(args, "backbone_ch", 256)
        max_len    = getattr(args, "max_len", 14)
        nhead      = getattr(args, "nhead", 4)
        num_layers = getattr(args, "num_layers", 2)
        dropout    = getattr(args, "dropout", 0.1)
        label_smoothing = getattr(args, "label_smoothing", 0.1)

        self.model = SmallLPRNAR(
            vocab_size=vocab_size,
            d_model=d_model,
            backbone_ch=backbone_ch,
            max_len=max_len,
            nhead=nhead,
            num_layers=num_layers,
            dropout=dropout,
        )

        # CrossEntropy — bỏ qua pad (index 0), label smoothing để tránh overfit
        self.criterion = nn.CrossEntropyLoss(
            ignore_index=PAD_IDX,
            label_smoothing=label_smoothing,
        )

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.model(images)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def training_step(self, batch, batch_idx: int) -> torch.Tensor:
        images, targets, target_lengths = batch
        B, max_len = targets.shape

        logits = self.model(images)      # (B, max_len, vocab_size)

        # CrossEntropy nhận (B*L, V) và (B*L,)
        loss = self.criterion(
            logits.reshape(-1, self.model.vocab_size),
            targets.reshape(-1),
        )

        acc = self._nar_accuracy(logits, targets, target_lengths)

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("train_acc", acc, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)

        if batch_idx == 0 and self.current_epoch % 5 == 0:
            self._log_sample(logits[0:1], targets[0], target_lengths[0], prefix="TRAIN")

        return loss

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validation_step(self, batch, batch_idx: int) -> torch.Tensor:
        images, targets, target_lengths = batch

        logits = self.model(images)
        loss = self.criterion(
            logits.reshape(-1, self.model.vocab_size),
            targets.reshape(-1),
        )
        acc = self._nar_accuracy(logits, targets, target_lengths)

        self.log("val_loss", loss, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("val_acc", acc, on_epoch=True, prog_bar=True, sync_dist=True)

        if batch_idx == 0:
            self._log_sample(logits[0:1], targets[0], target_lengths[0], prefix="VAL")

        return acc

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _nar_accuracy(
        self,
        logits: torch.Tensor,           # (B, max_len, V)
        targets: torch.Tensor,          # (B, max_len) padded
        target_lengths: torch.Tensor,   # (B,)
    ) -> torch.Tensor:
        """
        Exact-match accuracy.
        So sánh chuỗi đoán ra (sau khi bỏ pad) với ground truth.
        """
        preds = logits.argmax(dim=-1)   # (B, max_len)
        correct = 0
        B = preds.size(0)
        for i in range(B):
            length = int(target_lengths[i].item())
            pred_str = self._decode(preds[i], length)
            gt_str = self._decode(targets[i], length)
            if pred_str == gt_str:
                correct += 1
        return torch.tensor(correct / B, dtype=torch.float32)

    def _decode(self, ids: torch.Tensor, length: int) -> str:
        """
        Decode `length` vị trí đầu tiên (không lọ pad).
        Lưu ý: KHÔNG filter PAD_IDX ở giữa chuỗi (chỉ cắt theo `length`).
        Nếu model đoán 0 ở giữa, cượng độ mapping được giữ nguyên để exact-match fail đúng.
        """
        return "".join(
            self.args.chars[int(i)] for i in ids[:length].tolist()
        )

    def _log_sample(
        self,
        logits_one: torch.Tensor,    # (1, max_len, V)
        gt_ids: torch.Tensor,        # (max_len,)
        gt_length: torch.Tensor,
        prefix: str = "SAMPLE",
    ) -> None:
        length = int(gt_length.item())
        pred_str = self._decode(logits_one[0].argmax(-1), length)
        gt_str = self._decode(gt_ids, length)
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
        warmup_epochs = getattr(self.args, "warmup_epochs", 5)
        total_epochs = getattr(self.args, "max_epochs", 200)

        # Linear warmup → CosineAnnealing
        def lr_lambda(epoch: int) -> float:
            if epoch < warmup_epochs:
                return (epoch + 1) / warmup_epochs  # linear 0 → 1
            progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch", "frequency": 1},
        }
