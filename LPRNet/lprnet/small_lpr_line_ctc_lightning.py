"""
Lightning module for SmallLPR-Line-CTC.
"""

from __future__ import annotations

from typing import List

import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F

from lprnet.small_lpr_line_ctc import SmallLPRLineCTC, ctc_decode_logits, line_ctc_greedy_decode
from lprnet.small_lpr_line_ctc_datamodule import LAYOUT_IGNORE_INDEX


def masked_layout_cross_entropy(
    layout_logits: torch.Tensor,
    layout_labels: torch.Tensor,
    *,
    ignore_index: int = LAYOUT_IGNORE_INDEX,
) -> torch.Tensor:
    valid = layout_labels != ignore_index
    if not bool(valid.any()):
        return layout_logits.sum() * 0.0
    return F.cross_entropy(layout_logits[valid], layout_labels[valid])


class SmallLPRLineCTCLightning(L.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.save_hyperparameters()

        vocab_size = len(args.chars)
        self.use_global_head = bool(getattr(args, "use_global_head", True))
        self.model = SmallLPRLineCTC(
            vocab_size=vocab_size,
            d_model=int(getattr(args, "d_model", 256)),
            backbone_ch=int(getattr(args, "backbone_ch", 256)),
            line_prior_strength=float(getattr(args, "line_prior_strength", 1.0)),
            use_stn=bool(getattr(args, "use_stn", True)),
            use_pos_enc=bool(getattr(args, "use_pos_enc", True)),
            use_global_head=self.use_global_head,
        )
        self.ctc = nn.CTCLoss(blank=0, zero_infinity=True)
        self.global_loss_weight = float(getattr(args, "global_loss_weight", 1.0))
        self.one_line_loss_weight = float(getattr(args, "one_line_loss_weight", 1.0))
        self.top_loss_weight = float(getattr(args, "top_loss_weight", 1.0))
        self.bottom_loss_weight = float(getattr(args, "bottom_loss_weight", 1.0))
        self.layout_loss_weight = float(getattr(args, "layout_loss_weight", 0.2))
        self.two_line_threshold = float(getattr(args, "two_line_threshold", 0.5))
        self.line_separator = getattr(args, "line_separator", "[SEP]")
        self.decode_mode = getattr(args, "decode_mode", "layout")
        if self.decode_mode not in {"global", "layout"}:
            raise ValueError("decode_mode must be either 'global' or 'layout'")
        if not self.use_global_head and self.global_loss_weight != 0.0:
            raise ValueError("Disabling the global head requires global_loss_weight=0.")
        if not self.use_global_head and self.decode_mode == "global":
            raise ValueError("decode_mode='global' requires the global head to be enabled.")

    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        return self.model(images)

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        outputs = self.model(batch["images"])
        losses = self._losses(outputs, batch)
        loss = losses["loss"]
        batch_size = batch["images"].size(0)

        self.log(
            "train_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
            batch_size=batch_size,
        )
        if self.use_global_head:
            self.log(
                "train_global_ctc_loss",
                losses["global_ctc_loss"],
                on_epoch=True,
                sync_dist=True,
                batch_size=batch_size,
            )
        self.log(
            "train_one_line_ctc_loss",
            losses["one_line_ctc_loss"],
            on_epoch=True,
            sync_dist=True,
            batch_size=batch_size,
        )
        self.log(
            "train_top_ctc_loss",
            losses["top_ctc_loss"],
            on_epoch=True,
            sync_dist=True,
            batch_size=batch_size,
        )
        self.log(
            "train_bottom_ctc_loss",
            losses["bottom_ctc_loss"],
            on_epoch=True,
            sync_dist=True,
            batch_size=batch_size,
        )
        self.log(
            "train_layout_loss",
            losses["layout_loss"],
            on_epoch=True,
            sync_dist=True,
            batch_size=batch_size,
        )
        self.log(
            "train_layout_acc",
            self._layout_accuracy(outputs, batch),
            on_epoch=True,
            sync_dist=True,
            batch_size=batch_size,
        )

        if batch_idx == 0 and self.current_epoch % 5 == 0:
            self._log_sample(outputs, batch, prefix="TRAIN")
        return loss

    def validation_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        outputs = self.model(batch["images"])
        losses = self._losses(outputs, batch)
        val_acc = self._line_accuracy(outputs, batch)
        layout_acc = self._layout_accuracy(outputs, batch)
        batch_size = batch["images"].size(0)

        self.log(
            "val_loss",
            losses["loss"],
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
            batch_size=batch_size,
        )
        self.log(
            "val_acc",
            val_acc,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
            batch_size=batch_size,
        )
        if self.use_global_head:
            self.log(
                "val_global_acc",
                self._global_accuracy(outputs, batch),
                on_epoch=True,
                sync_dist=True,
                batch_size=batch_size,
            )
        self.log(
            "val_layout_acc",
            layout_acc,
            on_epoch=True,
            sync_dist=True,
            batch_size=batch_size,
        )
        if self.use_global_head:
            self.log(
                "val_global_ctc_loss",
                losses["global_ctc_loss"],
                on_epoch=True,
                sync_dist=True,
                batch_size=batch_size,
            )
        self.log(
            "val_one_line_ctc_loss",
            losses["one_line_ctc_loss"],
            on_epoch=True,
            sync_dist=True,
            batch_size=batch_size,
        )
        self.log(
            "val_top_ctc_loss",
            losses["top_ctc_loss"],
            on_epoch=True,
            sync_dist=True,
            batch_size=batch_size,
        )
        self.log(
            "val_bottom_ctc_loss",
            losses["bottom_ctc_loss"],
            on_epoch=True,
            sync_dist=True,
            batch_size=batch_size,
        )
        self.log(
            "val_layout_loss",
            losses["layout_loss"],
            on_epoch=True,
            sync_dist=True,
            batch_size=batch_size,
        )

        if batch_idx == 0:
            self._log_sample(outputs, batch, prefix="VAL")
        return val_acc

    def _losses(self, outputs: dict[str, torch.Tensor], batch: dict) -> dict[str, torch.Tensor]:
        if self.use_global_head:
            global_loss = self._ctc_loss(
                outputs["global_logits"],
                batch["global_targets"],
                batch["global_input_lengths"],
                batch["global_lengths"],
            )
        else:
            global_loss = outputs["one_line_logits"].sum() * 0.0
        one_line_loss = self._masked_ctc_loss(
            outputs["one_line_logits"],
            batch["one_line_targets"],
            batch["one_line_lengths"],
            batch["one_line_loss_mask"],
        )
        top_loss = self._masked_ctc_loss(
            outputs["top_logits"],
            batch["top_targets"],
            batch["top_lengths"],
            batch["top_loss_mask"],
        )
        bottom_loss = self._masked_ctc_loss(
            outputs["bottom_logits"],
            batch["bottom_targets"],
            batch["bottom_lengths"],
            batch["bottom_loss_mask"],
        )
        layout_loss = masked_layout_cross_entropy(
            outputs["layout_logits"],
            batch["layout_labels"].to(outputs["layout_logits"].device),
        )
        total = (
            self.global_loss_weight * global_loss
            + self.one_line_loss_weight * one_line_loss
            + self.top_loss_weight * top_loss
            + self.bottom_loss_weight * bottom_loss
            + self.layout_loss_weight * layout_loss
        )
        return {
            "loss": total,
            "global_ctc_loss": global_loss,
            "one_line_ctc_loss": one_line_loss,
            "top_ctc_loss": top_loss,
            "bottom_ctc_loss": bottom_loss,
            "layout_loss": layout_loss,
        }

    def _ctc_loss(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        input_lengths: torch.Tensor,
        target_lengths: torch.Tensor,
    ) -> torch.Tensor:
        log_probs = logits.log_softmax(2).permute(1, 0, 2)
        return self.ctc(
            log_probs,
            targets.to(logits.device),
            input_lengths.to(logits.device),
            target_lengths.to(logits.device),
        )

    def _masked_ctc_loss(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        target_lengths: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        selected_targets, selected_lengths, selected_mask = _select_ctc_targets(
            targets.to(logits.device),
            target_lengths.to(logits.device),
            mask.to(logits.device),
        )
        if selected_mask.numel() == 0:
            return logits.sum() * 0.0
        selected_logits = logits[selected_mask]
        input_lengths = torch.full(
            (selected_logits.size(0),),
            selected_logits.size(1),
            dtype=torch.long,
            device=logits.device,
        )
        return self._ctc_loss(selected_logits, selected_targets, input_lengths, selected_lengths)

    def _line_accuracy(self, outputs: dict[str, torch.Tensor], batch: dict) -> torch.Tensor:
        if self.decode_mode == "global":
            preds = ctc_decode_logits(outputs["global_logits"], self.args.chars)
        else:
            preds = line_ctc_greedy_decode(
                outputs,
                self.args.chars,
                two_line_threshold=self.two_line_threshold,
                line_separator=self.line_separator,
            )
        return _exact_match_accuracy(preds, batch["texts"], outputs["layout_logits"].device)

    def _global_accuracy(self, outputs: dict[str, torch.Tensor], batch: dict) -> torch.Tensor:
        preds = ctc_decode_logits(outputs["global_logits"], self.args.chars)
        return _exact_match_accuracy(preds, batch["texts"], outputs["global_logits"].device)

    def _layout_accuracy(self, outputs: dict[str, torch.Tensor], batch: dict) -> torch.Tensor:
        labels = batch["layout_labels"].to(outputs["layout_logits"].device)
        valid = labels != LAYOUT_IGNORE_INDEX
        if not bool(valid.any()):
            return outputs["layout_logits"].sum() * 0.0
        preds = outputs["layout_logits"].argmax(dim=-1)
        return (preds[valid] == labels[valid]).float().mean()

    def _log_sample(self, outputs: dict[str, torch.Tensor], batch: dict, prefix: str) -> None:
        if self.decode_mode == "global":
            pred = ctc_decode_logits(outputs["global_logits"][0:1], self.args.chars)[0]
        else:
            pred = line_ctc_greedy_decode(
                {key: value[0:1] for key, value in outputs.items()},
                self.args.chars,
                two_line_threshold=self.two_line_threshold,
                line_separator=self.line_separator,
            )[0]
        gt = batch["texts"][0]
        layout_prob = torch.softmax(outputs["layout_logits"][0], dim=-1)[1].detach().cpu().item()
        print(
            f"\r[{prefix} Ep{self.current_epoch}] GT: {gt:16} | Pred: {pred:16} | p2={layout_prob:.3f}",
            end="",
        )

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.args.lr,
            weight_decay=self.args.weight_decay,
        )
        scheduler_name = getattr(self.args, "scheduler", "cosine")
        min_lr = getattr(self.args, "min_lr", 1e-6)
        if scheduler_name == "constant":
            return optimizer
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


def _select_ctc_targets(
    targets: torch.Tensor,
    lengths: torch.Tensor,
    mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    pieces: List[torch.Tensor] = []
    selected_lengths: List[int] = []
    selected_indices: List[int] = []
    offset = 0
    for idx, length_tensor in enumerate(lengths.tolist()):
        length = int(length_tensor)
        keep = bool(mask[idx].item()) and length > 0
        if keep:
            pieces.append(targets[offset : offset + length])
            selected_lengths.append(length)
            selected_indices.append(idx)
        offset += length

    if not selected_indices:
        return (
            torch.empty((0,), dtype=torch.long, device=targets.device),
            torch.empty((0,), dtype=torch.long, device=targets.device),
            torch.empty((0,), dtype=torch.long, device=mask.device),
        )
    return (
        torch.cat(pieces, dim=0),
        torch.tensor(selected_lengths, dtype=torch.long, device=targets.device),
        torch.tensor(selected_indices, dtype=torch.long, device=mask.device),
    )


def _exact_match_accuracy(preds: List[str], targets: List[str], device: torch.device) -> torch.Tensor:
    if not preds:
        return torch.tensor(0.0, dtype=torch.float32, device=device)
    correct = sum(pred == target for pred, target in zip(preds, targets, strict=False))
    return torch.tensor(correct / len(preds), dtype=torch.float32, device=device)


__all__ = ["SmallLPRLineCTCLightning", "masked_layout_cross_entropy"]
