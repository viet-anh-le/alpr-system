from __future__ import annotations

from typing import Any

import lightning as L
import torch
import torch.nn as nn

from ocr.models.slot_lpr import SlotLPR
from ocr.train.slot_lpr_datamodule import decode_slot_tokens


class SlotLPRLightning(L.LightningModule):
    def __init__(self, args: Any) -> None:
        super().__init__()
        self.args = args
        self.save_hyperparameters(vars(args) if hasattr(args, "__dict__") else args)

        self.vocab_size = len(args.chars)
        self.model = SlotLPR(
            vocab_size=self.vocab_size,
            max_slots=args.max_slots,
            d_model=args.d_model,
            decoder_layers=args.decoder_layers,
            nhead=getattr(args, "nhead", 4),
            dropout=getattr(args, "dropout_rate", 0.1),
            use_stn=getattr(args, "use_stn", True),
        )

        self.slot_criterion = nn.CrossEntropyLoss(
            ignore_index=0,
            label_smoothing=float(getattr(args, "label_smoothing", 0.05)),
        )
        self.layout_criterion = nn.CrossEntropyLoss()
        self.layout_loss_weight = float(getattr(args, "layout_loss_weight", 0.2))

    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        return self.model(images)

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        outputs = self.model(batch["image"])
        loss, metrics = self._shared_loss(outputs, batch)
        batch_size = batch["image"].size(0)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=batch_size)
        self.log("train_slot_loss", metrics["slot_loss"], on_epoch=True, sync_dist=True, batch_size=batch_size)
        self.log("train_layout_loss", metrics["layout_loss"], on_epoch=True, sync_dist=True, batch_size=batch_size)
        self.log("train_char_acc", metrics["char_acc"], on_step=True, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=batch_size)
        self.log("train_eos_acc", metrics["eos_acc"], on_epoch=True, sync_dist=True, batch_size=batch_size)
        self.log("train_layout_acc", metrics["layout_acc"], on_epoch=True, sync_dist=True, batch_size=batch_size)
        return loss

    def validation_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        outputs = self.model(batch["image"])
        loss, metrics = self._shared_loss(outputs, batch)
        exact_acc = self._exact_match(outputs["slot_logits"], batch["text"])
        batch_size = batch["image"].size(0)

        self.log("val_loss", loss, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=batch_size)
        self.log("val_slot_loss", metrics["slot_loss"], on_epoch=True, sync_dist=True, batch_size=batch_size)
        self.log("val_layout_loss", metrics["layout_loss"], on_epoch=True, sync_dist=True, batch_size=batch_size)
        self.log("val_char_acc", metrics["char_acc"], on_epoch=True, prog_bar=True, sync_dist=True, batch_size=batch_size)
        self.log("val_eos_acc", metrics["eos_acc"], on_epoch=True, prog_bar=True, sync_dist=True, batch_size=batch_size)
        self.log("val_layout_acc", metrics["layout_acc"], on_epoch=True, sync_dist=True, batch_size=batch_size)
        self.log("val_acc", exact_acc, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=batch_size)

        if batch_idx == 0:
            pred = self.decode_logits(outputs["slot_logits"][:1])[0]
            print(f"\n[VAL] GT: {batch['text'][0]:13} | Pred: {pred:13}")
        return loss

    def test_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        outputs = self.model(batch["image"])
        loss, metrics = self._shared_loss(outputs, batch)
        exact_acc = self._exact_match(outputs["slot_logits"], batch["text"])
        batch_size = batch["image"].size(0)
        self.log("test_loss", loss, on_epoch=True, sync_dist=True, batch_size=batch_size)
        self.log("test_char_acc", metrics["char_acc"], on_epoch=True, sync_dist=True, batch_size=batch_size)
        self.log("test_eos_acc", metrics["eos_acc"], on_epoch=True, sync_dist=True, batch_size=batch_size)
        self.log("test_layout_acc", metrics["layout_acc"], on_epoch=True, sync_dist=True, batch_size=batch_size)
        self.log("test_acc", exact_acc, on_epoch=True, sync_dist=True, batch_size=batch_size)
        return loss

    def _shared_loss(
        self,
        outputs: dict[str, torch.Tensor],
        batch: dict,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        slot_logits = outputs["slot_logits"]
        layout_logits = outputs["layout_logits"]
        slots = batch["slots"]
        layout = batch["layout"]

        slot_loss = self.slot_criterion(
            slot_logits.reshape(-1, self.vocab_size),
            slots.reshape(-1),
        )
        layout_loss = self.layout_criterion(layout_logits, layout)
        loss = slot_loss + self.layout_loss_weight * layout_loss

        pred_slots = slot_logits.argmax(dim=-1)
        content_mask = (slots != 0) & (slots != 2)
        char_acc = (pred_slots[content_mask] == slots[content_mask]).float().mean()
        eos_mask = slots == 2
        eos_acc = (pred_slots[eos_mask] == 2).float().mean()
        layout_acc = (layout_logits.argmax(dim=-1) == layout).float().mean()
        return loss, {
            "slot_loss": slot_loss.detach(),
            "layout_loss": layout_loss.detach(),
            "char_acc": char_acc.detach(),
            "eos_acc": eos_acc.detach(),
            "layout_acc": layout_acc.detach(),
        }

    def decode_logits(self, slot_logits: torch.Tensor) -> list[str]:
        pred_slots = slot_logits.argmax(dim=-1).detach().cpu()
        return [decode_slot_tokens(row, self.args.chars) for row in pred_slots]

    def _exact_match(self, slot_logits: torch.Tensor, texts: list[str]) -> torch.Tensor:
        preds = self.decode_logits(slot_logits)
        correct = sum(pred == target for pred, target in zip(preds, texts))
        return torch.tensor(correct / max(1, len(texts)), device=slot_logits.device)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=float(self.args.lr),
            weight_decay=float(self.args.weight_decay),
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=int(getattr(self.args, "scheduler_t0", 20)),
            T_mult=2,
            eta_min=float(getattr(self.args, "eta_min", 1e-6)),
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch", "frequency": 1},
        }
