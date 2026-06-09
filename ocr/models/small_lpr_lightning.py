import lightning as L
import torch
import torch.nn as nn

from .small_lpr import SmallLPR


class SmallLPRLightning(L.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.save_hyperparameters()

        self.vocab_size = len(args.chars)

        self.model = SmallLPR(
            vocab_size=self.vocab_size,
            d_model=384,
            max_seq_len=args.max_seq_len,
            start_token_idx=1,
            end_token_idx=2,
            use_pretrained_decoder=True,
        )

        self.criterion = nn.CrossEntropyLoss(ignore_index=0, label_smoothing=0.1)

    def forward(self, images, targets=None):
        return self.model(images, targets)

    def training_step(self, batch, batch_idx):
        images, targets, _ = batch
        logits = self.model(images, targets)
        tgt_output = targets[:, 1:]
        loss = self.criterion(logits.reshape(-1, self.vocab_size), tgt_output.reshape(-1))

        preds = logits.argmax(dim=-1)
        mask = tgt_output != 0
        acc = (preds[mask] == tgt_output[mask]).float().mean()

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("train_acc", acc, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)

        if batch_idx == 0 and self.current_epoch % 10 == 0:
            print(f"\n[TRAIN Epoch {self.current_epoch}]", end=" ")
            self._log_examples(tgt_output[0], preds[0])
            print("")

        return loss

    def validation_step(self, batch, batch_idx):
        images, targets, _ = batch

        logits = self.model(images, targets)
        tgt_output = targets[:, 1:]
        loss = self.criterion(logits.reshape(-1, self.vocab_size), tgt_output.reshape(-1))

        generated = self.model(images)
        gt_content = targets[:, 1:]
        pred_content = generated[:, 1:]

        correct_count = 0
        batch_size = gt_content.size(0)
        for i in range(batch_size):
            if self._decode_seq(gt_content[i]) == self._decode_seq(pred_content[i]):
                correct_count += 1
        acc = torch.tensor(correct_count / batch_size, device=gt_content.device)

        self.log("val_loss", loss, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("val_acc", acc, on_epoch=True, prog_bar=True, sync_dist=True)

        if batch_idx == 0:
            self._log_examples(gt_content[0], pred_content[0])

        return acc

    def _decode_seq(self, tokens):
        res = []
        for c in tokens:
            c = int(c)
            if c == 2:
                break
            if c not in [0, 1]:
                res.append(self.args.chars[c])
        return "".join(res)

    def _log_examples(self, gt, pred):
        gt_str = "".join([self.args.chars[c] for c in gt if c not in [0, 1, 2]])
        pred_str = self._decode_seq(pred)
        print(f"\r[Sample] GT: {gt_str:12} | Pred: {pred_str:12}", end="")

    def configure_optimizers(self):
        """
        3 nhóm differential LR:
        1. Backbone (scratch): LR full
        2. Decoder pretrained (self-attn/FFN): LR thấp (0.05×), nhưng đang frozen nên ít tác dụng
        3. Decoder scratch (cross-attn, embedding, output): LR full
        """
        backbone_params = []
        decoder_pretrained = []
        scratch_params = []

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if name.startswith("backbone") or name.startswith("proj") or name.startswith("pos_enc_2d"):
                backbone_params.append(param)
            elif "decoder.self_attn" in name or "decoder.linear1" in name or "decoder.linear2" in name or "decoder.norm" in name:
                decoder_pretrained.append(param)
            else:
                scratch_params.append(param)

        param_groups = [
            {"params": backbone_params, "lr": self.args.lr, "name": "backbone"},
            {"params": decoder_pretrained, "lr": self.args.lr * 0.05, "name": "decoder_pretrained"},
            {"params": scratch_params, "lr": self.args.lr, "name": "scratch"},
        ]
        param_groups = [g for g in param_groups if len(g["params"]) > 0]

        optimizer = torch.optim.AdamW(param_groups, weight_decay=self.args.weight_decay)

        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=20, T_mult=2, eta_min=1e-6
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch", "frequency": 1},
        }
