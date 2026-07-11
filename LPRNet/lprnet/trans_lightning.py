import lightning as L
import torch
import torch.nn as nn
from lprnet import TransLPRNet


class TransLightningModule(L.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.save_hyperparameters()

        # vocab_size includes PAD, SOS, EOS
        self.vocab_size = len(args.chars)

        self.model = TransLPRNet(
            vocab_size=self.vocab_size,
            target_size=args.img_size,
            max_seq_len=args.max_seq_len,
            start_token_idx=1,  # <SOS> is at index 1
            use_pretrained=True,
        )

        # Label Smoothing added as requested
        self.criterion = nn.CrossEntropyLoss(
            ignore_index=0, label_smoothing=0.1
        )  # <PAD> is at index 0

    def forward(self, images, targets=None):
        return self.model(images, targets)

    def training_step(self, batch, batch_idx):
        images, targets, _ = batch

        logits = self.model(images, targets)
        tgt_output = targets[:, 1:]

        loss = self.criterion(logits.reshape(-1, self.vocab_size), tgt_output.reshape(-1))

        # Calculate Token-level Accuracy for training (fast)
        preds = logits.argmax(dim=-1)
        mask = tgt_output != 0  # Ignore PAD tokens
        acc = (preds[mask] == tgt_output[mask]).float().mean()

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("train_acc", acc, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)

        # --- THÊM ĐOẠN NÀY ĐỂ XEM DỰ ĐOÁN LÚC TRAIN ---
        # Chỉ in ở batch đầu tiên và mỗi 10 epoch in 1 lần để tránh trôi màn hình
        if batch_idx == 0 and self.current_epoch % 10 == 0:
            print(f"\n[TRAIN Epoch {self.current_epoch}]", end=" ")
            self._log_examples(tgt_output[0], preds[0])
            print("")  # Xống dòng
        # ---------------------------------------------

        return loss

    def validation_step(self, batch, batch_idx):
        images, targets, _ = batch

        # 1. Calculate val_loss using Teacher Forcing (for convergence monitoring)
        logits = self.model(images, targets)
        tgt_output = targets[:, 1:]
        loss = self.criterion(logits.reshape(-1, self.vocab_size), tgt_output.reshape(-1))

        # 2. Autoregressive decoding for real-world accuracy
        generated = self.model(images)

        # We compare from index 1 to end (excluding SOS)
        gt_content = targets[:, 1:]
        pred_content = generated[:, 1:]

        # Sequence Accuracy (Exact Match)
        correct_count = 0
        batch_size = gt_content.size(0)

        for i in range(batch_size):
            gt_str = self._decode_seq(gt_content[i])
            pred_str = self._decode_seq(pred_content[i])
            if gt_str == pred_str:
                correct_count += 1

        correct = torch.tensor(correct_count / batch_size, device=gt_content.device)

        self.log("val_loss", loss, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("val_acc", correct, on_epoch=True, prog_bar=True, sync_dist=True)

        # Logging some examples
        if batch_idx == 0:
            self._log_examples(gt_content[0], pred_content[0])

        return correct

    def _decode_seq(self, tokens):
        """Hàm dịch tensor thành chuỗi, tự động dừng khi gặp thẻ <EOS> (2)"""
        res = []
        for c in tokens:
            c = int(c)
            if c == 2:
                break
            if c not in [0, 1]:
                res.append(self.args.chars[c])
        return "".join(res)

    def _log_examples(self, gt, pred):
        # Indices 0: PAD, 1: SOS, 2: EOS
        gt_str = "".join([self.args.chars[c] for c in gt if c not in [0, 1, 2]])
        pred_str = self._decode_seq(pred)

        if self.trainer.logger and hasattr(self.trainer.logger, "experiment"):
            # If using WandB or similar
            try:
                import wandb

                if isinstance(self.trainer.logger.experiment, wandb.sdk.wandb_run.Run):
                    self.trainer.logger.experiment.log(
                        {"samples": wandb.Table(columns=["GT", "Pred"], data=[[gt_str, pred_str]])}
                    )
            except ImportError:
                pass

        # Always print to console for visibility
        print(f"\r[Sample] GT: {gt_str:10} | Pred: {pred_str:10}", end="")

    # def configure_optimizers(self):
    #     optimizer = torch.optim.AdamW(
    #         self.model.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay
    #     )

    #     scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    #         optimizer, T_max=self.args.max_epochs, eta_min=1e-6
    #     )

    #     return [optimizer], [scheduler]

    # Differential learning rate — chi tiết theo vai trò mỗi component
    def configure_optimizers(self):
        """
        Chiến lược tối ưu hóa differential LR với 4 nhóm:
        1. Backbone encoder (pretrained MobileViT): LR rất thấp (0.01x)
        2. Projection head + Memory adapter (mới): LR trung bình (0.5x)
        3. Decoder frozen → unfrozen (pretrained MiniLMv2): LR thấp (0.05x)
        4. PTN + Decoder new layers (cross-attn, output head): LR cao (1x)
        """
        backbone_params = []       # MobileViT backbone — pretrained, cần LR nhỏ nhất
        projection_params = []     # Encoder projection + memory adapter — mới nhưng gắn backbone
        decoder_pretrained = []    # Decoder self-attn, FFN (pretrained weights)
        scratch_params = []        # PTN, cross-attention, output head — train từ đầu

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue

            if "encoder.backbone" in name:
                backbone_params.append(param)
            elif "encoder.proj_conv" in name or "encoder.bn" in name or "encoder.linear_proj" in name or "memory_adapter" in name:
                projection_params.append(param)
            elif "decoder.self_attn" in name or "decoder.linear1" in name or "decoder.linear2" in name or "decoder.norm" in name:
                decoder_pretrained.append(param)
            else:
                # PTN, decoder cross-attn, embedding, fc_out, memory_pe
                scratch_params.append(param)

        param_groups = [
            {"params": backbone_params, "lr": self.args.lr * 0.01, "name": "backbone"},
            {"params": projection_params, "lr": self.args.lr * 0.5, "name": "projection"},
            {"params": decoder_pretrained, "lr": self.args.lr * 0.05, "name": "decoder_pretrained"},
            {"params": scratch_params, "lr": self.args.lr, "name": "scratch"},
        ]

        # Lọc bỏ nhóm rỗng (vd: decoder_pretrained bị freeze hoàn toàn)
        param_groups = [g for g in param_groups if len(g["params"]) > 0]

        optimizer = torch.optim.AdamW(
            param_groups,
            weight_decay=self.args.weight_decay,
        )

        # OneCycleLR: SOTA scheduler cho fine-tuning — warmup tự động + annealing
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=[g["lr"] for g in param_groups],
            total_steps=self.trainer.estimated_stepping_batches,
            pct_start=0.05,  # 5% warmup
            anneal_strategy="cos",
            div_factor=10,    # initial_lr = max_lr / 10
            final_div_factor=10,   # final_lr = initial_lr / 10 (backbone LR không về 0)
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }
