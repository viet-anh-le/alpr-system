import numpy as np
from cv2 import resize, INTER_LANCZOS4
from typing import Optional
from argparse import Namespace
import matplotlib.pyplot as plt
import os
import numpy as np
import cv2

import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L

from lprnet.utils import decode, accuracy

import matplotlib

matplotlib.use("Agg")


def sparse_tuple_for_ctc(t_length, lengths):
    input_lengths = []
    target_lengths = []

    for ch in lengths:
        input_lengths.append(t_length)
        target_lengths.append(ch)

    return torch.tensor(input_lengths), torch.tensor(target_lengths)


class MixConv2d(nn.Module):
    def __init__(self, in_ch, out_ch, kernels=[3, 5], stride=1):
        super().__init__()
        n = len(kernels)
        self.splits = [out_ch // n] * (n - 1) + [out_ch - (out_ch // n) * (n - 1)]
        self.convs = nn.ModuleList(
            [
                nn.Conv2d(in_ch, self.splits[i], k, stride, k // 2, groups=1, bias=False)
                for i, k in enumerate(kernels)
            ]
        )

    def forward(self, x):
        return torch.cat([conv(x) for conv in self.convs], dim=1)


class SEBlock(nn.Module):
    def __init__(self, ch, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(ch, ch // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(ch // reduction, ch, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class CBAM(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.ca = SEBlock(ch)
        self.sa = nn.Sequential(nn.Conv2d(2, 1, 7, padding=3, bias=False), nn.Sigmoid())

    def forward(self, x):
        x = self.ca(x)
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        sa_out = self.sa(torch.cat([avg_out, max_out], dim=1))
        return x * sa_out


class _STNet(nn.Module):
    def __init__(self):
        super(_STNet, self).__init__()

        # Spatial transformer localization-network
        self.localization = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3),
            nn.MaxPool2d(2, stride=2),
            nn.Mish(True),
            nn.Conv2d(32, 32, kernel_size=5),
            nn.MaxPool2d(3, stride=3),
            nn.Mish(True),
        )
        # Regressor for the 3x2 affine matrix
        self.fc_loc = nn.Sequential(nn.Linear(32 * 15 * 6, 32), nn.Mish(True), nn.Linear(32, 3 * 2))
        # Initialize the weights/bias with identity transformation
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    def forward(self, x):
        xs = self.localization(x)
        xs = xs.view(-1, 32 * 15 * 6)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)

        grid = F.affine_grid(theta, x.size(), align_corners=True)
        x = F.grid_sample(x, grid, align_corners=True)

        return x


class ImprovedBasicBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(ImprovedBasicBlock, self).__init__()
        self.block = nn.Sequential(
            MixConv2d(in_ch, out_ch // 4),
            nn.BatchNorm2d(out_ch // 4),
            nn.Mish(),
            nn.Conv2d(out_ch // 4, out_ch // 4, kernel_size=(3, 1), padding=(1, 0)),
            nn.BatchNorm2d(out_ch // 4),
            nn.Mish(),
            nn.Conv2d(out_ch // 4, out_ch // 4, kernel_size=(1, 3), padding=(0, 1)),
            nn.BatchNorm2d(out_ch // 4),
            nn.Mish(),
            nn.Conv2d(out_ch // 4, out_ch, kernel_size=1),
            nn.BatchNorm2d(out_ch),
            nn.Mish(),
            CBAM(out_ch),
        )

    def forward(self, x):
        out = self.block(x)
        return out


class _LPRNet(nn.Module):
    def __init__(self, class_num, dropout_rate):
        super(_LPRNet, self).__init__()
        self.class_num = class_num
        self.backbone = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1),
            nn.BatchNorm2d(num_features=64),
            nn.Mish(),
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 1, 1)),
            ImprovedBasicBlock(in_ch=64, out_ch=128),
            nn.BatchNorm2d(num_features=128),
            nn.Mish(),
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(2, 1, 2)),
            ImprovedBasicBlock(in_ch=64, out_ch=256),
            nn.BatchNorm2d(num_features=256),
            nn.Mish(),
            ImprovedBasicBlock(in_ch=256, out_ch=256),
            nn.BatchNorm2d(num_features=256),
            nn.Mish(),
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(4, 2, 2)),
            nn.Dropout(dropout_rate),
            nn.Conv2d(in_channels=64, out_channels=256, kernel_size=(2, 4), stride=1),
            nn.BatchNorm2d(num_features=256),
            nn.Mish(),
            nn.Dropout(dropout_rate),
            nn.Conv2d(in_channels=256, out_channels=class_num, kernel_size=(12, 2), stride=1),
            nn.BatchNorm2d(num_features=class_num),
            nn.Mish(),
        )
        self.container = nn.Sequential(
            nn.Conv2d(
                in_channels=256 + class_num + 128 + 64,
                out_channels=self.class_num,
                kernel_size=(1, 1),
                stride=(1, 1),
            ),
        )

    def forward(self, x):
        keep_features = list()
        for i, layer in enumerate(self.backbone.children()):
            x = layer(x)
            if i in [2, 6, 13, 22]:  # [2, 4, 8, 11, 22]
                keep_features.append(x)

        global_context = list()
        for i, f in enumerate(keep_features):
            if i in [0, 1]:
                f = nn.AvgPool2d(kernel_size=5, stride=5)(f)
            if i in [2]:
                f = nn.AvgPool2d(kernel_size=(4, 10), stride=(5, 2))(f)
            f_pow = torch.pow(f, 2)
            f_mean = torch.mean(f_pow)
            f = torch.div(f, f_mean + 1e-7)
            global_context.append(f)

        x = torch.cat(global_context, 1)
        x = self.container(x)
        logits = torch.mean(x, dim=2)

        return logits


class LPRNet(L.LightningModule):
    def __init__(self, args: Optional[Namespace] = None):
        super().__init__()
        self.save_hyperparameters(args)
        self.STNet = _STNet()
        self.LPRNet = _LPRNet(
            class_num=len(self.hparams.chars), dropout_rate=self.hparams.dropout_rate
        )

    def forward(self, x):
        # return self.LPRNet(self.STNet(x))
        return self.LPRNet(x)

    def training_step(self, batch, batch_idx):
        imgs, labels, lengths = batch
        # ---------------Bắt đầu visualize-----------------------------------
        # if batch_idx == 0:
        #     # Lấy ảnh đầu tiên trong batch
        #     single_img = imgs[0:1]

        #     # Đưa qua STNet để xem ảnh được nắn
        #     stnet_out = self.STNet(single_img)

        #     # Lấy feature map từ lớp số 2 (Conv đầu tiên)
        #     features = stnet_out
        #     for i in range(3):
        #         features = self.LPRNet.backbone[i](features)

        #     # Chuyển tensor sang numpy để vẽ 16 channels đầu
        #     fm_numpy = features[0, :16].detach().cpu().numpy()
        #     img_numpy = stnet_out[0].detach().cpu().permute(1, 2, 0).numpy()

        #     # Normalize ảnh gốc về [0, 1] để tránh lỗi hiển thị
        #     img_numpy = (img_numpy - np.min(img_numpy)) / (
        #         np.max(img_numpy) - np.min(img_numpy) + 1e-7
        #     )

        #     fig, axes = plt.subplots(4, 4, figsize=(12, 8))
        #     fig.suptitle(f"Feature Maps - Layer 2 - Epoch {self.current_epoch}")

        #     for i, ax in enumerate(axes.flat):
        #         ax.imshow(fm_numpy[i], cmap="viridis")
        #         ax.axis("off")

        #     # Lưu ảnh Feature map
        #     os.makedirs("visualizations", exist_ok=True)
        #     plt.savefig(f"visualizations/feature_maps_epoch_{self.current_epoch:03d}.png")
        #     plt.close(fig)

        #     # Lưu ảnh sau STNet
        #     plt.imshow(img_numpy)
        #     plt.title(f"Ảnh qua STNet - Epoch {self.current_epoch}")
        #     plt.savefig(f"visualizations/stnet_out_epoch_{self.current_epoch:03d}.png")
        #     plt.close()
        # # ------------------Kết thúc visualize-----------------------------

        # ---------------Bắt đầu visualize layer cuối---------------------------
        if batch_idx == 0:
            raw_tensor = imgs[0].detach().cpu().numpy()

            raw_img = np.transpose(raw_tensor, (1, 2, 0))
            # Hủy chuẩn hóa: img = (img - 127.5) * 0.0078125 -> img = img / 0.0078125 + 127.5
            raw_img = raw_img / 0.0078125 + 127.5
            raw_img = raw_img.astype(np.uint8)

            raw_img_rgb = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB)

            plt.imshow(raw_img_rgb)
            plt.title(f"Ảnh đầu vào Dataloader - Epoch {self.current_epoch}")
            plt.axis("off")
            plt.savefig(f"visualizations/1_dataloader_epoch_{self.current_epoch:03d}.png")
            plt.close()

            single_img = imgs[0:1]

            # Đưa qua STNet để xem ảnh được nắn
            stnet_out = self.STNet(single_img)

            # Lấy feature map từ lớp số 2 (Conv đầu tiên)
            features = stnet_out
            for i in range(3):
                features = self.LPRNet.backbone[i](features)

            # Chuyển tensor sang numpy để vẽ 16 channels đầu
            fm_numpy = features[0, :16].detach().cpu().numpy()
            img_numpy = stnet_out[0].detach().cpu().permute(1, 2, 0).numpy()

            # Normalize ảnh gốc về [0, 1] để tránh lỗi hiển thị
            img_numpy = (img_numpy - np.min(img_numpy)) / (
                np.max(img_numpy) - np.min(img_numpy) + 1e-7
            )

            fig, axes = plt.subplots(4, 4, figsize=(12, 8))
            fig.suptitle(f"Feature Maps - Layer 2 - Epoch {self.current_epoch}")

            for i, ax in enumerate(axes.flat):
                ax.imshow(fm_numpy[i], cmap="viridis")
                ax.axis("off")

            # Lưu ảnh Feature map
            os.makedirs("visualizations", exist_ok=True)
            plt.savefig(f"visualizations/feature_maps_epoch_{self.current_epoch:03d}.png")
            plt.close(fig)

            # Lưu ảnh sau STNet
            plt.imshow(img_numpy)
            plt.title(f"Ảnh qua STNet - Epoch {self.current_epoch}")
            plt.savefig(f"visualizations/stnet_out_epoch_{self.current_epoch:03d}.png")
            plt.close()

            activation = {}

            def get_activation(name):
                def hook(model, input, output):
                    activation[name] = output.detach()

                return hook

            handle = self.LPRNet.container.register_forward_hook(get_activation("last_layer"))

            single_img = imgs[0:1]
            _ = self(single_img)

            handle.remove()

            fm_numpy = activation["last_layer"][0, :16].cpu().numpy()

            fig, axes = plt.subplots(4, 4, figsize=(12, 6))
            fig.suptitle(f"Feature Maps - Lớp Cuối (Container) - Epoch {self.current_epoch}")

            for i, ax in enumerate(axes.flat):
                ax.imshow(fm_numpy[i], cmap="viridis", aspect="auto")
                ax.axis("off")

            os.makedirs("visualizations", exist_ok=True)
            plt.savefig(f"visualizations/last_layer_epoch_{self.current_epoch:03d}.png")
            plt.close(fig)
        # -----------------Kết thúc visualize layer cuối------------------------------

        logits = self(imgs)
        log_probs = logits.permute(2, 0, 1)
        log_probs = log_probs.log_softmax(2).requires_grad_()
        input_lengths, target_lengths = sparse_tuple_for_ctc(self.hparams.t_length, lengths)
        loss = F.ctc_loss(
            log_probs=log_probs,
            targets=labels,
            input_lengths=input_lengths,
            target_lengths=target_lengths,
            blank=len(self.hparams.chars) - 1,
            reduction="mean",
        )
        acc = accuracy(logits, labels, lengths, self.hparams.chars, self.current_epoch)

        self.log("train-loss", abs(loss), prog_bar=True, logger=True, sync_dist=True)
        self.log("train-acc", acc, prog_bar=True, logger=True, sync_dist=True)

        return loss

    def validation_step(self, batch, batch_idx):
        imgs, labels, lengths = batch

        logits = self(imgs)
        log_probs = logits.permute(2, 0, 1)
        log_probs = log_probs.log_softmax(2).requires_grad_()
        input_lengths, target_lengths = sparse_tuple_for_ctc(self.hparams.t_length, lengths)
        loss = F.ctc_loss(
            log_probs=log_probs,
            targets=labels,
            input_lengths=input_lengths,
            target_lengths=target_lengths,
            blank=len(self.hparams.chars) - 1,
            reduction="mean",
        )
        acc = accuracy(logits, labels, lengths, self.hparams.chars, self.current_epoch)

        self.log("val-loss", abs(loss), prog_bar=True, logger=True, sync_dist=True)
        self.log("val-acc", acc, prog_bar=True, logger=True, sync_dist=True)

    def test_step(self, batch, batch_idx):
        imgs, labels, lengths = batch
        import time

        start = time.time()
        logits = self(imgs)
        log_probs = logits.permute(2, 0, 1)
        log_probs = log_probs.log_softmax(2).requires_grad_()
        input_lengths, target_lengths = sparse_tuple_for_ctc(self.hparams.t_length, lengths)
        loss = F.ctc_loss(
            log_probs=log_probs,
            targets=labels,
            input_lengths=input_lengths,
            target_lengths=target_lengths,
            blank=len(self.hparams.chars) - 1,
            reduction="mean",
        )
        acc = accuracy(logits, labels, lengths, self.hparams.chars, self.current_epoch)
        end = time.time()

        self.log("test-loss", abs(loss), prog_bar=True, logger=True, sync_dist=True)
        self.log("test-acc", acc, prog_bar=True, logger=True, sync_dist=True)
        self.log("test-time", end - start, prog_bar=True, logger=True, sync_dist=True)

    def predict_step(self, batch, batch_idx, dataloader_idx: int = 0):
        imgs, labels, lengths = batch

        logits = self(imgs)
        preds = logits.cpu().detach().numpy()  # (batch size, 68, 18)
        predict, _ = decode(preds, self.chars)  # list of predict output

        return predict

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            [
                {
                    "params": self.STNet.parameters(),
                    "weight_decay": self.hparams.weight_decay,
                },
                {"params": self.LPRNet.parameters()},
            ],
            lr=self.hparams.lr,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, 10, 2, 0.0001, -1
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
                "monitor": "val-loss",
                "strict": True,
                "name": "lr",
            },
        }
