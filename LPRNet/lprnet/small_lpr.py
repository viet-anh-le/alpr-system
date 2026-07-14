"""
SmallLPR — OCR biển số cho ảnh đầu vào nhỏ (≥ 8×20 px), xử lý cả biển 1 dòng và 2 dòng.

Thiết kế:
  smart_resize (48×96) → CSM-style CNN backbone (stride 8) → 2D PE → MiniLMv2 decoder.

Không flatten feature map thành 1D như LPRNet/CSM_LPRNet → cross-attention decoder
tự học reading-order (top row → bottom row) cho biển 2 dòng qua 2D positional encoding.
"""

from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .components import MiniLMv2Decoder
from .csm_lprnet import CBAM, MixConv2d

# =============================================================================
# STN — Spatial Transformer Network, cho input 48×96
# Luồng kích thước localization network (H×W):
#   48×96 → Conv3×3 → 46×94 → MaxPool2 → 23×47
#         → Conv5×5 → 19×43 → MaxPool3 → 6×14  → FC: 32×6×14 = 2688
# =============================================================================


class _STNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.localization = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3),
            nn.MaxPool2d(2, stride=2),
            nn.Mish(inplace=True),
            nn.Conv2d(32, 32, kernel_size=5),
            nn.MaxPool2d(3, stride=3),
            nn.Mish(inplace=True),
        )
        self.fc_loc = nn.Sequential(
            nn.Linear(32 * 6 * 14, 32),
            nn.Mish(inplace=True),
            nn.Linear(32, 6),
        )
        self.fc_loc[-1].weight.data.zero_()
        self.fc_loc[-1].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    def forward(self, x):
        xs = self.localization(x)
        xs = xs.reshape(-1, 32 * 6 * 14)
        theta = self.fc_loc(xs).view(-1, 2, 3)
        grid = F.affine_grid(theta, x.size(), align_corners=True)
        return F.grid_sample(x, grid, align_corners=True)


# =============================================================================
# Backbone — LPRNet small_basic_block + CBAM
# =============================================================================


class SmallBasicBlockCBAM(nn.Module):
    """
    Bottleneck 1×1 → 3×1 → 1×3 → 1×1 (LPRNet) + CBAM attention (CSM_LPRNet).
    Residual connection khi channel in == out.
    """

    def __init__(self, ch_in: int, ch_out: int):
        super().__init__()
        mid = ch_out // 4
        self.block = nn.Sequential(
            MixConv2d(ch_in, mid, kernels=[3, 5]),
            # nn.Conv2d(ch_in, mid, 3, padding=1, bias=False),
            nn.BatchNorm2d(mid),
            nn.Mish(inplace=True),
            nn.Conv2d(mid, mid, (3, 1), padding=(1, 0)),
            nn.BatchNorm2d(mid),
            nn.Mish(inplace=True),
            nn.Conv2d(mid, mid, (1, 3), padding=(0, 1)),
            nn.BatchNorm2d(mid),
            nn.Mish(inplace=True),
            nn.Conv2d(mid, ch_out, 1),
            nn.BatchNorm2d(ch_out),
        )
        self.cbam = CBAM(ch_out)
        # self.cbam = nn.Identity()
        self.residual = ch_in == ch_out

    def forward(self, x):
        out = self.cbam(self.block(x))
        return F.mish(out + x) if self.residual else F.mish(out)


class SmallLPRBackbone(nn.Module):
    """
    Input:  (B, 3, 48, 96)
    Output: (B, 256, 6, 12)   — stride 8, 2D spatial giữ nguyên (không collapse)
    """

    def __init__(self, out_channels: int = 256):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.Mish(inplace=True),
        )
        # 48×96 → 24×48
        self.stage1 = nn.Sequential(
            SmallBasicBlockCBAM(64, 128),
            nn.MaxPool2d(2, 2),
        )
        # 24×48 → 12×24
        self.stage2 = nn.Sequential(
            SmallBasicBlockCBAM(128, 256),
            SmallBasicBlockCBAM(256, 256),
            nn.MaxPool2d(2, 2),
        )
        # 12×24 → 6×12
        self.stage3 = nn.Sequential(
            SmallBasicBlockCBAM(256, out_channels),
            SmallBasicBlockCBAM(out_channels, out_channels),
            nn.MaxPool2d(2, 2),
        )
        self.out_channels = out_channels

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        return x


# =============================================================================
# 2D Positional Encoding — tách row_pe + col_pe
# =============================================================================


class LearnablePositional2D(nn.Module):
    """row_pe + col_pe → mỗi token biết (row, col) trong feature map."""

    def __init__(self, max_h: int, max_w: int, d_model: int):
        super().__init__()
        self.row_pe = nn.Parameter(torch.randn(1, max_h, 1, d_model) * 0.02)
        self.col_pe = nn.Parameter(torch.randn(1, 1, max_w, d_model) * 0.02)

    def forward(self, x):
        # x: (B, H, W, d)
        H, W = x.size(1), x.size(2)
        return x + self.row_pe[:, :H] + self.col_pe[:, :, :W]


# =============================================================================
# Smart resize — aspect-preserve + zero-pad
# =============================================================================


def smart_resize(img: np.ndarray, target_hw: Tuple[int, int] = (48, 96)) -> np.ndarray:
    """BGR HxWxC uint8 → (Ht, Wt, 3) uint8. Lanczos4 khi upsample, INTER_AREA khi downsample."""
    import cv2

    Ht, Wt = target_hw
    h, w = img.shape[:2]
    scale = min(Wt / w, Ht / h)
    new_w, new_h = max(1, int(round(w * scale))), max(1, int(round(h * scale)))

    interp = cv2.INTER_LANCZOS4 if scale > 1 else cv2.INTER_AREA
    resized = cv2.resize(img, (new_w, new_h), interpolation=interp)

    canvas = np.zeros((Ht, Wt, 3), dtype=img.dtype)
    y0 = (Ht - new_h) // 2
    x0 = (Wt - new_w) // 2
    canvas[y0 : y0 + new_h, x0 : x0 + new_w] = resized
    return canvas


# =============================================================================
# SmallLPR — end-to-end model
# =============================================================================


class SmallLPR(nn.Module):
    """
    Forward modes:
      - Có `targets` (training): teacher forcing, trả về logits (B, L-1, V)
      - Không có targets (inference): autoregressive, trả về tokens (B, max_seq_len)
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 384,
        max_seq_len: int = 14,
        start_token_idx: int = 1,
        end_token_idx: int = 2,
        use_pretrained_decoder: bool = True,
    ):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.start_token_idx = start_token_idx
        self.end_token_idx = end_token_idx

        self.stn = _STNet()
        self.backbone = SmallLPRBackbone(out_channels=256)
        self.proj = nn.Conv2d(256, d_model, 1)
        self.pos_enc_2d = LearnablePositional2D(max_h=8, max_w=16, d_model=d_model)

        self.decoder = MiniLMv2Decoder(
            vocab_size=vocab_size,
            d_model=d_model,
            nhead=4,
            num_layers=4,
            dim_feedforward=d_model * 4,
            max_seq_len=max_seq_len,
        )

        if use_pretrained_decoder:
            self.decoder.init_from_pretrained()
            # Freeze self-attn/FFN (giống TransLPRNet); cross-attn + embedding vẫn trainable
            for name, param in self.decoder.named_parameters():
                if "self_attn" in name or "linear1" in name or "linear2" in name or "norm" in name:
                    param.requires_grad = False

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        images = self.stn(images)
        feat = self.backbone(images)  # (B, 256, 6, 12)
        feat = self.proj(feat)  # (B, d, 6, 12)
        feat = feat.permute(0, 2, 3, 1)  # (B, 6, 12, d)
        feat = self.pos_enc_2d(feat)
        B, H, W, D = feat.shape
        return feat.reshape(B, H * W, D)  # (B, 72, d)

    def forward(self, images: torch.Tensor, targets: torch.Tensor = None):
        memory = self.encode(images)

        if targets is not None:
            tgt_input = targets[:, :-1]
            return self.decoder(tgt_tokens=tgt_input, memory_features=memory)

        B = memory.size(0)
        device = memory.device
        tokens = torch.full((B, 1), self.start_token_idx, dtype=torch.long, device=device)
        finished = torch.zeros(B, dtype=torch.bool, device=device)

        for _ in range(self.max_seq_len - 1):
            logits = self.decoder(tgt_tokens=tokens, memory_features=memory)
            next_tok = logits[:, -1].argmax(-1, keepdim=True)
            tokens = torch.cat([tokens, next_tok], dim=1)
            finished = finished | (next_tok.squeeze(-1) == self.end_token_idx)
            if finished.all():
                break

        return tokens
