"""
SmallLPR-CTC — OCR biển số dùng kiến trúc CTC (Connectionist Temporal Classification).

So với SmallLPR (Autoregressive Decoder):
  - Giữ nguyên STN + Backbone CNN + 2D Positional Encoding.
  - Bỏ MiniLMv2Decoder → thay bằng một Linear head đơn giản.
  - Forward chỉ cần 1 lần GPU call duy nhất (không có vòng lặp Python).
  - Decode bằng Greedy CTC: argmax dọc T → bỏ blank (index 0) và ký tự lặp liên tiếp.

Luồng kích thước:
  Input (B, 3, 48, 96)
    → STN          → (B, 3, 48, 96)
    → Backbone     → (B, 256, 6, 12)
    → proj_conv    → (B, d_model, 6, 12)
    → 2D-PE        → (B, 6, 12, d_model)
    → reshape      → (B, 72, d_model)       ← T = 72 time steps
    → CTC head     → (B, 72, vocab_size)    ← logits đưa vào CTCLoss
"""

from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# Tái sử dụng backbone / STN / 2D-PE từ small_lpr (không copy, chỉ import)
from .small_lpr import (
    LearnablePositional2D,
    SmallLPRBackbone,
    _STNet,
    smart_resize,
)

# Số time steps cố định sau khi qua backbone 48×96 với stride 8:  6×12 = 72
_T_STEPS: int = 72  # 6 rows × 12 cols


# =============================================================================
# Utility: Greedy CTC decode (module-level, dùng được cả ngoài Lightning)
# =============================================================================


def ctc_greedy_decode(logits: torch.Tensor, chars: List[str]) -> List[str]:
    """
    Greedy CTC decode.

    Args:
        logits: (B, T, C) — output thô của model (chưa softmax).
        chars:  list ký tự, chars[0] là blank token.

    Returns:
        list[str] độ dài B.
    """
    n_vocab = len(chars)
    # argmax dọc theo chiều class C, clamp để tránh OOB khi vocab_size != len(chars)
    preds = logits.argmax(dim=-1).clamp(0, n_vocab - 1)  # (B, T)
    results: List[str] = []
    for seq in preds:
        chars_out: List[str] = []
        prev = -1
        for token in seq.tolist():
            if token != prev:
                if token != 0:  # 0 là blank
                    chars_out.append(chars[token])
                prev = token
        results.append("".join(chars_out))
    return results


# =============================================================================
# SmallLPRCTC — model chính
# =============================================================================


class SmallLPRCTC(nn.Module):
    """
    SmallLPR với CTC head thay thế Transformer Decoder.

    Args:
        vocab_size:  Số ký tự trong charset (bao gồm cả blank ở index 0).
        d_model:     Chiều embedding. Mặc định 256 (nhỏ hơn bản AR để nhanh hơn).
        backbone_ch: Số channels đầu ra của backbone. Mặc định 256.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        backbone_ch: int = 256,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model

        self.stn = _STNet()
        self.backbone = SmallLPRBackbone(out_channels=backbone_ch)
        self.proj = nn.Conv2d(backbone_ch, d_model, kernel_size=1, bias=False)
        self.pos_enc_2d = LearnablePositional2D(max_h=8, max_w=16, d_model=d_model)

        # CTC head: linear projection từ d_model → vocab (bao gồm blank)
        self.head = nn.Linear(d_model, vocab_size)

        # Khởi tạo head nhỏ để tránh gradient explode ngay lúc đầu train
        nn.init.trunc_normal_(self.head.weight, std=0.02)
        nn.init.zeros_(self.head.bias)

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        """
        Trả về sequence features (B, T, d_model), T = 72.
        """
        images = self.stn(images)
        feat = self.backbone(images)           # (B, backbone_ch, 6, 12)
        feat = self.proj(feat)                 # (B, d_model, 6, 12)
        feat = feat.permute(0, 2, 3, 1)        # (B, 6, 12, d_model)
        feat = self.pos_enc_2d(feat)
        B, H, W, D = feat.shape
        return feat.reshape(B, H * W, D)       # (B, 72, d_model)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Training forward — trả về raw logits (B, T, vocab_size).
        Không dùng softmax / log_softmax ở đây; Lightning module xử lý.
        """
        memory = self.encode(images)           # (B, 72, d_model)
        logits = self.head(memory)             # (B, 72, vocab_size)
        return logits

    @torch.no_grad()
    def greedy_decode(self, images: torch.Tensor, chars: List[str]) -> List[str]:
        """
        Inference: trả về list chuỗi ký tự đã decode.
        """
        self.eval()
        logits = self.forward(images)          # (B, 72, vocab_size)
        return ctc_greedy_decode(logits, chars)


# =============================================================================
# smart_resize — re-export để training script / benchmark dùng tiện
# =============================================================================
__all__ = ["SmallLPRCTC", "ctc_greedy_decode", "smart_resize", "_T_STEPS"]
