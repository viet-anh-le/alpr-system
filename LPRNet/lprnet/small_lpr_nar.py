"""
SmallLPR-NAR — Non-Autoregressive OCR biển số. (v2 — fixed)

Thay đổi so với v1:
  - NARDecoderLayer: thêm Self-Attention trước Cross-Attention.
    Các position queries giờ có thể giao tiếp với nhau (biết "hàng xóm" đang
    đoán gì) → học inter-character dependency tốt hơn nhiều.
  - pos_queries khởi tạo với std=0.1 thay vì 0.02 → attention score không đồng
    đều ngay từ đầu, gradient flow mạnh hơn.
  - Thêm query_norm trước khi broadcast pos_queries để chuẩn hoá giá trị ban đầu.

Luồng kích thước:
  Input (B, 3, 48, 96)
    → STN          → (B, 3, 48, 96)
    → Backbone     → (B, 256, 6, 12)
    → proj_conv    → (B, d_model, 6, 12)
    → 2D-PE        → (B, 6, 12, d_model)
    → reshape      → (B, 72, d_model)     ← memory
    → NARDecoder   → (B, MAX_LEN, vocab_size)   ← logits

Loss: CrossEntropy với ignore_index=0 (pad token).
Charset: index 0 = <pad>, index 1..N = ký tự thật.
"""

from __future__ import annotations

from typing import List

import math
import torch
import torch.nn as nn

from .small_lpr import (
    LearnablePositional2D,
    SmallLPRBackbone,
    _STNet,
    smart_resize,
)


# =============================================================================
# NARDecoderLayer — Self-Attention + Cross-Attention + FFN (Pre-norm)
# =============================================================================


class NARDecoderLayer(nn.Module):
    """
    Một lớp decoder NAR với đầy đủ 3 sub-layers (giống PARSeq/BERT decoder):
      1. Self-Attention (Q/K/V = pos_queries) — inter-position communication
      2. Cross-Attention (Q = pos_queries, K/V = encoder memory) — look at image
      3. FFN

    Dùng Pre-norm để huấn luyện ổn định hơn.
    """

    def __init__(self, d_model: int, nhead: int, dropout: float = 0.1):
        super().__init__()

        # 1. Self-attention (không mask — tất cả positions attend lẫn nhau)
        self.self_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        # 2. Cross-attention (Q=pos_queries, K/V=encoder memory)
        self.cross_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        # 3. FFN
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
        )

        self.norm0 = nn.LayerNorm(d_model)   # for self-attn
        self.norm1 = nn.LayerNorm(d_model)   # for cross-attn
        self.norm2 = nn.LayerNorm(d_model)   # for FFN
        self.drop = nn.Dropout(dropout)

    def forward(self, q: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
        # --- Sub-layer 1: Self-Attention (Pre-norm) ---
        q_norm = self.norm0(q)
        q2, _ = self.self_attn(q_norm, q_norm, q_norm)
        q = q + self.drop(q2)

        # --- Sub-layer 2: Cross-Attention (Pre-norm) ---
        q2, _ = self.cross_attn(self.norm1(q), kv, kv)
        q = q + self.drop(q2)

        # --- Sub-layer 3: FFN (Pre-norm) ---
        q = q + self.drop(self.ffn(self.norm2(q)))
        return q


# =============================================================================
# NARDecoder — Full decoder block
# =============================================================================


class NARDecoder(nn.Module):
    """
    Non-autoregressive decoder.

    Args:
        max_len:    Số vị trí output tối đa.
        d_model:    Chiều embedding.
        vocab_size: Số lớp đầu ra (bao gồm <pad> ở index 0).
        nhead:      Số attention heads.
        num_layers: Số lớp decoder (khuyến nghị ≥ 4).
        dropout:    Dropout rate.
    """

    def __init__(
        self,
        max_len: int,
        d_model: int,
        vocab_size: int,
        nhead: int = 4,
        num_layers: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.max_len = max_len

        # Learned position queries — mỗi row = 1 output slot
        self.pos_queries = nn.Embedding(max_len, d_model)

        self.layers = nn.ModuleList(
            [NARDecoderLayer(d_model, nhead, dropout) for _ in range(num_layers)]
        )
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)

        self._init_weights()

    def _init_weights(self) -> None:
        # std=0.1 thay vì 0.02: queries đủ khác biệt ngay từ đầu để attention
        # score không đồng đều, gradient flow mạnh hơn từ epoch đầu tiên.
        nn.init.trunc_normal_(self.pos_queries.weight, std=0.1)
        nn.init.trunc_normal_(self.head.weight, std=0.02)
        nn.init.zeros_(self.head.bias)

    def forward(self, memory: torch.Tensor) -> torch.Tensor:
        """
        Args:
            memory: (B, T, d_model) — encoder output
        Returns:
            logits: (B, max_len, vocab_size)
        """
        B = memory.size(0)
        idx = torch.arange(self.max_len, device=memory.device)
        q = self.pos_queries(idx).unsqueeze(0).expand(B, -1, -1)  # (B, L, D)

        x = q
        for layer in self.layers:
            x = layer(x, memory)
        x = self.norm(x)
        return self.head(x)  # (B, max_len, vocab_size)


# =============================================================================
# SmallLPRNAR — Model chính
# =============================================================================


class SmallLPRNAR(nn.Module):
    """
    SmallLPR với NAR (Non-Autoregressive) decoder.

    Args:
        vocab_size:  Số ký tự (bao gồm <pad> ở index 0).
        d_model:     Chiều embedding. Mặc định 256.
        backbone_ch: Số channels backbone. Mặc định 256.
        max_len:     Số vị trí output tối đa. Mặc định 14.
        nhead:       Số attention heads.
        num_layers:  Số lớp NARDecoder (khuyến nghị ≥ 4).
        dropout:     Dropout rate.
    """

    DEFAULT_MAX_LEN: int = 14

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        backbone_ch: int = 256,
        max_len: int = DEFAULT_MAX_LEN,
        nhead: int = 4,
        num_layers: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_len = max_len

        # Encoder (giống SmallLPR / SmallLPR-CTC)
        self.stn = _STNet()
        self.backbone = SmallLPRBackbone(out_channels=backbone_ch)
        self.proj = nn.Conv2d(backbone_ch, d_model, kernel_size=1, bias=False)
        self.pos_enc_2d = LearnablePositional2D(max_h=8, max_w=16, d_model=d_model)
        self.enc_norm = nn.LayerNorm(d_model)  # Thêm LayerNorm cho encoder output

        # NAR decoder
        self.decoder = NARDecoder(
            max_len=max_len,
            d_model=d_model,
            vocab_size=vocab_size,
            nhead=nhead,
            num_layers=num_layers,
            dropout=dropout,
        )

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        """Trả về encoder output (B, 72, d_model)."""
        x = self.stn(images)
        x = self.backbone(x)            # (B, C, 6, 12)
        x = self.proj(x)                # (B, D, 6, 12)
        x = x.permute(0, 2, 3, 1)      # (B, 6, 12, D)
        x = self.pos_enc_2d(x)
        B, H, W, D = x.shape
        x = x.reshape(B, H * W, D)     # (B, 72, D)
        return self.enc_norm(x)        # Chuẩn hoá encoder features bằng LayerNorm

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Training/inference forward.
        Returns:
            logits: (B, max_len, vocab_size)
        """
        memory = self.encode(images)
        return self.decoder(memory)

    @torch.no_grad()
    def predict(self, images: torch.Tensor, chars: List[str]) -> List[str]:
        """Inference: trả về list chuỗi ký tự đã decode (bỏ pad token)."""
        self.eval()
        logits = self.forward(images)
        ids = logits.argmax(-1)          # (B, max_len)
        return [
            "".join(chars[i] for i in seq.tolist() if i != 0)
            for seq in ids
        ]


__all__ = ["SmallLPRNAR", "NARDecoder", "NARDecoderLayer", "smart_resize"]
