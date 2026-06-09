"""SlotLPR — non-autoregressive single-frame license plate OCR.

The model keeps SmallLPR's 2D visual encoding path, then predicts all character
slots in parallel with learned queries.  A small layout head predicts whether
the source crop is one-line or two-line; it is an auxiliary signal only.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .small_lpr import LearnablePositional2D, SmallLPRBackbone, _STNet


class SlotLPR(nn.Module):
    """Parallel slot decoder for structured license-plate OCR."""

    def __init__(
        self,
        vocab_size: int,
        *,
        max_slots: int = 13,
        d_model: int = 256,
        decoder_layers: int = 2,
        nhead: int = 4,
        dim_feedforward: int | None = None,
        dropout: float = 0.1,
        use_stn: bool = True,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.max_slots = max_slots
        self.d_model = d_model

        self.stn = _STNet() if use_stn else nn.Identity()
        self.backbone = SmallLPRBackbone(out_channels=256)
        self.proj = nn.Conv2d(256, d_model, kernel_size=1)
        self.pos_enc_2d = LearnablePositional2D(max_h=8, max_w=16, d_model=d_model)

        ff_dim = dim_feedforward or d_model * 4
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.slot_decoder = nn.TransformerDecoder(decoder_layer, num_layers=decoder_layers)
        self.slot_queries = nn.Parameter(torch.randn(1, max_slots, d_model) * 0.02)
        self.slot_norm = nn.LayerNorm(d_model)
        self.slot_heads = nn.ModuleList(nn.Linear(d_model, vocab_size) for _ in range(max_slots))

        self.layout_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 2),
        )

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        """Return 2D visual tokens as ``(B, 72, d_model)`` for 48x96 input."""
        images = self.stn(images)
        feat = self.backbone(images)
        feat = self.proj(feat)
        feat = feat.permute(0, 2, 3, 1)
        feat = self.pos_enc_2d(feat)
        batch, height, width, dim = feat.shape
        return feat.reshape(batch, height * width, dim)

    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        memory = self.encode(images)
        queries = self.slot_queries.expand(memory.size(0), -1, -1)
        slot_features = self.slot_decoder(tgt=queries, memory=memory)
        slot_features = self.slot_norm(slot_features)
        slot_logits = torch.stack(
            [head(slot_features[:, idx]) for idx, head in enumerate(self.slot_heads)],
            dim=1,
        )
        pooled = memory.mean(dim=1)
        return {
            "slot_logits": slot_logits,
            "layout_logits": self.layout_head(pooled),
        }
