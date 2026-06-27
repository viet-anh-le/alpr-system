"""
SmallLPR-Line-CTC.

Segment-free OCR model for Vietnamese plates with an auxiliary visual layout
classifier and soft line attention.  It does not use character boxes or plate
aspect-ratio rules.
"""

from __future__ import annotations

from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from .small_lpr import (
    LearnablePositional2D,
    SmallLPRBackbone,
    _STNet,
    smart_resize,
)

_GLOBAL_T_STEPS: int = 72
_ONE_LINE_T_STEPS: int = 16
_LINE_T_STEPS: int = 12


def ctc_decode_logits(logits: torch.Tensor, chars: List[str]) -> List[str]:
    """Greedy CTC decode for logits shaped (B, T, C)."""
    n_vocab = len(chars)
    preds = logits.argmax(dim=-1).clamp(0, n_vocab - 1)
    results: List[str] = []
    for seq in preds:
        out: List[str] = []
        prev = -1
        for token in seq.tolist():
            if token != prev:
                if token != 0:
                    out.append(chars[token])
                prev = token
        results.append("".join(out))
    return results


def line_ctc_greedy_decode(
    outputs: Dict[str, torch.Tensor],
    chars: List[str],
    *,
    two_line_threshold: float = 0.5,
    line_separator: str = "[SEP]",
) -> List[str]:
    """Decode with visual layout logits: one-line head or top+separator+bottom heads."""
    batch_size = outputs["global_logits"].size(0)
    one_line_logits = outputs.get("one_line_logits", outputs["global_logits"])
    one_line_texts = ctc_decode_logits(one_line_logits, chars)
    top_texts = ctc_decode_logits(outputs["top_logits"], chars)
    bottom_texts = ctc_decode_logits(outputs["bottom_logits"], chars)
    layout_probs = torch.softmax(outputs["layout_logits"], dim=-1)

    decoded: List[str] = []
    for idx in range(batch_size):
        if float(layout_probs[idx, 1]) >= two_line_threshold:
            decoded.append(f"{top_texts[idx]}{line_separator}{bottom_texts[idx]}")
        else:
            decoded.append(one_line_texts[idx])
    return decoded


class SmallLPRLineCTC(nn.Module):
    """SmallLPR encoder with global CTC, two soft line CTC heads, and layout head."""

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        backbone_ch: int = 256,
        line_prior_strength: float = 1.0,
        use_stn: bool = True,
        use_pos_enc: bool = True,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.line_prior_strength = float(line_prior_strength)
        self.use_stn = bool(use_stn)
        self.use_pos_enc = bool(use_pos_enc)

        self.stn = _STNet()
        self.backbone = SmallLPRBackbone(out_channels=backbone_ch)
        self.proj = nn.Conv2d(backbone_ch, d_model, kernel_size=1, bias=False)
        self.pos_enc_2d = LearnablePositional2D(max_h=8, max_w=16, d_model=d_model)
        self.enc_norm = nn.LayerNorm(d_model)

        self.global_head = nn.Linear(d_model, vocab_size)
        self.one_line_head = nn.Linear(d_model, vocab_size)
        self.top_head = nn.Linear(d_model, vocab_size)
        self.bottom_head = nn.Linear(d_model, vocab_size)
        self.layout_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 2),
        )
        self.one_line_attention = nn.Conv2d(d_model, 1, kernel_size=1)
        self.line_attention = nn.Conv2d(d_model, 2, kernel_size=1)
        self._init_heads()

    def _init_heads(self) -> None:
        for head in (self.global_head, self.one_line_head, self.top_head, self.bottom_head):
            nn.init.trunc_normal_(head.weight, std=0.02)
            nn.init.zeros_(head.bias)
        nn.init.zeros_(self.one_line_attention.weight)
        nn.init.zeros_(self.one_line_attention.bias)
        nn.init.zeros_(self.line_attention.weight)
        nn.init.zeros_(self.line_attention.bias)

    def encode_2d(self, images: torch.Tensor) -> torch.Tensor:
        """Return encoded feature map shaped (B, H, W, D)."""
        x = self.stn(images) if self.use_stn else images
        x = self.backbone(x)
        x = self.proj(x)
        x = x.permute(0, 2, 3, 1)
        if self.use_pos_enc:
            x = self.pos_enc_2d(x)
        return self.enc_norm(x)

    def _one_line_features(self, feat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feat_bdhw = feat.permute(0, 3, 1, 2)
        attn_logits = self.one_line_attention(feat_bdhw)
        attention = torch.softmax(attn_logits, dim=2)
        line_feat = torch.einsum("bdhw,bqhw->bqwd", feat_bdhw, attention)
        line_feat = line_feat[:, 0]
        if line_feat.size(1) != _ONE_LINE_T_STEPS:
            line_feat = F.interpolate(
                line_feat.permute(0, 2, 1),
                size=_ONE_LINE_T_STEPS,
                mode="linear",
                align_corners=False,
            ).permute(0, 2, 1)
        return line_feat, attention[:, 0]

    def _line_features(self, feat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feat_bdhw = feat.permute(0, 3, 1, 2)
        _, _, height, _ = feat_bdhw.shape
        attn_logits = self.line_attention(feat_bdhw)
        if self.line_prior_strength != 0.0:
            y = torch.linspace(
                -1.0,
                1.0,
                height,
                device=feat.device,
                dtype=feat.dtype,
            ).view(1, 1, height, 1)
            prior = torch.cat((-y, y), dim=1) * self.line_prior_strength
            attn_logits = attn_logits + prior
        attention = torch.softmax(attn_logits, dim=2)
        line_feat = torch.einsum("bdhw,blhw->blwd", feat_bdhw, attention)
        return line_feat, attention

    def forward(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        feat = self.encode_2d(images)
        batch, height, width, dim = feat.shape

        global_seq = feat.reshape(batch, height * width, dim)
        global_logits = self.global_head(global_seq)

        pooled = feat.mean(dim=(1, 2))
        layout_logits = self.layout_head(pooled)

        one_line_feat, one_line_attention = self._one_line_features(feat)
        one_line_logits = self.one_line_head(one_line_feat)

        line_feat, attention = self._line_features(feat)
        top_logits = self.top_head(line_feat[:, 0])
        bottom_logits = self.bottom_head(line_feat[:, 1])

        return {
            "global_logits": global_logits,
            "one_line_logits": one_line_logits,
            "top_logits": top_logits,
            "bottom_logits": bottom_logits,
            "layout_logits": layout_logits,
            "one_line_attention": one_line_attention,
            "top_attention": attention[:, 0],
            "bottom_attention": attention[:, 1],
        }

    @torch.no_grad()
    def greedy_decode(self, images: torch.Tensor, chars: List[str]) -> List[str]:
        self.eval()
        return line_ctc_greedy_decode(self.forward(images), chars)


__all__ = [
    "SmallLPRLineCTC",
    "ctc_decode_logits",
    "line_ctc_greedy_decode",
    "smart_resize",
    "_GLOBAL_T_STEPS",
    "_ONE_LINE_T_STEPS",
    "_LINE_T_STEPS",
]
