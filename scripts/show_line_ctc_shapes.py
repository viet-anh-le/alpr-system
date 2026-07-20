"""Show tensor shapes at each stage of SmallLPR-Line-CTC.

Runs one forward pass on a dummy plate batch and prints the shape after every
major block plus every output head. Nothing is trained/loaded — this is a
structural inspection, so it needs no checkpoint or dataset.

Usage:
    python scripts/show_line_ctc_shapes.py                 # default batch=2
    python scripts/show_line_ctc_shapes.py --batch 4
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "LPRNet"))

from lprnet.small_lpr_line_ctc import SmallLPRLineCTC  # noqa: E402

CONFIG = ROOT / "LPRNet" / "config" / "small_lpr_line_ctc_config.yaml"


def _row(name: str, shape) -> None:
    print(f"  {name:<34} {tuple(shape)}")


def _fmt(n: int) -> str:
    """Human-readable count, e.g. 1_234_567 -> '1,234,567 (1.23 M)'."""
    return f"{n:,} ({n / 1e6:.2f} M)"


def show_param_counts(model) -> None:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # float32 -> 4 bytes/param
    size_mb = total * 4 / (1024 ** 2)

    print("\nPARAMETERS")
    print(f"  {'total params':<34} {_fmt(total)}")
    print(f"  {'trainable params':<34} {_fmt(trainable)}")
    print(f"  {'approx size (fp32)':<34} {size_mb:.2f} MB")

    print("\n  per top-level module:")
    for name, module in model.named_children():
        n = sum(p.numel() for p in module.parameters())
        if n:
            print(f"    {name:<28} {_fmt(n)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=2)
    args = parser.parse_args()

    chars = yaml.unsafe_load(CONFIG.read_text())["chars"]
    vocab = len(chars)

    model = SmallLPRLineCTC(vocab_size=vocab).eval()
    B = args.batch
    x = torch.randn(B, 3, 48, 96)  # (B, 3, H=48, W=96)

    print(f"vocab (len chars) = {vocab}   (index 0 = '<blank>')")

    show_param_counts(model)

    print(f"\nINPUT  (B, 3, H, W)")
    _row("input", x.shape)

    # ── intermediate blocks via forward hooks ────────────────────────────────
    print("\nBACKBONE / ENCODER")
    hooks = []
    for name, module in [
        ("stn", model.stn),
        ("backbone.stem", model.backbone.stem),
        ("backbone.stage1", model.backbone.stage1),
        ("backbone.stage2", model.backbone.stage2),
        ("backbone.stage3", model.backbone.stage3),
        ("proj (Conv1x1)", model.proj),
    ]:
        hooks.append(module.register_forward_hook(
            lambda _m, _i, out, n=name: _row(n, out.shape)
        ))

    with torch.no_grad():
        feat = model.encode_2d(x)          # triggers hooks above
    for h in hooks:
        h.remove()
    _row("encode_2d -> feat (B,H,W,D)", feat.shape)

    # ── output heads ─────────────────────────────────────────────────────────
    with torch.no_grad():
        out = model(x)

    print("\nOUTPUT HEADS  (CTC logits: (B, T, vocab);  T = so timesteps)")
    order = [
        ("layout_logits  (1-line vs 2-line)", "layout_logits"),
        ("global_logits", "global_logits"),
        ("one_line_logits", "one_line_logits"),
        ("top_logits", "top_logits"),
        ("bottom_logits", "bottom_logits"),
    ]
    for label, key in order:
        if key in out:
            _row(label, out[key].shape)

    # ── decoded strings ──────────────────────────────────────────────────────
    print("\nGREEDY DECODE (dummy input -> nonsense text, chi de xem pipeline chay)")
    preds = model.greedy_decode(x, chars)
    for i, p in enumerate(preds):
        print(f"  sample[{i}] -> {p!r}")


if __name__ == "__main__":
    main()
