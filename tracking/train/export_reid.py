"""
tracking/train/export_reid.py — Export trained VehicleReIDNet for inference.

boxmot (used by Ultralytics tracking) loads ReID models as either TorchScript
or ONNX. Raw state-dict .pt files are NOT supported — this script performs the
conversion after training is complete.

Outputs (written to weights/tracking/ by default):
  vehicle_reid.onnx          ← primary: loaded by api/core/tracker_adapter.py at startup
  vehicle_reid.torchscript   ← fallback: BoxMOT auto-detects format

Usage:
  python -m tracking.train.export_reid
  python -m tracking.train.export_reid --weights weights/tracking/vehicle_reid.pt
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch

from tracking.models.reid_net import VehicleReIDNet

logger = logging.getLogger(__name__)

_INPUT_SHAPE = (1, 3, 256, 128)   # (B, C, H, W) — standard Re-ID crop


def export_onnx(model: torch.nn.Module, out_path: Path) -> None:
    dummy = torch.zeros(_INPUT_SHAPE)
    torch.onnx.export(
        model,
        dummy,
        str(out_path),
        input_names=["input"],
        output_names=["output"],
        opset_version=12,
        dynamic_axes={
            "input":  {0: "batch", 2: "height", 3: "width"},  # accept any crop size
            "output": {0: "batch"},
        },
    )
    logger.info("ONNX  → %s", out_path)


def export_torchscript(model: torch.nn.Module, out_path: Path) -> None:
    dummy = torch.zeros(_INPUT_SHAPE)
    traced = torch.jit.trace(model, dummy)
    torch.jit.save(traced, str(out_path))
    logger.info("TorchScript → %s", out_path)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export VehicleReIDNet for tracking inference")
    p.add_argument(
        "--weights",
        type=Path,
        default=Path("weights/tracking/vehicle_reid.pt"),
        help="State-dict .pt file produced by train_reid.py",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("weights/tracking"),
    )
    p.add_argument("--embedding-dim", type=int, default=128)
    p.add_argument("--no-onnx",         action="store_true")
    p.add_argument("--no-torchscript",  action="store_true")
    return p.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = _parse_args()

    if not args.weights.exists():
        raise FileNotFoundError(f"Weights not found: {args.weights}")

    # Load state dict — num_ids=None removes the classifier head for inference
    model = VehicleReIDNet(embedding_dim=args.embedding_dim, num_ids=None)
    raw = torch.load(args.weights, map_location="cpu", weights_only=True)
    # Support both raw state-dicts and training checkpoints produced by train_reid.py
    # Training checkpoints are dicts with keys: "model", "optimiser", "scheduler", "epoch", ...
    state = raw.get("model", raw) if isinstance(raw, dict) and "model" in raw else raw
    # Strip classifier keys if present (they were only used during training)
    state = {k: v for k, v in state.items() if not k.startswith("classifier")}
    model.load_state_dict(state, strict=False)
    model.eval()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    if not args.no_onnx:
        export_onnx(model, args.output_dir / "vehicle_reid.onnx")

    if not args.no_torchscript:
        export_torchscript(model, args.output_dir / "vehicle_reid.torchscript")

    logger.info("Export complete.")


if __name__ == "__main__":
    main()
