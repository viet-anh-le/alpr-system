"""
Training script cho SmallLPR-NAR.

Sử dụng:
    python scripts/train_small_lpr_nar.py [options]

Ví dụ:
    python scripts/train_small_lpr_nar.py
    python scripts/train_small_lpr_nar.py --epochs 100 --batch-size 64
    python scripts/train_small_lpr_nar.py --num-layers 3 --nhead 8
"""

from __future__ import annotations

import argparse
import sys
from argparse import Namespace
from datetime import datetime
from pathlib import Path

import lightning as L
import torch
import yaml
from lightning.pytorch.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
    RichProgressBar,
)
from lightning.pytorch.loggers import CSVLogger

# ─── sys.path ─────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]
LPRNET_ROOT = ROOT / "LPRNet"
if str(LPRNET_ROOT) not in sys.path:
    sys.path.insert(0, str(LPRNET_ROOT))

from lprnet.small_lpr_nar_datamodule import SmallLPRNARDataModule   # noqa: E402
from lprnet.small_lpr_nar_lightning import SmallLPRNARLightning      # noqa: E402

DEFAULT_CONFIG = ROOT / "LPRNet" / "config" / "small_lpr_nar_config.yaml"


# =============================================================================
# CLI
# =============================================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train SmallLPR-NAR (Non-Autoregressive) trên OCR biển số."
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--data-root", default=None, help="Ghi đè train_dir/valid_dir trong config.")
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--valid-split", default="valid")
    parser.add_argument("--out-dir", default="weights/ocr/small_lpr_nar")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--resume", default=None, help="Resume từ checkpoint Lightning.")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--d-model", type=int, default=None, dest="d_model")
    parser.add_argument("--max-len", type=int, default=None, dest="max_len")
    parser.add_argument("--num-layers", type=int, default=None, dest="num_layers")
    parser.add_argument("--nhead", type=int, default=None)
    parser.add_argument("--precision", default="32")
    parser.add_argument("--devices", default="1")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--accumulate-grad", type=int, default=None, dest="accumulate_grad", help="Số batch tích lũy gradient trước khi update.")
    return parser.parse_args()


# =============================================================================
# Config loading
# =============================================================================


def _abs(value: str | Path) -> Path:
    p = Path(value)
    return p if p.is_absolute() else ROOT / p


def load_config(cli: argparse.Namespace) -> Namespace:
    with Path(cli.config).open("r", encoding="utf-8") as f:
        data = yaml.load(f, Loader=yaml.FullLoader)

    # Data dirs
    if cli.data_root is not None:
        data_root = _abs(cli.data_root)
        data["train_dir"] = str(data_root / cli.train_split)
        data["valid_dir"] = str(data_root / cli.valid_split)
        data["test_dir"] = str(data_root / cli.valid_split)
    else:
        if not Path(data["train_dir"]).is_absolute():
            data["train_dir"] = str(ROOT / data["train_dir"])
            data["valid_dir"] = str(ROOT / data["valid_dir"])
            data["test_dir"] = str(ROOT / data["test_dir"])

    # CLI overrides
    overrides = {
        "max_epochs": cli.epochs,
        "batch_size": cli.batch_size,
        "lr": cli.lr,
        "d_model": cli.d_model,
        "max_len": cli.max_len,
        "num_layers": cli.num_layers,
        "nhead": cli.nhead,
    }
    for key, val in overrides.items():
        if val is not None:
            data[key] = val

    run_name = cli.run_name or datetime.now().strftime("nar_%Y%m%d_%H%M%S")
    ckpt_dir = _abs(cli.out_dir) / run_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    data["saving_ckpt"] = str(ckpt_dir)

    return Namespace(**data)


def _parse_devices(value: str) -> int | str:
    return int(value) if value.isdigit() else value


# =============================================================================
# Main training loop
# =============================================================================


def train() -> None:
    cli = parse_args()
    L.seed_everything(cli.seed, workers=True)
    torch.serialization.add_safe_globals([Namespace])

    args = load_config(cli)

    print("=" * 60)
    print("SmallLPR-NAR Training (Non-Autoregressive)")
    print(f"  Train dir  : {args.train_dir}")
    print(f"  Valid dir  : {args.valid_dir}")
    print(f"  Checkpoints: {args.saving_ckpt}")
    print(f"  Vocab size : {len(args.chars)}")
    print(f"  d_model    : {getattr(args, 'd_model', 256)}")
    print(f"  max_len    : {getattr(args, 'max_len', 14)}")
    print(f"  num_layers : {getattr(args, 'num_layers', 2)}")
    print(f"  Batch size : {args.batch_size}")
    print(f"  LR         : {args.lr}")
    print("=" * 60)

    model = SmallLPRNARLightning(args)
    datamodule = SmallLPRNARDataModule(args)

    checkpoint_cb = ModelCheckpoint(
        dirpath=args.saving_ckpt,
        monitor="val_acc",
        mode="max",
        filename="small_lpr_nar-{epoch:03d}-{val_acc:.4f}",
        save_top_k=3,
        save_last=True,
    )

    accumulate_grad = cli.accumulate_grad or getattr(args, "accumulate_grad_batches", 1)

    trainer = L.Trainer(
        accelerator="auto",
        devices=_parse_devices(cli.devices),
        precision=cli.precision,
        max_epochs=args.max_epochs,
        gradient_clip_val=args.gradient_clip_val,
        accumulate_grad_batches=accumulate_grad,
        logger=CSVLogger(save_dir=args.saving_ckpt, name="logs"),
        callbacks=[
            RichProgressBar(),
            checkpoint_cb,
            EarlyStopping(monitor="val_acc", mode="max", patience=40, verbose=True),
            LearningRateMonitor(logging_interval="epoch"),
        ],
        log_every_n_steps=10,
    )

    trainer.fit(model, datamodule=datamodule, ckpt_path=cli.resume)
    print(f"\nBest checkpoint: {checkpoint_cb.best_model_path}")
    print(f"Best val_acc   : {checkpoint_cb.best_model_score:.4f}")


if __name__ == "__main__":
    train()
