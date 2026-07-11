"""
Training script cho SmallLPR-CTC.

Sử dụng:
    python scripts/train_small_lpr_ctc.py [options]

Ví dụ:
    python scripts/train_small_lpr_ctc.py
    python scripts/train_small_lpr_ctc.py --epochs 100 --batch-size 64
    python scripts/train_small_lpr_ctc.py --data-root data/datasets/ocr --devices 1
    python scripts/train_small_lpr_ctc.py \
        --finetune-from weights/ocr/small_lpr_ctc/.../small_lpr_ctc-epoch=055-val_acc=0.9358.ckpt \
        --lr 0.0001 --run-name ctc_finetune_ep55_lr1e4
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
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint, RichProgressBar
from lightning.pytorch.loggers import CSVLogger

# ─── Thiết lập sys.path ───────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]
LPRNET_ROOT = ROOT / "LPRNet"
if str(LPRNET_ROOT) not in sys.path:
    sys.path.insert(0, str(LPRNET_ROOT))

from lprnet.small_lpr_ctc_datamodule import SmallLPRCTCDataModule   # noqa: E402
from lprnet.small_lpr_ctc_lightning import SmallLPRCTCLightning     # noqa: E402

# ─── Default paths ────────────────────────────────────────────────────────────
DEFAULT_CONFIG = ROOT / "LPRNet" / "config" / "small_lpr_ctc_config.yaml"


# =============================================================================
# CLI
# =============================================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train SmallLPR-CTC (CTC head) trên OCR biển số.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="Path đến file YAML config.")
    parser.add_argument("--data-root", default=None, help="Root chứa train/ và valid/. Ghi đè config.")
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--valid-split", default="valid")
    parser.add_argument("--out-dir", default="weights/ocr/small_lpr_ctc")
    parser.add_argument("--run-name", default=None, help="Tên sub-directory cho checkpoint/log.")
    parser.add_argument(
        "--resume",
        default=None,
        help="Resume đầy đủ từ checkpoint Lightning, gồm optimizer/scheduler state.",
    )
    parser.add_argument(
        "--finetune-from",
        default=None,
        help="Load weight-only từ checkpoint rồi train run mới với optimizer/scheduler mới.",
    )
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--d-model", type=int, default=None, dest="d_model")
    parser.add_argument("--precision", default="32")
    parser.add_argument("--devices", default="1")
    parser.add_argument("--seed", type=int, default=42)
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
    else:
        # config có thể dùng path tương đối từ ROOT
        data_root = ROOT
        # Nếu train_dir trong config là tương đối, resolve từ ROOT
        if not Path(data["train_dir"]).is_absolute():
            data["train_dir"] = str(ROOT / data["train_dir"])
            data["valid_dir"] = str(ROOT / data["valid_dir"])
            data["test_dir"]  = str(ROOT / data["test_dir"])
        # Đặt lại về ROOT để bỏ qua logic bên dưới
        data_root = None

    if data_root is not None:
        data["train_dir"] = str(data_root / cli.train_split)
        data["valid_dir"] = str(data_root / cli.valid_split)
        data["test_dir"]  = str(data_root / cli.valid_split)

    # Override từ CLI
    if cli.epochs is not None:
        data["max_epochs"] = cli.epochs
    if cli.batch_size is not None:
        data["batch_size"] = cli.batch_size
    if cli.lr is not None:
        data["lr"] = cli.lr
    if cli.d_model is not None:
        data["d_model"] = cli.d_model

    run_name = cli.run_name or datetime.now().strftime("ctc_%Y%m%d_%H%M%S")
    ckpt_dir = _abs(cli.out_dir) / run_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    data["saving_ckpt"] = str(ckpt_dir)

    return Namespace(**data)


def _parse_devices(value: str) -> int | str:
    return int(value) if value.isdigit() else value


def _validate_checkpoint_mode(cli: argparse.Namespace) -> None:
    if cli.resume and cli.finetune_from:
        raise ValueError("Chỉ dùng một trong hai: --resume hoặc --finetune-from.")


def _resolve_checkpoint(value: str | None) -> str | None:
    return str(_abs(value)) if value else None


def _build_model(cli: argparse.Namespace, args: Namespace) -> tuple[SmallLPRCTCLightning, str | None]:
    resume_ckpt = _resolve_checkpoint(cli.resume)
    finetune_ckpt = _resolve_checkpoint(cli.finetune_from)

    if finetune_ckpt:
        print(f"  Finetune from: {finetune_ckpt}")
        model = SmallLPRCTCLightning.load_from_checkpoint(finetune_ckpt, args=args)
        return model, None

    return SmallLPRCTCLightning(args), resume_ckpt


# =============================================================================
# Main training loop
# =============================================================================


def train() -> None:
    cli = parse_args()
    _validate_checkpoint_mode(cli)
    L.seed_everything(cli.seed, workers=True)
    torch.serialization.add_safe_globals([Namespace])

    args = load_config(cli)

    print("=" * 60)
    print("SmallLPR-CTC Training")
    print(f"  Train dir : {args.train_dir}")
    print(f"  Valid dir : {args.valid_dir}")
    print(f"  Checkpoints: {args.saving_ckpt}")
    print(f"  Vocab size : {len(args.chars)}")
    print(f"  d_model    : {getattr(args, 'd_model', 256)}")
    print(f"  Batch size : {args.batch_size}")
    print(f"  LR         : {args.lr}")
    print(f"  Scheduler  : {getattr(args, 'scheduler', 'cosine')}")
    print("=" * 60)

    model, resume_ckpt = _build_model(cli, args)
    datamodule = SmallLPRCTCDataModule(args)

    checkpoint_cb = ModelCheckpoint(
        dirpath=args.saving_ckpt,
        monitor="val_acc",
        mode="max",
        filename="small_lpr_ctc-{epoch:03d}-{val_acc:.4f}",
        save_top_k=3,
        save_last=True,
    )

    trainer = L.Trainer(
        accelerator="auto",
        devices=_parse_devices(cli.devices),
        precision=cli.precision,
        max_epochs=args.max_epochs,
        gradient_clip_val=args.gradient_clip_val,
        logger=CSVLogger(save_dir=args.saving_ckpt, name="logs"),
        callbacks=[
            RichProgressBar(),
            checkpoint_cb,
            EarlyStopping(monitor="val_acc", mode="max", patience=40, verbose=True),
            LearningRateMonitor(logging_interval="epoch"),
        ],
        log_every_n_steps=10,
    )

    trainer.fit(model, datamodule=datamodule, ckpt_path=resume_ckpt)
    print(f"\nBest checkpoint: {checkpoint_cb.best_model_path}")
    print(f"Best val_acc   : {checkpoint_cb.best_model_score:.4f}")


if __name__ == "__main__":
    train()
