"""
Train the original LPRNet implementation on the current OCR crop dataset.

This runner keeps LPRNet's original blank-token convention: the blank token is
the last item in ``chars``. For the OCR dataset used in this project, labels are
normalized to alphanumeric text only by the datamodule when ``label_mode`` is
``alnum``.
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


ROOT = Path(__file__).resolve().parents[1]
LPRNET_ROOT = ROOT / "LPRNet"
if str(LPRNET_ROOT) not in sys.path:
    sys.path.insert(0, str(LPRNET_ROOT))

from lprnet.datamodule import DataModule  # noqa: E402
from lprnet.lprnet import LPRNet  # noqa: E402


DEFAULT_CONFIG = ROOT / "LPRNet" / "config" / "lprnet_ocr_config.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train original LPRNet on the OCR crop dataset.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="Path to YAML config.")
    parser.add_argument("--data-root", default=None, help="Root containing train/ and valid/.")
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--valid-split", default="valid")
    parser.add_argument("--out-dir", default="weights/ocr/lprnet")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--precision", default="32")
    parser.add_argument("--devices", default="1")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _abs(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def load_config(cli: argparse.Namespace) -> Namespace:
    with Path(cli.config).open("r", encoding="utf-8") as handle:
        data = yaml.load(handle, Loader=yaml.FullLoader)

    if cli.data_root is not None:
        data_root = _abs(cli.data_root)
        data["train_dir"] = str(data_root / cli.train_split)
        data["valid_dir"] = str(data_root / cli.valid_split)
        data["test_dir"] = str(data_root / cli.valid_split)
    else:
        for key in ("train_dir", "valid_dir", "test_dir"):
            if key in data and not Path(data[key]).is_absolute():
                data[key] = str(ROOT / data[key])

    if cli.epochs is not None:
        data["max_epochs"] = cli.epochs
    if cli.batch_size is not None:
        data["batch_size"] = cli.batch_size
    if cli.lr is not None:
        data["lr"] = cli.lr

    run_name = cli.run_name or datetime.now().strftime("lprnet_%Y%m%d_%H%M%S")
    checkpoint_dir = _abs(cli.out_dir) / run_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    data["saving_ckpt"] = str(checkpoint_dir)
    return Namespace(**data)


def _parse_devices(value: str) -> int | str:
    return int(value) if value.isdigit() else value


def train() -> None:
    cli = parse_args()
    L.seed_everything(cli.seed, workers=True)
    torch.serialization.add_safe_globals([Namespace])
    args = load_config(cli)

    print("=" * 60)
    print("LPRNet OCR Training")
    print(f"  Train dir  : {args.train_dir}")
    print(f"  Valid dir  : {args.valid_dir}")
    print(f"  Checkpoints: {args.saving_ckpt}")
    print(f"  Label mode : {getattr(args, 'label_mode', 'raw')}")
    print(f"  Image size : {args.img_size}")
    print(f"  T length   : {args.t_length}")
    print(f"  Vocab size : {len(args.chars)}")
    print(f"  Blank token: {args.chars[-1]}")
    print(f"  Batch size : {args.batch_size}")
    print(f"  LR         : {args.lr}")
    print("=" * 60)

    model = LPRNet(args)
    datamodule = DataModule(args)

    checkpoint_cb = ModelCheckpoint(
        dirpath=args.saving_ckpt,
        monitor="val-acc",
        mode="max",
        filename="lprnet-{epoch:03d}-{val-acc:.4f}",
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
            EarlyStopping(monitor="val-acc", mode="max", patience=40, verbose=True),
            LearningRateMonitor(logging_interval="epoch"),
        ],
        log_every_n_steps=10,
    )

    trainer.fit(model=model, datamodule=datamodule)
    print(f"\nBest checkpoint: {checkpoint_cb.best_model_path}")
    print(f"Best val-acc   : {checkpoint_cb.best_model_score:.4f}")


if __name__ == "__main__":
    train()
