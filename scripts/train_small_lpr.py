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

from lprnet.small_lpr_datamodule import SmallLPRDataModule  # noqa: E402
from lprnet.small_lpr_lightning import SmallLPRLightning  # noqa: E402


DEFAULT_CONFIG = ROOT / "LPRNet" / "config" / "small_lpr_config.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train SmallLPR on filename-labeled OCR plate crops.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="SmallLPR YAML config.")
    parser.add_argument("--data-root", default="data/datasets/ocr", help="Dataset root containing train/ and valid/.")
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--valid-split", default="valid")
    parser.add_argument("--out-dir", default="weights/ocr/small_lpr")
    parser.add_argument("--run-name", default=None, help="Checkpoint/log subdirectory name.")
    parser.add_argument("--resume", default=None, help="Resume from a SmallLPR Lightning checkpoint.")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--precision", default="32")
    parser.add_argument("--devices", default="1")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def path_from_root(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def load_config(cli: argparse.Namespace) -> Namespace:
    with Path(cli.config).open("r", encoding="utf-8") as handle:
        data = yaml.load(handle, Loader=yaml.FullLoader)

    data_root = path_from_root(cli.data_root)
    data["train_dir"] = str(data_root / cli.train_split)
    data["valid_dir"] = str(data_root / cli.valid_split)
    data["test_dir"] = str(data_root / cli.valid_split)

    if cli.epochs is not None:
        data["max_epochs"] = cli.epochs
    if cli.batch_size is not None:
        data["batch_size"] = cli.batch_size
    if cli.lr is not None:
        data["lr"] = cli.lr

    run_name = cli.run_name or datetime.now().strftime("small_lpr_%Y%m%d_%H%M%S")
    ckpt_dir = path_from_root(cli.out_dir) / run_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    data["saving_ckpt"] = str(ckpt_dir)
    return Namespace(**data)


def parse_devices(value: str) -> int | str:
    return int(value) if value.isdigit() else value


def train() -> None:
    cli = parse_args()
    L.seed_everything(cli.seed, workers=True)
    torch.serialization.add_safe_globals([Namespace])

    args = load_config(cli)
    model = SmallLPRLightning(args)
    datamodule = SmallLPRDataModule(args)

    checkpoint = ModelCheckpoint(
        dirpath=args.saving_ckpt,
        monitor="val_acc",
        mode="max",
        filename="small_lpr-{epoch:03d}-{val_acc:.4f}",
        save_top_k=3,
        save_last=True,
    )
    trainer = L.Trainer(
        accelerator="auto",
        devices=parse_devices(cli.devices),
        precision=cli.precision,
        max_epochs=args.max_epochs,
        gradient_clip_val=args.gradient_clip_val,
        logger=CSVLogger(save_dir=args.saving_ckpt, name="logs"),
        callbacks=[
            RichProgressBar(),
            checkpoint,
            EarlyStopping(monitor="val_acc", mode="max", patience=50, verbose=True),
            LearningRateMonitor(logging_interval="step"),
        ],
    )

    print(f"Training SmallLPR with train_dir={args.train_dir}")
    print(f"Validation dir={args.valid_dir}")
    print(f"Checkpoints={args.saving_ckpt}")
    trainer.fit(model, datamodule=datamodule, ckpt_path=cli.resume)
    print(f"Best checkpoint: {checkpoint.best_model_path}")


if __name__ == "__main__":
    train()
