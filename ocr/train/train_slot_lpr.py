from __future__ import annotations

import argparse
import time
from argparse import Namespace
from pathlib import Path

import lightning as L
import torch
import yaml
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint

from ocr.train.slot_lpr_datamodule import SlotLPRDataModule
from ocr.train.slot_lpr_lightning import SlotLPRLightning


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "configs" / "ocr" / "slot_lpr.yaml"


def load_args(config_path: str | Path, overrides: argparse.Namespace) -> Namespace:
    with open(config_path, "r", encoding="utf-8") as handle:
        data = yaml.load(handle, Loader=yaml.FullLoader)

    for key in (
        "max_epochs",
        "batch_size",
        "num_workers",
        "subset_size",
        "precision",
        "lr",
    ):
        value = getattr(overrides, key, None)
        if value is not None:
            data[key] = value

    for key in ("train_dir", "valid_dir", "test_dir", "saving_ckpt"):
        data[key] = str((ROOT / data[key]).resolve()) if not Path(data[key]).is_absolute() else data[key]

    return Namespace(**data)


def build_trainer(args: Namespace) -> L.Trainer:
    ckpt_dir = Path(args.saving_ckpt)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = ModelCheckpoint(
        dirpath=ckpt_dir,
        filename="slot-lpr-{epoch:03d}-{val_acc:.4f}",
        monitor="val_acc",
        mode="max",
        save_top_k=3,
        save_last=True,
    )
    callbacks = [checkpoint, LearningRateMonitor(logging_interval="epoch")]
    return L.Trainer(
        accelerator="auto",
        devices="auto",
        precision=args.precision,
        max_epochs=args.max_epochs,
        gradient_clip_val=args.gradient_clip_val,
        callbacks=callbacks,
        log_every_n_steps=20,
    )


def train(args: Namespace) -> None:
    L.seed_everything(args.seed, workers=True)
    datamodule = SlotLPRDataModule(args)
    model = SlotLPRLightning(args)
    trainer = build_trainer(args)

    start = time.time()
    trainer.fit(model, datamodule=datamodule)
    elapsed = time.time() - start
    print(f"Training finished in {elapsed / 60:.2f} minutes")
    print(f"Best checkpoint: {trainer.checkpoint_callback.best_model_path}")
    print(f"Best val_acc: {float(trainer.checkpoint_callback.best_model_score or 0):.4f}")


def parse_cli() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train SlotLPR OCR model.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="Path to SlotLPR YAML config.")
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--subset-size", type=int, default=None, help="Use first N images per split.")
    parser.add_argument("--precision", default=None)
    parser.add_argument("--lr", type=float, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    cli = parse_cli()
    train(load_args(cli.config, cli))
