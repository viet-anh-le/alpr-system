"""
Train SmallLPR-Line-CTC on data/datasets/ocr.

Examples:
    python scripts/train_small_lpr_line_ctc.py
    python scripts/train_small_lpr_line_ctc.py --epochs 100 --batch-size 64
    python scripts/train_small_lpr_line_ctc.py --data-root data/datasets/ocr --devices 1
    python scripts/train_small_lpr_line_ctc.py --config LPRNet/config/small_lpr_line_ctc_no_global_config.yaml
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

from lprnet.small_lpr_line_ctc_datamodule import SmallLPRLineCTCDataModule  # noqa: E402
from lprnet.small_lpr_line_ctc_lightning import SmallLPRLineCTCLightning  # noqa: E402

DEFAULT_CONFIG = ROOT / "LPRNet" / "config" / "small_lpr_line_ctc_config.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train SmallLPR-Line-CTC on Vietnamese plate OCR crops.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="Path to YAML config.")
    parser.add_argument("--data-root", default=None, help="Root containing train/ and valid/ splits.")
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--valid-split", default="valid")
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Checkpoint root. Defaults to saving_ckpt from the selected config.",
    )
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--resume", default=None, help="Resume from a Lightning checkpoint.")
    parser.add_argument(
        "--init-from",
        default=None,
        help="Initialize model weights from a checkpoint, then start a fresh optimizer/trainer run.",
    )
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--d-model", type=int, default=None, dest="d_model")
    parser.add_argument("--exclude-paths-file", default=None, dest="exclude_paths_file")
    parser.add_argument("--global-loss-weight", type=float, default=None, dest="global_loss_weight")
    parser.add_argument("--one-line-loss-weight", type=float, default=None, dest="one_line_loss_weight")
    parser.add_argument("--top-loss-weight", type=float, default=None, dest="top_loss_weight")
    parser.add_argument("--bottom-loss-weight", type=float, default=None, dest="bottom_loss_weight")
    parser.add_argument("--layout-loss-weight", type=float, default=None, dest="layout_loss_weight")
    parser.add_argument("--line-prior-strength", type=float, default=None, dest="line_prior_strength")
    parser.add_argument("--label-mode", choices=("raw", "alnum"), default=None)
    parser.add_argument("--line-separator", default=None)
    parser.add_argument("--decode-mode", choices=("global", "layout"), default=None)
    parser.add_argument("--augment", dest="augment", action="store_true", default=None)
    parser.add_argument("--no-augment", dest="augment", action="store_false")
    parser.add_argument("--use-stn", dest="use_stn", action="store_true", default=None)
    parser.add_argument("--no-use-stn", dest="use_stn", action="store_false")
    parser.add_argument("--use-pos-enc", dest="use_pos_enc", action="store_true", default=None)
    parser.add_argument("--no-use-pos-enc", dest="use_pos_enc", action="store_false")
    parser.add_argument("--use-global-head", dest="use_global_head", action="store_true", default=None)
    parser.add_argument("--no-use-global-head", dest="use_global_head", action="store_false")
    parser.add_argument("--precision", default="32")
    parser.add_argument("--devices", default="1")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--accumulate-grad", type=int, default=None, dest="accumulate_grad")
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
            if not Path(data[key]).is_absolute():
                data[key] = str(ROOT / data[key])

    overrides = {
        "max_epochs": cli.epochs,
        "batch_size": cli.batch_size,
        "lr": cli.lr,
        "d_model": cli.d_model,
        "exclude_paths_file": cli.exclude_paths_file,
        "global_loss_weight": cli.global_loss_weight,
        "one_line_loss_weight": cli.one_line_loss_weight,
        "top_loss_weight": cli.top_loss_weight,
        "bottom_loss_weight": cli.bottom_loss_weight,
        "layout_loss_weight": cli.layout_loss_weight,
        "line_prior_strength": cli.line_prior_strength,
        "label_mode": cli.label_mode,
        "line_separator": cli.line_separator,
        "decode_mode": cli.decode_mode,
        "augment": cli.augment,
        "use_stn": cli.use_stn,
        "use_pos_enc": cli.use_pos_enc,
        "use_global_head": cli.use_global_head,
    }
    for key, value in overrides.items():
        if value is not None:
            data[key] = value

    run_name = cli.run_name or datetime.now().strftime("line_ctc_%Y%m%d_%H%M%S")
    output_root = cli.out_dir or data.get("saving_ckpt", "weights/ocr/small_lpr_line_ctc")
    ckpt_dir = _abs(output_root) / run_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    data["saving_ckpt"] = str(ckpt_dir)
    return Namespace(**data)


def _parse_devices(value: str) -> int | str:
    return int(value) if value.isdigit() else value


def _resolve_checkpoint(value: str | None) -> str | None:
    return str(_abs(value)) if value else None


def _validate_checkpoint_mode(cli: argparse.Namespace) -> None:
    if cli.resume and cli.init_from:
        raise ValueError("Use only one of --resume or --init-from.")


def _with_legacy_one_line_weights(
    model: SmallLPRLineCTCLightning,
    state_dict: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    compatible = dict(state_dict)
    current = model.state_dict()
    if "model.one_line_head.weight" not in compatible and "model.global_head.weight" in compatible:
        compatible["model.one_line_head.weight"] = compatible["model.global_head.weight"].clone()
        compatible["model.one_line_head.bias"] = compatible["model.global_head.bias"].clone()
        compatible["model.one_line_attention.weight"] = current["model.one_line_attention.weight"].clone()
        compatible["model.one_line_attention.bias"] = current["model.one_line_attention.bias"].clone()
        print("  Legacy init: copied global_head -> one_line_head and kept fresh one_line_attention.")
    if not model.use_global_head:
        compatible.pop("model.global_head.weight", None)
        compatible.pop("model.global_head.bias", None)
    return compatible


def _load_init_weights(model: SmallLPRLineCTCLightning, checkpoint: str) -> None:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state_dict = payload["state_dict"] if "state_dict" in payload else payload
    compatible = _with_legacy_one_line_weights(model, state_dict)
    missing, unexpected = model.load_state_dict(compatible, strict=False)
    if unexpected:
        raise RuntimeError(f"Unexpected keys while loading {checkpoint}: {unexpected}")
    if missing:
        raise RuntimeError(f"Missing keys while loading {checkpoint}: {missing}")


def _build_model(cli: argparse.Namespace, args: Namespace) -> tuple[SmallLPRLineCTCLightning, str | None]:
    resume_ckpt = _resolve_checkpoint(cli.resume)
    init_ckpt = _resolve_checkpoint(cli.init_from)
    model = SmallLPRLineCTCLightning(args)
    if init_ckpt:
        print(f"  Init from  : {init_ckpt}")
        _load_init_weights(model, init_ckpt)
        return model, None
    return model, resume_ckpt


def train() -> None:
    cli = parse_args()
    _validate_checkpoint_mode(cli)
    L.seed_everything(cli.seed, workers=True)
    torch.serialization.add_safe_globals([Namespace])

    args = load_config(cli)

    print("=" * 60)
    print("SmallLPR-Line-CTC Training")
    print(f"  Train dir : {args.train_dir}")
    print(f"  Valid dir : {args.valid_dir}")
    print(f"  Checkpoints: {args.saving_ckpt}")
    print(f"  Vocab size : {len(args.chars)}")
    print(f"  d_model    : {getattr(args, 'd_model', 256)}")
    print(f"  Batch size : {args.batch_size}")
    print(f"  LR         : {args.lr}")
    print(f"  Augment    : {getattr(args, 'augment', True)}")
    print(f"  Use STN    : {getattr(args, 'use_stn', True)}")
    print(f"  Use 2D PE  : {getattr(args, 'use_pos_enc', True)}")
    print(f"  Global head: {getattr(args, 'use_global_head', True)}")
    print(f"  Label mode : {getattr(args, 'label_mode', 'raw')}")
    print(f"  Line sep   : {getattr(args, 'line_separator', '[SEP]')!r}")
    print(f"  Decode mode: {getattr(args, 'decode_mode', 'layout')}")
    print(f"  Global loss: {getattr(args, 'global_loss_weight', 1.0)}")
    print(f"  One-line loss: {getattr(args, 'one_line_loss_weight', 1.0)}")
    print(f"  Top loss   : {getattr(args, 'top_loss_weight', 1.0)}")
    print(f"  Bottom loss: {getattr(args, 'bottom_loss_weight', 1.0)}")
    print(f"  Layout loss: {getattr(args, 'layout_loss_weight', 0.2)}")
    print("=" * 60)

    model, resume_ckpt = _build_model(cli, args)
    datamodule = SmallLPRLineCTCDataModule(args)

    checkpoint_cb = ModelCheckpoint(
        dirpath=args.saving_ckpt,
        monitor="val_acc",
        mode="max",
        filename="small_lpr_line_ctc-{epoch:03d}-{val_acc:.4f}",
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

    trainer.fit(model, datamodule=datamodule, ckpt_path=resume_ckpt)
    print(f"\nBest checkpoint: {checkpoint_cb.best_model_path}")
    print(f"Best val_acc   : {checkpoint_cb.best_model_score:.4f}")


if __name__ == "__main__":
    train()
