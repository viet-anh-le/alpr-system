from __future__ import annotations

import argparse
import csv
import random
import sys
from contextlib import nullcontext
from pathlib import Path
from typing import Iterator

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ocr.parseq_dataset import (  # noqa: E402
    DEFAULT_PARSEQ_VN_CHARSET,
    FilenamePlateDataset,
    make_parseq_transform,
    parseq_collate,
    scan_split_stats,
)
from ocr.parseq_model import (  # noqa: E402
    checkpoint_payload,
    configure_parseq_charset,
    load_parseq_from_torchhub,
    predict_strings,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune PARSeq on filename-labeled OCR plate crops.")
    parser.add_argument("--data-root", default="data/datasets/ocr", help="Dataset root containing train/ and valid/.")
    parser.add_argument("--train-split", default="train", help="Train subdirectory under data-root.")
    parser.add_argument("--valid-split", default="valid", help="Validation subdirectory under data-root.")
    parser.add_argument("--out-dir", default="weights/ocr/parseq", help="Directory for checkpoints and logs.")
    parser.add_argument("--run-name", default="parseq_vn_plate", help="Checkpoint/log filename prefix.")
    parser.add_argument("--variant", default="parseq", choices=["parseq", "parseq_tiny", "parseq_patch16_224"])
    parser.add_argument("--pretrained", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--decode-ar", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--refine-iters", type=int, default=1)
    parser.add_argument("--charset", default=DEFAULT_PARSEQ_VN_CHARSET)
    parser.add_argument("--max-label-length", type=int, default=25)
    parser.add_argument("--image-width", type=int, default=128)
    parser.add_argument("--image-height", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--min-epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--lr-factor", type=float, default=0.5)
    parser.add_argument("--lr-patience", type=int, default=4)
    parser.add_argument("--early-stop-patience", type=int, default=12)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu", "mps"])
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--subset-size", type=int, default=None, help="Use first N images per split for a smoke run.")
    parser.add_argument("--skip-invalid-labels", action="store_true")
    parser.add_argument("--resume", default=None, help="Resume model weights from a checkpoint produced by this script.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)
    paths = resolve_paths(args)
    paths["out_dir"].mkdir(parents=True, exist_ok=True)

    print_dataset_summary(args, paths)
    train_loader, valid_loader = build_loaders(args, paths, device)

    model = load_parseq_from_torchhub(
        variant=args.variant,
        pretrained=args.pretrained,
        decode_ar=args.decode_ar,
        refine_iters=args.refine_iters,
    )
    resize_report = configure_parseq_charset(model, args.charset)
    print(
        "PARSeq charset:",
        resize_report.charset,
        "| copied head rows:",
        resize_report.copied_head_rows,
        "| copied embedding rows:",
        resize_report.copied_embedding_rows,
        "| initialized:",
        ",".join(resize_report.initialized_tokens) or "none",
    )
    model.to(device)

    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["state_dict"], strict=True)
        print(f"Loaded resume checkpoint: {args.resume}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=args.lr_factor,
        patience=args.lr_patience,
    )

    best_metric = -1.0
    bad_epochs = 0
    ckpt_path = paths["out_dir"] / f"{args.run_name}_best.pt"
    last_path = paths["out_dir"] / f"{args.run_name}_last.pt"
    log_path = paths["out_dir"] / f"{args.run_name}_log.csv"
    write_log_header(log_path)

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device, args)
        val_seq_acc, val_char_acc = evaluate(model, valid_loader, device)
        scheduler.step(val_seq_acc)
        lr = optimizer.param_groups[0]["lr"]

        append_log(log_path, epoch, train_loss, val_seq_acc, val_char_acc, lr)
        print(
            f"Epoch {epoch:03d}/{args.epochs} | "
            f"loss={train_loss:.4f} | val_seq_acc={val_seq_acc:.4f} | "
            f"val_char_acc={val_char_acc:.4f} | lr={lr:.2e}"
        )

        if val_seq_acc > best_metric:
            best_metric = val_seq_acc
            bad_epochs = 0
            torch.save(
                checkpoint_payload(model=model, optimizer=optimizer, epoch=epoch, args=args, best_metric=best_metric),
                ckpt_path,
            )
            print(f"Saved best checkpoint: {ckpt_path}")
        elif epoch >= args.min_epochs:
            bad_epochs += 1
            if bad_epochs >= args.early_stop_patience:
                print(f"Early stop at epoch {epoch}; best val_seq_acc={best_metric:.4f}")
                break

        torch.save(
            checkpoint_payload(model=model, optimizer=optimizer, epoch=epoch, args=args, best_metric=best_metric),
            last_path,
        )

    print(f"Best checkpoint: {ckpt_path}")
    print(f"Training log: {log_path}")


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false.")
    return torch.device(requested)


def resolve_paths(args: argparse.Namespace) -> dict[str, Path]:
    data_root = path_from_root(args.data_root)
    return {
        "data_root": data_root,
        "train_dir": data_root / args.train_split,
        "valid_dir": data_root / args.valid_split,
        "out_dir": path_from_root(args.out_dir),
    }


def path_from_root(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def print_dataset_summary(args: argparse.Namespace, paths: dict[str, Path]) -> None:
    train_stats = scan_split_stats(paths["train_dir"], charset=args.charset)
    valid_stats = scan_split_stats(paths["valid_dir"], charset=args.charset)
    print(
        f"Train: {train_stats.count} images | max label len={train_stats.max_label_length} | chars={train_stats.charset}"
    )
    print(
        f"Valid: {valid_stats.count} images | max label len={valid_stats.max_label_length} | chars={valid_stats.charset}"
    )


def build_loaders(
    args: argparse.Namespace,
    paths: dict[str, Path],
    device: torch.device,
) -> tuple[DataLoader, DataLoader]:
    train_transform = make_parseq_transform(
        image_width=args.image_width,
        image_height=args.image_height,
        augment=True,
    )
    eval_transform = make_parseq_transform(
        image_width=args.image_width,
        image_height=args.image_height,
        augment=False,
    )
    train_dataset = FilenamePlateDataset(
        paths["train_dir"],
        charset=args.charset,
        max_label_length=args.max_label_length,
        transform=train_transform,
        subset_size=args.subset_size,
        skip_invalid=args.skip_invalid_labels,
    )
    valid_dataset = FilenamePlateDataset(
        paths["valid_dir"],
        charset=args.charset,
        max_label_length=args.max_label_length,
        transform=eval_transform,
        subset_size=args.subset_size,
        skip_invalid=args.skip_invalid_labels,
    )
    pin_memory = device.type == "cuda"
    return (
        DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            collate_fn=parseq_collate,
            pin_memory=pin_memory,
        ),
        DataLoader(
            valid_dataset,
            batch_size=args.batch_size * 2,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=parseq_collate,
            pin_memory=pin_memory,
        ),
    )


def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    args: argparse.Namespace,
) -> float:
    model.train()
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp and device.type == "cuda")
    total_loss = 0.0
    batches = 0
    for images, labels, _paths in loader:
        images = images.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with autocast_context(args.amp, device):
            _logits, loss, _loss_numel = model.forward_logits_loss(images, labels)
        scaler.scale(loss).backward()
        if args.grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        total_loss += float(loss.detach().cpu())
        batches += 1
    return total_loss / max(batches, 1)


def autocast_context(enabled: bool, device: torch.device) -> Iterator:
    if enabled and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


@torch.no_grad()
def evaluate(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> tuple[float, float]:
    model.eval()
    total_seq = 0
    correct_seq = 0
    correct_char = 0
    total_char = 0
    for images, labels, _paths in loader:
        images = images.to(device, non_blocking=True)
        preds = predict_strings(model, images)
        correct_seq += sum(pred == target for pred, target in zip(preds, labels, strict=False))
        total_seq += len(labels)
        char_ok, char_total = positional_char_accuracy(preds, labels)
        correct_char += char_ok
        total_char += char_total
    return (
        correct_seq / total_seq if total_seq else 0.0,
        correct_char / total_char if total_char else 0.0,
    )


def positional_char_accuracy(preds: list[str], targets: list[str]) -> tuple[int, int]:
    correct = 0
    total = 0
    for pred, target in zip(preds, targets, strict=False):
        width = max(len(pred), len(target))
        total += width
        correct += sum(pred[index] == target[index] for index in range(min(len(pred), len(target))))
    return correct, total


def write_log_header(path: Path) -> None:
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["epoch", "train_loss", "val_seq_acc", "val_char_acc", "lr"])


def append_log(path: Path, epoch: int, train_loss: float, val_seq_acc: float, val_char_acc: float, lr: float) -> None:
    with open(path, "a", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([epoch, f"{train_loss:.6f}", f"{val_seq_acc:.6f}", f"{val_char_acc:.6f}", f"{lr:.8g}"])


if __name__ == "__main__":
    main()
