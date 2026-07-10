from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

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
)
from ocr.parseq_model import load_parseq_checkpoint, predict_strings_with_confidence  # noqa: E402
from ocr.train_parseq import path_from_root, positional_char_accuracy, resolve_device  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a fine-tuned PARSeq OCR checkpoint.")
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint produced by ocr.train_parseq.")
    parser.add_argument("--data-root", default="data/datasets/ocr")
    parser.add_argument("--split", default="valid")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu", "mps"])
    parser.add_argument("--charset", default=None, help="Override checkpoint charset if needed.")
    parser.add_argument("--variant", default="parseq")
    parser.add_argument("--decode-ar", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--refine-iters", type=int, default=1)
    parser.add_argument("--image-width", type=int, default=None)
    parser.add_argument("--image-height", type=int, default=None)
    parser.add_argument("--max-label-length", type=int, default=None)
    parser.add_argument("--preds-csv", default=None)
    parser.add_argument("--subset-size", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    checkpoint_path = path_from_root(args.checkpoint)
    model, checkpoint = load_parseq_checkpoint(
        checkpoint_path,
        variant=args.variant,
        charset=args.charset,
        decode_ar=args.decode_ar,
        refine_iters=args.refine_iters,
        device=device,
    )
    model.eval()

    image_width = int(args.image_width or checkpoint.get("image_width", 128))
    image_height = int(args.image_height or checkpoint.get("image_height", 32))
    max_label_length = int(args.max_label_length or checkpoint.get("max_label_length", 25))
    charset = args.charset or checkpoint.get("charset", DEFAULT_PARSEQ_VN_CHARSET)

    split_dir = path_from_root(args.data_root) / args.split
    dataset = FilenamePlateDataset(
        split_dir,
        charset=charset,
        max_label_length=max_label_length,
        transform=make_parseq_transform(image_width=image_width, image_height=image_height, augment=False),
        subset_size=args.subset_size,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=parseq_collate,
        pin_memory=device.type == "cuda",
    )

    seq_acc, char_acc, rows = evaluate_with_rows(model, loader, device)
    print(f"Samples: {len(dataset)}")
    print(f"seq_acc: {seq_acc:.4f}")
    print(f"char_acc: {char_acc:.4f}")

    if args.preds_csv:
        out_path = path_from_root(args.preds_csv)
    else:
        out_path = checkpoint_path.with_name(f"{checkpoint_path.stem}_{args.split}_preds.csv")
    write_predictions(out_path, rows)
    print(f"Predictions CSV: {out_path}")


@torch.no_grad()
def evaluate_with_rows(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[float, float, list[dict[str, str | float]]]:
    total_seq = 0
    correct_seq = 0
    correct_char = 0
    total_char = 0
    rows: list[dict[str, str | float]] = []
    for images, labels, paths in loader:
        images = images.to(device, non_blocking=True)
        preds, confidences = predict_strings_with_confidence(model, images)
        correct_seq += sum(pred == target for pred, target in zip(preds, labels, strict=False))
        total_seq += len(labels)
        char_ok, char_total = positional_char_accuracy(preds, labels)
        correct_char += char_ok
        total_char += char_total
        for path, target, pred, confidence in zip(paths, labels, preds, confidences, strict=False):
            rows.append(
                {
                    "path": path,
                    "target": target,
                    "prediction": pred,
                    "correct": str(pred == target),
                    "confidence": float(confidence),
                }
            )
    return (
        correct_seq / total_seq if total_seq else 0.0,
        correct_char / total_char if total_char else 0.0,
        rows,
    )


def write_predictions(path: Path, rows: list[dict[str, str | float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["path", "target", "prediction", "correct", "confidence"])
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
