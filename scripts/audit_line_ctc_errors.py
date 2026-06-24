"""
Audit SmallLPR-Line-CTC validation errors.

The script writes:
  - errors.csv: one row per wrong prediction
  - summary.md: aggregate metrics and actionable error clusters
  - samples/: optional copied example crops grouped by first error category

Example:
    /home/vietanh/anaconda3/envs/myenv/bin/python scripts/audit_line_ctc_errors.py \
        --checkpoint weights/ocr/small_lpr_line_ctc/line_ctc_reviewed_v1/small_lpr_line_ctc-epoch=048-val_acc=0.9222.ckpt
"""

from __future__ import annotations

import argparse
import csv
import os
import shutil
import sys
from argparse import Namespace
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

os.environ.setdefault("MPLCONFIGDIR", "/tmp")
os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")

import cv2
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
LPRNET_ROOT = ROOT / "LPRNet"
if str(LPRNET_ROOT) not in sys.path:
    sys.path.insert(0, str(LPRNET_ROOT))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from api.core.plate_format import is_vn_plate_text
from lprnet.small_lpr_line_ctc import ctc_decode_logits, line_ctc_greedy_decode
from lprnet.small_lpr_line_ctc_datamodule import (
    SmallLPRLineCTCDataset,
    collate_fn_line_ctc,
)
from lprnet.small_lpr_line_ctc_lightning import SmallLPRLineCTCLightning

DEFAULT_CONFIG = ROOT / "LPRNet/config/small_lpr_line_ctc_config.yaml"
DEFAULT_CKPT = (
    ROOT
    / "weights/ocr/small_lpr_line_ctc/line_ctc_cleaned_20260618_053309/small_lpr_line_ctc-epoch=015-val_acc=0.9399.ckpt"
)
DEFAULT_OUT_DIR = ROOT / "weights/ocr/small_lpr_line_ctc/line_ctc_reviewed_v1/error_audit"
SEPARATORS = {"-", ".", "[SEP]"}
DIGIT_TO_LETTER = {"0": "O", "1": "I", "2": "Z", "5": "S", "6": "G", "8": "B"}
LETTER_TO_DIGIT = {value: key for key, value in DIGIT_TO_LETTER.items()}


@dataclass(frozen=True)
class EditAudit:
    distance: int
    substitutions: tuple[tuple[str, str], ...]
    missing_gt: tuple[str, ...]
    extra_pred: tuple[str, ...]


@dataclass(frozen=True)
class ImageStats:
    width: int
    height: int
    aspect: float
    brightness: float
    contrast: float
    laplacian_var: float
    flags: tuple[str, ...]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit SmallLPR-Line-CTC OCR errors.")
    parser.add_argument("--checkpoint", default=str(DEFAULT_CKPT), help="Lightning checkpoint.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="Training YAML config.")
    parser.add_argument("--split", default="valid", choices=("train", "valid", "test"))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR), help="Audit output directory.")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument(
        "--max-samples", type=int, default=None, help="Limit samples for quick audits."
    )
    parser.add_argument(
        "--copy-samples", type=int, default=80, help="Number of wrong crops to copy."
    )
    parser.add_argument("--rare-threshold", type=int, default=200)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument(
        "--train-augment",
        action="store_true",
        help="Keep random train augmentations while auditing the train split.",
    )
    return parser.parse_args()


def main() -> None:
    cli = parse_args()
    out_dir = resolve_path(cli.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    config = load_config(resolve_path(cli.config), cli.batch_size)
    args = Namespace(**config)
    device = choose_device(cli.device)
    model = load_model(resolve_path(cli.checkpoint), args, device)

    dataset = prepare_dataset_for_audit(
        SmallLPRLineCTCDataset(args, cli.split),
        split=cli.split,
        use_train_augment=cli.train_augment,
    )
    train_counts = collect_train_char_counts(args)
    rows, counters = audit_dataset(
        model=model,
        dataset=dataset,
        args=args,
        device=device,
        batch_size=cli.batch_size,
        max_samples=cli.max_samples,
        rare_threshold=cli.rare_threshold,
        train_counts=train_counts,
    )

    write_csv(out_dir / "errors.csv", rows)
    write_summary(
        out_dir / "summary.md",
        rows=rows,
        counters=counters,
        checkpoint=resolve_path(cli.checkpoint),
        split=cli.split,
        dataset_size=(
            len(dataset) if cli.max_samples is None else min(len(dataset), cli.max_samples)
        ),
        device=device,
    )
    if cli.copy_samples > 0:
        copy_error_samples(rows, out_dir / "samples", limit=cli.copy_samples)

    print(f"Audit written to: {out_dir}")
    print(f"  errors.csv: {out_dir / 'errors.csv'}")
    print(f"  summary.md: {out_dir / 'summary.md'}")


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def load_config(path: Path, batch_size: int) -> dict:
    config = yaml.load(path.read_text(encoding="utf-8"), Loader=yaml.FullLoader)
    for key in ("train_dir", "valid_dir", "test_dir"):
        if not Path(config[key]).is_absolute():
            config[key] = str(ROOT / config[key])
    config["batch_size"] = batch_size
    config["num_workers"] = 0
    return config


def choose_device(value: str) -> torch.device:
    if value == "cpu":
        return torch.device("cpu")
    if value == "cuda":
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def prepare_dataset_for_audit(
    dataset: SmallLPRLineCTCDataset,
    *,
    split: str,
    use_train_augment: bool,
) -> SmallLPRLineCTCDataset:
    if split == "train" and not use_train_augment:
        dataset.transform = None
        dataset.img_paths = sorted(dataset.img_paths)
        print("Train audit uses raw crops: disabled random train augmentation.")
    return dataset


def load_model(checkpoint: Path, args: Namespace, device: torch.device) -> SmallLPRLineCTCLightning:
    model = SmallLPRLineCTCLightning(args).to(device).eval()
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    state_dict = dict(payload["state_dict"])
    legacy_one_line_decode = "model.one_line_head.weight" not in state_dict
    if legacy_one_line_decode and "model.global_head.weight" in state_dict:
        current = model.state_dict()
        state_dict["model.one_line_head.weight"] = state_dict["model.global_head.weight"].clone()
        state_dict["model.one_line_head.bias"] = state_dict["model.global_head.bias"].clone()
        state_dict["model.one_line_attention.weight"] = current[
            "model.one_line_attention.weight"
        ].clone()
        state_dict["model.one_line_attention.bias"] = current[
            "model.one_line_attention.bias"
        ].clone()
        print("Legacy checkpoint detected: auditing one-line decode with global_logits.")
    model.load_state_dict(state_dict, strict=True)
    model._legacy_global_one_line_decode = legacy_one_line_decode
    return model


def collect_train_char_counts(args: Namespace) -> Counter:
    counts: Counter = Counter()
    train_dir = Path(args.train_dir)
    if not train_dir.exists():
        return counts
    for path in train_dir.iterdir():
        if not path.is_file():
            continue
        for token in tokenize(path.stem.split("#")[0].upper()):
            if token not in SEPARATORS:
                counts[token] += 1
    return counts


def audit_dataset(
    *,
    model: SmallLPRLineCTCLightning,
    dataset: SmallLPRLineCTCDataset,
    args: Namespace,
    device: torch.device,
    batch_size: int,
    max_samples: int | None,
    rare_threshold: int,
    train_counts: Counter,
) -> tuple[list[dict[str, object]], Counter]:
    rows: list[dict[str, object]] = []
    counters: Counter = Counter()
    by_slice: dict[str, Counter] = defaultdict(Counter)

    total = len(dataset) if max_samples is None else min(len(dataset), max_samples)
    with torch.no_grad():
        for start in range(0, total, batch_size):
            stop = min(total, start + batch_size)
            items = [dataset[index] for index in range(start, stop)]
            batch = collate_fn_line_ctc(items)
            outputs = model.model(batch["images"].to(device))
            decode_outputs = outputs
            if getattr(model, "_legacy_global_one_line_decode", False):
                decode_outputs = dict(outputs)
                decode_outputs["one_line_logits"] = outputs["global_logits"]
            line_preds = line_ctc_greedy_decode(
                decode_outputs,
                args.chars,
                two_line_threshold=float(getattr(args, "two_line_threshold", 0.5)),
            )
            global_preds = ctc_decode_logits(outputs["global_logits"], args.chars)
            layout_probs = torch.softmax(outputs["layout_logits"], dim=-1).detach().cpu()
            layout_preds = layout_probs.argmax(dim=-1).tolist()

            for local_idx, gt in enumerate(batch["texts"]):
                index = start + local_idx
                source_path = Path(dataset.img_paths[index])
                pred = line_preds[local_idx]
                global_pred = global_preds[local_idx]
                has_sep = bool(batch["has_sep"][local_idx].item())
                layout = "two_line" if has_sep else "one_line"
                expected_layout_id = 1 if has_sep else 0
                layout_pred_id = int(layout_preds[local_idx])
                ok = pred == gt

                counters["total"] += 1
                counters["correct"] += int(ok)
                counters["global_correct"] += int(global_pred == gt)
                counters["layout_correct"] += int(layout_pred_id == expected_layout_id)
                by_slice[layout]["total"] += 1
                by_slice[layout]["correct"] += int(ok)
                by_slice[layout]["global_correct"] += int(global_pred == gt)
                by_slice[layout]["layout_correct"] += int(layout_pred_id == expected_layout_id)

                if ok:
                    continue

                edit = edit_audit(tokenize(pred), tokenize(gt))
                image_stats = inspect_image(source_path)
                categories = categorize_error(
                    gt=gt,
                    pred=pred,
                    global_pred=global_pred,
                    expected_layout_id=expected_layout_id,
                    layout_pred_id=layout_pred_id,
                    edit=edit,
                    image_flags=image_stats.flags,
                    rare_threshold=rare_threshold,
                    train_counts=train_counts,
                )
                counters["wrong"] += 1
                counters["edit_sum"] += edit.distance
                for category in categories:
                    counters[f"category:{category}"] += 1
                for pred_token, gt_token in edit.substitutions:
                    counters[f"sub:{pred_token}->{gt_token}"] += 1
                for token in edit.missing_gt:
                    counters[f"missing:{token}"] += 1
                for token in edit.extra_pred:
                    counters[f"extra:{token}"] += 1

                rows.append(
                    {
                        "index": index,
                        "path": str(source_path),
                        "gt": gt,
                        "pred": pred,
                        "global_pred": global_pred,
                        "layout": layout,
                        "layout_pred": "two_line" if layout_pred_id == 1 else "one_line",
                        "p_two_line": round(float(layout_probs[local_idx, 1]), 6),
                        "edit_distance": edit.distance,
                        "pred_len": len(tokenize(pred)),
                        "gt_len": len(tokenize(gt)),
                        "categories": "|".join(categories),
                        "substitutions": "|".join(f"{p}->{g}" for p, g in edit.substitutions),
                        "missing_gt": "|".join(edit.missing_gt),
                        "extra_pred": "|".join(edit.extra_pred),
                        "pred_valid_format": is_vn_plate_text(pred),
                        "gt_valid_format": is_vn_plate_text(gt),
                        "global_was_correct": global_pred == gt,
                        "width": image_stats.width,
                        "height": image_stats.height,
                        "aspect": round(image_stats.aspect, 4),
                        "brightness": round(image_stats.brightness, 3),
                        "contrast": round(image_stats.contrast, 3),
                        "laplacian_var": round(image_stats.laplacian_var, 3),
                        "image_flags": "|".join(image_stats.flags),
                    }
                )

    for slice_name, slice_counter in by_slice.items():
        for key, value in slice_counter.items():
            counters[f"{slice_name}:{key}"] = value
    return rows, counters


def tokenize(text: str) -> list[str]:
    tokens: list[str] = []
    idx = 0
    while idx < len(text):
        if text.startswith("[SEP]", idx):
            tokens.append("[SEP]")
            idx += len("[SEP]")
        else:
            tokens.append(text[idx])
            idx += 1
    return tokens


def edit_audit(pred_tokens: list[str], gt_tokens: list[str]) -> EditAudit:
    n = len(pred_tokens)
    m = len(gt_tokens)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    back = [[""] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        dp[i][0] = i
        back[i][0] = "delete"
    for j in range(1, m + 1):
        dp[0][j] = j
        back[0][j] = "insert"
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if pred_tokens[i - 1] == gt_tokens[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
                back[i][j] = "match"
                continue
            choices = (
                (dp[i - 1][j - 1] + 1, "substitute"),
                (dp[i - 1][j] + 1, "delete"),
                (dp[i][j - 1] + 1, "insert"),
            )
            dp[i][j], back[i][j] = min(choices, key=lambda item: item[0])

    i, j = n, m
    substitutions: list[tuple[str, str]] = []
    missing_gt: list[str] = []
    extra_pred: list[str] = []
    while i > 0 or j > 0:
        action = back[i][j]
        if action == "match":
            i -= 1
            j -= 1
        elif action == "substitute":
            substitutions.append((pred_tokens[i - 1], gt_tokens[j - 1]))
            i -= 1
            j -= 1
        elif action == "delete":
            extra_pred.append(pred_tokens[i - 1])
            i -= 1
        elif action == "insert":
            missing_gt.append(gt_tokens[j - 1])
            j -= 1
        else:
            break

    return EditAudit(
        distance=dp[n][m],
        substitutions=tuple(reversed(substitutions)),
        missing_gt=tuple(reversed(missing_gt)),
        extra_pred=tuple(reversed(extra_pred)),
    )


def inspect_image(path: Path) -> ImageStats:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        return ImageStats(0, 0, 0.0, 0.0, 0.0, 0.0, ("read_error",))
    height, width = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    brightness = float(gray.mean())
    contrast = float(gray.std())
    laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    aspect = width / height if height > 0 else 0.0
    flags: list[str] = []
    if width < 48 or height < 20:
        flags.append("low_res")
    if aspect < 0.9:
        flags.append("tall_crop")
    if aspect > 4.5:
        flags.append("very_wide_crop")
    if contrast < 22:
        flags.append("low_contrast")
    if brightness < 55:
        flags.append("dark")
    if brightness > 210:
        flags.append("bright")
    if laplacian_var < 80:
        flags.append("blur_or_smooth")
    return ImageStats(width, height, aspect, brightness, contrast, laplacian_var, tuple(flags))


def categorize_error(
    *,
    gt: str,
    pred: str,
    global_pred: str,
    expected_layout_id: int,
    layout_pred_id: int,
    edit: EditAudit,
    image_flags: Iterable[str],
    rare_threshold: int,
    train_counts: Counter,
) -> tuple[str, ...]:
    categories: list[str] = []
    gt_tokens = tokenize(gt)
    pred_tokens = tokenize(pred)
    literal_touched = any(
        token in SEPARATORS for token in [*edit.missing_gt, *edit.extra_pred]
    ) or any(p in SEPARATORS or g in SEPARATORS for p, g in edit.substitutions)

    if layout_pred_id != expected_layout_id:
        categories.append("layout_error")
    if len(pred_tokens) != len(gt_tokens):
        categories.append("length_error")
    if literal_touched:
        categories.append("separator_error")
    if any({p, g} == {"-", "[SEP]"} for p, g in edit.substitutions):
        categories.append("dash_sep_confusion")
    if any(is_digit_letter_confusion(p, g) for p, g in edit.substitutions):
        categories.append("digit_letter_confusion")
    if any(p.isdigit() and g.isdigit() for p, g in edit.substitutions):
        categories.append("digit_digit_confusion")
    if any(p.isalpha() and g.isalpha() and p != g for p, g in edit.substitutions):
        categories.append("letter_letter_confusion")
    if any("Đ" in pair for pair in edit.substitutions):
        categories.append("diacritic_confusion")
    if global_pred == gt and pred != gt:
        categories.append("line_decode_regression")
    if not is_vn_plate_text(pred):
        categories.append("invalid_pred_format")
    if not is_vn_plate_text(gt):
        categories.append("gt_outside_format_templates")

    rare_gt = [
        token
        for token in gt_tokens
        if token not in SEPARATORS and train_counts.get(token, 0) < rare_threshold
    ]
    if rare_gt:
        categories.append("rare_gt_char")

    for flag in image_flags:
        categories.append(f"image_{flag}")
    return tuple(categories or ["uncategorized"])


def is_digit_letter_confusion(pred: str, gt: str) -> bool:
    if pred in DIGIT_TO_LETTER and DIGIT_TO_LETTER[pred] == gt:
        return True
    if pred in LETTER_TO_DIGIT and LETTER_TO_DIGIT[pred] == gt:
        return True
    return (pred.isdigit() and gt.isalpha()) or (pred.isalpha() and gt.isdigit())


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_summary(
    path: Path,
    *,
    rows: list[dict[str, object]],
    counters: Counter,
    checkpoint: Path,
    split: str,
    dataset_size: int,
    device: torch.device,
) -> None:
    wrong = counters["wrong"]
    total = counters["total"]
    correct = counters["correct"]
    lines = [
        "# SmallLPR-Line-CTC Error Audit",
        "",
        f"- Checkpoint: `{checkpoint}`",
        f"- Split: `{split}`",
        f"- Device: `{device}`",
        f"- Evaluated samples: {total} (dataset target: {dataset_size})",
        f"- Exact-match accuracy: {safe_pct(correct, total)} ({correct}/{total})",
        f"- Wrong predictions: {wrong}",
        f"- Global CTC accuracy: {safe_pct(counters['global_correct'], total)}",
        f"- Layout accuracy: {safe_pct(counters['layout_correct'], total)}",
        f"- Mean edit distance per wrong sample: {safe_div(counters['edit_sum'], wrong):.3f}",
        "",
        "## Layout Slices",
        "",
    ]
    for name in ("one_line", "two_line"):
        slice_total = counters[f"{name}:total"]
        lines.append(
            f"- {name}: acc {safe_pct(counters[f'{name}:correct'], slice_total)}, "
            f"global {safe_pct(counters[f'{name}:global_correct'], slice_total)}, "
            f"layout {safe_pct(counters[f'{name}:layout_correct'], slice_total)} "
            f"({slice_total} samples)"
        )

    lines.extend(
        [
            "",
            "## Error Categories",
            "",
            *format_counter_lines(counters, "category:", wrong, limit=30),
            "",
            "## Top Substitutions",
            "",
            *format_counter_lines(counters, "sub:", wrong, limit=25),
            "",
            "## Top Missing GT Tokens",
            "",
            *format_counter_lines(counters, "missing:", wrong, limit=20),
            "",
            "## Top Extra Pred Tokens",
            "",
            *format_counter_lines(counters, "extra:", wrong, limit=20),
            "",
            "## Suggested Next Experiments",
            "",
            "- Inspect `samples/` visually and mark label/crop-noise cases before changing architecture.",
            "- Compare `line_decode_regression` rows against global predictions; these are likely fixable in decoding/routing.",
            "- Add beam search plus Vietnamese plate template filtering for separator and length errors.",
            "- Increase CTC horizontal resolution or adopt SVTRv2-style multi-size resizing for low-resolution and one-line failures.",
            "- Try DCTC/Focal CTC on the current model before a larger transformer rewrite.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def format_counter_lines(
    counter: Counter, prefix: str, denominator: int, *, limit: int
) -> list[str]:
    items = [
        (key.removeprefix(prefix), value)
        for key, value in counter.items()
        if key.startswith(prefix)
    ]
    items.sort(key=lambda item: (-item[1], item[0]))
    if not items:
        return ["- None"]
    return [
        f"- `{name}`: {value} ({safe_pct(value, denominator)})" for name, value in items[:limit]
    ]


def safe_pct(numerator: int | float, denominator: int | float) -> str:
    if denominator == 0:
        return "0.00%"
    return f"{100.0 * float(numerator) / float(denominator):.2f}%"


def safe_div(numerator: int | float, denominator: int | float) -> float:
    if denominator == 0:
        return 0.0
    return float(numerator) / float(denominator)


def copy_error_samples(rows: list[dict[str, object]], out_dir: Path, *, limit: int) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for row in sorted(
        rows, key=lambda item: (-int(item["edit_distance"]), str(item["categories"]))
    )[:limit]:
        categories = str(row["categories"]).split("|")
        category = categories[0] if categories else "uncategorized"
        category_dir = out_dir / safe_name(category)
        category_dir.mkdir(parents=True, exist_ok=True)
        source = Path(str(row["path"]))
        target = category_dir / (
            f"idx_{int(row['index']):05d}_"
            f"gt_{safe_name(str(row['gt']))}_pred_{safe_name(str(row['pred']))}"
            f"{source.suffix.lower() or '.jpg'}"
        )
        if source.exists():
            shutil.copy2(source, target)


def safe_name(value: str) -> str:
    safe = []
    for char in value:
        safe.append(char if char.isalnum() or char in {"-", "_"} else "_")
    return "".join(safe)[:160] or "empty"


if __name__ == "__main__":
    main()
