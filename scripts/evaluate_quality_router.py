#!/usr/bin/env python3
"""Evaluate the Plate Quality Router with confusion matrices.

Examples:
  # Evaluate a YOLO classification router on an ImageFolder test split.
  python scripts/evaluate_quality_router.py \
    --model runs/classify/plate_quality_legibility4/weights/best.pt \
    --data data/lplcv2_quality/legibility4 \
    --split test \
    --output-dir runs/eval/quality_router

  # Evaluate from an existing CSV with true/predicted labels.
  python scripts/evaluate_quality_router.py \
    --predictions runs/eval/router_predictions.csv \
    --output-dir runs/eval/quality_router
"""
from __future__ import annotations

import argparse
import csv
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

LEGIBILITY_LABELS = ("illegible", "poor", "good", "perfect")
BINARY_LABELS = ("unsuitable", "suitable")
SUITABLE_LABELS = {"good", "perfect"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

_LABEL_ALIASES = {
    "0": "illegible",
    "1": "poor",
    "2": "good",
    "3": "perfect",
    "unreadable": "illegible",
    "bad": "poor",
}


def normalize_label(value: Any) -> str:
    text = str(value).strip().lower()
    text = _LABEL_ALIASES.get(text, text)
    if text not in LEGIBILITY_LABELS:
        raise ValueError(f"Unknown legibility label: {value!r}")
    return text


def binary_label(label: str) -> str:
    return "suitable" if normalize_label(label) in SUITABLE_LABELS else "unsuitable"


def build_confusion_matrix(
    y_true: Iterable[str],
    y_pred: Iterable[str],
    *,
    labels: tuple[str, ...],
) -> list[list[int]]:
    index = {label: i for i, label in enumerate(labels)}
    matrix = [[0 for _ in labels] for _ in labels]
    for true_label, pred_label in zip(y_true, y_pred):
        true_norm = normalize_label(true_label) if labels == LEGIBILITY_LABELS else str(true_label)
        pred_norm = normalize_label(pred_label) if labels == LEGIBILITY_LABELS else str(pred_label)
        if true_norm not in index:
            raise ValueError(f"Unknown true label {true_norm!r}; expected one of {labels}")
        if pred_norm not in index:
            raise ValueError(f"Unknown predicted label {pred_norm!r}; expected one of {labels}")
        matrix[index[true_norm]][index[pred_norm]] += 1
    return matrix


def compute_class_metrics(
    matrix: list[list[int]],
    labels: tuple[str, ...],
) -> dict[str, dict[str, float | int]]:
    metrics: dict[str, dict[str, float | int]] = {}
    for i, label in enumerate(labels):
        tp = matrix[i][i]
        support = sum(matrix[i])
        predicted = sum(row[i] for row in matrix)
        fp = predicted - tp
        fn = support - tp
        precision = _safe_div(tp, tp + fp)
        recall = _safe_div(tp, tp + fn)
        f1 = _safe_div(2 * precision * recall, precision + recall)
        metrics[label] = {
            "precision": round(precision, 6),
            "recall": round(recall, 6),
            "f1": round(f1, 6),
            "support": support,
        }
    return metrics


def evaluate_predictions(
    y_true: list[str],
    y_pred: list[str],
) -> dict[str, Any]:
    if len(y_true) != len(y_pred):
        raise ValueError(f"y_true/y_pred length mismatch: {len(y_true)} != {len(y_pred)}")
    if not y_true:
        raise ValueError("No predictions to evaluate")

    true_norm = [normalize_label(label) for label in y_true]
    pred_norm = [normalize_label(label) for label in y_pred]
    matrix = build_confusion_matrix(true_norm, pred_norm, labels=LEGIBILITY_LABELS)
    per_class = compute_class_metrics(matrix, LEGIBILITY_LABELS)

    true_binary = [binary_label(label) for label in true_norm]
    pred_binary = [binary_label(label) for label in pred_norm]
    binary_matrix = build_binary_confusion_matrix(true_binary, pred_binary)
    binary_per_class = compute_binary_class_metrics(binary_matrix)

    false_suitable_count = sum(
        1
        for true_label, pred_label in zip(true_norm, pred_norm)
        if true_label not in SUITABLE_LABELS and pred_label in SUITABLE_LABELS
    )
    unsafe_gt_count = sum(1 for label in true_norm if label not in SUITABLE_LABELS)

    report = {
        "n": len(true_norm),
        "labels": list(LEGIBILITY_LABELS),
        "confusion_matrix": matrix,
        "per_class": per_class,
        "accuracy": round(_accuracy(matrix), 6),
        "macro_f1": round(_macro_metric(per_class, "f1"), 6),
        "macro_recall": round(_macro_metric(per_class, "recall"), 6),
        "binary": {
            "labels": list(BINARY_LABELS),
            "confusion_matrix": binary_matrix,
            "per_class": binary_per_class,
            "accuracy": round(_accuracy(binary_matrix), 6),
            "macro_f1": round(_macro_metric(binary_per_class, "f1"), 6),
            "false_suitable_count": false_suitable_count,
            "unsuitable_gt_count": unsafe_gt_count,
            "false_suitable_rate": round(_safe_div(false_suitable_count, unsafe_gt_count), 6),
        },
        "critical": {
            "poor_recall": per_class["poor"]["recall"],
            "illegible_recall": per_class["illegible"]["recall"],
            "false_suitable_rate": round(_safe_div(false_suitable_count, unsafe_gt_count), 6),
        },
    }
    return report


def build_binary_confusion_matrix(
    y_true: Iterable[str],
    y_pred: Iterable[str],
) -> list[list[int]]:
    index = {label: i for i, label in enumerate(BINARY_LABELS)}
    matrix = [[0 for _ in BINARY_LABELS] for _ in BINARY_LABELS]
    for true_label, pred_label in zip(y_true, y_pred):
        matrix[index[true_label]][index[pred_label]] += 1
    return matrix


def compute_binary_class_metrics(
    matrix: list[list[int]],
) -> dict[str, dict[str, float | int]]:
    return compute_class_metrics(matrix, BINARY_LABELS)


def load_predictions_csv(path: Path) -> list[tuple[str, str, str]]:
    rows: list[tuple[str, str, str]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            true_raw = _first_field(row, ("true", "label", "gt", "ground_truth", "y_true"))
            pred_raw = _first_field(row, ("pred", "prediction", "predicted", "y_pred"))
            if true_raw is None or pred_raw is None:
                raise ValueError(
                    "Predictions CSV must contain true/label/gt and pred/prediction columns"
                )
            image_path = _first_field(row, ("path", "image", "image_path", "file")) or ""
            rows.append((normalize_label(true_raw), normalize_label(pred_raw), image_path))
    return rows


def discover_imagefolder_samples(
    data_dir: Path,
    *,
    split: str = "test",
) -> list[tuple[Path, str]]:
    root = data_dir / split if split and (data_dir / split).exists() else data_dir
    samples: list[tuple[Path, str]] = []
    for class_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        try:
            label = normalize_label(class_dir.name)
        except ValueError:
            continue
        for image_path in sorted(class_dir.rglob("*")):
            if image_path.is_file() and image_path.suffix.lower() in IMAGE_EXTENSIONS:
                samples.append((image_path, label))
    return samples


def predict_with_yolo(
    model_path: Path,
    samples: list[tuple[Path, str]],
    *,
    imgsz: int,
    device: str,
    batch: int,
) -> list[tuple[str, str, str]]:
    from ultralytics import YOLO

    model = YOLO(str(model_path))
    rows: list[tuple[str, str, str]] = []
    for start in range(0, len(samples), batch):
        chunk = samples[start : start + batch]
        paths = [str(path) for path, _ in chunk]
        results = model.predict(
            source=paths,
            imgsz=imgsz,
            device=device or None,
            verbose=False,
        )
        for (path, true_label), result in zip(chunk, results):
            pred_label = _result_label(result)
            rows.append((true_label, pred_label, str(path)))
    return rows


def save_report(
    report: dict[str, Any],
    output_dir: Path,
    *,
    predictions: list[tuple[str, str, str]] | None = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    _write_matrix_csv(
        output_dir / "confusion_matrix.csv",
        report["labels"],
        report["confusion_matrix"],
    )
    _write_matrix_csv(
        output_dir / "binary_confusion_matrix.csv",
        report["binary"]["labels"],
        report["binary"]["confusion_matrix"],
    )
    if predictions is not None:
        with (output_dir / "predictions.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["path", "true", "pred"])
            writer.writeheader()
            for true_label, pred_label, path in predictions:
                writer.writerow({"path": path, "true": true_label, "pred": pred_label})


def print_summary(report: dict[str, Any]) -> None:
    print("Quality Router Evaluation")
    print(f"n={report['n']} accuracy={report['accuracy']:.4f} macro_f1={report['macro_f1']:.4f}")
    print(
        "critical: "
        f"poor_recall={report['critical']['poor_recall']:.4f} "
        f"illegible_recall={report['critical']['illegible_recall']:.4f} "
        f"false_suitable_rate={report['critical']['false_suitable_rate']:.4f}"
    )
    print("confusion matrix rows=true cols=pred:")
    print("true\\pred," + ",".join(report["labels"]))
    for label, row in zip(report["labels"], report["confusion_matrix"]):
        print(label + "," + ",".join(str(value) for value in row))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--model", type=Path, help="YOLO-cls router checkpoint.")
    source.add_argument("--predictions", type=Path, help="CSV with true and pred columns.")
    parser.add_argument("--data", type=Path, help="ImageFolder dataset root for --model.")
    parser.add_argument("--split", default="test", help="Split directory name under --data.")
    parser.add_argument("--output-dir", type=Path, default=Path("runs/eval/quality_router"))
    parser.add_argument("--imgsz", type=int, default=96)
    parser.add_argument("--device", default="")
    parser.add_argument("--batch", type=int, default=64)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.predictions is not None:
        rows = load_predictions_csv(args.predictions)
    else:
        if args.data is None:
            raise SystemExit("--data is required when evaluating --model")
        samples = discover_imagefolder_samples(args.data, split=args.split)
        if not samples:
            raise SystemExit(f"No samples found under {args.data} split={args.split}")
        rows = predict_with_yolo(
            args.model,
            samples,
            imgsz=args.imgsz,
            device=args.device,
            batch=args.batch,
        )

    y_true = [true_label for true_label, _, _ in rows]
    y_pred = [pred_label for _, pred_label, _ in rows]
    report = evaluate_predictions(y_true, y_pred)
    save_report(report, args.output_dir, predictions=rows)
    print_summary(report)
    print(f"Wrote report to {args.output_dir}")


def _safe_div(num: float, den: float) -> float:
    return float(num) / float(den) if den else 0.0


def _accuracy(matrix: list[list[int]]) -> float:
    total = sum(sum(row) for row in matrix)
    correct = sum(matrix[i][i] for i in range(len(matrix)))
    return _safe_div(correct, total)


def _macro_metric(
    metrics: dict[str, dict[str, float | int]],
    key: str,
) -> float:
    return _safe_div(sum(float(values[key]) for values in metrics.values()), len(metrics))


def _first_field(row: dict[str, str], names: tuple[str, ...]) -> str | None:
    for name in names:
        if name in row and row[name] not in (None, ""):
            return row[name]
    return None


def _result_label(result: Any) -> str:
    probs = getattr(result, "probs", None)
    if probs is None:
        raise ValueError("YOLO result has no classification probabilities")
    top1 = int(getattr(probs, "top1"))
    names = getattr(result, "names", None) or {}
    if isinstance(names, dict):
        raw = names.get(top1, str(top1))
    else:
        raw = names[top1]
    return normalize_label(raw)


def _write_matrix_csv(path: Path, labels: list[str], matrix: list[list[int]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["true\\pred", *labels])
        for label, row in zip(labels, matrix):
            writer.writerow([label, *row])


if __name__ == "__main__":
    main()
