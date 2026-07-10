"""
Summarize SmallLPR-Line-CTC ablation results into CSV and Markdown tables.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MATRIX = ROOT / "LPRNet/config/small_lpr_line_ctc_ablation.yaml"
DEFAULT_ROOT = ROOT / "weights/ocr/small_lpr_line_ctc_ablation"
DEFAULT_OUT = DEFAULT_ROOT / "summary"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize SmallLPR-Line-CTC ablation results.")
    parser.add_argument("--matrix", default=str(DEFAULT_MATRIX))
    parser.add_argument("--root", default=str(DEFAULT_ROOT))
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    return parser.parse_args()


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def load_matrix(path: str | Path) -> dict[str, Any]:
    data = yaml.safe_load(resolve_path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not isinstance(data.get("runs"), list):
        raise ValueError("Ablation matrix must contain a top-level runs list.")
    return data


def best_metrics_from_csv(path: Path) -> dict[str, float | int | str]:
    best: dict[str, float | int | str] | None = None
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if not row.get("val_acc"):
                continue
            current = {
                "epoch": int(float(row.get("epoch", "0") or 0)),
                "step": int(float(row.get("step", "0") or 0)),
                "val_acc": float(row["val_acc"]),
            }
            for key in (
                "val_loss",
                "val_global_acc",
                "val_layout_acc",
                "val_global_ctc_loss",
                "val_one_line_ctc_loss",
                "val_top_ctc_loss",
                "val_bottom_ctc_loss",
                "val_layout_loss",
            ):
                if row.get(key):
                    current[key] = float(row[key])
            if best is None or float(current["val_acc"]) > float(best["val_acc"]):
                best = current
    if best is None:
        raise ValueError(f"No val_acc rows found in {path}")
    return best


def collect_rows(matrix: dict[str, Any], root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for run in matrix.get("runs", []):
        run_id = run["id"]
        metrics = _metrics_for_run(root, run)
        rows.append(
            {
                "run": run_id,
                "description": run.get("description", ""),
                "data_aug": bool(run.get("augment", True)),
                "stn": bool(run.get("use_stn", True)),
                "pos_enc": bool(run.get("use_pos_enc", True)),
                "layout_heads": run.get("decode_mode", "layout") == "layout",
                "vertical_prior": float(run.get("line_prior_strength", 1.0)) != 0.0,
                "format_correction": bool(run.get("format_correction", False)),
                "accuracy": _accuracy(metrics),
                "val_acc": metrics.get("val_acc"),
                "exact_acc": metrics.get("exact_acc"),
                "global_acc": metrics.get("global_acc"),
                "layout_acc": metrics.get("layout_acc"),
                "valid_format_rate": metrics.get("valid_format_rate"),
                "val_loss": metrics.get("val_loss"),
                "epoch": metrics.get("epoch"),
                "status": metrics.get("status", "ok"),
            }
        )
    return rows


def _metrics_for_run(root: Path, run: dict[str, Any]) -> dict[str, Any]:
    eval_json = root / run["id"] / "eval_metrics.json"
    if eval_json.exists():
        return json.loads(eval_json.read_text(encoding="utf-8"))
    if run.get("source_run"):
        source_eval = root / str(run["source_run"]) / "eval_metrics.json"
        if source_eval.exists() and not run.get("format_correction", False):
            return json.loads(source_eval.read_text(encoding="utf-8"))
    metrics_csv = _find_metrics_csv(root / run.get("source_run", run["id"]))
    if metrics_csv is None:
        return {"status": "missing"}
    return best_metrics_from_csv(metrics_csv)


def _find_metrics_csv(run_dir: Path) -> Path | None:
    candidates = sorted(run_dir.glob("logs/version_*/metrics.csv"))
    if not candidates:
        return None
    return candidates[-1]


def _accuracy(metrics: dict[str, Any]) -> float | None:
    value = metrics.get("exact_acc", metrics.get("val_acc"))
    return float(value) if value is not None else None


def _mark(value: bool) -> str:
    return "✓" if value else ""


def _percent(value: float | None) -> str:
    if value is None:
        return ""
    return f"{100.0 * value:.2f}"


def write_outputs(rows: list[dict[str, Any]], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "ablation_results.csv"
    fieldnames = [
        "run",
        "description",
        "data_aug",
        "stn",
        "pos_enc",
        "layout_heads",
        "vertical_prior",
        "format_correction",
        "accuracy",
        "val_acc",
        "exact_acc",
        "global_acc",
        "layout_acc",
        "valid_format_rate",
        "val_loss",
        "epoch",
        "status",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    md_path = out_dir / "ablation_results.md"
    lines = [
        "| Run | Data aug. | STN | 2D PE | Layout heads | Vertical prior | Format correction | Accuracy (%) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {run} | {data_aug} | {stn} | {pos_enc} | {layout_heads} | {vertical_prior} | {format_correction} | {acc} |".format(
                run=row["run"],
                data_aug=_mark(row["data_aug"]),
                stn=_mark(row["stn"]),
                pos_enc=_mark(row["pos_enc"]),
                layout_heads=_mark(row["layout_heads"]),
                vertical_prior=_mark(row["vertical_prior"]),
                format_correction=_mark(row["format_correction"]),
                acc=_percent(row["accuracy"]),
            )
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    opts = parse_args()
    matrix = load_matrix(opts.matrix)
    rows = collect_rows(matrix, resolve_path(opts.root))
    write_outputs(rows, resolve_path(opts.out))
    print(f"Wrote {resolve_path(opts.out) / 'ablation_results.csv'}")
    print(f"Wrote {resolve_path(opts.out) / 'ablation_results.md'}")


if __name__ == "__main__":
    main()
