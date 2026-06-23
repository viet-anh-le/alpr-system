"""
Run SmallLPR-Line-CTC ablation experiments from a YAML matrix.

Examples:
    /home/vietanh/anaconda3/envs/myenv/bin/python scripts/run_small_lpr_line_ctc_ablation.py \
        --matrix LPRNet/config/small_lpr_line_ctc_ablation.yaml --dry-run
"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MATRIX = ROOT / "LPRNet/config/small_lpr_line_ctc_ablation.yaml"
DEFAULT_OUT_DIR = ROOT / "weights/ocr/small_lpr_line_ctc_ablation"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run SmallLPR-Line-CTC ablation matrix.")
    parser.add_argument("--matrix", default=str(DEFAULT_MATRIX), help="Ablation YAML matrix.")
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR), help="Root output directory.")
    parser.add_argument("--only", default="", help="Comma-separated run IDs to execute.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing.")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--devices", default=None)
    parser.add_argument("--precision", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--config", default=None, help="Override base training config.")
    parser.add_argument(
        "--skip-eval",
        action="store_true",
        help="Only launch training commands; do not run evaluation commands.",
    )
    return parser.parse_args()


def resolve_path(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else ROOT / candidate


def load_matrix(path: str | Path) -> dict[str, Any]:
    data = yaml.safe_load(resolve_path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not isinstance(data.get("runs"), list):
        raise ValueError("Ablation matrix must contain a top-level runs list.")
    for run in data["runs"]:
        validate_run(run)
    return data


def validate_run(run: dict[str, Any]) -> None:
    if not run.get("id"):
        raise ValueError("Each ablation run must define an id.")
    if "resume" in run or "init_from" in run or "init-from" in run:
        raise ValueError(f"{run['id']} must train from scratch; remove resume/init_from.")
    if run.get("decode_mode", "layout") not in {"global", "layout"}:
        raise ValueError(f"{run['id']} decode_mode must be 'global' or 'layout'.")


def select_runs(matrix: dict[str, Any], only: str) -> list[dict[str, Any]]:
    runs = list(matrix.get("runs", []))
    if not only:
        return runs
    requested = [part.strip() for part in only.split(",") if part.strip()]
    available = {run["id"]: run for run in runs}
    missing = [run_id for run_id in requested if run_id not in available]
    if missing:
        raise ValueError(f"Unknown ablation run id(s): {', '.join(missing)}")
    return [available[run_id] for run_id in requested]


def _default(matrix: dict[str, Any], name: str, fallback: Any = None) -> Any:
    return matrix.get("defaults", {}).get(name, fallback)


def _bool_flag(command: list[str], enabled: bool, positive: str, negative: str) -> None:
    command.append(positive if enabled else negative)


def _loss_weight(run: dict[str, Any], name: str, default: float) -> float:
    return float(run.get("loss_weights", {}).get(name, default))


def build_train_command(
    matrix: dict[str, Any],
    run: dict[str, Any],
    opts: argparse.Namespace,
) -> list[str]:
    python_bin = str(matrix.get("python", "/home/vietanh/anaconda3/envs/myenv/bin/python"))
    out_dir = str(resolve_path(opts.out_dir))
    config = opts.config or _default(matrix, "config", "LPRNet/config/small_lpr_line_ctc_config.yaml")
    epochs = opts.epochs if opts.epochs is not None else _default(matrix, "epochs", 50)
    devices = opts.devices if opts.devices is not None else _default(matrix, "devices", "1")
    precision = opts.precision if opts.precision is not None else _default(matrix, "precision", "32")
    seed = opts.seed if opts.seed is not None else _default(matrix, "seed", 42)

    command = [
        python_bin,
        "scripts/train_small_lpr_line_ctc.py",
        "--config",
        str(config),
        "--data-root",
        str(_default(matrix, "data_root", "data/datasets/ocr")),
        "--out-dir",
        out_dir,
        "--run-name",
        run["id"],
        "--epochs",
        str(epochs),
        "--batch-size",
        str(_default(matrix, "batch_size", 64)),
        "--lr",
        str(_default(matrix, "lr", 0.0003)),
        "--devices",
        str(devices),
        "--precision",
        str(precision),
        "--seed",
        str(seed),
        "--accumulate-grad",
        str(_default(matrix, "accumulate_grad", 1)),
        "--decode-mode",
        str(run.get("decode_mode", "layout")),
        "--line-prior-strength",
        str(float(run.get("line_prior_strength", 1.0))),
        "--global-loss-weight",
        str(_loss_weight(run, "global", 1.0)),
        "--one-line-loss-weight",
        str(_loss_weight(run, "one_line", 1.0)),
        "--top-loss-weight",
        str(_loss_weight(run, "top", 1.0)),
        "--bottom-loss-weight",
        str(_loss_weight(run, "bottom", 1.0)),
        "--layout-loss-weight",
        str(_loss_weight(run, "layout", 0.2)),
    ]
    _bool_flag(command, bool(run.get("augment", True)), "--augment", "--no-augment")
    _bool_flag(command, bool(run.get("use_stn", True)), "--use-stn", "--no-use-stn")
    _bool_flag(command, bool(run.get("use_pos_enc", True)), "--use-pos-enc", "--no-use-pos-enc")
    return command


def build_eval_command(
    matrix: dict[str, Any],
    run: dict[str, Any],
    opts: argparse.Namespace,
) -> list[str] | None:
    python_bin = str(matrix.get("python", "/home/vietanh/anaconda3/envs/myenv/bin/python"))
    out_root = resolve_path(opts.out_dir)
    source_run = run.get("source_run", run["id"])
    ckpt = _best_checkpoint(out_root / source_run)
    if ckpt is None:
        return None
    output_json = out_root / run["id"] / "eval_metrics.json"
    command = [
        python_bin,
        "scripts/evaluate_small_lpr_line_ctc.py",
        "--checkpoint",
        str(ckpt),
        "--output",
        str(output_json),
        "--decode-mode",
        str(run.get("decode_mode", "layout")),
    ]
    if run.get("format_correction", False):
        command.append("--format-correction")
    return command


def _best_checkpoint(run_dir: Path) -> Path | None:
    ckpts = sorted(run_dir.glob("small_lpr_line_ctc-epoch=*-val_acc=*.ckpt"))
    if not ckpts:
        return None
    return max(ckpts, key=lambda path: _val_acc_from_name(path.name))


def _val_acc_from_name(name: str) -> float:
    marker = "val_acc="
    if marker not in name:
        return -1.0
    tail = name.split(marker, 1)[1].removesuffix(".ckpt")
    try:
        return float(tail)
    except ValueError:
        return -1.0


def run_command(command: list[str], *, dry_run: bool) -> None:
    rendered = " ".join(shlex.quote(part) for part in command)
    print(rendered)
    if dry_run:
        return
    env = os.environ.copy()
    env.setdefault("MPLCONFIGDIR", "/tmp")
    env.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")
    subprocess.run(command, cwd=ROOT, env=env, check=True)


def main() -> None:
    opts = parse_args()
    matrix = load_matrix(opts.matrix)
    runs = select_runs(matrix, opts.only)
    for run in runs:
        if run.get("train", True):
            run_command(build_train_command(matrix, run, opts), dry_run=opts.dry_run)
        if opts.skip_eval:
            continue
        eval_command = build_eval_command(matrix, run, opts)
        if eval_command is not None:
            run_command(eval_command, dry_run=opts.dry_run)
        elif run.get("format_correction", False):
            print(f"# Skipping eval for {run['id']}: source checkpoint not found yet.")


if __name__ == "__main__":
    main()
