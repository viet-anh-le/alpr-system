"""
Evaluate a SmallLPR-Line-CTC checkpoint on the OCR validation split.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from argparse import Namespace
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp")
os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")

import lightning as L
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
LPRNET_ROOT = ROOT / "LPRNet"
if str(LPRNET_ROOT) not in sys.path:
    sys.path.insert(0, str(LPRNET_ROOT))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from api.core.ocr_ambiguity import correct_ambiguous_chars  # noqa: E402
from api.core.plate_format import is_vn_plate_chars  # noqa: E402
from lprnet.small_lpr_line_ctc import ctc_decode_logits, line_ctc_greedy_decode  # noqa: E402
from lprnet.small_lpr_line_ctc_datamodule import (  # noqa: E402
    LAYOUT_ONE_LINE,
    LAYOUT_TWO_LINE,
    SmallLPRLineCTCDataset,
    collate_fn_line_ctc,
)
from lprnet.small_lpr_line_ctc_lightning import SmallLPRLineCTCLightning  # noqa: E402

DEFAULT_CONFIG = ROOT / "LPRNet/config/small_lpr_line_ctc_config.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate SmallLPR-Line-CTC checkpoint.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--split", default="valid", choices=("train", "valid", "test"))
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--decode-mode", choices=("global", "layout"), default=None)
    parser.add_argument("--format-correction", action="store_true")
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def choose_device(value: str) -> torch.device:
    if value == "cpu":
        return torch.device("cpu")
    if value == "cuda":
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_config(path: Path, batch_size: int, decode_mode: str | None) -> Namespace:
    data = yaml.load(path.read_text(encoding="utf-8"), Loader=yaml.FullLoader)
    for key in ("train_dir", "valid_dir", "test_dir"):
        if key in data and not Path(data[key]).is_absolute():
            data[key] = str(ROOT / data[key])
    data["batch_size"] = batch_size
    data["num_workers"] = 0
    data["augment"] = False
    if decode_mode is not None:
        data["decode_mode"] = decode_mode
    return Namespace(**data)


def load_model(checkpoint: Path, args: Namespace, device: torch.device) -> SmallLPRLineCTCLightning:
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    ckpt_args = _checkpoint_args(payload)
    for name in (
        "chars",
        "d_model",
        "backbone_ch",
        "line_prior_strength",
        "use_stn",
        "use_pos_enc",
        "two_line_threshold",
        "global_loss_weight",
        "one_line_loss_weight",
        "top_loss_weight",
        "bottom_loss_weight",
        "layout_loss_weight",
    ):
        value = _arg_value(ckpt_args, name, None)
        if value is not None:
            setattr(args, name, value)
    if getattr(args, "decode_mode", None) is None:
        ckpt_decode_mode = _arg_value(ckpt_args, "decode_mode", None)
        if ckpt_decode_mode is not None:
            setattr(args, "decode_mode", ckpt_decode_mode)
    model = SmallLPRLineCTCLightning(args).to(device).eval()
    model.load_state_dict(payload["state_dict"], strict=True)
    return model


def _checkpoint_args(checkpoint: dict[str, Any]) -> object:
    hyper_parameters = checkpoint.get("hyper_parameters", {})
    if isinstance(hyper_parameters, dict):
        return hyper_parameters.get("args", hyper_parameters)
    return getattr(hyper_parameters, "args", hyper_parameters)


def _arg_value(args: object, name: str, default: object) -> object:
    if isinstance(args, dict):
        return args.get(name, default)
    return getattr(args, name, default)


def char_probs_to_text(char_probs: list[tuple[str, float]]) -> str:
    return "".join(token for token, _prob in char_probs)


def logits_to_char_probs(logits: torch.Tensor, chars: list[str]) -> list[list[tuple[str, float]]]:
    probs = torch.softmax(logits, dim=-1)
    token_ids = probs.argmax(dim=-1)
    token_probs = probs.max(dim=-1).values
    decoded: list[list[tuple[str, float]]] = []
    for seq_ids, seq_probs in zip(token_ids, token_probs, strict=False):
        sequence: list[tuple[str, float]] = []
        previous = -1
        for token_tensor, prob_tensor in zip(seq_ids, seq_probs, strict=False):
            token = int(token_tensor)
            if token != previous and token != 0 and token < len(chars):
                sequence.append((chars[token], float(prob_tensor)))
            previous = token
        decoded.append(sequence)
    return decoded


def layout_char_probs(outputs: dict[str, torch.Tensor], chars: list[str], threshold: float) -> list[list[tuple[str, float]]]:
    one_line = logits_to_char_probs(outputs.get("one_line_logits", outputs["global_logits"]), chars)
    top = logits_to_char_probs(outputs["top_logits"], chars)
    bottom = logits_to_char_probs(outputs["bottom_logits"], chars)
    layout_probs = torch.softmax(outputs["layout_logits"], dim=-1)
    results: list[list[tuple[str, float]]] = []
    for idx, layout_prob in enumerate(layout_probs):
        if float(layout_prob[1]) >= threshold:
            results.append([*top[idx], ("[SEP]", float(layout_prob[1])), *bottom[idx]])
        else:
            results.append(one_line[idx])
    return results


@torch.no_grad()
def evaluate(
    model: SmallLPRLineCTCLightning,
    dataset: SmallLPRLineCTCDataset,
    args: Namespace,
    *,
    device: torch.device,
    batch_size: int,
    format_correction: bool,
) -> dict[str, float | int | str]:
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        collate_fn=collate_fn_line_ctc,
    )
    totals = {
        "samples": 0,
        "exact": 0,
        "global": 0,
        "layout": 0,
        "layout_valid": 0,
        "one_line": 0,
        "one_line_total": 0,
        "two_line": 0,
        "two_line_total": 0,
        "valid_format": 0,
    }
    val_loss_sum = 0.0

    for batch in loader:
        images = batch["images"].to(device)
        outputs = model(images)
        losses = model._losses(outputs, {key: value for key, value in batch.items()})
        batch_size_actual = images.size(0)
        val_loss_sum += float(losses["loss"].detach().cpu()) * batch_size_actual

        global_texts = ctc_decode_logits(outputs["global_logits"], args.chars)
        layout_texts = line_ctc_greedy_decode(
            outputs,
            args.chars,
            two_line_threshold=float(getattr(args, "two_line_threshold", 0.5)),
        )
        char_probs = (
            logits_to_char_probs(outputs["global_logits"], args.chars)
            if model.decode_mode == "global"
            else layout_char_probs(
                outputs,
                args.chars,
                float(getattr(args, "two_line_threshold", 0.5)),
            )
        )
        if format_correction:
            char_probs = [correct_ambiguous_chars(chars).char_probs for chars in char_probs]
        pred_texts = [char_probs_to_text(chars) for chars in char_probs]

        labels = batch["layout_labels"].tolist()
        layout_preds = outputs["layout_logits"].argmax(dim=-1).detach().cpu().tolist()
        for idx, gt in enumerate(batch["texts"]):
            totals["samples"] += 1
            totals["global"] += int(global_texts[idx] == gt)
            totals["exact"] += int(pred_texts[idx] == gt)
            totals["valid_format"] += int(is_vn_plate_chars(char_probs[idx]))
            if labels[idx] in (LAYOUT_ONE_LINE, LAYOUT_TWO_LINE):
                totals["layout_valid"] += 1
                totals["layout"] += int(layout_preds[idx] == labels[idx])
            if labels[idx] == LAYOUT_ONE_LINE:
                totals["one_line_total"] += 1
                totals["one_line"] += int(pred_texts[idx] == gt)
            if labels[idx] == LAYOUT_TWO_LINE:
                totals["two_line_total"] += 1
                totals["two_line"] += int(pred_texts[idx] == gt)

    samples = max(1, totals["samples"])
    return {
        "samples": totals["samples"],
        "exact_acc": totals["exact"] / samples,
        "global_acc": totals["global"] / samples,
        "layout_acc": totals["layout"] / max(1, totals["layout_valid"]),
        "one_line_acc": totals["one_line"] / max(1, totals["one_line_total"]),
        "two_line_acc": totals["two_line"] / max(1, totals["two_line_total"]),
        "valid_format_rate": totals["valid_format"] / samples,
        "val_loss": val_loss_sum / samples,
        "decode_mode": model.decode_mode,
        "format_correction": bool(format_correction),
    }


def main() -> None:
    cli = parse_args()
    device = choose_device(cli.device)
    args = load_config(resolve_path(cli.config), cli.batch_size, cli.decode_mode)
    torch.serialization.add_safe_globals([Namespace])
    L.seed_everything(42, workers=True)
    model = load_model(resolve_path(cli.checkpoint), args, device)
    dataset = SmallLPRLineCTCDataset(args, cli.split)
    dataset.transform = None
    metrics = evaluate(
        model,
        dataset,
        args,
        device=device,
        batch_size=cli.batch_size,
        format_correction=cli.format_correction,
    )
    output = resolve_path(cli.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
