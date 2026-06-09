from __future__ import annotations

from collections import Counter
from pathlib import Path

import cv2
import torch

from ocr.train.slot_lpr_datamodule import decode_slot_tokens


def per_slot_accuracy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    pad_id: int = 0,
    eos_id: int | None = None,
) -> torch.Tensor:
    preds = logits.argmax(dim=-1)
    mask = targets != pad_id
    if eos_id is not None:
        mask = mask & (targets != eos_id)
    correct = ((preds == targets) & mask).sum(dim=0)
    total = mask.sum(dim=0).clamp_min(1)
    return correct.float() / total.float()


def valid_token_indices(
    target: torch.Tensor,
    pad_id: int = 0,
    eos_id: int | None = None,
) -> torch.Tensor:
    mask = target != pad_id
    if eos_id is not None:
        mask = mask & (target != eos_id)
    return mask.nonzero(as_tuple=False).flatten()


def collect_edge_confusions(
    preds: torch.Tensor,
    targets: torch.Tensor,
    id2char: list[str],
    pad_id: int = 0,
    eos_id: int | None = None,
) -> tuple[Counter, Counter]:
    first_counter: Counter = Counter()
    last_counter: Counter = Counter()

    for pred_seq, target_seq in zip(preds.cpu(), targets.cpu()):
        valid_indices = valid_token_indices(target_seq, pad_id=pad_id, eos_id=eos_id)
        if len(valid_indices) == 0:
            continue
        first_idx = int(valid_indices[0])
        last_idx = int(valid_indices[-1])
        for idx, counter in ((first_idx, first_counter), (last_idx, last_counter)):
            gt = int(target_seq[idx])
            pr = int(pred_seq[idx])
            if gt != pr:
                counter[(id2char[gt], id2char[pr])] += 1
    return first_counter, last_counter


def edge_error_flags(
    pred_tokens: torch.Tensor,
    target_tokens: torch.Tensor,
    pad_id: int = 0,
    eos_id: int | None = None,
) -> tuple[bool, bool, bool]:
    valid_indices = valid_token_indices(target_tokens.cpu(), pad_id=pad_id, eos_id=eos_id)
    if len(valid_indices) == 0:
        return False, False, False

    first_idx = int(valid_indices[0])
    last_idx = int(valid_indices[-1])
    first_wrong = int(pred_tokens[first_idx]) != int(target_tokens[first_idx])
    last_wrong = int(pred_tokens[last_idx]) != int(target_tokens[last_idx])

    tail = pred_tokens[last_idx + 1 :]
    extra_tail = bool(((tail != pad_id) if eos_id is None else ((tail != pad_id) & (tail != eos_id))).any())
    return first_wrong, last_wrong, extra_tail


def slot_probability_rows(
    logits: torch.Tensor,
    target_tokens: torch.Tensor,
    chars: list[str],
    *,
    top_k: int = 5,
) -> list[dict]:
    probs = logits.softmax(dim=-1).detach().cpu()
    pred_tokens = probs.argmax(dim=-1)
    rows: list[dict] = []
    for slot_idx, slot_probs in enumerate(probs):
        top_probs, top_ids = torch.topk(slot_probs, k=min(top_k, slot_probs.numel()))
        rows.append(
            {
                "slot": slot_idx,
                "target_id": int(target_tokens[slot_idx]),
                "target": chars[int(target_tokens[slot_idx])],
                "pred_id": int(pred_tokens[slot_idx]),
                "pred": chars[int(pred_tokens[slot_idx])],
                "pred_prob": float(slot_probs[int(pred_tokens[slot_idx])]),
                "top": [
                    {"char": chars[int(idx)], "prob": float(prob)}
                    for idx, prob in zip(top_ids, top_probs)
                ],
            }
        )
    return rows


def decode_batch(logits: torch.Tensor, chars: list[str]) -> list[str]:
    pred_tokens = logits.argmax(dim=-1).detach().cpu()
    return [decode_slot_tokens(row, chars) for row in pred_tokens]


def write_edge_error_image(
    source_path: str | Path,
    output_dir: str | Path,
    *,
    index: int,
    gt: str,
    pred: str,
) -> Path:
    image = cv2.imread(str(source_path))
    if image is None:
        raise RuntimeError(f"Failed to read image: {source_path}")
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    safe_gt = _safe_name(gt)
    safe_pred = _safe_name(pred)
    out_path = out_dir / f"idx_{index:05d}_gt_{safe_gt}_pred_{safe_pred}.png"
    cv2.imwrite(str(out_path), image)
    return out_path


def format_counter(counter: Counter, limit: int = 20) -> list[dict[str, object]]:
    return [
        {"gt": gt, "pred": pred, "count": count}
        for (gt, pred), count in counter.most_common(limit)
    ]


def _safe_name(text: str) -> str:
    keep = []
    for ch in text:
        keep.append(ch if ch.isalnum() or ch in {"-", "_"} else "_")
    return "".join(keep) or "empty"
