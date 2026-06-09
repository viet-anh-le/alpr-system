from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from ocr.parseq_dataset import DEFAULT_PARSEQ_VN_CHARSET


@dataclass(frozen=True)
class CharsetResizeReport:
    charset: str
    copied_head_rows: int
    copied_embedding_rows: int
    initialized_tokens: tuple[str, ...]


def load_parseq_from_torchhub(
    *,
    variant: str = "parseq",
    pretrained: bool = True,
    decode_ar: bool = True,
    refine_iters: int = 1,
) -> torch.nn.Module:
    try:
        import timm  # noqa: F401
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "PARSeq via torch.hub requires timm. Install it in your env with: "
            "/home/vietanh/anaconda3/envs/myenv/bin/python -m pip install timm"
        ) from exc

    return torch.hub.load(
        "baudm/parseq",
        variant,
        pretrained=pretrained,
        decode_ar=decode_ar,
        refine_iters=refine_iters,
        trust_repo=True,
        skip_validation=True,
    )


def configure_parseq_charset(
    model: torch.nn.Module,
    charset: str = DEFAULT_PARSEQ_VN_CHARSET,
) -> CharsetResizeReport:
    tokenizer_class, charset_adapter_class = _load_parseq_tokenizer_classes()
    old_tokenizer = model.tokenizer
    old_tokens = tuple(old_tokenizer._itos)
    new_tokenizer = tokenizer_class(charset)
    new_tokens = tuple(new_tokenizer._itos)

    if old_tokens == new_tokens:
        return CharsetResizeReport(charset, 0, 0, ())

    model.tokenizer = new_tokenizer
    model.charset_adapter = charset_adapter_class(charset)
    model.bos_id = new_tokenizer.bos_id
    model.eos_id = new_tokenizer.eos_id
    model.pad_id = new_tokenizer.pad_id

    inner = _inner_parseq_model(model)
    head_report = _resize_head(inner, old_tokens, new_tokens)
    embed_report = _resize_text_embedding(inner, old_tokens, new_tokens)

    _set_hparam(model, "charset_train", charset)
    _set_hparam(model, "charset_test", charset)

    initialized = tuple(sorted(set(head_report["initialized"]) | set(embed_report["initialized"])))
    return CharsetResizeReport(
        charset=charset,
        copied_head_rows=head_report["copied"],
        copied_embedding_rows=embed_report["copied"],
        initialized_tokens=initialized,
    )


@torch.no_grad()
def predict_strings(model: torch.nn.Module, images: torch.Tensor) -> list[str]:
    logits = model(images)
    probs = logits.softmax(-1)
    labels, _ = model.tokenizer.decode(probs)
    return labels


@torch.no_grad()
def predict_strings_with_confidence(model: torch.nn.Module, images: torch.Tensor) -> tuple[list[str], list[float]]:
    logits = model(images)
    probs = logits.softmax(-1)
    labels, batch_probs = model.tokenizer.decode(probs)
    confidences: list[float] = []
    for label, char_probs in zip(labels, batch_probs):
        probs_for_label = char_probs[: len(label)]
        confidences.append(sequence_confidence(probs_for_label))
    return labels, confidences


def sequence_confidence(probs: torch.Tensor) -> float:
    if probs.numel() == 0:
        return 0.0
    eps = 1e-12
    log_sum = torch.log(probs.clamp_min(eps)).sum().item()
    return math.exp(log_sum / probs.numel())


def checkpoint_payload(
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None,
    epoch: int,
    args: Any,
    best_metric: float,
) -> dict[str, Any]:
    return {
        "state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "epoch": epoch,
        "best_metric": best_metric,
        "charset": args.charset,
        "variant": args.variant,
        "pretrained": bool(args.pretrained),
        "decode_ar": bool(args.decode_ar),
        "refine_iters": int(args.refine_iters),
        "image_width": int(args.image_width),
        "image_height": int(args.image_height),
        "max_label_length": int(args.max_label_length),
    }


def load_parseq_checkpoint(
    checkpoint_path: str | Path,
    *,
    variant: str = "parseq",
    charset: str | None = None,
    decode_ar: bool = True,
    refine_iters: int = 1,
    device: torch.device | str = "cpu",
) -> tuple[torch.nn.Module, dict[str, Any]]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = load_parseq_from_torchhub(
        variant=checkpoint.get("variant", variant),
        pretrained=False,
        decode_ar=checkpoint.get("decode_ar", decode_ar),
        refine_iters=checkpoint.get("refine_iters", refine_iters),
    )
    configure_parseq_charset(model, charset or checkpoint.get("charset", DEFAULT_PARSEQ_VN_CHARSET))
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    return model.to(device), checkpoint


def _load_parseq_tokenizer_classes():
    try:
        from strhub.data.utils import CharsetAdapter, Tokenizer
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Could not import PARSeq's strhub tokenizer. Load PARSeq from torch.hub first "
            "or install the baudm/parseq package in this environment."
        ) from exc
    return Tokenizer, CharsetAdapter


def _inner_parseq_model(model: torch.nn.Module) -> torch.nn.Module:
    if not hasattr(model, "model"):
        raise AttributeError("Expected a PARSeq Lightning module with a .model attribute.")
    return model.model


def _resize_head(inner: torch.nn.Module, old_tokens: tuple[str, ...], new_tokens: tuple[str, ...]) -> dict[str, Any]:
    old_head = inner.head
    old_predict_tokens = old_tokens[: old_head.out_features]
    new_out_features = len(new_tokens) - 2
    new_predict_tokens = new_tokens[:new_out_features]

    new_head = nn.Linear(
        old_head.in_features,
        new_out_features,
        bias=old_head.bias is not None,
        device=old_head.weight.device,
        dtype=old_head.weight.dtype,
    )
    nn.init.trunc_normal_(new_head.weight, std=0.02)
    if new_head.bias is not None:
        nn.init.zeros_(new_head.bias)

    copied, initialized = _copy_token_rows(
        new_weight=new_head.weight,
        old_weight=old_head.weight,
        new_tokens=new_predict_tokens,
        old_tokens=old_predict_tokens,
    )
    if old_head.bias is not None and new_head.bias is not None:
        _copy_token_rows(
            new_weight=new_head.bias.unsqueeze(1),
            old_weight=old_head.bias.unsqueeze(1),
            new_tokens=new_predict_tokens,
            old_tokens=old_predict_tokens,
        )
    inner.head = new_head
    return {"copied": copied, "initialized": initialized}


def _resize_text_embedding(
    inner: torch.nn.Module,
    old_tokens: tuple[str, ...],
    new_tokens: tuple[str, ...],
) -> dict[str, Any]:
    old_embed = inner.text_embed
    old_weight = old_embed.embedding.weight
    new_embed = type(old_embed)(
        len(new_tokens),
        old_embed.embed_dim,
    ).to(device=old_weight.device, dtype=old_weight.dtype)
    nn.init.trunc_normal_(new_embed.embedding.weight, std=0.02)

    copied, initialized = _copy_token_rows(
        new_weight=new_embed.embedding.weight,
        old_weight=old_weight,
        new_tokens=new_tokens,
        old_tokens=old_tokens,
    )
    inner.text_embed = new_embed
    return {"copied": copied, "initialized": initialized}


@torch.no_grad()
def _copy_token_rows(
    *,
    new_weight: torch.Tensor,
    old_weight: torch.Tensor,
    new_tokens: tuple[str, ...],
    old_tokens: tuple[str, ...],
) -> tuple[int, tuple[str, ...]]:
    old_index = {token: index for index, token in enumerate(old_tokens)}
    copied = 0
    initialized: list[str] = []
    for new_index, token in enumerate(new_tokens):
        source_index = old_index.get(token)
        if source_index is None and token == "Đ":
            source_index = old_index.get("D")
        if source_index is None:
            initialized.append(token)
            continue
        new_weight[new_index].copy_(old_weight[source_index])
        copied += 1
    return copied, tuple(initialized)


def _set_hparam(model: torch.nn.Module, key: str, value: Any) -> None:
    hparams = getattr(model, "hparams", None)
    if hparams is None:
        return
    try:
        setattr(hparams, key, value)
    except Exception:
        try:
            hparams[key] = value
        except Exception:
            return
