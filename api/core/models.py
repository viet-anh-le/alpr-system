"""
core/models.py — Model loading, preprocessing, and OCR batch inference.

Exposes ModelBundle (a typed container for all models) so downstream code
never touches raw globals.
"""

from __future__ import annotations

import logging
import sys
from argparse import Namespace
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np
import torch
from PIL import Image
from ultralytics import YOLO

from .quality_router import PlateQualityRouter
from .tracker_adapter import VehicleTracker
from .config import (
    CHARS,
    CONF_THRESHOLD,
    EOS_IDX,
    IMG_H,
    IMG_W,
    OCR_BACKEND,
    PAD_IDX,
    PARSEQ_IMAGE_H,
    PARSEQ_IMAGE_W,
    PLATE_MODEL_PATH,
    REID_DEVICE,
    REID_MODEL_PATH,
    ROOT,
    SMALL_LPR_LINE_CTC_CKPT_PATH,
    SOS_IDX,
    VEHICLE_DETECTOR_BACKEND,
    VEHICLE_MODEL_PATH,
    YOLOV5_CHAR_CKPT_PATH,
    YOLOV5_OBJECT_CKPT_PATH,
    normalize_ocr_backend,
)

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from .yolov5_vehicle import YOLOv5VehicleDetector

# Allow loading checkpoints that contain argparse.Namespace objects
torch.serialization.add_safe_globals([Namespace])

# Make LPRNet importable
sys.path.insert(0, str(ROOT / "LPRNet"))
from lprnet import SmallLPR  # noqa: E402
from lprnet.small_lpr import smart_resize  # noqa: E402
from lprnet.small_lpr_ctc import SmallLPRCTC  # noqa: E402
from lprnet.small_lpr_line_ctc import SmallLPRLineCTC  # noqa: E402


@dataclass(frozen=True)
class ParseqOcrModel:
    model: torch.nn.Module
    image_width: int
    image_height: int

    def eval(self) -> "ParseqOcrModel":
        self.model.eval()
        return self


@dataclass(frozen=True)
class SmallLprCtcOcrModel:
    model: torch.nn.Module
    chars: list[str]

    def eval(self) -> "SmallLprCtcOcrModel":
        self.model.eval()
        return self


@dataclass(frozen=True)
class SmallLprLineCtcOcrModel:
    model: torch.nn.Module
    chars: list[str]
    two_line_threshold: float = 0.5
    line_separator: str = "[SEP]"

    def eval(self) -> "SmallLprLineCtcOcrModel":
        self.model.eval()
        return self


@dataclass
class ModelBundle:
    device: torch.device
    vehicle: YOLO | YOLOv5VehicleDetector
    plate: YOLO
    ocr: SmallLPR | ParseqOcrModel | SmallLprCtcOcrModel | SmallLprLineCtcOcrModel
    reid_weights: Path
    tracker_device: str
    tracker_half: bool = False
    quality_router: PlateQualityRouter | None = None
    ocr_backend: str = OCR_BACKEND
    ocr_yolov5: object | None = None  # YOLOv5CharOcrModel
    yolov5_object: object | None = None  # YOLOv5 object model

    def create_vehicle_tracker(self) -> VehicleTracker:
        """Create a fresh VehicleTracker instance for a single session.

        Each video processing session MUST use its own tracker because
        BoT-SORT maintains stateful Kalman filters and track IDs that
        would conflict if shared across concurrent sessions.
        """
        return VehicleTracker(
            reid_weights=self.reid_weights,
            device=self.tracker_device,
            half=self.tracker_half,
        )


def load_yolov5_vehicle_detector(
    checkpoint_path: str | Path,
    *,
    device: torch.device,
) -> "YOLOv5VehicleDetector":
    from .yolov5_vehicle import load_yolov5_vehicle_detector as _load

    return _load(checkpoint_path, device=device)


def load_models() -> ModelBundle:
    configured_ocr_backend = normalize_ocr_backend(OCR_BACKEND)
    ocr_backend = "smalllpr_line_ctc"
    if configured_ocr_backend == "vietnamese_yolov5":
        logger.info(
            "OCR_BACKEND=vietnamese_yolov5 uses the dedicated YOLOv5 Vietnamese pipeline; "
            "loading SmallLPR-Line-CTC as the single-frame OCR fallback."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Loading models on %s…", device)

    use_yolov5_vehicle = VEHICLE_DETECTOR_BACKEND == "yolov5" or (
        Path(VEHICLE_MODEL_PATH).resolve() == Path(YOLOV5_OBJECT_CKPT_PATH).resolve()
    )
    if use_yolov5_vehicle:
        vehicle = load_yolov5_vehicle_detector(VEHICLE_MODEL_PATH, device=device)
    else:
        vehicle = YOLO(str(VEHICLE_MODEL_PATH))
    plate = YOLO(str(PLATE_MODEL_PATH))

    ocr = load_small_lpr_line_ctc_model(SMALL_LPR_LINE_CTC_CKPT_PATH, device=device)

    logger.info(
        "Single-frame OCR ready (%s); track-level results use cached OCR voting.",
        ocr_backend,
    )

    # VehicleTracker config — actual instances are created per-session
    # via ModelBundle.create_vehicle_tracker() to avoid stateful conflicts
    # between concurrent video processing jobs.
    reid_weights = REID_MODEL_PATH
    tracker_device = REID_DEVICE if REID_DEVICE and REID_DEVICE != "auto" else str(device)
    tracker_half = False
    logger.info("Vehicle tracker config stored (per-session instances will be created on demand).")
    quality_router = PlateQualityRouter.from_env(device=device)
    logger.info("Plate quality router ready.")

    logger.info("All models ready.")

    ocr_yolov5 = None
    yolov5_object = None
    if YOLOV5_CHAR_CKPT_PATH.exists():
        from api.core.ocr_yolov5 import load_yolov5_char_model

        ocr_yolov5 = load_yolov5_char_model(YOLOV5_CHAR_CKPT_PATH, device=device)
        logger.info("YOLOv5 Character Detection model ready.")

    if use_yolov5_vehicle:
        yolov5_object = vehicle
        logger.info("YOLOv5 Object Detection model ready as vehicle detector.")
    elif YOLOV5_OBJECT_CKPT_PATH.exists():
        from api.core.ocr_yolov5 import load_yolov5_object_model

        yolov5_object = load_yolov5_object_model(YOLOV5_OBJECT_CKPT_PATH, device=device)
        logger.info("YOLOv5 Object Detection model ready.")

    return ModelBundle(
        device=device,
        vehicle=vehicle,
        plate=plate,
        ocr=ocr,
        reid_weights=reid_weights,
        tracker_device=tracker_device,
        tracker_half=tracker_half,
        quality_router=quality_router,
        ocr_backend=ocr_backend,
        ocr_yolov5=ocr_yolov5,
        yolov5_object=yolov5_object,
    )


def select_ocr_model(
    models: ModelBundle | object,
    ocr_backend: str = "default",
) -> object:
    model_backend = getattr(models, "ocr_backend", "smalllpr_line_ctc")
    if not isinstance(model_backend, str):
        model_backend = "smalllpr_line_ctc"
    normalize_ocr_backend(ocr_backend if ocr_backend != "default" else model_backend)
    return getattr(models, "ocr")


def load_small_lpr_model(checkpoint_path: str | Path, *, device: torch.device) -> SmallLPR:
    ocr = (
        SmallLPR(vocab_size=len(CHARS), max_seq_len=14, use_pretrained_decoder=False)
        .to(device)
        .eval()
    )
    ckpt = torch.load(str(checkpoint_path), map_location=device, weights_only=False)
    sd = ckpt.get("state_dict", ckpt)
    sd = {k.removeprefix("model."): v for k, v in sd.items()}
    ocr.load_state_dict(sd)
    return ocr


def load_parseq_ocr_model(checkpoint_path: str | Path, *, device: torch.device) -> ParseqOcrModel:
    from ocr.parseq_model import load_parseq_checkpoint

    model, checkpoint = load_parseq_checkpoint(checkpoint_path, device=device)
    model.eval()
    return ParseqOcrModel(
        model=model,
        image_width=int(checkpoint.get("image_width", PARSEQ_IMAGE_W)),
        image_height=int(checkpoint.get("image_height", PARSEQ_IMAGE_H)),
    )


def load_small_lpr_ctc_model(
    checkpoint_path: str | Path, *, device: torch.device
) -> SmallLprCtcOcrModel:
    ckpt = torch.load(str(checkpoint_path), map_location=device, weights_only=False)
    sd = ckpt.get("state_dict", ckpt)
    sd = {k.removeprefix("model."): v for k, v in sd.items()}
    args = _checkpoint_args(ckpt)
    chars = list(_arg_value(args, "chars", []) or [])
    if not chars:
        raise ValueError("SmallLPR-CTC checkpoint must include hyper_parameters.args.chars")

    model = (
        SmallLPRCTC(
            vocab_size=len(chars),
            d_model=int(_arg_value(args, "d_model", 256)),
            backbone_ch=int(_arg_value(args, "backbone_ch", 256)),
        )
        .to(device)
        .eval()
    )
    model.load_state_dict(sd)
    return SmallLprCtcOcrModel(model=model, chars=chars)


def load_small_lpr_line_ctc_model(
    checkpoint_path: str | Path, *, device: torch.device
) -> SmallLprLineCtcOcrModel:
    ckpt = torch.load(str(checkpoint_path), map_location=device, weights_only=False)
    sd = ckpt.get("state_dict", ckpt)
    sd = {k.removeprefix("model."): v for k, v in sd.items()}
    args = _checkpoint_args(ckpt)
    chars = list(_arg_value(args, "chars", []) or [])
    if not chars:
        raise ValueError("SmallLPR-Line-CTC checkpoint must include hyper_parameters.args.chars")

    model = (
        SmallLPRLineCTC(
            vocab_size=len(chars),
            d_model=int(_arg_value(args, "d_model", 256)),
            backbone_ch=int(_arg_value(args, "backbone_ch", 256)),
            line_prior_strength=float(_arg_value(args, "line_prior_strength", 1.0)),
            use_stn=bool(_arg_value(args, "use_stn", True)),
            use_pos_enc=bool(_arg_value(args, "use_pos_enc", True)),
            use_global_head=bool(_arg_value(args, "use_global_head", True)),
        )
        .to(device)
        .eval()
    )
    model.load_state_dict(sd)
    return SmallLprLineCtcOcrModel(
        model=model,
        chars=chars,
        two_line_threshold=float(_arg_value(args, "two_line_threshold", 0.5)),
        line_separator=str(_arg_value(args, "line_separator", "[SEP]")),
    )


def _checkpoint_args(checkpoint: dict) -> object:
    hyper_parameters = checkpoint.get("hyper_parameters", {})
    if isinstance(hyper_parameters, dict):
        return hyper_parameters.get("args", hyper_parameters)
    return getattr(hyper_parameters, "args", hyper_parameters)


def _arg_value(args: object, name: str, default: object) -> object:
    if isinstance(args, dict):
        return args.get(name, default)
    return getattr(args, name, default)


def preprocess_plate(
    bgr: np.ndarray,
    *,
    backend: str | None = None,
    image_width: int | None = None,
    image_height: int | None = None,
) -> torch.Tensor:
    """BGR crop → normalized OCR input tensor."""
    normalize_ocr_backend(backend or OCR_BACKEND)
    return preprocess_plate_small_lpr(bgr)


def preprocess_plate_for_model(
    model: SmallLPR
    | ParseqOcrModel
    | SmallLprCtcOcrModel
    | SmallLprLineCtcOcrModel
    | object,
    bgr: np.ndarray,
) -> torch.Tensor:
    if _is_parseq_ocr_model(model):
        return preprocess_plate_parseq(
            bgr,
            image_width=model.image_width,
            image_height=model.image_height,
        )
    if _is_yolov5_char_model(model):
        from api.core.ocr_yolov5 import preprocess_plate_yolov5

        return preprocess_plate_yolov5(bgr)
    return preprocess_plate_small_lpr(bgr)


def preprocess_plate_small_lpr(bgr: np.ndarray) -> torch.Tensor:
    """BGR crop → SmallLPR tensor in [-1, 1]."""
    img = smart_resize(bgr, target_hw=(IMG_H, IMG_W)).astype("float32")
    img = (img - 127.5) * 0.0078125
    return torch.from_numpy(img.transpose(2, 0, 1))


def preprocess_plate_parseq(
    bgr: np.ndarray,
    *,
    image_width: int = PARSEQ_IMAGE_W,
    image_height: int = PARSEQ_IMAGE_H,
) -> torch.Tensor:
    """BGR crop → PARSeq ImageNet-normalized tensor."""
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(rgb)
    return _parseq_transform(image_width, image_height)(image)


@lru_cache(maxsize=8)
def _parseq_transform(image_width: int, image_height: int):
    from ocr.parseq_dataset import make_parseq_transform

    return make_parseq_transform(
        image_width=image_width,
        image_height=image_height,
        augment=False,
    )


@torch.no_grad()
def ocr_batch(
    model: SmallLPR | ParseqOcrModel | SmallLprCtcOcrModel | SmallLprLineCtcOcrModel,
    images: torch.Tensor,
    device: torch.device,
) -> list[tuple[list[tuple[str, float]], bool]]:
    if _is_parseq_ocr_model(model):
        return parseq_ocr_batch(model, images, device)
    if _is_small_lpr_line_ctc_model(model):
        return small_lpr_line_ctc_ocr_batch(model, images, device)
    if _is_small_lpr_ctc_model(model):
        return small_lpr_ctc_ocr_batch(model, images, device)
    if _is_yolov5_char_model(model):
        from api.core.ocr_yolov5 import yolov5_char_ocr_batch

        return yolov5_char_ocr_batch(model, images, device)
    return small_lpr_ocr_batch(model, images, device)


def _is_parseq_ocr_model(model: object) -> bool:
    return isinstance(model, ParseqOcrModel) or (
        type(model).__name__ == "ParseqOcrModel"
        and hasattr(model, "model")
        and hasattr(model, "image_width")
        and hasattr(model, "image_height")
    )


def _is_small_lpr_ctc_model(model: object) -> bool:
    return isinstance(model, SmallLprCtcOcrModel) or (
        type(model).__name__ == "SmallLprCtcOcrModel"
        and hasattr(model, "model")
        and hasattr(model, "chars")
    )


def _is_small_lpr_line_ctc_model(model: object) -> bool:
    return isinstance(model, SmallLprLineCtcOcrModel) or (
        type(model).__name__ == "SmallLprLineCtcOcrModel"
        and hasattr(model, "model")
        and hasattr(model, "chars")
    )


def _is_yolov5_char_model(model: object) -> bool:
    return type(model).__name__ == "YOLOv5CharOcrModel" and hasattr(model, "model")


@torch.no_grad()
def small_lpr_ocr_batch(
    model: SmallLPR,
    images: torch.Tensor,
    device: torch.device,
) -> list[tuple[list[tuple[str, float]], bool]]:
    """
    Autoregressive decode for a batch of preprocessed plate tensors.

    Returns [(char_probs, all_confident), ...] where
      char_probs     = [(char, prob), ...]
      all_confident  = True iff every char prob ≥ CONF_THRESHOLD
    """
    memory = model.encode(images)
    B = memory.size(0)
    tokens = torch.full((B, 1), SOS_IDX, dtype=torch.long, device=device)
    finished = torch.zeros(B, dtype=torch.bool, device=device)
    per_chars: list[list[tuple[str, float]]] = [[] for _ in range(B)]

    for _ in range(model.max_seq_len - 1):
        logits = model.decoder(tgt_tokens=tokens, memory_features=memory)
        probs = torch.softmax(logits[:, -1], dim=-1)
        next_tok = probs.argmax(-1)
        max_prob = probs.max(-1).values
        tokens = torch.cat([tokens, next_tok.unsqueeze(1)], dim=1)

        for b in range(B):
            if not finished[b]:
                t = int(next_tok[b])
                if t == EOS_IDX:
                    finished[b] = True
                elif t not in (PAD_IDX, SOS_IDX):
                    per_chars[b].append((CHARS[t], float(max_prob[b])))

        if finished.all():
            break

    return [
        (chars, bool(chars) and all(p >= CONF_THRESHOLD for _, p in chars)) for chars in per_chars
    ]


@torch.no_grad()
def small_lpr_ctc_ocr_batch(
    wrapper: SmallLprCtcOcrModel,
    images: torch.Tensor,
    device: torch.device,
) -> list[tuple[list[tuple[str, float]], bool]]:
    model = wrapper.model.to(device).eval()
    logits = model(images.to(device, non_blocking=True)) 
    probs = torch.softmax(logits, dim=-1) 
    token_ids = probs.argmax(dim=-1) 
    token_probs = probs.max(dim=-1).values 

    results: list[tuple[list[tuple[str, float]], bool]] = []
    for seq_ids, seq_probs in zip(token_ids, token_probs, strict=False):
        chars: list[tuple[str, float]] = []
        prev_token = -1
        for token_id_tensor, prob_tensor in zip(seq_ids, seq_probs, strict=False):
            token_id = int(token_id_tensor)
            if token_id != prev_token and token_id != 0 and token_id < len(wrapper.chars):
                chars.append((wrapper.chars[token_id], float(prob_tensor)))
            prev_token = token_id
        all_confident = bool(chars) and all(prob >= CONF_THRESHOLD for _, prob in chars)
        results.append((chars, all_confident))
    return results


@torch.no_grad()
def small_lpr_line_ctc_ocr_batch(
    wrapper: SmallLprLineCtcOcrModel,
    images: torch.Tensor,
    device: torch.device,
) -> list[tuple[list[tuple[str, float]], bool]]:
    model = wrapper.model.to(device).eval()
    outputs = model(images.to(device, non_blocking=True))
    layout_probs = torch.softmax(outputs["layout_logits"], dim=-1)
    one_line_logits = outputs.get("one_line_logits")
    if one_line_logits is None:
        one_line_logits = outputs["global_logits"]

    one_line_chars = _ctc_logits_to_char_probs(one_line_logits, wrapper.chars)
    top_chars = _ctc_logits_to_char_probs(outputs["top_logits"], wrapper.chars)
    bottom_chars = _ctc_logits_to_char_probs(outputs["bottom_logits"], wrapper.chars)

    results: list[tuple[list[tuple[str, float]], bool]] = []
    for idx, layout_prob in enumerate(layout_probs):
        if float(layout_prob[1]) >= wrapper.two_line_threshold:
            separator = (
                [(wrapper.line_separator, float(layout_prob[1]))]
                if wrapper.line_separator
                else []
            )
            chars = [*top_chars[idx], *separator, *bottom_chars[idx]]
        else:
            chars = one_line_chars[idx]
        all_confident = bool(chars) and all(prob >= CONF_THRESHOLD for _, prob in chars)
        results.append((chars, all_confident))
    return results


def _ctc_logits_to_char_probs(
    logits: torch.Tensor,
    chars: list[str],
) -> list[list[tuple[str, float]]]:
    probs = torch.softmax(logits, dim=-1)
    token_ids = probs.argmax(dim=-1)
    token_probs = probs.max(dim=-1).values

    decoded: list[list[tuple[str, float]]] = []
    for seq_ids, seq_probs in zip(token_ids, token_probs, strict=False):
        sequence: list[tuple[str, float]] = []
        prev_token = -1
        for token_id_tensor, prob_tensor in zip(seq_ids, seq_probs, strict=False):
            token_id = int(token_id_tensor)
            if token_id != prev_token and token_id != 0 and token_id < len(chars):
                sequence.append((chars[token_id], float(prob_tensor)))
            prev_token = token_id
        decoded.append(sequence)
    return decoded


@torch.no_grad()
def parseq_ocr_batch(
    wrapper: ParseqOcrModel,
    images: torch.Tensor,
    device: torch.device,
) -> list[tuple[list[tuple[str, float]], bool]]:
    model = wrapper.model.to(device).eval()
    logits = model(images.to(device, non_blocking=True))
    probs = torch.softmax(logits, dim=-1)
    labels, batch_probs = model.tokenizer.decode(probs)
    results: list[tuple[list[tuple[str, float]], bool]] = []
    for label, char_probs_tensor in zip(labels, batch_probs, strict=False):
        chars = parseq_label_to_char_probs(label, char_probs_tensor)
        all_confident = bool(chars) and all(prob >= CONF_THRESHOLD for _, prob in chars)
        results.append((chars, all_confident))
    return results


def parseq_label_to_char_probs(
    label: str,
    char_probs: torch.Tensor | list[float] | tuple[float, ...],
) -> list[tuple[str, float]]:
    probs = _prob_list(char_probs)
    chars: list[tuple[str, float]] = []
    i = 0
    while i < len(label):
        if label.startswith("[SEP]", i):
            sep_probs = probs[i : i + len("[SEP]")]
            chars.append(("[SEP]", _mean_prob(sep_probs)))
            i += len("[SEP]")
            continue
        chars.append((label[i], probs[i] if i < len(probs) else 0.0))
        i += 1
    return chars


def _prob_list(char_probs: torch.Tensor | list[float] | tuple[float, ...]) -> list[float]:
    if isinstance(char_probs, torch.Tensor):
        values = char_probs.detach().cpu().flatten().tolist()
    else:
        values = list(char_probs)
    return [float(value) for value in values]


def _mean_prob(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / len(values))
