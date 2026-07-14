from __future__ import annotations

# ruff: noqa: E402 -- executable scripts add the repository root to sys.path first.

import argparse
import json
import os
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", "/tmp")
os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "LPRNet") not in sys.path:
    sys.path.insert(0, str(ROOT / "LPRNet"))

DEFAULT_PLATE_WEIGHTS = (
    ROOT / "runs/obb/experiments/detection/lp_detection_obb_merged/weights/best.pt"
)
DEFAULT_LEGIBILITY_WEIGHTS = (
    ROOT / "runs/classify/runs/classify/legibility_finetuned_vn/weights/best.pt"
)
DEFAULT_OCR_CHECKPOINT = (
    ROOT
    / "weights/ocr/small_lpr_line_ctc/line_ctc_cleaned_20260618_061855"
    / "small_lpr_line_ctc-epoch=008-val_acc=0.9501.ckpt"
)
DEFAULT_OUTPUT_DIR = ROOT / "data/outputs/single_image_inference"


@dataclass(frozen=True)
class PlateCandidate:
    index: int
    kind: str
    class_id: int
    class_name: str
    confidence: float
    box: list[int]
    crop: Any
    points: list[list[float]] | None = None
    crop_path: Path | None = None

    def to_json(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "index": self.index,
            "type": self.kind,
            "class_id": self.class_id,
            "class_name": self.class_name,
            "confidence": round(float(self.confidence), 6),
            "box": [int(value) for value in self.box],
        }
        if self.points is not None:
            payload["points"] = [
                [round(float(x), 2), round(float(y), 2)] for x, y in self.points
            ]
        if self.crop_path is not None:
            payload["crop_path"] = str(self.crop_path)
        return payload


def resolve_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else ROOT / path


def validate_existing_file(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")
    if not path.is_file():
        raise ValueError(f"{label} is not a file: {path}")


def _prediction_kwargs(
    *,
    device: str | None = None,
    imgsz: int | None = None,
    conf: float | None = None,
    half: bool | None = None,
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {"verbose": False}
    if imgsz is not None:
        kwargs["imgsz"] = int(imgsz)
    if conf is not None:
        kwargs["conf"] = float(conf)
    if device not in (None, "", "auto"):
        kwargs["device"] = device
    if half is not None:
        kwargs["half"] = bool(half)
    return kwargs


def _normalize_names(names: Any) -> dict[int, str]:
    if isinstance(names, dict):
        normalized: dict[int, str] = {}
        for key, value in names.items():
            try:
                normalized[int(key)] = str(value)
            except (TypeError, ValueError):
                continue
        return normalized
    if isinstance(names, (list, tuple)):
        return {index: str(value) for index, value in enumerate(names)}
    return {}


def _clip_box(box: list[int], image_shape: tuple[int, ...]) -> list[int]:
    height, width = image_shape[:2]
    x1, y1, x2, y2 = box
    return [
        max(0, min(width, int(x1))),
        max(0, min(height, int(y1))),
        max(0, min(width, int(x2))),
        max(0, min(height, int(y2))),
    ]


def _crop_axis_aligned(image_bgr: Any, box: list[int]) -> Any:
    x1, y1, x2, y2 = _clip_box(box, image_bgr.shape)
    return image_bgr[y1:y2, x1:x2].copy()


def read_image(path: Path) -> Any:
    import cv2

    image = cv2.imread(str(path))
    if image is None:
        raise ValueError(f"Could not read image: {path}")
    return image


def load_yolo_model(weights_path: Path) -> Any:
    from ultralytics import YOLO

    validate_existing_file(weights_path, "YOLO weights")
    return YOLO(str(weights_path))


def detect_plate_candidates(
    image_bgr: Any,
    detector: Any,
    *,
    conf: float,
    imgsz: int,
    device: str | None,
) -> list[PlateCandidate]:
    import cv2
    import numpy as np

    from api.core.video_processor import warp_plate_crop

    result = detector.predict(
        image_bgr,
        **_prediction_kwargs(device=device, imgsz=imgsz, conf=conf),
    )[0]
    names = _normalize_names(getattr(result, "names", None) or getattr(detector, "names", None))

    obb = getattr(result, "obb", None)
    if obb is not None and getattr(obb, "xyxyxyxy", None) is not None:
        points = obb.xyxyxyxy.detach().cpu().numpy().astype(np.float32)
        confs = (
            obb.conf.detach().cpu().numpy()
            if getattr(obb, "conf", None) is not None
            else np.ones((len(points),), dtype=np.float32)
        )
        classes = (
            obb.cls.detach().cpu().numpy().astype(int)
            if getattr(obb, "cls", None) is not None
            else np.zeros((len(points),), dtype=int)
        )
        candidates: list[PlateCandidate] = []
        for index, (pts, score, class_id) in enumerate(zip(points, confs, classes)):
            x, y, w, h = cv2.boundingRect(pts.astype(np.int32))
            box = _clip_box([x, y, x + w, y + h], image_bgr.shape)
            crop = warp_plate_crop(image_bgr, pts)
            if getattr(crop, "size", 0) == 0:
                crop = _crop_axis_aligned(image_bgr, box)
            if getattr(crop, "size", 0) == 0:
                continue
            candidates.append(
                PlateCandidate(
                    index=index,
                    kind="obb",
                    class_id=int(class_id),
                    class_name=names.get(int(class_id), "plate"),
                    confidence=float(score),
                    box=box,
                    points=pts.round(2).tolist(),
                    crop=crop,
                )
            )
        return _reindex_by_confidence(candidates)

    boxes = getattr(result, "boxes", None)
    if boxes is None or getattr(boxes, "xyxy", None) is None:
        return []

    xyxy = boxes.xyxy.detach().cpu().numpy()
    confs = (
        boxes.conf.detach().cpu().numpy()
        if getattr(boxes, "conf", None) is not None
        else np.ones((len(xyxy),), dtype=np.float32)
    )
    classes = (
        boxes.cls.detach().cpu().numpy().astype(int)
        if getattr(boxes, "cls", None) is not None
        else np.zeros((len(xyxy),), dtype=int)
    )
    candidates = []
    for index, (box_values, score, class_id) in enumerate(zip(xyxy, confs, classes)):
        box = _clip_box([int(round(float(value))) for value in box_values], image_bgr.shape)
        crop = _crop_axis_aligned(image_bgr, box)
        if getattr(crop, "size", 0) == 0:
            continue
        candidates.append(
            PlateCandidate(
                index=index,
                kind="xyxy",
                class_id=int(class_id),
                class_name=names.get(int(class_id), "plate"),
                confidence=float(score),
                box=box,
                crop=crop,
            )
        )
    return _reindex_by_confidence(candidates)


def _reindex_by_confidence(candidates: list[PlateCandidate]) -> list[PlateCandidate]:
    ordered = sorted(candidates, key=lambda item: item.confidence, reverse=True)
    return [replace(candidate, index=index) for index, candidate in enumerate(ordered)]


def classify_legibility_crop(
    crop_bgr: Any,
    classifier: Any,
    *,
    imgsz: int,
    device: str | None,
) -> dict[str, Any]:
    result = classifier.predict(
        crop_bgr,
        **_prediction_kwargs(device=device, imgsz=imgsz),
    )[0]
    probs = getattr(result, "probs", None)
    if probs is None or getattr(probs, "data", None) is None:
        raise RuntimeError("Legibility model did not return classification probabilities.")

    values = probs.data.detach().cpu().float().tolist()
    names = _normalize_names(getattr(result, "names", None) or getattr(classifier, "names", None))
    scores = {
        names.get(index, str(index)): round(float(score), 6)
        for index, score in enumerate(values)
    }
    scores = dict(sorted(scores.items(), key=lambda item: item[1], reverse=True))
    label, confidence = next(iter(scores.items())) if scores else ("", 0.0)
    normalized_label = label.strip().lower()
    return {
        "label": label,
        "confidence": confidence,
        "quality_bin": "suitable"
        if normalized_label in {"perfect", "good"}
        else "unsuitable",
        "scores": scores,
    }


def choose_torch_device(device: str | None) -> Any:
    import torch

    if device in (None, "", "auto"):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if str(device).isdigit():
        return torch.device(f"cuda:{device}" if torch.cuda.is_available() else "cpu")
    return torch.device(str(device))


def load_line_ctc_ocr(checkpoint_path: Path, *, device: Any) -> Any:
    from argparse import Namespace

    import torch

    from api.core.models import load_small_lpr_line_ctc_model

    validate_existing_file(checkpoint_path, "SmallLPR-Line-CTC checkpoint")
    torch.serialization.add_safe_globals([Namespace])
    return load_small_lpr_line_ctc_model(checkpoint_path, device=device)


def ocr_line_ctc_crop(crop_bgr: Any, ocr_model: Any, *, device: Any) -> dict[str, Any]:
    import torch

    from api.core.models import ocr_batch, preprocess_plate_for_model
    from api.core.plate_format import (
        chars_to_display_text,
        chars_to_text,
        is_vn_plate_chars,
        mean_confidence,
    )

    tensor = preprocess_plate_for_model(ocr_model, crop_bgr).unsqueeze(0).to(device)
    with torch.inference_mode():
        char_probs, all_confident = ocr_batch(ocr_model, tensor, device)[0]

    return {
        "text": chars_to_text(char_probs),
        "display_text": chars_to_display_text(char_probs),
        "mean_confidence": round(float(mean_confidence(char_probs)), 6),
        "all_confident": bool(all_confident),
        "valid_format": bool(is_vn_plate_chars(char_probs)),
        "char_probs": [
            {"char": char, "prob": round(float(prob), 6)}
            for char, prob in char_probs
        ],
    }


def save_crop(crop_bgr: Any, output_dir: Path, stem: str, index: int) -> Path:
    import cv2

    crop_dir = output_dir / "crops"
    crop_dir.mkdir(parents=True, exist_ok=True)
    crop_path = crop_dir / f"{stem}_plate_{index:02d}.jpg"
    if not cv2.imwrite(str(crop_path), crop_bgr):
        raise RuntimeError(f"Could not write crop: {crop_path}")
    return crop_path


def draw_candidates(
    image_bgr: Any,
    candidates: list[PlateCandidate],
    detections: list[dict[str, Any]],
) -> Any:
    import cv2
    import numpy as np

    annotated = image_bgr.copy()
    by_index = {int(item["index"]): item for item in detections}
    for candidate in candidates:
        payload = by_index.get(candidate.index, {})
        ocr_text = payload.get("ocr", {}).get("display_text", "")
        legibility = payload.get("legibility", {}).get("label", "")
        label_parts = [
            f"#{candidate.index}",
            f"{candidate.confidence:.2f}",
            str(ocr_text),
            str(legibility),
        ]
        label = " ".join(part for part in label_parts if part)

        if candidate.points is not None:
            pts = np.array(candidate.points, dtype=np.int32)
            cv2.polylines(annotated, [pts], isClosed=True, color=(0, 220, 0), thickness=2)
            x = int(pts[:, 0].min())
            y = int(pts[:, 1].min())
        else:
            x1, y1, x2, y2 = candidate.box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 220, 0), 2)
            x, y = x1, y1
        _draw_label(annotated, x, y, label)
    return annotated


def _draw_label(image_bgr: Any, x: int, y: int, label: str) -> None:
    import cv2

    if not label:
        return
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.52
    thickness = 1
    (text_w, text_h), baseline = cv2.getTextSize(label, font, scale, thickness)
    top = max(0, y - text_h - baseline - 6)
    cv2.rectangle(
        image_bgr,
        (x, top),
        (x + text_w + 8, top + text_h + baseline + 6),
        (0, 220, 0),
        -1,
    )
    cv2.putText(
        image_bgr,
        label,
        (x + 4, top + text_h + 2),
        font,
        scale,
        (0, 0, 0),
        thickness,
        cv2.LINE_AA,
    )


def write_image(path: Path, image_bgr: Any) -> None:
    import cv2

    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image_bgr):
        raise RuntimeError(f"Could not write image: {path}")


def _default_json_output(args: argparse.Namespace, image_path: Path) -> Path:
    return resolve_path(args.json_output) if args.json_output else args.output_dir / f"{image_path.stem}_{args.mode}.json"


def _default_annotated_output(args: argparse.Namespace, image_path: Path) -> Path:
    if args.annotated_output:
        return resolve_path(args.annotated_output)
    return args.output_dir / f"{image_path.stem}_{args.mode}_annotated.jpg"


def _base_result(args: argparse.Namespace, image_path: Path) -> dict[str, Any]:
    return {
        "image": str(image_path),
        "mode": args.mode,
        "models": {
            "plate_detection": str(args.plate_weights),
            "legibility": str(args.legibility_weights),
            "ocr": str(args.ocr_checkpoint),
        },
    }


def run_detection(args: argparse.Namespace, image_path: Path, image_bgr: Any) -> dict[str, Any]:
    detector = load_yolo_model(args.plate_weights)
    candidates = detect_plate_candidates(
        image_bgr,
        detector,
        conf=args.det_conf,
        imgsz=args.det_imgsz,
        device=args.device,
    )
    if args.max_plates is not None:
        candidates = candidates[: args.max_plates]

    output_dir = args.output_dir
    detections: list[dict[str, Any]] = []
    for candidate in candidates:
        if args.save_crops:
            candidate = replace(
                candidate,
                crop_path=save_crop(candidate.crop, output_dir, image_path.stem, candidate.index),
            )
        detections.append(candidate.to_json())

    annotated_output = _default_annotated_output(args, image_path)
    write_image(annotated_output, draw_candidates(image_bgr, candidates, detections))

    return {
        **_base_result(args, image_path),
        "detections": detections,
        "annotated_output": str(annotated_output),
    }


def run_classification(args: argparse.Namespace, image_path: Path, image_bgr: Any) -> dict[str, Any]:
    classifier = load_yolo_model(args.legibility_weights)
    return {
        **_base_result(args, image_path),
        "legibility": classify_legibility_crop(
            image_bgr,
            classifier,
            imgsz=args.classify_imgsz,
            device=args.device,
        ),
    }


def run_ocr(args: argparse.Namespace, image_path: Path, image_bgr: Any) -> dict[str, Any]:
    torch_device = choose_torch_device(args.device)
    ocr_model = load_line_ctc_ocr(args.ocr_checkpoint, device=torch_device)
    return {
        **_base_result(args, image_path),
        "device": str(torch_device),
        "ocr": ocr_line_ctc_crop(image_bgr, ocr_model, device=torch_device),
    }


def run_alpr(args: argparse.Namespace, image_path: Path, image_bgr: Any) -> dict[str, Any]:
    classifier = load_yolo_model(args.legibility_weights)
    torch_device = choose_torch_device(args.device)
    ocr_model = load_line_ctc_ocr(args.ocr_checkpoint, device=torch_device)

    if args.crop_is_plate:
        candidates = [
            PlateCandidate(
                index=0,
                kind="input_crop",
                class_id=0,
                class_name="plate",
                confidence=1.0,
                box=[0, 0, int(image_bgr.shape[1]), int(image_bgr.shape[0])],
                crop=image_bgr,
            )
        ]
    else:
        detector = load_yolo_model(args.plate_weights)
        candidates = detect_plate_candidates(
            image_bgr,
            detector,
            conf=args.det_conf,
            imgsz=args.det_imgsz,
            device=args.device,
        )
    if args.max_plates is not None:
        candidates = candidates[: args.max_plates]

    detections: list[dict[str, Any]] = []
    for candidate in candidates:
        if args.save_crops:
            candidate = replace(
                candidate,
                crop_path=save_crop(candidate.crop, args.output_dir, image_path.stem, candidate.index),
            )
        payload = candidate.to_json()
        payload["legibility"] = classify_legibility_crop(
            candidate.crop,
            classifier,
            imgsz=args.classify_imgsz,
            device=args.device,
        )
        payload["ocr"] = ocr_line_ctc_crop(candidate.crop, ocr_model, device=torch_device)
        detections.append(payload)

    annotated_output = _default_annotated_output(args, image_path)
    write_image(annotated_output, draw_candidates(image_bgr, candidates, detections))

    return {
        **_base_result(args, image_path),
        "device": str(torch_device),
        "crop_is_plate": bool(args.crop_is_plate),
        "detections": detections,
        "annotated_output": str(annotated_output),
    }


def persist_and_print(result: dict[str, Any], args: argparse.Namespace, image_path: Path) -> None:
    json_output = _default_json_output(args, image_path)
    json_output.parent.mkdir(parents=True, exist_ok=True)
    result = {**result, "json_output": str(json_output)}
    text = json.dumps(result, ensure_ascii=False, indent=None if args.compact else 2)
    json_output.write_text(text + "\n", encoding="utf-8")
    print(text)


def parse_args(argv: list[str] | None = None, *, default_mode: str = "alpr") -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run ALPR single-image inference: plate OBB detection, "
            "legibility classification, and SmallLPR-Line-CTC OCR."
        )
    )
    parser.add_argument("image", type=Path, help="Input full-frame image or plate crop.")
    parser.add_argument(
        "--mode",
        choices=("alpr", "detect", "classify", "ocr"),
        default=default_mode,
        help=f"Inference mode. Default: {default_mode}",
    )
    parser.add_argument("--plate-weights", type=Path, default=DEFAULT_PLATE_WEIGHTS)
    parser.add_argument("--legibility-weights", type=Path, default=DEFAULT_LEGIBILITY_WEIGHTS)
    parser.add_argument("--ocr-checkpoint", type=Path, default=DEFAULT_OCR_CHECKPOINT)
    parser.add_argument("--det-conf", type=float, default=0.25)
    parser.add_argument("--det-imgsz", type=int, default=1280)
    parser.add_argument("--classify-imgsz", type=int, default=64)
    parser.add_argument(
        "--device",
        default="auto",
        help="Ultralytics/PyTorch device: auto, cpu, cuda, cuda:0, or 0.",
    )
    parser.add_argument(
        "--crop-is-plate",
        action="store_true",
        help="In alpr mode, skip detection and treat the input image as one plate crop.",
    )
    parser.add_argument("--max-plates", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--json-output", type=Path, default=None)
    parser.add_argument("--annotated-output", type=Path, default=None)
    parser.add_argument("--save-crops", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--compact", action="store_true", help="Print compact JSON.")
    args = parser.parse_args(argv)

    args.image = resolve_path(args.image)
    args.plate_weights = resolve_path(args.plate_weights)
    args.legibility_weights = resolve_path(args.legibility_weights)
    args.ocr_checkpoint = resolve_path(args.ocr_checkpoint)
    args.output_dir = resolve_path(args.output_dir)
    if args.max_plates is not None and args.max_plates < 1:
        raise ValueError("--max-plates must be >= 1")
    return args


def main(argv: list[str] | None = None, *, default_mode: str = "alpr") -> None:
    args = parse_args(argv, default_mode=default_mode)
    validate_existing_file(args.image, "Input image")
    image_bgr = read_image(args.image)

    if args.mode == "detect":
        result = run_detection(args, args.image, image_bgr)
    elif args.mode == "classify":
        result = run_classification(args, args.image, image_bgr)
    elif args.mode == "ocr":
        result = run_ocr(args, args.image, image_bgr)
    else:
        result = run_alpr(args, args.image, image_bgr)
    persist_and_print(result, args, args.image)


if __name__ == "__main__":
    main()
