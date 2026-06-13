#!/usr/bin/env python3
"""Prepare LPLCv2-style crops for plate quality classifier training.

The official LPLCv2 data is distributed by request, so this converter accepts
common JSON shapes instead of assuming a single private annotation schema.
It writes two image-folder datasets:

  output/legibility4/{illegible,poor,good,perfect}/...
  output/binary/{suitable,unsuitable}/...
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any, Literal

import cv2

LEGIBILITY_NAMES = {
    0: "illegible",
    1: "poor",
    2: "good",
    3: "perfect",
}

SUITABLE = {"good", "perfect"}


def normalize_legibility(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip().lower()
        if text.isdigit():
            return LEGIBILITY_NAMES.get(int(text))
        aliases = {
            "0": "illegible",
            "1": "poor",
            "2": "good",
            "3": "perfect",
            "unreadable": "illegible",
            "bad": "poor",
        }
        return aliases.get(text, text if text in set(LEGIBILITY_NAMES.values()) else None)
    if isinstance(value, (int, float)):
        return LEGIBILITY_NAMES.get(int(value))
    return None


def iter_plate_annotations(payload: Any) -> Iterable[dict[str, Any]]:
    if isinstance(payload, list):
        records = payload
    elif isinstance(payload, Mapping):
        records = _first_list(payload, ("annotations", "records", "images", "data", "items"))
        if not records:
            records = [
                {"file_name": image_path, **image_record}
                for image_path, image_record in payload.items()
                if isinstance(image_record, Mapping)
            ]
    else:
        records = []

    for record in records:
        if not isinstance(record, Mapping):
            continue
        image_path = _first_value(record, ("image", "image_path", "img_path", "file_name", "filename", "path"))
        plates = _first_list(record, ("plates", "license_plates", "lps", "objects", "annotations", "anns"))
        if plates:
            for plate in plates:
                if isinstance(plate, Mapping):
                    yield {
                        **plate,
                        "_image_path": image_path or _first_value(
                            plate,
                            ("image", "image_path", "file_name", "filename", "path"),
                        ),
                        "_image_meta": {
                            key: value
                            for key, value in record.items()
                            if key not in {"plates", "license_plates", "lps", "objects", "annotations", "anns"}
                        },
                    }
        else:
            yield {**record, "_image_path": image_path}


def extract_bbox(
    annotation: Mapping[str, Any],
    *,
    bbox_format: Literal["auto", "xywh", "xyxy"] = "auto",
) -> tuple[int, int, int, int] | None:
    raw = _first_value(annotation, ("bbox", "box", "lp_bbox", "plate_bbox", "rect"))
    if raw is None and all(key in annotation for key in ("x", "y", "w", "h")):
        raw = [annotation["x"], annotation["y"], annotation["w"], annotation["h"]]
        bbox_format = "xywh" if bbox_format == "auto" else bbox_format
    if raw is None and all(key in annotation for key in ("x1", "y1", "x2", "y2")):
        raw = [annotation["x1"], annotation["y1"], annotation["x2"], annotation["y2"]]
        bbox_format = "xyxy" if bbox_format == "auto" else bbox_format
    if raw is None and isinstance(annotation.get("xy"), (list, tuple)):
        xy = annotation["xy"]
        if len(xy) >= 4 and len(xy) % 2 == 0:
            xs = [float(xy[i]) for i in range(0, len(xy), 2)]
            ys = [float(xy[i]) for i in range(1, len(xy), 2)]
            return tuple(int(round(v)) for v in (min(xs), min(ys), max(xs), max(ys)))
    if not isinstance(raw, (list, tuple)) or len(raw) < 4:
        return None

    x1, y1, v3, v4 = (float(raw[i]) for i in range(4))
    if bbox_format == "xywh":
        x2, y2 = x1 + v3, y1 + v4
    elif bbox_format == "xyxy":
        x2, y2 = v3, v4
    elif v3 <= x1 or v4 <= y1:
        x2, y2 = x1 + v3, y1 + v4
    else:
        x2, y2 = v3, v4

    return tuple(int(round(v)) for v in (x1, y1, x2, y2))


def convert_dataset(
    annotations_path: Path,
    image_root: Path,
    output_dir: Path,
    *,
    bbox_format: Literal["auto", "xywh", "xyxy"] = "auto",
    limit: int | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    payload = json.loads(annotations_path.read_text(encoding="utf-8"))
    counts: Counter[str] = Counter()
    skipped: Counter[str] = Counter()

    for index, annotation in enumerate(iter_plate_annotations(payload)):
        if limit is not None and index >= limit:
            break

        label = _annotation_legibility(annotation)
        if label is None:
            skipped["missing_legibility"] += 1
            continue

        image_path = annotation.get("_image_path")
        if not image_path:
            skipped["missing_image_path"] += 1
            continue
        src = image_root / str(image_path)
        if not src.exists():
            skipped["missing_image_file"] += 1
            continue

        bbox = extract_bbox(annotation, bbox_format=bbox_format)
        if bbox is None:
            skipped["missing_bbox"] += 1
            continue

        image = cv2.imread(str(src))
        if image is None:
            skipped["bad_image"] += 1
            continue
        crop = _crop_bbox(image, bbox)
        if crop is None:
            skipped["bad_bbox"] += 1
            continue

        counts[label] += 1
        if dry_run:
            continue

        stem = f"{src.stem}_{index:06d}.jpg"
        legibility_dir = output_dir / "legibility4" / label
        binary_dir = output_dir / "binary" / ("suitable" if label in SUITABLE else "unsuitable")
        legibility_dir.mkdir(parents=True, exist_ok=True)
        binary_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(legibility_dir / stem), crop)
        cv2.imwrite(str(binary_dir / stem), crop)

    summary = {
        "annotations": str(annotations_path),
        "image_root": str(image_root),
        "output_dir": str(output_dir),
        "counts": dict(counts),
        "skipped": dict(skipped),
    }
    if not dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def _annotation_legibility(annotation: Mapping[str, Any]) -> str | None:
    for key in ("legibility", "readability", "quality", "class", "label", "leg"):
        value = annotation.get(key)
        label = normalize_legibility(value)
        if label is not None:
            return label
    return None


def _crop_bbox(image, bbox: tuple[int, int, int, int]):
    h, w = image.shape[:2]
    x1, y1, x2, y2 = bbox
    x1 = max(0, min(w, x1))
    y1 = max(0, min(h, y1))
    x2 = max(0, min(w, x2))
    y2 = max(0, min(h, y2))
    if x2 <= x1 or y2 <= y1:
        return None
    return image[y1:y2, x1:x2]


def _first_value(record: Mapping[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        if key in record and record[key] not in (None, ""):
            return record[key]
    return None


def _first_list(record: Mapping[str, Any], keys: tuple[str, ...]) -> list[Any]:
    for key in keys:
        value = record.get(key)
        if isinstance(value, list):
            return value
    return []


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bbox-format", choices=("auto", "xywh", "xyxy"), default="auto")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = convert_dataset(
        args.annotations,
        args.image_root,
        args.output_dir,
        bbox_format=args.bbox_format,
        limit=args.limit,
        dry_run=args.dry_run,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
