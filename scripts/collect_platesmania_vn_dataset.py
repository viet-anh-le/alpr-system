from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import time
from dataclasses import asdict, dataclass, replace
from html import escape
from html.parser import HTMLParser
from pathlib import Path
from typing import Callable, Iterable
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin, urlparse
from urllib.request import Request, urlopen

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_DIR = ROOT / "data" / "raw" / "platesmania_vn"
DEFAULT_HTML_DIR = DEFAULT_OUTPUT_DIR / "html_pages"
DEFAULT_PLATE_WEIGHTS = ROOT / "weights" / "detection" / "best.pt"
PLATESMANIA_VN_GALLERY = "https://platesmania.com/vn/gallery"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
ALLOWED_DOWNLOAD_HOST_SUFFIX = "platesmania.com"
KILLBOT_MARKERS = (
    "killbot",
    "user verification",
    "checking your browser",
    "verify you are human",
)


class PlatesmaniaBlockedError(RuntimeError):
    """Raised when Platesmania returns bot-verification or blocked content."""


@dataclass(frozen=True)
class PlateRecord:
    record_id: str
    page_url: str
    detail_url: str
    vehicle_image_url: str
    plate_ref_url: str | None
    plate_text_raw: str
    plate_text_normalized: str
    vehicle_image_path: Path | None = None
    plate_crop_path: Path | None = None
    status: str = "discovered"
    review_reason: str = ""
    detector_confidence: float | None = None
    bbox_xyxy: tuple[int, int, int, int] | None = None
    obb_points: tuple[tuple[float, float], ...] | None = None
    split: str | None = None


@dataclass(frozen=True)
class PlateDetection:
    confidence: float
    class_id: int
    class_name: str
    box_xyxy: tuple[int, int, int, int]
    points: tuple[
        tuple[float, float], tuple[float, float], tuple[float, float], tuple[float, float]
    ]


@dataclass(frozen=True)
class _ImageCandidate:
    href: str | None
    src: str
    alt: str


class _PlatesmaniaImageParser(HTMLParser):
    def __init__(self, page_url: str) -> None:
        super().__init__(convert_charrefs=True)
        self.page_url = page_url
        self.current_href: str | None = None
        self.images: list[_ImageCandidate] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attr_map = {name.lower(): value for name, value in attrs if value is not None}
        if tag.lower() == "a":
            href = attr_map.get("href")
            self.current_href = urljoin(self.page_url, href) if href else None
            return

        if tag.lower() != "img":
            return

        src = (
            attr_map.get("src")
            or attr_map.get("data-src")
            or attr_map.get("data-original")
            or attr_map.get("data-lazy-src")
        )
        if not src:
            return

        self.images.append(
            _ImageCandidate(
                href=self.current_href,
                src=urljoin(self.page_url, src),
                alt=attr_map.get("alt", ""),
            )
        )

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() == "a":
            self.current_href = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect Platesmania Vietnam full-frame images and auto-label license plates."
    )
    parser.add_argument("--source", choices=["direct", "html", "mixed"], default="html")
    parser.add_argument("--html-dir", type=Path, default=DEFAULT_HTML_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--crawl-mode", choices=["gallery", "province-search"], default="gallery")
    parser.add_argument("--all-vietnam", action="store_true")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--end-index", type=int, default=None)
    parser.add_argument("--max-pages", type=int, default=200)
    parser.add_argument("--province-start", type=int, default=11)
    parser.add_argument("--province-end", type=int, default=99)
    parser.add_argument("--search-start-min", type=int, default=0)
    parser.add_argument("--search-start-max", type=int, default=100)
    parser.add_argument("--max-records", type=int, default=None)
    parser.add_argument("--plate-weights", type=Path, default=DEFAULT_PLATE_WEIGHTS)
    parser.add_argument("--plate-conf", type=float, default=0.25)
    parser.add_argument("--accept-conf", type=float, default=0.50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--delay", type=float, default=2.0)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--resume", action="store_true", help="Continue writing into an existing output directory."
    )
    parser.add_argument(
        "--skip-detection",
        action="store_true",
        help="Download frames only; skip YOLO labels/crops.",
    )
    parser.add_argument(
        "--download-html-only",
        action="store_true",
        help="Fetch gallery pages into --html-dir and exit.",
    )
    parser.add_argument(
        "--html-save-mode",
        choices=["compact", "full", "records"],
        default="compact",
        help="How --download-html-only stores fetched pages. compact keeps only image/detail snippets.",
    )
    return parser.parse_args()


def normalize_plate_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().upper())


def _is_http_url(url: str) -> bool:
    parsed = urlparse(url)
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def _is_platesmania_host(url: str) -> bool:
    parsed = urlparse(url)
    hostname = (parsed.hostname or "").lower()
    return hostname == ALLOWED_DOWNLOAD_HOST_SUFFIX or hostname.endswith(
        f".{ALLOWED_DOWNLOAD_HOST_SUFFIX}"
    )


def _is_supported_image_url(url: str) -> bool:
    return _is_http_url(url) and Path(urlparse(url).path).suffix.lower() in IMAGE_SUFFIXES


def _is_plate_ref_url(url: str) -> bool:
    return "/inf/" in urlparse(url).path


def _is_vehicle_image_url(url: str) -> bool:
    return _is_supported_image_url(url) and not _is_plate_ref_url(url)


def _is_detail_url(url: str) -> bool:
    return re.search(r"/vn/nomer\d+", urlparse(url).path) is not None


def _plate_text_from_alt(alt: str) -> str | None:
    normalized = normalize_plate_text(alt)
    if not normalized:
        return None
    if not any(char.isdigit() for char in normalized):
        return None
    if re.fullmatch(r"[0-9A-ZĐ.\- ]+", normalized) is None:
        return None
    return normalized


def _record_id_from_detail(detail_url: str, fallback: str) -> str:
    match = re.search(r"(nomer\d+)", urlparse(detail_url).path)
    if match:
        return match.group(1)
    digest = hashlib.sha1(fallback.encode("utf-8")).hexdigest()[:12]
    return f"plate_{digest}"


def safe_stem(value: str) -> str:
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")
    return stem or hashlib.sha1(value.encode("utf-8")).hexdigest()[:12]


def parse_gallery_html(html: str, *, page_url: str) -> list[PlateRecord]:
    parser = _PlatesmaniaImageParser(page_url=page_url)
    parser.feed(html)

    grouped: dict[str, list[_ImageCandidate]] = {}
    for image in parser.images:
        if image.href is None or not _is_detail_url(image.href):
            continue
        grouped.setdefault(image.href, []).append(image)

    records: list[PlateRecord] = []
    for detail_url, images in grouped.items():
        vehicle = next((image for image in images if _is_vehicle_image_url(image.src)), None)
        plate_ref = next((image for image in images if _is_plate_ref_url(image.src)), None)
        if vehicle is None or plate_ref is None:
            continue

        plate_text = _plate_text_from_alt(plate_ref.alt)
        if plate_text is None:
            continue

        record_id = _record_id_from_detail(detail_url, fallback=f"{detail_url}:{vehicle.src}")
        records.append(
            PlateRecord(
                record_id=safe_stem(record_id),
                page_url=page_url,
                detail_url=detail_url,
                vehicle_image_url=vehicle.src,
                plate_ref_url=plate_ref.src,
                plate_text_raw=plate_ref.alt.strip(),
                plate_text_normalized=plate_text,
            )
        )

    return records


def record_to_compact_html(record: PlateRecord) -> str:
    vehicle_alt = record.record_id
    parts = [
        '<div class="platesmania-record">',
        f'  <a href="{escape(record.detail_url, quote=True)}">',
        f'    <img src="{escape(record.vehicle_image_url, quote=True)}" alt="{escape(vehicle_alt, quote=True)}">',
        "  </a>",
    ]
    if record.plate_ref_url is not None:
        parts.extend(
            [
                f'  <a href="{escape(record.detail_url, quote=True)}">',
                (
                    f'    <img src="{escape(record.plate_ref_url, quote=True)}" '
                    f'alt="{escape(record.plate_text_raw, quote=True)}">'
                ),
                "  </a>",
            ]
        )
    parts.append("</div>")
    return "\n".join(parts)


def records_to_compact_html(records: list[PlateRecord], *, page_url: str) -> str:
    body = "\n".join(record_to_compact_html(record) for record in records)
    return "\n".join(
        [
            "<!doctype html>",
            '<html lang="en">',
            "<head>",
            '  <meta charset="utf-8">',
            f'  <meta name="source-url" content="{escape(page_url, quote=True)}">',
            "</head>",
            "<body>",
            body,
            "</body>",
            "</html>",
            "",
        ]
    )


def urls_to_download(record: PlateRecord) -> list[str]:
    return [record.vehicle_image_url]


def build_gallery_url(index: int) -> str:
    if index < 0:
        raise ValueError("Gallery index must be non-negative.")
    if index == 0:
        return PLATESMANIA_VN_GALLERY
    return f"{PLATESMANIA_VN_GALLERY}-{index}"


def build_province_search_url(nomer: int, start: int) -> str:
    if not 0 <= start:
        raise ValueError("Search start must be non-negative.")
    if not 0 <= nomer:
        raise ValueError("Province code must be non-negative.")
    return f"https://platesmania.com/vn/gallery.php?&nomer={nomer}&start={start}"


def _check_not_blocked(text: str, *, source: str) -> None:
    lowered = text.lower()
    if any(marker in lowered for marker in KILLBOT_MARKERS):
        raise PlatesmaniaBlockedError(f"Blocked by bot verification while reading {source}")


def default_fetcher(url: str, timeout: float) -> bytes:
    if not _is_http_url(url) or not _is_platesmania_host(url):
        raise ValueError(f"Refusing to fetch non-Platesmania URL: {url}")

    headers = {
        "User-Agent": "ALPR-Vietnamese-dataset-collector/1.0 (+research; respectful rate limit)",
        "Accept": "image/avif,image/webp,image/apng,image/*,*/*;q=0.8",
    }
    request = Request(url, headers=headers)
    try:
        with urlopen(request, timeout=timeout) as response:
            return response.read()
    except HTTPError as exc:
        if exc.code in {401, 403, 429}:
            raise PlatesmaniaBlockedError(f"HTTP {exc.code} while fetching {url}") from exc
        raise
    except URLError as exc:
        raise RuntimeError(f"Could not fetch {url}: {exc}") from exc


def fetch_text(url: str, timeout: float) -> str:
    if not _is_http_url(url) or not _is_platesmania_host(url):
        raise ValueError(f"Refusing to fetch non-Platesmania URL: {url}")

    headers = {
        "User-Agent": "ALPR-Vietnamese-dataset-collector/1.0 (+research; respectful rate limit)",
        "Accept": "text/html,application/xhtml+xml",
    }
    request = Request(url, headers=headers)
    try:
        with urlopen(request, timeout=timeout) as response:
            data = response.read()
    except HTTPError as exc:
        if exc.code in {401, 403, 429}:
            raise PlatesmaniaBlockedError(f"HTTP {exc.code} while fetching {url}") from exc
        raise
    except URLError as exc:
        raise RuntimeError(f"Could not fetch {url}: {exc}") from exc

    text = data.decode("utf-8", errors="replace")
    _check_not_blocked(text, source=url)
    return text


def _image_suffix_from_url(url: str) -> str:
    suffix = Path(urlparse(url).path).suffix.lower()
    return suffix if suffix in IMAGE_SUFFIXES else ".jpg"


def _decode_image_bytes(data: bytes, *, source: str) -> np.ndarray:
    image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None or image.size == 0:
        raise ValueError(f"Downloaded content is not a readable image: {source}")
    return image


def download_vehicle_image(
    record: PlateRecord,
    output_dir: Path,
    *,
    fetcher: Callable[[str, float], bytes] = default_fetcher,
    timeout: float = 30.0,
    overwrite: bool = False,
) -> Path:
    if not _is_supported_image_url(record.vehicle_image_url):
        raise ValueError(f"Unsupported vehicle image URL: {record.vehicle_image_url}")
    if not _is_platesmania_host(record.vehicle_image_url):
        raise ValueError(f"Refusing to download non-Platesmania image: {record.vehicle_image_url}")

    image_dir = output_dir / "downloads" / "full_frames"
    image_dir.mkdir(parents=True, exist_ok=True)
    image_path = (
        image_dir
        / f"{safe_stem(record.record_id)}{_image_suffix_from_url(record.vehicle_image_url)}"
    )
    if image_path.exists() and not overwrite:
        return image_path

    data = fetcher(record.vehicle_image_url, timeout)
    _decode_image_bytes(data, source=record.vehicle_image_url)
    image_path.write_bytes(data)
    return image_path


def _order_plate_points(points: np.ndarray) -> np.ndarray:
    src = points.astype(np.float32)
    sums = src.sum(axis=1)
    diffs = np.diff(src, axis=1).ravel()
    top_left = src[np.argmin(sums)]
    bottom_right = src[np.argmax(sums)]
    top_right = src[np.argmin(diffs)]
    bottom_left = src[np.argmax(diffs)]
    return np.array([top_left, top_right, bottom_right, bottom_left], dtype=np.float32)


def warp_plate_crop(frame: np.ndarray, points: Iterable[Iterable[float]]) -> np.ndarray:
    src = np.array(list(points), dtype=np.float32).reshape(4, 2)
    ordered = _order_plate_points(src)
    top_left, top_right, bottom_right, bottom_left = ordered
    width = int(
        round(max(np.linalg.norm(top_right - top_left), np.linalg.norm(bottom_right - bottom_left)))
    )
    height = int(
        round(max(np.linalg.norm(bottom_left - top_left), np.linalg.norm(bottom_right - top_right)))
    )
    if width < 1 or height < 1:
        return np.zeros((0, 0, 3), dtype=np.uint8)

    dst = np.array(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype=np.float32
    )
    matrix = cv2.getPerspectiveTransform(ordered, dst)
    return cv2.warpPerspective(frame, matrix, (width, height))


def _box_from_points(points: np.ndarray) -> tuple[int, int, int, int]:
    x, y, w, h = cv2.boundingRect(points.astype(np.int32))
    return (int(x), int(y), int(x + w), int(y + h))


def extract_plate_detections(result: object, *, min_conf: float = 0.0) -> list[PlateDetection]:
    detections: list[PlateDetection] = []
    names = getattr(result, "names", {})
    names = names if isinstance(names, dict) else {}

    obb = getattr(result, "obb", None)
    if obb is not None and getattr(obb, "xyxyxyxy", None) is not None:
        points_array = obb.xyxyxyxy.cpu().numpy().astype(np.float32)
        confs = (
            obb.conf.cpu().numpy()
            if getattr(obb, "conf", None) is not None
            else np.ones((len(points_array),))
        )
        classes = (
            obb.cls.cpu().numpy().astype(int)
            if getattr(obb, "cls", None) is not None
            else np.zeros((len(points_array),), dtype=int)
        )
        for points, confidence, class_id in zip(points_array, confs, classes):
            if float(confidence) < min_conf:
                continue
            detections.append(
                PlateDetection(
                    confidence=float(confidence),
                    class_id=int(class_id),
                    class_name=str(names.get(int(class_id), "plate")),
                    box_xyxy=_box_from_points(points),
                    points=tuple((float(x), float(y)) for x, y in points),  # type: ignore[arg-type]
                )
            )
        return detections

    boxes_obj = getattr(result, "boxes", None)
    if boxes_obj is None or getattr(boxes_obj, "xyxy", None) is None:
        return detections

    boxes = boxes_obj.xyxy.cpu().numpy().astype(np.float32)
    confs = (
        boxes_obj.conf.cpu().numpy()
        if getattr(boxes_obj, "conf", None) is not None
        else np.ones((len(boxes),))
    )
    classes = (
        boxes_obj.cls.cpu().numpy().astype(int)
        if getattr(boxes_obj, "cls", None) is not None
        else np.zeros((len(boxes),), dtype=int)
    )
    for box, confidence, class_id in zip(boxes, confs, classes):
        if float(confidence) < min_conf:
            continue
        x1, y1, x2, y2 = box.tolist()
        points = ((x1, y1), (x2, y1), (x2, y2), (x1, y2))
        detections.append(
            PlateDetection(
                confidence=float(confidence),
                class_id=int(class_id),
                class_name=str(names.get(int(class_id), "plate")),
                box_xyxy=(int(x1), int(y1), int(x2), int(y2)),
                points=tuple((float(x), float(y)) for x, y in points),  # type: ignore[arg-type]
            )
        )
    return detections


def load_yolo_model(weights_path: Path) -> object:
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise ImportError("Install ultralytics from requirements.txt to run detection.") from exc

    if not weights_path.exists():
        raise FileNotFoundError(f"Plate detector weights not found: {weights_path}")
    return YOLO(str(weights_path))


def detect_plates(
    frame: np.ndarray, model: object, *, conf: float, imgsz: int
) -> list[PlateDetection]:
    results = model.predict(frame, conf=conf, imgsz=imgsz, verbose=False)
    if not results:
        return []
    return extract_plate_detections(results[0], min_conf=conf)


def _ensure_split_dirs(output_dir: Path, group: str, split: str) -> tuple[Path, Path]:
    images_dir = output_dir / group / "images" / split
    labels_dir = output_dir / group / "labels" / split
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    return images_dir, labels_dir


def write_ocr_sample(
    output_dir: Path, record: PlateRecord, crop: np.ndarray, *, split: str
) -> tuple[Path, Path]:
    if crop.size == 0:
        raise ValueError(f"Cannot write empty OCR crop for {record.record_id}")
    images_dir, labels_dir = _ensure_split_dirs(output_dir, "ocr", split)
    stem = safe_stem(record.record_id)
    image_path = images_dir / f"{stem}.jpg"
    label_path = labels_dir / f"{stem}.txt"
    if not cv2.imwrite(str(image_path), crop):
        raise RuntimeError(f"Could not write OCR crop: {image_path}")
    label_path.write_text(f"{record.plate_text_normalized}\n", encoding="utf-8")
    return image_path, label_path


def _normalize_points(
    points: Iterable[Iterable[float]], *, image_width: int, image_height: int
) -> list[float]:
    if image_width <= 0 or image_height <= 0:
        raise ValueError("Image dimensions must be positive.")
    normalized: list[float] = []
    for x, y in points:
        clipped_x = min(max(float(x), 0.0), float(image_width))
        clipped_y = min(max(float(y), 0.0), float(image_height))
        normalized.extend([clipped_x / image_width, clipped_y / image_height])
    return normalized


def build_yolo_obb_line(detection: PlateDetection, *, image_width: int, image_height: int) -> str:
    normalized = _normalize_points(
        detection.points,
        image_width=image_width,
        image_height=image_height,
    )
    return f"{detection.class_id} " + " ".join(f"{value:.6f}" for value in normalized)


def write_detection_sample(
    output_dir: Path,
    record: PlateRecord,
    source_image: Path,
    detections: list[PlateDetection],
    *,
    split: str,
) -> tuple[Path, Path]:
    image = cv2.imread(str(source_image))
    if image is None or image.size == 0:
        raise ValueError(f"Could not read source image: {source_image}")
    image_height, image_width = image.shape[:2]

    images_dir, labels_dir = _ensure_split_dirs(output_dir, "detection", split)
    suffix = (
        source_image.suffix.lower() if source_image.suffix.lower() in IMAGE_SUFFIXES else ".jpg"
    )
    stem = safe_stem(record.record_id)
    image_path = images_dir / f"{stem}{suffix}"
    label_path = labels_dir / f"{stem}.txt"
    shutil.copy2(source_image, image_path)

    label_lines = [
        build_yolo_obb_line(detection, image_width=image_width, image_height=image_height)
        for detection in detections
    ]
    label_path.write_text("\n".join(label_lines) + ("\n" if label_lines else ""), encoding="utf-8")
    return image_path, label_path


def write_detection_yaml(output_dir: Path, *, names: dict[int, str]) -> Path:
    yaml_path = output_dir / "detection" / "data.yaml"
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    ordered = [names[index] for index in sorted(names)]
    names_block = "\n".join(f"  {index}: {name}" for index, name in enumerate(ordered))
    lines = [
        "train: images/train",
        "val: images/val",
        f"nc: {len(ordered)}",
        "names:",
        names_block,
        "",
    ]
    yaml_path.write_text("\n".join(lines), encoding="utf-8")
    return yaml_path


def split_for_record(record_id: str, *, val_ratio: float, seed: int) -> str:
    if not 0.0 <= val_ratio < 1.0:
        raise ValueError("val_ratio must be in [0, 1).")
    digest = hashlib.sha256(f"{seed}:{record_id}".encode("utf-8")).hexdigest()
    value = int(digest[:8], 16) / 0xFFFFFFFF
    return "val" if value < val_ratio else "train"


def write_pending_review(
    output_dir: Path,
    record: PlateRecord,
    source_image: Path,
    *,
    reason: str,
    crop: np.ndarray | None = None,
) -> Path:
    review_dir = output_dir / "review" / "pending_review"
    review_dir.mkdir(parents=True, exist_ok=True)
    stem = safe_stem(record.record_id)
    image_path = review_dir / f"{stem}.jpg"
    if crop is not None and crop.size > 0:
        if not cv2.imwrite(str(image_path), crop):
            raise RuntimeError(f"Could not write review crop: {image_path}")
    else:
        shutil.copy2(source_image, image_path)
    (review_dir / f"{stem}.txt").write_text(reason + "\n", encoding="utf-8")
    return image_path


def _manifest_value(value: object) -> object:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return [_manifest_value(item) for item in value]
    return value


def record_to_manifest(record: PlateRecord) -> dict[str, object]:
    raw = asdict(record)
    return {key: _manifest_value(value) for key, value in raw.items()}


def record_from_manifest(raw: dict[str, object]) -> PlateRecord:
    required = {
        "record_id",
        "page_url",
        "detail_url",
        "vehicle_image_url",
        "plate_text_raw",
        "plate_text_normalized",
    }
    missing = sorted(key for key in required if not raw.get(key))
    if missing:
        raise ValueError(f"Record is missing required fields: {', '.join(missing)}")
    plate_ref_url = raw.get("plate_ref_url")
    return PlateRecord(
        record_id=safe_stem(str(raw["record_id"])),
        page_url=str(raw["page_url"]),
        detail_url=str(raw["detail_url"]),
        vehicle_image_url=str(raw["vehicle_image_url"]),
        plate_ref_url=str(plate_ref_url) if plate_ref_url else None,
        plate_text_raw=str(raw["plate_text_raw"]),
        plate_text_normalized=normalize_plate_text(str(raw["plate_text_normalized"])),
    )


def write_manifest(output_dir: Path, records: list[PlateRecord]) -> Path:
    manifest_dir = output_dir / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = manifest_dir / "records.jsonl"
    with manifest_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record_to_manifest(record), ensure_ascii=False) + "\n")
    return manifest_path


def _read_records_jsonl(path: Path) -> list[PlateRecord]:
    records: list[PlateRecord] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                raw = json.loads(stripped)
                if not isinstance(raw, dict):
                    raise ValueError("JSONL row must be an object")
                records.append(record_from_manifest(raw))
            except Exception as exc:
                raise ValueError(f"Invalid record in {path}:{line_number}: {exc}") from exc
    return records


def _iter_html_pages(html_dir: Path) -> list[tuple[str, str]]:
    if not html_dir.exists():
        raise FileNotFoundError(f"HTML directory not found: {html_dir}")
    pages: list[tuple[str, str]] = []
    html_paths = sorted([*html_dir.glob("*.html"), *html_dir.glob("*.htm")])
    if not html_paths:
        raise FileNotFoundError(f"No .html/.htm files found in HTML directory: {html_dir}")
    for path in html_paths:
        html = path.read_text(encoding="utf-8", errors="replace")
        _check_not_blocked(html, source=str(path))
        pages.append((PLATESMANIA_VN_GALLERY, html))
    return pages


def _iter_direct_pages(args: argparse.Namespace) -> Iterable[tuple[str, str]]:
    for _, url in page_requests(args):
        html = fetch_text(url, timeout=args.timeout)
        yield url, html
        if args.delay > 0:
            time.sleep(args.delay)


def gallery_indexes(args: argparse.Namespace) -> range:
    if args.end_index is not None:
        return range(args.start_index, args.end_index + 1)
    if args.all_vietnam:
        return range(args.start_index, args.start_index + args.max_pages)
    return range(args.start_index, args.start_index + 1)


def page_requests(args: argparse.Namespace) -> Iterable[tuple[str, str]]:
    crawl_mode = getattr(args, "crawl_mode", "gallery")
    if crawl_mode == "province-search":
        for nomer in range(args.province_start, args.province_end + 1):
            for start in range(args.search_start_min, args.search_start_max + 1):
                yield f"nomer-{nomer}-start-{start}", build_province_search_url(nomer, start)
        return

    for index in gallery_indexes(args):
        slug = "gallery" if index == 0 else f"gallery-{index}"
        yield slug, build_gallery_url(index)


def _write_fetched_page(
    html_dir: Path,
    *,
    filename_stem: str,
    page_url: str,
    html: str,
    records: list[PlateRecord],
    save_mode: str,
) -> None:
    html_dir.mkdir(parents=True, exist_ok=True)
    if save_mode == "full":
        (html_dir / f"{filename_stem}.html").write_text(html, encoding="utf-8")
        return

    if save_mode == "compact":
        compact = records_to_compact_html(records, page_url=page_url)
        (html_dir / f"{filename_stem}.html").write_text(compact, encoding="utf-8")
        return

    records_path = html_dir / "gallery_records.jsonl"
    with records_path.open("a", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record_to_manifest(record), ensure_ascii=False) + "\n")


def download_gallery_html_pages(
    args: argparse.Namespace,
    *,
    fetcher: Callable[[str, float], str] = fetch_text,
) -> dict[str, int]:
    html_dir = args.html_dir.expanduser().resolve()
    if args.html_save_mode == "records":
        records_path = html_dir / "gallery_records.jsonl"
        if records_path.exists():
            records_path.unlink()

    total_pages = 0
    total_records = 0
    empty_pages = 0
    for filename_stem, page_url in page_requests(args):
        html = fetcher(page_url, args.timeout)
        _check_not_blocked(html, source=page_url)
        records = parse_gallery_html(html, page_url=page_url)
        _write_fetched_page(
            html_dir,
            filename_stem=filename_stem,
            page_url=page_url,
            html=html,
            records=records,
            save_mode=args.html_save_mode,
        )
        total_pages += 1
        total_records += len(records)

        print(f"saved_page={page_url} records={len(records)} mode={args.html_save_mode}")
        if not records and args.all_vietnam:
            empty_pages += 1
            if empty_pages >= 2:
                break
        else:
            empty_pages = 0
        if args.max_records is not None and total_records >= args.max_records:
            break
        if args.delay > 0:
            time.sleep(args.delay)
    return {"pages": total_pages, "records": total_records}


def _add_page_records(
    records_by_id: dict[str, PlateRecord],
    vehicle_image_urls: set[str],
    page_url: str,
    html: str,
    *,
    max_records: int | None,
) -> int:
    page_records = parse_gallery_html(html, page_url=page_url)
    for record in page_records:
        add_record_if_new(records_by_id, vehicle_image_urls, record)
        if max_records is not None and len(records_by_id) >= max_records:
            break
    return len(page_records)


def add_record_if_new(
    records_by_id: dict[str, PlateRecord],
    vehicle_image_urls: set[str],
    record: PlateRecord,
) -> bool:
    if record.record_id in records_by_id or record.vehicle_image_url in vehicle_image_urls:
        return False
    records_by_id[record.record_id] = record
    vehicle_image_urls.add(record.vehicle_image_url)
    return True


def collect_records_from_sources(args: argparse.Namespace) -> list[PlateRecord]:
    records_by_id: dict[str, PlateRecord] = {}
    vehicle_image_urls: set[str] = set()

    if args.source in {"html", "mixed"}:
        records_jsonl = args.html_dir / "gallery_records.jsonl"
        if records_jsonl.exists():
            for record in _read_records_jsonl(records_jsonl):
                add_record_if_new(records_by_id, vehicle_image_urls, record)
                if args.max_records is not None and len(records_by_id) >= args.max_records:
                    return list(records_by_id.values())
            if args.source == "html":
                return list(records_by_id.values())

        for page_url, html in _iter_html_pages(args.html_dir):
            _add_page_records(
                records_by_id,
                vehicle_image_urls,
                page_url,
                html,
                max_records=args.max_records,
            )
            if args.max_records is not None and len(records_by_id) >= args.max_records:
                return list(records_by_id.values())

    if args.source in {"direct", "mixed"}:
        empty_pages = 0
        for page_url, html in _iter_direct_pages(args):
            page_count = _add_page_records(
                records_by_id,
                vehicle_image_urls,
                page_url,
                html,
                max_records=args.max_records,
            )
            if page_count == 0 and args.all_vietnam:
                empty_pages += 1
                if empty_pages >= 2:
                    break
            else:
                empty_pages = 0
            if args.max_records is not None and len(records_by_id) >= args.max_records:
                return list(records_by_id.values())

    return list(records_by_id.values())


def _best_detection(detections: list[PlateDetection]) -> PlateDetection | None:
    if not detections:
        return None
    return max(detections, key=lambda detection: detection.confidence)


def process_record(
    record: PlateRecord,
    output_dir: Path,
    *,
    plate_model: object | None,
    args: argparse.Namespace,
) -> PlateRecord:
    try:
        image_path = download_vehicle_image(
            record,
            output_dir,
            timeout=args.timeout,
            overwrite=args.overwrite,
        )
    except Exception as exc:
        return replace(record, status="download_failed", review_reason=str(exc))

    record = replace(record, vehicle_image_path=image_path)
    if args.skip_detection or plate_model is None:
        return replace(record, status="downloaded")

    frame = cv2.imread(str(image_path))
    if frame is None or frame.size == 0:
        return replace(record, status="pending_review", review_reason="full_frame_unreadable")

    detections = detect_plates(frame, plate_model, conf=args.plate_conf, imgsz=args.imgsz)
    split = split_for_record(record.record_id, val_ratio=args.val_ratio, seed=args.seed)
    if detections:
        write_detection_sample(output_dir, record, image_path, detections, split=split)

    detection = _best_detection(detections)
    if detection is None:
        write_pending_review(output_dir, record, image_path, reason="no_plate_detection")
        return replace(
            record, status="no_detection", review_reason="no_plate_detection", split=split
        )

    crop = warp_plate_crop(frame, detection.points)
    updated = replace(
        record,
        split=split,
        detector_confidence=detection.confidence,
        bbox_xyxy=detection.box_xyxy,
        obb_points=detection.points,
    )

    if detection.confidence < args.accept_conf:
        review_path = write_pending_review(
            output_dir,
            updated,
            image_path,
            reason=f"low_detector_confidence:{detection.confidence:.4f}",
        )
        return replace(
            updated,
            status="pending_review",
            review_reason="low_detector_confidence",
            plate_crop_path=review_path,
        )

    try:
        crop_path, _ = write_ocr_sample(output_dir, updated, crop, split=split)
    except Exception as exc:
        review_path = write_pending_review(
            output_dir, updated, image_path, reason=str(exc), crop=crop
        )
        return replace(
            updated,
            status="pending_review",
            review_reason=str(exc),
            plate_crop_path=review_path,
        )

    return replace(updated, status="accepted", plate_crop_path=crop_path)


def summarize(records: list[PlateRecord]) -> dict[str, int]:
    summary: dict[str, int] = {}
    for record in records:
        summary[record.status] = summary.get(record.status, 0) + 1
    return summary


def _path_contains(parent: Path, child: Path) -> bool:
    try:
        child.resolve().relative_to(parent.resolve())
    except ValueError:
        return False
    return True


def prepare_output_dir(args: argparse.Namespace, output_dir: Path) -> None:
    html_input_inside_output = args.source in {"html", "mixed"} and _path_contains(
        output_dir, args.html_dir
    )
    if args.overwrite and html_input_inside_output:
        raise SystemExit(
            "Refusing --overwrite because --html-dir is inside --output-dir. "
            "Move saved HTML outside the output directory or omit --overwrite."
        )

    if args.overwrite and output_dir.exists():
        shutil.rmtree(output_dir)

    if output_dir.exists() and any(output_dir.iterdir()) and not args.resume and not args.overwrite:
        raise SystemExit(
            f"Output directory already exists: {output_dir}. Use --resume or --overwrite."
        )

    output_dir.mkdir(parents=True, exist_ok=True)


def main() -> None:
    args = parse_args()
    if args.download_html_only:
        try:
            summary = download_gallery_html_pages(args)
        except PlatesmaniaBlockedError as exc:
            raise SystemExit(
                f"{exc}\nCannot download gallery HTML automatically while verification is active."
            ) from exc
        print(f"html_dir={args.html_dir.expanduser().resolve()}")
        print("summary=" + json.dumps(summary, ensure_ascii=False, sort_keys=True))
        return

    output_dir = args.output_dir.expanduser().resolve()
    prepare_output_dir(args, output_dir)

    try:
        records = collect_records_from_sources(args)
    except PlatesmaniaBlockedError as exc:
        raise SystemExit(
            f"{exc}\nUse --source html with saved gallery HTML files instead of bypassing verification."
        ) from exc

    if not records:
        raise SystemExit("No Platesmania records found.")

    plate_model = (
        None if args.skip_detection else load_yolo_model(args.plate_weights.expanduser().resolve())
    )
    processed: list[PlateRecord] = []
    class_names: dict[int, str] = {0: "plate"}

    for index, record in enumerate(records, start=1):
        processed_record = process_record(record, output_dir, plate_model=plate_model, args=args)
        processed.append(processed_record)
        if processed_record.status == "accepted":
            print(
                f"[{index}/{len(records)}] accepted {record.record_id} {record.plate_text_normalized}"
            )
        else:
            print(
                f"[{index}/{len(records)}] {processed_record.status} {record.record_id}: {processed_record.review_reason}"
            )

    for record in processed:
        if record.obb_points is not None:
            class_names.setdefault(0, "plate")
    write_detection_yaml(output_dir, names=class_names)
    manifest_path = write_manifest(output_dir, processed)
    print(f"output_dir={output_dir}")
    print(f"manifest={manifest_path}")
    print("summary=" + json.dumps(summarize(processed), ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
