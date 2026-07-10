"""
core/config.py — All constants, paths, and hyperparameters.
Single source of truth; import from here everywhere.
"""

import os
import tempfile
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

# Project root: api/core/ → api/ → ALPR_Vietnamese/
ROOT = Path(__file__).resolve().parent.parent.parent


def _rooted_env_path(name: str, default: str) -> Path:
    value = os.environ.get(name, default)
    path = Path(value)
    if path.is_absolute():
        return path
    return ROOT / path


# ── MongoDB Atlas ─────────────────────────────────────────────────────────────
MONGODB_URI: str = os.environ.get("MONGODB_URI", "")
MONGODB_DB_NAME: str = os.environ.get("MONGODB_DB_NAME", "alpr_vn")

# ── Web / Auth ───────────────────────────────────────────────────────────────
WEB_ORIGIN: str = os.environ.get(
    "WEB_ORIGIN",
    "http://localhost:5173,http://127.0.0.1:5173,http://localhost:3000",
)
AUTH_SECRET_KEY: str = os.environ.get("AUTH_SECRET_KEY", "")
AUTH_COOKIE_NAME: str = os.environ.get("AUTH_COOKIE_NAME", "alpr_session")
AUTH_COOKIE_SECURE: bool = os.environ.get("AUTH_COOKIE_SECURE", "").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
AUTH_SESSION_TTL_HOURS: int = int(os.environ.get("AUTH_SESSION_TTL_HOURS", "24"))
CSRF_COOKIE_NAME: str = os.environ.get("CSRF_COOKIE_NAME", "alpr_csrf")
AUTH_COOKIE_SAMESITE: str = os.environ.get("AUTH_COOKIE_SAMESITE", "lax").strip().lower()
MAX_UPLOAD_MB: int = int(os.environ.get("MAX_UPLOAD_MB", "512"))


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _env_int_list(name: str, default: list[int]) -> list[int]:
    value = os.environ.get(name)
    if value is None:
        return list(default)
    try:
        return [int(part.strip()) for part in value.split(",") if part.strip()]
    except ValueError:
        return list(default)


# ── Model paths ───────────────────────────────────────────────────────────────
YOLOV5_OBJECT_DEFAULT = "references/Character-Time-series-Matching/Vietnamese/object.pt"
VEHICLE_DETECTOR_BACKEND = os.environ.get("VEHICLE_DETECTOR_BACKEND", "auto").strip().lower()
# VEHICLE_MODEL_PATH_DEFAULT = "weights/detection/vehicle_best.pt"
VEHICLE_MODEL_PATH = _rooted_env_path("VEHICLE_MODEL_PATH", YOLOV5_OBJECT_DEFAULT)
# PLATE_MODEL_PATH = ROOT / "weights/detection/best.pt"
PLATE_MODEL_PATH = _rooted_env_path(
    "PLATE_MODEL_PATH",
    "runs/obb/experiments/detection/lp_detection_obb_merged/weights/best.pt",
)
REID_MODEL_PATH = _rooted_env_path("REID_MODEL_PATH", "weights/tracking/vehicle_reid.onnx")
REID_DEVICE = os.environ.get("REID_DEVICE", "auto").strip().lower()
# Vehicle tracker backend: "botsort" (BoT-SORT + ReID + CMC, accurate) or
# "bytetrack" (IoU + Kalman only — much faster, no ReID/CMC). Switch via env var.
VEHICLE_TRACKER_TYPE = os.environ.get("VEHICLE_TRACKER_TYPE", "bytetrack").strip().lower()

OCR_BACKEND = os.environ.get("OCR_BACKEND", "smalllpr_line_ctc").strip().lower()


def normalize_ocr_backend(value: str) -> str:
    backend = value.strip().lower().replace("-", "_")
    if backend in {
        "default",
        "small_lpr_line_ctc",
        "smalllpr_line_ctc",
        "line_ctc",
    }:
        return "smalllpr_line_ctc"
    if backend == "vietnamese_yolov5":
        return "vietnamese_yolov5"
    raise ValueError("OCR_BACKEND must be one of: default, smalllpr_line_ctc, vietnamese_yolov5")


SMALL_LPR_CKPT_PATH = _rooted_env_path(
    "SMALL_LPR_CKPT_PATH",
    "weights/ocr/small_lpr-epoch=136-val_acc=0.914.ckpt",
)
SMALL_LPR_CTC_CKPT_PATH = _rooted_env_path(
    "SMALL_LPR_CTC_CKPT_PATH",
    "weights/ocr/small_lpr_ctc/ctc_20260609_155238/small_lpr_ctc-epoch=055-val_acc=0.9358.ckpt",
)
SMALL_LPR_LINE_CTC_CKPT_PATH = _rooted_env_path(
    "SMALL_LPR_LINE_CTC_CKPT_PATH",
    "weights/ocr/small_lpr_line_ctc/line_ctc_cleaned_20260618_061855/small_lpr_line_ctc-epoch=008-val_acc=0.9501.ckpt",
)
PARSEQ_OCR_CKPT_PATH = _rooted_env_path(
    "PARSEQ_OCR_CKPT_PATH",
    "weights/ocr/parseq/parseq_vn_plate_best.pt",
)
YOLOV5_CHAR_CKPT_PATH = _rooted_env_path(
    "YOLOV5_CHAR_CKPT_PATH",
    "references/Character-Time-series-Matching/Vietnamese/char.pt",
)
YOLOV5_OBJECT_CKPT_PATH = _rooted_env_path(
    "YOLOV5_OBJECT_CKPT_PATH",
    YOLOV5_OBJECT_DEFAULT,
)
PARSEQ_IMAGE_W = int(os.environ.get("PARSEQ_IMAGE_W", "128"))
PARSEQ_IMAGE_H = int(os.environ.get("PARSEQ_IMAGE_H", "32"))

# Backward-compatible alias for scripts that still import OCR_CKPT_PATH.
OCR_CKPT_PATH = SMALL_LPR_LINE_CTC_CKPT_PATH

# ── Detection ─────────────────────────────────────────────────────────────────
# object.pt class IDs:
#   1: motorbike, 6: car, 7: truck, 8: van, 9: bus, 10: delivery tricycle
VEHICLE_CLASSES = _env_int_list("VEHICLE_CLASSES", [1, 6, 7, 8, 9, 10])

# ── OCR ───────────────────────────────────────────────────────────────────────
CONF_THRESHOLD = 0.90
IMG_W, IMG_H = 96, 48

CHARS = [
    "<PAD>",
    "<SOS>",
    "<EOS>",
    "0",
    "1",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9",
    "A",
    "B",
    "C",
    "D",
    "E",
    "F",
    "G",
    "H",
    "K",
    "L",
    "M",
    "N",
    "P",
    "Q",
    "R",
    "S",
    "T",
    "U",
    "V",
    "X",
    "Y",
    "Z",
    "Đ",
    "-",
    "_",
]
PAD_IDX, SOS_IDX, EOS_IDX = 0, 1, 2

# ── Video processing ──────────────────────────────────────────────────────────
FRAME_STRIDE = 1  # run plate detection every N-th frame
# 0 disables sampling so vehicle detection/tracking still runs on every frame.
ALPR_TARGET_PROCESS_FPS = _env_float("ALPR_TARGET_PROCESS_FPS", 0.0)
ALPR_PREVIEW_FPS = _env_float("ALPR_PREVIEW_FPS", 2.0)
ALPR_PREVIEW_MAX_WIDTH = _env_int("ALPR_PREVIEW_MAX_WIDTH", 960)
ALPR_PREVIEW_JPEG_QUALITY = _env_int("ALPR_PREVIEW_JPEG_QUALITY", 70)
ALPR_PREPROCESSED_VIDEO_DIR = _rooted_env_path(
    "ALPR_PREPROCESSED_VIDEO_DIR",
    str(Path(tempfile.gettempdir()) / "alpr_preprocessed"),
)
ALPR_PREPROCESSED_VIDEO_TTL_SEC = _env_float("ALPR_PREPROCESSED_VIDEO_TTL_SEC", 3600.0)
ALPR_PREPROCESSED_VIDEO_CLEANUP_INTERVAL_SEC = _env_float(
    "ALPR_PREPROCESSED_VIDEO_CLEANUP_INTERVAL_SEC",
    300.0,
)
ALPR_DEBUG_TIMINGS = _env_bool("ALPR_DEBUG_TIMINGS", False)
PLATE_PAD = 8  # context padding around plate crop (px)
CASCADE_VEHICLE_PAD_RATIO = 0.08  # context padding around vehicle crops
CASCADE_VEHICLE_PAD_MIN = 16  # min vehicle crop context padding (px)
ASSOCIATION_MATCH_FRAMES = _env_int("ASSOCIATION_MATCH_FRAMES", 2)
ASSOCIATION_AGREEMENT_RATIO = _env_float("ASSOCIATION_AGREEMENT_RATIO", 0.6)

# ── Anti-hallucination — Layer 1 (Pre-OCR quality gates) ─────────────────────
PLATE_DET_CONF = 0.50  # min YOLO OBB detection confidence
MIN_PLATE_W = 30  # min raw plate width  (px, before padding)
MIN_PLATE_H = 10  # min raw plate height (px, before padding)
BLUR_THRESHOLD = 80.0  # Laplacian variance; below → too blurry (hard gate)
LAP_MAX = 500.0  # Laplacian variance ceiling for quality score normalisation

# ── Anti-hallucination — Layer 3 (Temporal consistency) ──────────────────────
MIN_FRAME_VOTES = 2  # OCR frames required before marking a vehicle done

# ── Track buffer & lifecycle ──────────────────────────────────────────────────
MAX_BUFFER = 30  # max plate crops stored per track (also bounds the voting set)
MIN_FRAMES_FOR_OCR = 2  # min buffered frames before final track-level OCR vote
LOST_THRESHOLD = 30  # consecutive missing strides before track is finalised
TOP_K_FRAMES = 10
# ── Multi-cluster voting ──────────────────────────────────────────────────────
MAX_CLUSTERS = 3  # max distinct plate clusters within one track buffer
CLUSTER_SIMILARITY_THRESHOLD = 0.6  # min normalised Levenshtein similarity to merge clusters

# ── UI display labels ─────────────────────────────────────────────────────────
VN_CLASS = {
    "car": "Ô tô",
    "bus": "Xe buýt",
    "truck": "Xe tải",
    "van": "Xe tải nhỏ",
    "motorcycle": "Xe máy",
    "motorbike_rider": "Xe máy",
    "motorbike": "Xe máy",
    "delivery tricycle": "Xe ba gác",
}
