"""
core/config.py — All constants, paths, and hyperparameters.
Single source of truth; import from here everywhere.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

# Project root: api/core/ → api/ → ALPR_Vietnamese/
ROOT = Path(__file__).resolve().parent.parent.parent

# ── MongoDB Atlas ─────────────────────────────────────────────────────────────
MONGODB_URI: str = os.environ.get("MONGODB_URI", "")
MONGODB_DB_NAME: str = os.environ.get("MONGODB_DB_NAME", "alpr_vn")


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


# ── Model paths ───────────────────────────────────────────────────────────────
VEHICLE_MODEL_PATH = ROOT / "weights/detection/vehicle_best.pt"
# PLATE_MODEL_PATH = ROOT / "weights/detection/best.pt"
PLATE_MODEL_PATH = ROOT / "runs/obb/experiments/detection/lp_detection_obb_merged/weights/best.pt"

OCR_BACKEND = os.environ.get("OCR_BACKEND", "smalllpr_ctc").strip().lower()
SMALL_LPR_CKPT_PATH = ROOT / os.environ.get(
    "SMALL_LPR_CKPT_PATH",
    "weights/ocr/small_lpr-epoch=136-val_acc=0.914.ckpt",
)
SMALL_LPR_CTC_CKPT_PATH = ROOT / os.environ.get(
    "SMALL_LPR_CTC_CKPT_PATH",
    "weights/ocr/small_lpr_ctc/ctc_20260609_155238/small_lpr_ctc-epoch=055-val_acc=0.9358.ckpt",
)
SMALL_LPR_NAR_CKPT_PATH = ROOT / os.environ.get(
    "SMALL_LPR_NAR_CKPT_PATH",
    "weights/ocr/small_lpr_nar/nar_20260608_123600/small_lpr_nar-epoch=085-val_acc=0.9581.ckpt",
)
PARSEQ_OCR_CKPT_PATH = ROOT / os.environ.get(
    "PARSEQ_OCR_CKPT_PATH",
    "weights/ocr/parseq/parseq_vn_plate_best.pt",
)
YOLOV5_CHAR_CKPT_PATH = ROOT / os.environ.get(
    "YOLOV5_CHAR_CKPT_PATH",
    "references/Character-Time-series-Matching/Vietnamese/char.pt",
)
YOLOV5_OBJECT_CKPT_PATH = ROOT / os.environ.get(
    "YOLOV5_OBJECT_CKPT_PATH",
    "references/Character-Time-series-Matching/Vietnamese/object.pt",
)
PARSEQ_IMAGE_W = int(os.environ.get("PARSEQ_IMAGE_W", "128"))
PARSEQ_IMAGE_H = int(os.environ.get("PARSEQ_IMAGE_H", "32"))

# Backward-compatible alias for scripts that still import OCR_CKPT_PATH.
OCR_CKPT_PATH = (
    PARSEQ_OCR_CKPT_PATH
    if OCR_BACKEND == "parseq"
    else (
        SMALL_LPR_CTC_CKPT_PATH
        if OCR_BACKEND in {"smalllpr_ctc", "small_lpr_ctc", "ctc"}
        else (
            SMALL_LPR_NAR_CKPT_PATH
            if OCR_BACKEND in {"smalllpr_nar", "small_lpr_nar", "nar"}
            else SMALL_LPR_CKPT_PATH
        )
    )
)

# ── Detection ─────────────────────────────────────────────────────────────────
# vehicle_best.pt class IDs (5-class custom detector):
#   0: car, 1: bus, 2: truck, 3: motorcycle, 4: motorbike_rider
VEHICLE_CLASSES = [0, 1, 2, 3, 4]  # car, bus, truck, motorcycle, motorbike_rider

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
ALPR_PREVIEW_FPS = _env_float("ALPR_PREVIEW_FPS", 0.0)
ALPR_DEBUG_TIMINGS = _env_bool("ALPR_DEBUG_TIMINGS", False)
PLATE_PAD = 8  # context padding around plate crop (px)
CASCADE_VEHICLE_PAD_RATIO = 0.08  # context padding around vehicle crops
CASCADE_VEHICLE_PAD_MIN = 16  # min vehicle crop context padding (px)
CASCADE_PLATE_TRACK_IOU = 0.30  # IoU threshold for cascade plate track continuity
CASCADE_PLATE_TRACK_BUFFER = 15  # frames to retain unmatched cascade plate tracks

# ── Anti-hallucination — Layer 1 (Pre-OCR quality gates) ─────────────────────
PLATE_DET_CONF = 0.50  # min YOLO OBB detection confidence
MIN_PLATE_W = 30  # min raw plate width  (px, before padding)
MIN_PLATE_H = 15  # min raw plate height (px, before padding)
BLUR_THRESHOLD = 80.0  # Laplacian variance; below → too blurry (hard gate)
LAP_MAX = 500.0  # Laplacian variance ceiling for quality score normalisation

# ── Anti-hallucination — Layer 3 (Temporal consistency) ──────────────────────
MIN_FRAME_VOTES = 2  # OCR frames required before marking a vehicle done

# ── Track buffer & lifecycle ──────────────────────────────────────────────────
MAX_BUFFER = 10  # max plate crops stored per track
MIN_FRAMES_FOR_OCR = 3  # min buffered frames before final track-level OCR vote
LOST_THRESHOLD = 5  # consecutive missing strides before track is finalised
TOP_K_FRAMES = 5  # how many top-quality OCR frames to pass to voting

# ── UI display labels ─────────────────────────────────────────────────────────
VN_CLASS = {
    "car": "Ô tô",
    "bus": "Xe buýt",
    "truck": "Xe tải",
    "motorcycle": "Xe máy",
    "motorbike_rider": "Xe máy",
}
