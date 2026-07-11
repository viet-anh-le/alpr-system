"""
detection/train.py — Finetune a YOLO vehicle detector.

Usage (from project root):
  # Step 1: convert BDD100K to YOLO format (one-time)
  python -m data.scripts.bdd100k_to_yolo

  # Step 2: finetune
  python -m detection.train \\
      --weights  weights/detection/yolov8n.pt \\
      --data     configs/detection/bdd100k.yaml \\
      --epochs   50 \\
      --batch    16 \\
      --freeze   10

After training the best checkpoint lands in:
  experiments/detection/<run-name>/weights/best.pt

Copy it to weights/detection/best_vehicle.pt and update
api/core/config.py:
  VEHICLE_MODEL_PATH = ROOT / "weights/detection/best_vehicle.pt"
  VEHICLE_CLASSES    = [0, 1, 2, 3]   # car, truck, bus, motorcycle
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parent.parent
    p = argparse.ArgumentParser(description="Finetune YOLO vehicle detector")
    p.add_argument(
        "--weights", type=Path,
        default=project_root / "weights/detection/yolov8n.pt",
        help="Pretrained YOLO checkpoint to finetune from",
    )
    p.add_argument(
        "--data", type=Path,
        default=project_root / "configs/detection/bdd100k.yaml",
        help="Dataset YAML (YOLO format)",
    )
    p.add_argument("--epochs",  type=int,   default=50)
    p.add_argument("--batch",   type=int,   default=16)
    p.add_argument("--imgsz",   type=int,   default=640)
    p.add_argument("--lr0",     type=float, default=1e-3,  help="Initial learning rate")
    p.add_argument("--lrf",     type=float, default=0.01,  help="Final LR as fraction of lr0")
    p.add_argument(
        "--freeze", type=int, default=10,
        help="Number of backbone layers to freeze (0 = full finetune)",
    )
    p.add_argument(
        "--project", type=Path,
        default=project_root / "experiments/detection",
        help="Directory for run logs and checkpoints",
    )
    p.add_argument("--name",    type=str,   default="bdd100k",  help="Run name")
    p.add_argument("--device",  type=str,   default="",         help="cuda device or cpu")
    p.add_argument("--workers", type=int,   default=8)
    return p.parse_args()


def finetune(
    weights: Path,
    data:    Path,
    epochs:  int   = 50,
    batch:   int   = 16,
    imgsz:   int   = 640,
    lr0:     float = 1e-3,
    lrf:     float = 0.01,
    freeze:  int   = 10,
    project: Path  = Path("experiments/detection"),
    name:    str   = "bdd100k",
    device:  str   = "",
    workers: int   = 8,
) -> Path:
    """
    Finetune a pretrained YOLO model and return the path to the best checkpoint.

    freeze=10  locks the first 10 backbone layers, letting only the detection
    head and upper backbone adapt — appropriate when the source domain (BDD100K)
    is close to the target domain and the dataset is small.
    Set freeze=0 for a full end-to-end finetune on larger datasets.
    """
    from ultralytics import YOLO

    if not weights.exists():
        raise FileNotFoundError(f"Weights not found: {weights}")
    if not data.exists():
        raise FileNotFoundError(f"Dataset YAML not found: {data}")

    logger.info("Loading pretrained weights from %s", weights)
    model = YOLO(str(weights))

    logger.info(
        "Finetuning — epochs=%d  batch=%d  imgsz=%d  freeze=%d  lr0=%.4g",
        epochs, batch, imgsz, freeze, lr0,
    )
    model.train(
        data    = str(data),
        epochs  = epochs,
        batch   = batch,
        imgsz   = imgsz,
        lr0     = lr0,
        lrf     = lrf,
        freeze  = freeze,
        project = str(project),
        name    = name,
        device  = device,
        workers = workers,
        # Augmentation — conservative for finetuning
        hsv_h   = 0.015,
        hsv_s   = 0.7,
        hsv_v   = 0.4,
        degrees = 0.0,
        flipud  = 0.0,
        fliplr  = 0.5,
        mosaic  = 1.0,
        mixup   = 0.0,
    )

    best_ckpt = project / name / "weights" / "best.pt"
    logger.info("Training complete. Best checkpoint: %s", best_ckpt)
    return best_ckpt


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = _parse_args()
    finetune(
        weights = args.weights,
        data    = args.data,
        epochs  = args.epochs,
        batch   = args.batch,
        imgsz   = args.imgsz,
        lr0     = args.lr0,
        lrf     = args.lrf,
        freeze  = args.freeze,
        project = args.project,
        name    = args.name,
        device  = args.device,
        workers = args.workers,
    )
