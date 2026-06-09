from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import cv2
import lightning as L
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

try:
    import albumentations as A
except ImportError:  # pragma: no cover - training env has albumentations
    A = None

from ocr.models.small_lpr import smart_resize
from ocr.models.utils import encode


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
LAYOUT_ONE_LINE = 0
LAYOUT_TWO_LINE = 1


def label_from_path(path: str | Path) -> str:
    label_with_layout = Path(path).stem.split("#", 1)[0]
    return label_with_layout.rsplit("&", 1)[0].upper()


def parse_layout_from_path(path: str | Path) -> int:
    label_with_layout = Path(path).stem.split("#", 1)[0]
    _, sep, layout_token = label_with_layout.rpartition("&")
    if not sep or layout_token not in {"1", "2"}:
        raise ValueError(
            "Filename must include manual layout annotation '&1' or '&2' "
            f"before optional image id '#...': {path}"
        )
    return LAYOUT_ONE_LINE if layout_token == "1" else LAYOUT_TWO_LINE


def encode_slot_target(
    text: str,
    chars: list[str],
    max_slots: int,
    *,
    eos_id: int = 2,
    pad_id: int = 0,
) -> list[int]:
    tokens = encode(text, chars)
    if len(tokens) + 1 > max_slots:
        raise ValueError(
            f"Label '{text}' needs {len(tokens) + 1} slots including EOS, "
            f"exceeds max_slots={max_slots}"
        )
    tokens = tokens + [eos_id]
    return tokens + [pad_id] * (max_slots - len(tokens))


def decode_slot_tokens(tokens: list[int] | torch.Tensor, chars: list[str]) -> str:
    result: list[str] = []
    for token in tokens:
        idx = int(token)
        if idx == 2:
            break
        if idx in (0, 1):
            continue
        result.append(chars[idx])
    return "".join(result)


def _image_paths(directory: str | Path) -> list[Path]:
    root = Path(directory)
    return sorted(p for p in root.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)


class SlotLPRDataset(Dataset):
    def __init__(self, args: Any, stage: str, *, augment: bool | None = None) -> None:
        self.args = args
        self.stage = stage

        if stage == "train":
            img_dir = args.train_dir
        elif stage == "valid":
            img_dir = args.valid_dir
        else:
            img_dir = args.test_dir

        paths = _image_paths(img_dir)
        subset_size = getattr(args, "subset_size", None)
        if subset_size:
            paths = paths[: int(subset_size)]

        min_w = int(getattr(args, "min_img_width", 1))
        min_h = int(getattr(args, "min_img_height", 1))
        self.img_paths: list[Path] = []
        for path in paths:
            width, height = self._read_image_size(path)
            if width >= min_w and height >= min_h:
                self.img_paths.append(path)

        if stage == "train":
            random.shuffle(self.img_paths)

        self.target_hw = (int(args.img_size[1]), int(args.img_size[0]))
        self.max_slots = int(args.max_slots)
        use_augment = stage == "train" if augment is None else augment
        self.transform = self._build_transform() if use_augment else None

    def __len__(self) -> int:
        return len(self.img_paths)

    def __getitem__(self, index: int) -> dict[str, Any]:
        path = self.img_paths[index]
        image = cv2.imread(str(path))
        if image is None:
            raise RuntimeError(f"Failed to read image: {path}")

        if self.transform is not None:
            image = self.transform(image=image)["image"]

        text = label_from_path(path)
        layout = parse_layout_from_path(path)
        slots = encode_slot_target(text, self.args.chars, self.max_slots)
        image = smart_resize(image, target_hw=self.target_hw)
        image = self._normalize(image)

        return {
            "image": torch.from_numpy(image),
            "slots": torch.tensor(slots, dtype=torch.long),
            "layout": int(layout),
            "length": min(len(text), self.max_slots),
            "text": text,
            "path": str(path),
        }

    @staticmethod
    def _read_image_size(path: Path) -> tuple[int, int]:
        image = cv2.imread(str(path))
        if image is None:
            return 0, 0
        height, width = image.shape[:2]
        return width, height

    @staticmethod
    def _normalize(image: np.ndarray) -> np.ndarray:
        image = image.astype(np.float32)
        image = (image - 127.5) * 0.0078125
        return np.transpose(image, (2, 0, 1))

    @staticmethod
    def _build_transform():
        if A is None:
            return None
        return A.Compose(
            [
                A.ShiftScaleRotate(
                    shift_limit=0.04,
                    scale_limit=0.06,
                    rotate_limit=4,
                    border_mode=cv2.BORDER_REPLICATE,
                    p=0.35,
                ),
                A.Perspective(scale=(0.015, 0.06), p=0.25),
                A.OneOf(
                    [
                        A.MotionBlur(blur_limit=5),
                        A.GaussianBlur(blur_limit=5),
                        A.GaussNoise(),
                    ],
                    p=0.35,
                ),
                A.RandomBrightnessContrast(
                    brightness_limit=0.18,
                    contrast_limit=0.18,
                    p=0.45,
                ),
            ]
        )


def slot_lpr_collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "image": torch.stack([item["image"] for item in batch]),
        "slots": torch.stack([item["slots"] for item in batch]),
        "layout": torch.tensor([item["layout"] for item in batch], dtype=torch.long),
        "length": torch.tensor([item["length"] for item in batch], dtype=torch.long),
        "text": [item["text"] for item in batch],
        "path": [item["path"] for item in batch],
    }


class SlotLPRDataModule(L.LightningDataModule):
    def __init__(self, args: Any) -> None:
        super().__init__()
        self.args = args

    def setup(self, stage: str) -> None:
        if stage in ("fit", None):
            self.train = SlotLPRDataset(self.args, "train")
            self.val = SlotLPRDataset(self.args, "valid", augment=False)
            print(f"train: {len(self.train)} | val: {len(self.val)}")
        if stage in ("test", None):
            self.test = SlotLPRDataset(self.args, "test", augment=False)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train,
            batch_size=self.args.batch_size,
            shuffle=True,
            num_workers=self.args.num_workers,
            collate_fn=slot_lpr_collate,
            pin_memory=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val,
            batch_size=self.args.batch_size,
            shuffle=False,
            num_workers=self.args.num_workers,
            collate_fn=slot_lpr_collate,
            pin_memory=True,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test,
            batch_size=self.args.batch_size,
            shuffle=False,
            num_workers=self.args.num_workers,
            collate_fn=slot_lpr_collate,
            pin_memory=True,
        )
