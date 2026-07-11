from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DEFAULT_PARSEQ_VN_CHARSET = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZĐ-."


@dataclass(frozen=True)
class OcrSplitStats:
    count: int
    max_label_length: int
    charset: str


def normalize_plate_label(label: str) -> str:
    return label.strip().upper().replace("[SEP]", "")


def label_from_path(path: str | Path) -> str:
    stem = Path(path).stem
    label_with_layout = stem.split("#", 1)[0]
    label = label_with_layout.rsplit("&", 1)[0]
    return normalize_plate_label(label)


def parse_layout_from_path(path: str | Path) -> str | None:
    stem = Path(path).stem
    label_with_layout = stem.split("#", 1)[0]
    _, sep, layout = label_with_layout.rpartition("&")
    if not sep:
        return None
    return layout if layout in {"1", "2"} else None


def unknown_chars(label: str, charset: str) -> set[str]:
    allowed = set(charset)
    return {char for char in label if char not in allowed}


def _image_paths(directory: str | Path) -> list[Path]:
    root = Path(directory)
    if not root.exists():
        raise FileNotFoundError(f"OCR image directory does not exist: {root}")
    return sorted(path for path in root.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)


def make_parseq_transform(
    *,
    image_width: int = 128,
    image_height: int = 32,
    augment: bool = False,
) -> transforms.Compose:
    steps: list[Callable] = [transforms.Resize((image_height, image_width))]
    if augment:
        steps.extend(
            [
                transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
                transforms.RandomAffine(
                    degrees=2,
                    translate=(0.02, 0.05),
                    scale=(0.95, 1.05),
                    fill=0,
                ),
                transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5)),
            ]
        )
    steps.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )
    return transforms.Compose(steps)


class FilenamePlateDataset(Dataset):
    def __init__(
        self,
        image_dir: str | Path,
        *,
        charset: str = DEFAULT_PARSEQ_VN_CHARSET,
        max_label_length: int = 25,
        transform: Callable | None = None,
        subset_size: int | None = None,
        skip_invalid: bool = False,
    ) -> None:
        self.image_dir = Path(image_dir)
        self.charset = charset
        self.max_label_length = max_label_length
        self.transform = transform or make_parseq_transform()
        self.samples: list[tuple[Path, str]] = []
        invalid: list[tuple[Path, str, str]] = []

        paths = _image_paths(self.image_dir)
        if subset_size is not None:
            paths = paths[:subset_size]

        for path in paths:
            label = label_from_path(path)
            bad_chars = unknown_chars(label, charset)
            too_long = len(label) > max_label_length
            if bad_chars or too_long:
                reason = (
                    f"unknown chars {''.join(sorted(bad_chars))!r}"
                    if bad_chars
                    else f"length {len(label)} > max_label_length={max_label_length}"
                )
                if skip_invalid:
                    invalid.append((path, label, reason))
                    continue
                raise ValueError(f"Invalid OCR label '{label}' in {path}: {reason}")
            self.samples.append((path, label))

        self.invalid_samples = tuple(invalid)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, str, str]:
        path, label = self.samples[index]
        image = Image.open(path).convert("RGB")
        image_tensor = self.transform(image)
        return image_tensor, label, str(path)


def parseq_collate(batch: list[tuple[torch.Tensor, str, str]]) -> tuple[torch.Tensor, list[str], list[str]]:
    images, labels, paths = zip(*batch)
    return torch.stack(list(images)), list(labels), list(paths)


def scan_split_stats(image_dir: str | Path, *, charset: str = DEFAULT_PARSEQ_VN_CHARSET) -> OcrSplitStats:
    labels = [label_from_path(path) for path in _image_paths(image_dir)]
    seen = sorted({char for label in labels for char in label})
    bad = sorted({char for char in seen if char not in set(charset)})
    if bad:
        raise ValueError(f"Dataset contains chars outside configured charset: {''.join(bad)}")
    return OcrSplitStats(
        count=len(labels),
        max_label_length=max((len(label) for label in labels), default=0),
        charset="".join(seen),
    )
