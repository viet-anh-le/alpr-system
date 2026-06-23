"""
SmallLPR-Line-CTC datamodule.

The OCR dataset is expected to live under data/datasets/ocr with labels encoded
in filenames.  Character boxes are not used.
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple

import albumentations as A
import cv2
import lightning as L
import numpy as np
import torch
from imutils import paths
from torch.utils.data import DataLoader, Dataset

from lprnet.small_lpr import smart_resize
from lprnet.small_lpr_line_ctc import _GLOBAL_T_STEPS, _LINE_T_STEPS, _ONE_LINE_T_STEPS
from lprnet.trans_datamodule import _read_image_size
from lprnet.utils import encode

LAYOUT_ONE_LINE: int = 0
LAYOUT_TWO_LINE: int = 1
LAYOUT_IGNORE_INDEX: int = -100


@dataclass(frozen=True)
class LineCtcLabel:
    global_text: str
    one_line_text: str
    top_text: str
    bottom_text: str
    global_label: List[int]
    one_line_label: List[int]
    top_label: List[int]
    bottom_label: List[int]
    layout_label: int
    layout_loss_mask: bool
    one_line_loss_mask: bool
    top_loss_mask: bool
    bottom_loss_mask: bool
    is_ambiguous_layout: bool
    has_sep: bool


def parse_line_ctc_label(label: str, chars: Sequence[str]) -> LineCtcLabel:
    text = label.upper()
    has_sep = "[SEP]" in text

    global_label = encode(text, list(chars))
    if has_sep:
        top_text, bottom_text = text.split("[SEP]", 1)
        one_line_text = ""
        layout_label = LAYOUT_TWO_LINE
        layout_loss_mask = True
        one_line_loss_mask = False
        top_loss_mask = True
        bottom_loss_mask = bool(bottom_text)
    else:
        one_line_text = text
        top_text = ""
        bottom_text = ""
        layout_label = LAYOUT_ONE_LINE
        layout_loss_mask = True
        one_line_loss_mask = True
        top_loss_mask = False
        bottom_loss_mask = False

    return LineCtcLabel(
        global_text=text,
        one_line_text=one_line_text,
        top_text=top_text,
        bottom_text=bottom_text,
        global_label=global_label,
        one_line_label=encode(one_line_text, list(chars)) if one_line_text else [],
        top_label=encode(top_text, list(chars)) if top_text else [],
        bottom_label=encode(bottom_text, list(chars)) if bottom_text else [],
        layout_label=layout_label,
        layout_loss_mask=layout_loss_mask,
        one_line_loss_mask=one_line_loss_mask,
        top_loss_mask=top_loss_mask,
        bottom_loss_mask=bottom_loss_mask,
        is_ambiguous_layout=False,
        has_sep=has_sep,
    )


def _flatten_targets(labels: List[List[int]]) -> torch.Tensor:
    flattened: List[int] = []
    for label in labels:
        flattened.extend(label)
    return torch.tensor(flattened, dtype=torch.long)


def collate_fn_line_ctc(batch: List[Tuple[np.ndarray, LineCtcLabel]]) -> dict:
    images, labels = zip(*batch)
    batch_size = len(batch)

    global_labels = [label.global_label for label in labels]
    one_line_labels = [label.one_line_label for label in labels]
    top_labels = [label.top_label for label in labels]
    bottom_labels = [label.bottom_label for label in labels]

    return {
        "images": torch.stack([torch.from_numpy(image) for image in images], dim=0),
        "global_targets": _flatten_targets(global_labels),
        "global_lengths": torch.tensor([len(label) for label in global_labels], dtype=torch.long),
        "global_input_lengths": torch.full((batch_size,), _GLOBAL_T_STEPS, dtype=torch.long),
        "one_line_targets": _flatten_targets(one_line_labels),
        "one_line_lengths": torch.tensor([len(label) for label in one_line_labels], dtype=torch.long),
        "one_line_input_lengths": torch.full((batch_size,), _ONE_LINE_T_STEPS, dtype=torch.long),
        "top_targets": _flatten_targets(top_labels),
        "top_lengths": torch.tensor([len(label) for label in top_labels], dtype=torch.long),
        "top_input_lengths": torch.full((batch_size,), _LINE_T_STEPS, dtype=torch.long),
        "bottom_targets": _flatten_targets(bottom_labels),
        "bottom_lengths": torch.tensor([len(label) for label in bottom_labels], dtype=torch.long),
        "bottom_input_lengths": torch.full((batch_size,), _LINE_T_STEPS, dtype=torch.long),
        "layout_labels": torch.tensor([label.layout_label for label in labels], dtype=torch.long),
        "layout_loss_mask": torch.tensor([label.layout_loss_mask for label in labels], dtype=torch.bool),
        "one_line_loss_mask": torch.tensor(
            [label.one_line_loss_mask for label in labels],
            dtype=torch.bool,
        ),
        "top_loss_mask": torch.tensor([label.top_loss_mask for label in labels], dtype=torch.bool),
        "bottom_loss_mask": torch.tensor([label.bottom_loss_mask for label in labels], dtype=torch.bool),
        "has_sep": torch.tensor([label.has_sep for label in labels], dtype=torch.bool),
        "is_ambiguous_layout": torch.tensor(
            [label.is_ambiguous_layout for label in labels],
            dtype=torch.bool,
        ),
        "texts": [label.global_text for label in labels],
    }


def _load_excluded_paths(
    exclude_paths_file: str | os.PathLike[str] | None,
    *,
    dataset_root: Path,
) -> set[Path]:
    if not exclude_paths_file:
        return set()
    path = Path(exclude_paths_file)
    if not path.is_absolute():
        path = Path.cwd() / path
    if not path.exists():
        return set()

    excluded: set[Path] = set()
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        candidate = Path(line)
        if not candidate.is_absolute():
            candidate = dataset_root / candidate
        excluded.add(candidate.resolve())
    return excluded


class SmallLPRLineCTCDataset(Dataset):
    def __init__(self, args, stage: str):
        self.args = args
        self.stage = stage

        if stage == "train":
            img_dir = args.train_dir
        elif stage == "valid":
            img_dir = args.valid_dir
        else:
            img_dir = args.test_dir

        all_paths = list(paths.list_images(img_dir))
        dataset_root = Path(img_dir).resolve().parent
        excluded_paths = _load_excluded_paths(
            getattr(args, "exclude_paths_file", None),
            dataset_root=dataset_root,
        )
        min_w = getattr(args, "min_img_width", 20)
        min_h = getattr(args, "min_img_height", 8)
        self.img_paths: List[str] = []
        skipped = 0
        excluded = 0
        for path in all_paths:
            if Path(path).resolve() in excluded_paths:
                excluded += 1
                continue
            width, height = _read_image_size(path)
            if width is not None and (width < min_w or height < min_h):
                skipped += 1
            else:
                self.img_paths.append(path)
        if skipped > 0:
            print(f"[LineCTC/{stage}] Skipped {skipped}/{len(all_paths)} tiny images")
        if excluded > 0:
            print(f"[LineCTC/{stage}] Excluded {excluded}/{len(all_paths)} reviewed bad samples")

        if stage == "train":
            random.shuffle(self.img_paths)

        self.target_hw: Tuple[int, int] = (args.img_size[1], args.img_size[0])
        use_augment = bool(getattr(args, "augment", True))
        self.transform = self._build_transform() if stage == "train" and use_augment else None

    def __len__(self) -> int:
        return len(self.img_paths)

    def __getitem__(self, index: int) -> Tuple[np.ndarray, LineCtcLabel]:
        filename = self.img_paths[index]
        image = cv2.imread(filename)
        if image is None:
            image = np.zeros((self.target_hw[0], self.target_hw[1], 3), dtype=np.uint8)

        if self.transform is not None:
            image = self.transform(image=image)["image"]

        image = smart_resize(image, target_hw=self.target_hw)
        image = self._normalize(image)

        basename = os.path.basename(filename)
        label_text = os.path.splitext(basename)[0].split("#")[0].upper()
        label = parse_line_ctc_label(label_text, self.args.chars)
        return image, label

    def _normalize(self, image: np.ndarray) -> np.ndarray:
        image = image.astype(np.float32)
        image = (image - 127.5) * 0.0078125
        return np.transpose(image, (2, 0, 1))

    def _build_transform(self):
        return A.Compose(
            [
                A.ShiftScaleRotate(
                    shift_limit=0.05,
                    scale_limit=0.05,
                    rotate_limit=5,
                    border_mode=cv2.BORDER_REPLICATE,
                    p=0.4,
                ),
                A.Perspective(scale=(0.02, 0.08), p=0.3),
                A.OneOf(
                    [
                        A.MotionBlur(blur_limit=5),
                        A.GaussianBlur(blur_limit=5),
                        A.GaussNoise(var_limit=(10.0, 40.0)),
                    ],
                    p=0.4,
                ),
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
                A.HueSaturationValue(
                    hue_shift_limit=10,
                    sat_shift_limit=20,
                    val_shift_limit=20,
                    p=0.3,
                ),
            ]
        )


class SmallLPRLineCTCDataModule(L.LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.args = args

    def setup(self, stage: str) -> None:
        if stage == "fit":
            self.train_ds = SmallLPRLineCTCDataset(self.args, "train")
            self.val_ds = SmallLPRLineCTCDataset(self.args, "valid")
            print(f"[LineCTC] Train: {len(self.train_ds)} | Val: {len(self.val_ds)}")
        if stage == "test":
            self.test_ds = SmallLPRLineCTCDataset(self.args, "test")
        if stage == "predict":
            self.predict_ds = SmallLPRLineCTCDataset(self.args, "predict")

    def _loader(self, dataset: Dataset, shuffle: bool) -> DataLoader:
        num_workers = getattr(self.args, "num_workers", 4)
        return DataLoader(
            dataset,
            batch_size=self.args.batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=collate_fn_line_ctc,
        )

    def train_dataloader(self) -> DataLoader:
        return self._loader(self.train_ds, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return self._loader(self.val_ds, shuffle=False)

    def test_dataloader(self) -> DataLoader:
        return self._loader(self.test_ds, shuffle=False)

    def predict_dataloader(self) -> DataLoader:
        return self._loader(self.predict_ds, shuffle=False)


__all__ = [
    "LAYOUT_IGNORE_INDEX",
    "LAYOUT_ONE_LINE",
    "LAYOUT_TWO_LINE",
    "LineCtcLabel",
    "SmallLPRLineCTCDataModule",
    "SmallLPRLineCTCDataset",
    "_load_excluded_paths",
    "collate_fn_line_ctc",
    "parse_line_ctc_label",
]
