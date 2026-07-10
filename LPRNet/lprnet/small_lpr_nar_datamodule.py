"""
SmallLPR-NAR Datamodule

Điểm khác biệt so với SmallLPRCTC:
  - Label được pad cố định đến max_len (không phải 1D concat).
    NAR có vị trí tường minh: position i → ký tự thứ i của biển số.
  - Index 0 = <pad> — CrossEntropy sẽ ignore vị trí này.
  - Không cần input_lengths / target_lengths (không có CTCLoss).
  - collate_fn_nar trả về (images, targets_padded, target_lengths).
    target_lengths để tính accuracy (bỏ padding khi so sánh chuỗi).

Lưu ý về thứ tự đọc biển 2 dòng:
  Tên file phải theo format dòng-trên_dòng-dưới, ví dụ "51A[SEP]12345".
  Với NAR, model học position-to-char mapping rõ ràng:
    pos 0 → '5', pos 1 → '1', pos 2 → 'A', pos 3 → '[SEP]', ...
"""

from __future__ import annotations

import os
import random
from typing import List, Tuple

import albumentations as A
import cv2
import lightning as L
import numpy as np
import torch
from imutils import paths
from torch.utils.data import DataLoader, Dataset

from lprnet.small_lpr import smart_resize
from lprnet.trans_datamodule import _read_image_size
from lprnet.utils import encode


PAD_IDX: int = 0  # index 0 = <pad>, khớp với CrossEntropy ignore_index


# =============================================================================
# collate_fn — dùng class thay lambda (lambda kông pickle được với num_workers>0)
# =============================================================================


class NARCollateFn:
    """Callable class cho collate_fn, picklable với multiprocessing."""

    def __init__(self, max_len: int):
        self.max_len = max_len

    def __call__(
        self,
        batch: List[Tuple[np.ndarray, List[int], int]],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return collate_fn_nar(batch, self.max_len)


def collate_fn_nar(
    batch: List[Tuple[np.ndarray, List[int], int]],
    max_len: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Returns:
        images:          (B, C, H, W) float32
        targets_padded:  (B, max_len) long — pad bằng PAD_IDX (0)
        target_lengths:  (B,) long — số ký tự thật (không kể pad)
    """
    imgs, padded_targets, tgt_lengths = [], [], []

    for img, label, length in batch:
        imgs.append(torch.from_numpy(img))

        # Cắt nếu label dài hơn max_len (hiếm nhưng cần xử lý)
        label = label[:max_len]
        pad_len = max_len - len(label)
        padded = label + [PAD_IDX] * pad_len

        padded_targets.append(torch.tensor(padded, dtype=torch.long))
        tgt_lengths.append(len(label))

    return (
        torch.stack(imgs, dim=0),
        torch.stack(padded_targets, dim=0),
        torch.tensor(tgt_lengths, dtype=torch.long),
    )


# =============================================================================
# Dataset
# =============================================================================


class SmallLPRNARDataset(Dataset):
    """
    Dataset cho SmallLPR-NAR.

    - Ảnh được smart_resize về (H=48, W=96) và normalize giống SmallLPR.
    - Label = encode(imgname, chars) — index bắt đầu từ 1 (index 0 = pad).
    - KHÔNG thêm SOS/EOS.
    - Trả về (img, label, len(label)) để collate_fn_nar dễ xử lý.
    """

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

        min_w = getattr(args, "min_img_width", 20)
        min_h = getattr(args, "min_img_height", 8)
        self.img_paths: List[str] = []
        skipped = 0
        for p in all_paths:
            w, h = _read_image_size(p)
            if w is not None and (w < min_w or h < min_h):
                skipped += 1
            else:
                self.img_paths.append(p)
        if skipped > 0:
            print(f"[NAR/{stage}] Bỏ qua {skipped}/{len(all_paths)} ảnh nhỏ hơn {min_w}×{min_h}px")

        if stage == "train":
            random.shuffle(self.img_paths)

        # img_size trong config là (W, H) → smart_resize nhận (H, W)
        self.target_hw: Tuple[int, int] = (args.img_size[1], args.img_size[0])

        if stage == "train":
            self.transform = A.Compose(
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
                        hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=20, p=0.3
                    ),
                ]
            )
        else:
            self.transform = None

    def __len__(self) -> int:
        return len(self.img_paths)

    def __getitem__(self, index: int) -> Tuple[np.ndarray, List[int], int]:
        filename = self.img_paths[index]
        img = cv2.imread(filename)
        if img is None:
            img = np.zeros((self.target_hw[0], self.target_hw[1], 3), dtype=np.uint8)

        if self.transform is not None:
            img = self.transform(image=img)["image"]

        img = smart_resize(img, target_hw=self.target_hw)
        img = self._normalize(img)

        basename = os.path.basename(filename)
        imgname = os.path.splitext(basename)[0].split("#")[0].upper()
        # encode trả về index ≥ 1 (index 0 là pad, không bao giờ xuất hiện trong label thật)
        label = encode(imgname, self.args.chars)

        return img, label, len(label)

    def _normalize(self, img: np.ndarray) -> np.ndarray:
        """BGR uint8 → float32 CHW, range ≈ [-1, 1]."""
        img = img.astype(np.float32)
        img = (img - 127.5) * 0.0078125
        return np.transpose(img, (2, 0, 1))


# =============================================================================
# DataModule
# =============================================================================


class SmallLPRNARDataModule(L.LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.max_len = getattr(args, "max_len", 14)

    def setup(self, stage: str) -> None:
        if stage == "fit":
            self.train_ds = SmallLPRNARDataset(self.args, "train")
            self.val_ds = SmallLPRNARDataset(self.args, "valid")
            print(f"[NAR] Train: {len(self.train_ds)} | Val: {len(self.val_ds)}")
        if stage == "test":
            self.test_ds = SmallLPRNARDataset(self.args, "test")
        if stage == "predict":
            self.predict_ds = SmallLPRNARDataset(self.args, "predict")

    def _loader(self, dataset: Dataset, shuffle: bool) -> DataLoader:
        num_workers = getattr(self.args, "num_workers", 4)
        return DataLoader(
            dataset,
            batch_size=self.args.batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=NARCollateFn(self.max_len),  # class, picklable
        )

    def train_dataloader(self) -> DataLoader:
        return self._loader(self.train_ds, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return self._loader(self.val_ds, shuffle=False)

    def test_dataloader(self) -> DataLoader:
        return self._loader(self.test_ds, shuffle=False)

    def predict_dataloader(self) -> DataLoader:
        return self._loader(self.predict_ds, shuffle=False)
