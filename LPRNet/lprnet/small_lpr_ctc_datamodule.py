"""
SmallLPR-CTC Datamodule

Điểm khác biệt so với SmallLPRDataset (bản Autoregressive):
  - __getitem__ KHÔNG thêm <SOS> / <EOS> vào label.
    Label = encode(imgname, chars) thuần túy (index 1..N, skip index 0 = blank).
  - collate_fn_ctc trả về targets dưới dạng 1D tensor nối liên tiếp —
    format chuẩn của nn.CTCLoss khi không padding.
  - input_lengths cố định = 72 (số time steps của backbone SmallLPR-CTC).
  - target_lengths = số ký tự thật của mỗi biển số trong batch.
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

from lprnet.small_lpr_ctc import _T_STEPS
from lprnet.small_lpr import smart_resize
from lprnet.trans_datamodule import _read_image_size
from lprnet.utils import encode


# =============================================================================
# collate_fn cho CTC — trả về targets 1D
# =============================================================================


def collate_fn_ctc(
    batch: List[Tuple[np.ndarray, List[int]]],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Returns:
        images:         (B, 3, H, W) float32
        targets_1d:     (sum_of_label_lengths,) long — tất cả labels nối liền
        input_lengths:  (B,) long — đều bằng _T_STEPS = 72
        target_lengths: (B,) long — số ký tự thật mỗi mẫu
    """
    imgs, all_targets, tgt_lengths = [], [], []
    for img, label in batch:
        imgs.append(torch.from_numpy(img))
        all_targets.extend(label)
        tgt_lengths.append(len(label))

    images = torch.stack(imgs, dim=0)
    targets_1d = torch.tensor(all_targets, dtype=torch.long)
    input_lengths = torch.full((len(batch),), _T_STEPS, dtype=torch.long)
    target_lengths = torch.tensor(tgt_lengths, dtype=torch.long)
    return images, targets_1d, input_lengths, target_lengths


# =============================================================================
# Dataset
# =============================================================================


class SmallLPRCTCDataset(Dataset):
    """
    Dataset cho SmallLPR-CTC.

    - Ảnh được smart_resize về (H=48, W=96), normalize giống bản AR.
    - Label KHÔNG có SOS/EOS. Index 0 là <blank> nên encode sẽ trả về index ≥ 1.
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
            print(f"[CTC/{stage}] Bỏ qua {skipped}/{len(all_paths)} ảnh nhỏ hơn {min_w}×{min_h}px")

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

    def __getitem__(self, index: int) -> Tuple[np.ndarray, List[int]]:
        filename = self.img_paths[index]
        img = cv2.imread(filename)
        if img is None:
            # Trường hợp file lỗi — trả về ảnh đen + label trống (xử lý sau trong training)
            img = np.zeros((self.target_hw[0], self.target_hw[1], 3), dtype=np.uint8)

        if self.transform is not None:
            img = self.transform(image=img)["image"]

        img = smart_resize(img, target_hw=self.target_hw)
        img = self._normalize(img)

        basename = os.path.basename(filename)
        imgname = os.path.splitext(basename)[0].split("#")[0].upper()
        # encode trả về list index ≥ 1 (index 0 là blank, không dùng làm ký tự)
        label = encode(imgname, self.args.chars)

        return img, label

    def _normalize(self, img: np.ndarray) -> np.ndarray:
        """BGR uint8 → float32 CHW, range ≈ [-1, 1]."""
        img = img.astype(np.float32)
        img = (img - 127.5) * 0.0078125
        return np.transpose(img, (2, 0, 1))


# =============================================================================
# DataModule
# =============================================================================


class SmallLPRCTCDataModule(L.LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.args = args

    def setup(self, stage: str) -> None:
        if stage == "fit":
            self.train_ds = SmallLPRCTCDataset(self.args, "train")
            self.val_ds = SmallLPRCTCDataset(self.args, "valid")
            print(f"[CTC] Train: {len(self.train_ds)} | Val: {len(self.val_ds)}")
        if stage == "test":
            self.test_ds = SmallLPRCTCDataset(self.args, "test")
        if stage == "predict":
            self.predict_ds = SmallLPRCTCDataset(self.args, "predict")

    def _loader(self, dataset: Dataset, shuffle: bool) -> DataLoader:
        num_workers = getattr(self.args, "num_workers", 4)
        return DataLoader(
            dataset,
            batch_size=self.args.batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=collate_fn_ctc,
        )

    def train_dataloader(self) -> DataLoader:
        return self._loader(self.train_ds, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return self._loader(self.val_ds, shuffle=False)

    def test_dataloader(self) -> DataLoader:
        return self._loader(self.test_ds, shuffle=False)

    def predict_dataloader(self) -> DataLoader:
        return self._loader(self.predict_ds, shuffle=False)
