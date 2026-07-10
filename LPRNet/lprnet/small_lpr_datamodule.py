import os
import random
import numpy as np
import cv2
from torch.utils.data import Dataset, DataLoader
from imutils import paths
import lightning as L
import torch
import albumentations as A

from lprnet.utils import encode
from lprnet.small_lpr import smart_resize
from lprnet.trans_datamodule import _read_image_size, collate_fn


class SmallLPRDataset(Dataset):
    """
    Dataset cho SmallLPR — dùng smart_resize (aspect-preserve + zero-pad) thay vì
    cv2.resize cứng để tránh distort biển số (đặc biệt biển 2 dòng).
    """

    def __init__(self, args, stage):
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
        self.img_paths = []
        skipped = 0
        for p in all_paths:
            w, h = _read_image_size(p)
            if w is not None and (w < min_w or h < min_h):
                skipped += 1
            else:
                self.img_paths.append(p)
        if skipped > 0:
            print(f"[{stage}] Filtered {skipped}/{len(all_paths)} images below {min_w}×{min_h}px")

        if stage == "train":
            random.shuffle(self.img_paths)

        # img_size trong config là (W, H) → smart_resize nhận (H, W)
        self.target_hw = (args.img_size[1], args.img_size[0])

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
                    # 2. Làm mờ & Nhiễu: Mô phỏng xe chạy nhanh (MotionBlur) hoặc camera noise
                    A.OneOf(
                        [
                            A.MotionBlur(blur_limit=5),
                            A.GaussianBlur(blur_limit=5),
                            A.GaussNoise(var_limit=(10.0, 40.0)),
                        ],
                        p=0.4,
                    ),
                    # 3. Ánh sáng & Màu sắc: Mô phỏng thời tiết nắng gắt, bóng râm, buổi tối
                    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
                    A.HueSaturationValue(
                        hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=20, p=0.3
                    ),
                    # # 4. Che khuất: Tạo các đốm đen nhỏ mô phỏng bùn đất, ốc vít bám trên chữ số
                    # A.CoarseDropout(
                    #     max_holes=4, max_height=4, max_width=4, min_holes=1, fill_value=0, p=0.3
                    # ),
                ]
            )
        else:
            self.transform = None

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        filename = self.img_paths[index]
        img = cv2.imread(filename)

        if self.transform is not None:
            augmented = self.transform(image=img)
            img = augmented["image"]

        img = smart_resize(img, target_hw=self.target_hw)
        img = self._normalize(img)

        basename = os.path.basename(filename)
        imgname = os.path.splitext(basename)[0].split("#")[0].upper()
        label = encode(imgname, self.args.chars)
        label = [1] + label + [2]  # <SOS> ... <EOS>

        return img, label, len(label)

    def _normalize(self, img):
        """BGR uint8 → normalized float32 CHW in range ~[-1, 1]."""
        img = img.astype(np.float32)
        img = (img - 127.5) * 0.0078125
        img = np.transpose(img, (2, 0, 1))
        return img


class SmallLPRDataModule(L.LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.args = args

    def setup(self, stage: str):
        if stage == "fit":
            self.train = SmallLPRDataset(self.args, "train")
            self.val = SmallLPRDataset(self.args, "valid")
            print(f"train: {len(self.train)}  |  val: {len(self.val)}")
        if stage == "test":
            self.test = SmallLPRDataset(self.args, "test")
        if stage == "predict":
            self.predict = SmallLPRDataset(self.args, "predict")

    def train_dataloader(self):
        return DataLoader(
            self.train,
            batch_size=self.args.batch_size,
            shuffle=True,
            num_workers=4,
            collate_fn=collate_fn,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val,
            batch_size=self.args.batch_size,
            shuffle=False,
            num_workers=4,
            collate_fn=collate_fn,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test,
            batch_size=self.args.batch_size,
            shuffle=False,
            num_workers=4,
            collate_fn=collate_fn,
        )

    def predict_dataloader(self):
        return DataLoader(
            self.predict,
            batch_size=self.args.batch_size,
            shuffle=False,
            num_workers=4,
            collate_fn=collate_fn,
        )
