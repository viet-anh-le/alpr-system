import os
import re
import struct
import random
import torch
import numpy as np
import cv2
from torch.utils.data import Dataset, DataLoader
from imutils import paths
import lightning as L

from lprnet.utils import encode


def _read_image_size(path):
    """Read (width, height) from image header without loading pixel data."""
    try:
        with open(path, 'rb') as f:
            header = f.read(32)
        if header[:8] == b'\x89PNG\r\n\x1a\n':
            w = struct.unpack('>I', header[16:20])[0]
            h = struct.unpack('>I', header[20:24])[0]
            return w, h
        if header[:2] == b'\xff\xd8':
            with open(path, 'rb') as f:
                data = f.read()
            i = 2
            while i < len(data):
                if data[i] != 0xFF:
                    break
                marker = data[i + 1]
                if marker in (0xC0, 0xC1, 0xC2):
                    return struct.unpack('>H', data[i + 7:i + 9])[0], struct.unpack('>H', data[i + 5:i + 7])[0]
                i += 2 + struct.unpack('>H', data[i + 2:i + 4])[0]
    except Exception:
        pass
    return None, None


def resize_pad(img, size):
    base_pic = np.zeros((size[1], size[0], 3), np.uint8)
    pic1 = img
    h, w = pic1.shape[:2]
    ash = size[1] / h
    asw = size[0] / w

    if asw < ash:
        sizeas = (int(w * asw), int(h * asw))
    else:
        sizeas = (int(w * ash), int(h * ash))

    pic1 = cv2.resize(pic1, dsize=sizeas)
    base_pic[
        int(size[1] / 2 - sizeas[1] / 2) : int(size[1] / 2 + sizeas[1] / 2),
        int(size[0] / 2 - sizeas[0] / 2) : int(size[0] / 2 + sizeas[0] / 2),
        :,
    ] = pic1

    return base_pic


def collate_fn(batch):
    imgs = []
    labels = []
    lengths = []

    max_seq_len = max(len(label) for _, label, _ in batch)

    for img, label, length in batch:
        imgs.append(torch.from_numpy(img))

        # Original label already has SOS and EOS (added in __getitem__)
        # Ensure label fits within max_seq_len
        effective_label = label[:max_seq_len]

        # Pad with PAD_IDX (0)
        padded_label = np.zeros(max_seq_len, dtype=np.int64)
        padded_label[: len(effective_label)] = effective_label

        labels.append(torch.from_numpy(padded_label))
        lengths.append(len(effective_label))

    return (torch.stack(imgs, 0), torch.stack(labels, 0), lengths)


class LPRNetDataset(Dataset):
    def __init__(self, args, stage, PreprocFun=None):
        self.args = args
        self.stage = stage
        self.img_paths = []
        self.img_size = self.args.img_size

        if stage == "train":
            self.img_dir = self.args.train_dir
        elif stage == "valid":
            self.img_dir = self.args.valid_dir
        elif stage == "test":
            self.img_dir = self.args.test_dir
        elif stage == "predict":
            self.img_dir = self.args.test_dir
        else:
            assert f"No Such Stage. Your input -> {self.stage}"

        all_paths = list(paths.list_images(self.img_dir))

        # Filter images that are too small to contain readable plate text.
        # Minimum 60×20px: below this, upsampling to 224×224 produces noise, not signal.
        min_w = getattr(args, 'min_img_width', 60)
        min_h = getattr(args, 'min_img_height', 20)
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

        if PreprocFun is not None:
            self.PreprocFun = PreprocFun
        else:
            self.PreprocFun = self.transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        filename = self.img_paths[index]
        Image = cv2.imread(filename)
        height, width, _ = Image.shape
        if height != self.img_size[1] or width != self.img_size[0]:
            Image = cv2.resize(Image, self.img_size, interpolation=cv2.INTER_CUBIC)
        Image = self.PreprocFun(Image)

        basename = os.path.basename(filename)
        imgname, suffix = os.path.splitext(basename)
        imgname = imgname.split("#")[0]
        imgname = imgname.upper()
        label = encode(imgname, self.args.chars)

        # Add SOS and EOS tokens for Transformer
        # SOS_IDX=1, EOS_IDX=2 as defined in trans_vietnam_config.yaml
        label = [1] + label + [2]

        if label:
            # Skip checking special tokens SOS/EOS in check() or update check()
            # The check function uses self.args.chars, which now includes <PAD>, <SOS>, <EOS>
            if not self.check(label):
                assert 0, f"{imgname} <- Error label ^~^!!!"

        return Image, label, len(label)

    def transform(self, img):
        """
        ImageNet normalization cho CVNets MobileViTv3-S.
        cv2.imread đọc ảnh dưới dạng BGR -> Cần convert sang RGB.
        Normalize: pixel / 255.0 → subtract ImageNet mean → divide ImageNet std
        """
        import cv2
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype("float32")
        img = img / 255.0

        # ImageNet normalization (RGB order chuẩn của torchvision/CVNets)
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)  # R, G, B
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)   # R, G, B
        img = (img - mean) / std

        img = np.transpose(img, (2, 0, 1))  # HWC → CHW

        return img

    def check(self, label):
        # Allow special tokens <PAD>, <SOS>, <EOS> in addition to alphanumeric and symbols
        # Note: self.args.chars now contains these special tokens at the beginning
        vietnam_plate_pattern = re.compile(r"^[0-9A-Z\-\.Đ|<PAD>|<SOS>|<EOS>]+$")
        label_str = "".join([self.args.chars[c] for c in label])
        return bool(vietnam_plate_pattern.match(label_str))


class DataModule(L.LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        print("dm loaded")
        # print(self.args)

    def setup(self, stage: str):
        if stage == "fit":
            self.train = LPRNetDataset(self.args, "train")
            print("train: ", len(self.train))
            self.val = LPRNetDataset(self.args, "valid")
            print("val: ", len(self.val))

        if stage == "test":
            self.test = LPRNetDataset(self.args, "test")

        if stage == "predict":
            self.predict = LPRNetDataset(self.args, "predict")

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
