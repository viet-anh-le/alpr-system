"""
tracking/train/dataloader.py — Dataset for vehicle Re-ID training.

Directory layout expected under root/{train,val,test}/:
  <split>/
    <vehicle_id>/          # one folder per identity
      frame_0001.jpg
      frame_0002.jpg
      ...

IDs are integer-encoded from sorted folder names per split.
Augmentation: random crop, colour jitter, horizontal flip, random erasing.
"""
from __future__ import annotations

import logging
import random
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

logger = logging.getLogger(__name__)

_CROP_SIZE = (256, 128)    # (H, W) — standard Re-ID crop

_TRAIN_TRANSFORM = transforms.Compose([
    transforms.Resize((_CROP_SIZE[0] + 16, _CROP_SIZE[1] + 8)),
    transforms.RandomCrop(_CROP_SIZE),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    transforms.RandomErasing(p=0.5, scale=(0.02, 0.25)),
])

_VAL_TRANSFORM = transforms.Compose([
    transforms.Resize(_CROP_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


class VehicleReIDDataset(Dataset):
    """
    Flat-indexed Re-ID dataset.

    Each __getitem__ returns (image_tensor, class_id, path_str) so the
    training loop can build triplet batches using class_id.
    """

    def __init__(self, root: Path, split: str = "train") -> None:
        split_dir = root / split
        if not split_dir.exists():
            raise FileNotFoundError(f"Split directory not found: {split_dir}")

        self.transform = _TRAIN_TRANSFORM if split == "train" else _VAL_TRANSFORM
        self.samples: list[tuple[Path, int]] = []

        id_dirs = sorted(p for p in split_dir.iterdir() if p.is_dir())
        if not id_dirs:
            raise ValueError(f"No identity sub-directories found in {split_dir}")

        for class_id, id_dir in enumerate(id_dirs):
            imgs = sorted(id_dir.glob("*.jpg")) + sorted(id_dir.glob("*.png"))
            for img_path in imgs:
                self.samples.append((img_path, class_id))

        self.num_ids = len(id_dirs)
        logger.info("ReID %s: %d images, %d identities", split, len(self.samples), self.num_ids)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        path, class_id = self.samples[idx]
        img = Image.open(path).convert("RGB")
        return self.transform(img), class_id


class TripletReIDDataset(Dataset):
    """
    On-the-fly triplet dataset.

    Each __getitem__ returns (anchor, positive, negative, id) where anchor and
    positive are different crops of the same vehicle identity and negative is a
    crop from a different identity.
    """

    def __init__(self, root: Path, split: str = "train") -> None:
        split_dir = root / split
        if not split_dir.exists():
            raise FileNotFoundError(f"Split directory not found: {split_dir}")

        self.transform = _TRAIN_TRANSFORM if split == "train" else _VAL_TRANSFORM

        id_dirs = sorted(p for p in split_dir.iterdir() if p.is_dir())
        self._by_id: list[list[Path]] = []
        for id_dir in id_dirs:
            imgs = sorted(id_dir.glob("*.jpg")) + sorted(id_dir.glob("*.png"))
            if len(imgs) >= 2:
                self._by_id.append(imgs)

        if len(self._by_id) < 2:
            raise ValueError(f"Need at least 2 identities with ≥2 images each in {split_dir}")

        self.num_ids = len(self._by_id)
        total = sum(len(v) for v in self._by_id)
        logger.info("TripletReID %s: %d images across %d identities", split, total, self.num_ids)

    def __len__(self) -> int:
        return sum(len(v) for v in self._by_id)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        # Pick anchor identity
        anc_id = idx % self.num_ids
        anc_imgs = self._by_id[anc_id]
        a_path, p_path = random.sample(anc_imgs, 2) if len(anc_imgs) >= 2 else (anc_imgs[0], anc_imgs[0])

        neg_id = random.choice([i for i in range(self.num_ids) if i != anc_id])
        n_path = random.choice(self._by_id[neg_id])

        def _load(p: Path) -> torch.Tensor:
            return self.transform(Image.open(p).convert("RGB"))

        return _load(a_path), _load(p_path), _load(n_path), anc_id


class _SamplesDataset(Dataset):
    """Minimal dataset wrapping a pre-built (Path, label) list with a transform."""

    def __init__(self, samples: list[tuple[Path, int]], transform) -> None:
        self.samples = samples
        self.transform = transform
        self.num_ids = len({lbl for _, lbl in samples})

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        path, class_id = self.samples[idx]
        return self.transform(Image.open(path).convert("RGB")), class_id


def build_reid_loader(
    root: Path,
    split: str,
    batch_size: int,
    num_workers: int = 4,
    triplet: bool = True,
) -> DataLoader:
    ds: Dataset
    if triplet and split == "train":
        ds = TripletReIDDataset(root, split)
    else:
        ds = VehicleReIDDataset(root, split)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=(split == "train"),
        num_workers=num_workers,
        pin_memory=True,
        drop_last=(split == "train"),
    )


def build_query_gallery_loaders(
    root: Path,
    split: str,
    batch_size: int,
    num_workers: int = 4,
) -> tuple[DataLoader, DataLoader]:
    """Build query and gallery DataLoaders from a split that has query/ and gallery/ subdirs.

    Both sets share a consistent label encoding derived from identity folder names,
    so a vehicle appearing in both query and gallery receives the same integer label.

    Args:
        root:        dataset root (e.g. data/datasets/tracking)
        split:       split directory name (e.g. "test")
        batch_size:  batch size for both loaders
        num_workers: DataLoader worker count

    Returns:
        query_loader, gallery_loader
    """
    split_dir = root / split

    # Collect all identity folder names across both subsets for a shared label map
    all_id_names: set[str] = set()
    for sub in ("query", "gallery"):
        sub_dir = split_dir / sub
        if sub_dir.exists():
            for d in sub_dir.iterdir():
                if d.is_dir():
                    all_id_names.add(d.name)

    if not all_id_names:
        raise ValueError(f"No identity directories found under {split_dir}/query or {split_dir}/gallery")

    id_map: dict[str, int] = {name: i for i, name in enumerate(sorted(all_id_names))}

    def _load_sub(sub: str) -> list[tuple[Path, int]]:
        sub_dir = split_dir / sub
        samples: list[tuple[Path, int]] = []
        if not sub_dir.exists():
            return samples
        for id_dir in sorted(sub_dir.iterdir()):
            if not id_dir.is_dir():
                continue
            label = id_map[id_dir.name]
            for img in sorted(id_dir.glob("*.jpg")) + sorted(id_dir.glob("*.png")):
                samples.append((img, label))
        return samples

    query_samples   = _load_sub("query")
    gallery_samples = _load_sub("gallery")
    logger.info(
        "%s/query: %d images | %s/gallery: %d images",
        split, len(query_samples), split, len(gallery_samples),
    )

    def _make_loader(samples: list[tuple[Path, int]]) -> DataLoader:
        ds = _SamplesDataset(samples, _VAL_TRANSFORM)
        return DataLoader(
            ds, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True, drop_last=False,
        )

    return _make_loader(query_samples), _make_loader(gallery_samples)
