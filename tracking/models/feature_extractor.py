"""
tracking/models/feature_extractor.py — Inference wrapper for vehicle Re-ID.

Accepts raw BGR numpy crops (OpenCV format) and returns L2-normalised
embeddings. Used by the tracking algorithms (DeepSORT / BoT-SORT) to produce
appearance descriptors for association.
"""
from __future__ import annotations

import numpy as np
import torch
from torchvision import transforms

_INFERENCE_TRANSFORM = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((256, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


class FeatureExtractor:
    """Wraps a Re-ID model for batch feature extraction from raw BGR crops.

    Args:
        model:  a VehicleReIDNet (or any nn.Module that returns L2-normalised
                embeddings when called without extra arguments)
        device: "cuda" or "cpu"
    """

    def __init__(self, model: torch.nn.Module, device: str = "cpu") -> None:
        self.model = model.to(device)
        self.model.eval()
        self._device = device
        # Resolve once at construction — avoids a full forward pass on every
        # empty-batch call, which is common in tracking (no detections in frame).
        dummy = torch.zeros(1, 3, 256, 128, device=device)
        with torch.no_grad():
            self._embedding_dim = int(self.model(dummy).shape[1])

    @property
    def embedding_dim(self) -> int:
        return self._embedding_dim

    @torch.no_grad()
    def __call__(self, crops: list[np.ndarray]) -> np.ndarray:
        """Extract L2-normalised embeddings from BGR uint8 crops.

        Args:
            crops: list of (H, W, 3) BGR uint8 numpy arrays (OpenCV format).
                   Can be any spatial size — resized to 256×128 internally.
        Returns:
            (N, D) float32 array of L2-normalised embeddings.
            Returns empty (0, D) array when crops is empty.
        """
        if not crops:
            return np.empty((0, self.embedding_dim), dtype=np.float32)

        tensors = [
            _INFERENCE_TRANSFORM(crop[:, :, ::-1].copy())  # BGR → RGB
            for crop in crops
        ]
        batch = torch.stack(tensors).to(self._device)
        embs: torch.Tensor = self.model(batch)  # (N, D), L2-normalised
        return embs.cpu().numpy().astype(np.float32)
