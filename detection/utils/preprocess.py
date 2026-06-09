from __future__ import annotations

import numpy as np


def resize_with_padding(image: np.ndarray, target_size: tuple[int, int]) -> np.ndarray:
    raise NotImplementedError


def normalize(image: np.ndarray) -> np.ndarray:
    raise NotImplementedError
