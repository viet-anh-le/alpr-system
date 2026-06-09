"""
tracking/train/sampler.py — PK sampler for batch-hard triplet mining.

Produces batches of exactly P × K samples: P identities with K images each.
This guarantees that every batch contains the positive pairs required for
online hard mining, unlike a standard random sampler.
"""
from __future__ import annotations

import random
from collections import defaultdict

from torch.utils.data import Sampler


class PKSampler(Sampler):
    """Sample P identities × K images per iteration.

    Identities with fewer than 2 images are excluded (no valid positive pair).
    Each call to __iter__ yields (n_ids // P) × P × K individual indices;
    the DataLoader groups consecutive P×K indices into one batch.
    """

    def __init__(self, labels: list[int], P: int, K: int) -> None:
        if P < 2:
            raise ValueError(f"P must be >= 2, got {P}")
        if K < 2:
            raise ValueError(f"K must be >= 2 for positive pairs, got {K}")
        self.P = P
        self.K = K

        groups: dict[int, list[int]] = defaultdict(list)
        for idx, label in enumerate(labels):
            groups[label].append(idx)

        self._groups: dict[int, list[int]] = {
            pid: idxs for pid, idxs in groups.items() if len(idxs) >= 2
        }
        self._ids: list[int] = sorted(self._groups)

        if len(self._ids) < P:
            raise ValueError(
                f"Need at least P={P} identities with >=2 images, found {len(self._ids)}"
            )

    def __len__(self) -> int:
        return (len(self._ids) // self.P) * self.P * self.K

    def __iter__(self):
        ids = self._ids.copy()
        random.shuffle(ids)
        for i in range(0, len(ids) - self.P + 1, self.P):
            batch_ids = ids[i : i + self.P]
            indices: list[int] = []
            for pid in batch_ids:
                pool = self._groups[pid]
                if len(pool) < self.K:
                    chosen = random.choices(pool, k=self.K)
                else:
                    chosen = random.sample(pool, self.K)
                indices.extend(chosen)
            yield from indices
