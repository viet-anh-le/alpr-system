"""
tracking/train/evaluate_reid.py — Re-ID evaluation metrics.

Computes CMC (Cumulative Matching Characteristic) and mAP (mean Average
Precision) on a query / gallery split. During training, the val split is
automatically divided into query (1 image per identity) and gallery (rest)
so we get a proxy metric without needing a dedicated query set.

All embeddings are assumed to be L2-normalised, so cosine similarity is used
(equivalent to negative L2 distance for normalised vectors and more numerically
stable).
"""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader


def extract_embeddings(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract L2-normalised embeddings for every sample in the loader.

    The model must return a single tensor (embeddings) when called without
    return_logits=True.

    Returns:
        features: (N, D) float32 array
        labels:   (N,)   int64 array
    """
    model.eval()
    all_feats: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []
    with torch.no_grad():
        for imgs, ids in loader:
            imgs = imgs.to(device)
            embs: torch.Tensor = model(imgs)  # (B, D)
            all_feats.append(embs.cpu().numpy())
            all_labels.append(np.asarray(ids))
    return np.concatenate(all_feats, axis=0), np.concatenate(all_labels, axis=0)


def _build_val_query_gallery(
    features: np.ndarray,
    labels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split val features into query (1 per ID) and gallery (rest).

    Identities with only 1 image are skipped — they cannot form a
    query/gallery pair and would produce undefined AP.

    Uses index arrays instead of list accumulation to avoid tripling peak
    memory when the feature matrix is large.

    Returns:
        (query_feats, query_labels, gallery_feats, gallery_labels)
    """
    query_idx: list[int] = []
    gallery_idx: list[int] = []

    for pid in np.unique(labels):
        idxs = np.where(labels == pid)[0]
        if len(idxs) < 2:
            continue
        query_idx.append(int(idxs[0]))
        gallery_idx.extend(idxs[1:].tolist())

    if not gallery_idx:
        D = features.shape[1]
        return (
            np.empty((0, D), dtype=np.float32),
            np.array([], dtype=np.int64),
            np.empty((0, D), dtype=np.float32),
            np.array([], dtype=np.int64),
        )

    qi = np.array(query_idx, dtype=np.int64)
    gi = np.array(gallery_idx, dtype=np.int64)
    return (
        features[qi].astype(np.float32),
        labels[qi].astype(np.int64),
        features[gi].astype(np.float32),
        labels[gi].astype(np.int64),
    )


def compute_cmc(
    query_feats: np.ndarray,
    gallery_feats: np.ndarray,
    query_labels: np.ndarray,
    gallery_labels: np.ndarray,
    topk: int = 10,
) -> tuple[dict[str, float], float]:
    """Compute CMC rank accuracies and mean Average Precision.

    Uses cosine similarity (features must be L2-normalised).

    Args:
        query_feats:   (Q, D) float32
        gallery_feats: (G, D) float32
        query_labels:  (Q,)   int
        gallery_labels:(G,)   int
        topk:          maximum rank to compute

    Returns:
        cmc:  {"rank_1": float, ..., "rank_{topk}": float}
        mAP:  float in [0, 1]
    """
    empty = {f"rank_{k}": 0.0 for k in range(1, topk + 1)}

    if gallery_feats.shape[0] == 0 or query_feats.shape[0] == 0:
        return empty, 0.0

    # Cosine similarity → convert to distance (lower = more similar)
    sim = query_feats @ gallery_feats.T  # (Q, G)
    dist = 1.0 - sim                     # (Q, G)

    num_query = query_feats.shape[0]
    correct = np.zeros(topk, dtype=np.float64)
    all_ap: list[float] = []

    for q in range(num_query):
        ranked_idx = np.argsort(dist[q])
        matched = gallery_labels[ranked_idx] == query_labels[q]

        # CMC: rank-k = 1 if first correct match is within top-k
        first = int(np.argmax(matched)) if matched.any() else topk
        for k in range(first, topk):
            correct[k] += 1.0

        # Average precision
        n_relevant = int(matched.sum())
        if n_relevant == 0:
            continue
        hits = 0
        ap = 0.0
        for r, m in enumerate(matched):
            if m:
                hits += 1
                ap += hits / (r + 1)
        all_ap.append(ap / n_relevant)

    cmc = {f"rank_{k + 1}": float(correct[k] / num_query) for k in range(topk)}
    mAP = float(np.mean(all_ap)) if all_ap else 0.0
    return cmc, mAP


def evaluate_reid_split(
    model: torch.nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    topk: int = 10,
) -> tuple[dict[str, float], float]:
    """End-to-end evaluation on a flat split (auto-builds query/gallery).

    Extracts embeddings from val_loader, splits 1 image per identity as query
    and the rest as gallery, then computes CMC + mAP.

    Returns:
        (cmc, mAP) — same as compute_cmc
    """
    features, labels = extract_embeddings(model, val_loader, device)
    qf, ql, gf, gl = _build_val_query_gallery(features, labels)
    return compute_cmc(qf, gf, ql, gl, topk=topk)


def evaluate_reid_query_gallery(
    model: torch.nn.Module,
    query_loader: DataLoader,
    gallery_loader: DataLoader,
    device: torch.device,
    topk: int = 10,
) -> tuple[dict[str, float], float]:
    """Evaluation using a pre-split query and gallery.

    Used for test splits that already provide separate query/ and gallery/
    subdirectories (standard ReID benchmark format).  Labels must be consistent
    between the two loaders (same vehicle → same integer label).

    Returns:
        (cmc, mAP) — same as compute_cmc
    """
    qf, ql = extract_embeddings(model, query_loader, device)
    gf, gl = extract_embeddings(model, gallery_loader, device)
    return compute_cmc(qf, gf, ql, gl, topk=topk)
