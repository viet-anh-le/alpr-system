from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image


def test_majority_vote() -> None:
    from ocr.postprocess.voting import majority_vote

    result = majority_vote(["51A12345", "51A12345", "51A12300"])
    assert result == "51A12345"


# ── PKSampler ─────────────────────────────────────────────────────────────────


def test_pk_sampler_total_length() -> None:
    from tracking.train.sampler import PKSampler

    labels = [i for i in range(10) for _ in range(5)]  # 10 IDs × 5 imgs
    sampler = PKSampler(labels, P=2, K=3)
    assert len(sampler) == 30  # (10 // 2) * 2 * 3
    assert len(list(iter(sampler))) == 30


def test_pk_sampler_excludes_singletons() -> None:
    from tracking.train.sampler import PKSampler

    labels = [0, 1, 1, 2, 2, 2]  # ID 0 → 1 image (excluded)
    sampler = PKSampler(labels, P=2, K=2)
    assert 0 not in sampler._ids
    assert set(sampler._ids) == {1, 2}


def test_pk_sampler_indices_are_valid() -> None:
    from tracking.train.sampler import PKSampler

    labels = [i for i in range(8) for _ in range(4)]
    sampler = PKSampler(labels, P=2, K=2)
    for idx in iter(sampler):
        assert 0 <= idx < len(labels)


def test_pk_sampler_rejects_small_p() -> None:
    from tracking.train.sampler import PKSampler

    with pytest.raises(ValueError, match="P must be >= 2"):
        PKSampler([0, 0, 1, 1], P=1, K=2)


def test_pk_sampler_rejects_small_k() -> None:
    from tracking.train.sampler import PKSampler

    with pytest.raises(ValueError, match="K must be >= 2"):
        PKSampler([0, 0, 1, 1], P=2, K=1)


# ── pairwise distance ─────────────────────────────────────────────────────────


def test_pairwise_squared_distance_self_is_zero() -> None:
    from tracking.train.loss import pairwise_squared_distance

    embs = torch.randn(5, 16)
    dist = pairwise_squared_distance(embs)
    assert dist.diagonal().abs().max().item() < 1e-5


def test_pairwise_squared_distance_symmetry() -> None:
    from tracking.train.loss import pairwise_squared_distance

    embs = torch.randn(4, 8)
    dist = pairwise_squared_distance(embs)
    assert torch.allclose(dist, dist.T, atol=1e-5)


def test_pairwise_squared_distance_nonnegative() -> None:
    from tracking.train.loss import pairwise_squared_distance

    embs = torch.randn(6, 32)
    dist = pairwise_squared_distance(embs)
    assert (dist >= 0).all()


# ── BatchHardTripletLoss ──────────────────────────────────────────────────────


def test_batch_hard_triplet_zero_when_trivially_separated() -> None:
    from tracking.train.loss import BatchHardTripletLoss

    loss_fn = BatchHardTripletLoss(margin=0.3)
    embs = torch.tensor([
        [1.0, 0.0, 0.0],
        [0.9, 0.1, 0.0],
        [-1.0, 0.0, 0.0],
        [-0.9, -0.1, 0.0],
    ])
    labels = torch.tensor([0, 0, 1, 1])
    loss, _ = loss_fn(embs, labels)
    assert loss.item() == pytest.approx(0.0, abs=1e-5)


def test_batch_hard_triplet_equals_margin_when_collapsed() -> None:
    from tracking.train.loss import BatchHardTripletLoss

    loss_fn = BatchHardTripletLoss(margin=0.3)
    embs = torch.zeros(4, 8)
    labels = torch.tensor([0, 0, 1, 1])
    loss, frac_active = loss_fn(embs, labels)
    # dist_ap = dist_an = 0 → loss = relu(0 - 0 + 0.3) = 0.3
    assert loss.item() == pytest.approx(0.3, abs=1e-5)
    assert frac_active.item() == pytest.approx(1.0)


def test_batch_hard_triplet_frac_active_in_range() -> None:
    from tracking.train.loss import BatchHardTripletLoss

    loss_fn = BatchHardTripletLoss(margin=0.3)
    embs = torch.randn(8, 16)
    labels = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])
    _, frac = loss_fn(embs, labels)
    assert 0.0 <= frac.item() <= 1.0


# ── CombinedReIDLoss ──────────────────────────────────────────────────────────


def test_combined_reid_loss_returns_all_stats() -> None:
    from tracking.train.loss import CombinedReIDLoss

    criterion = CombinedReIDLoss(margin=0.3, ce_weight=0.5)
    embs = torch.randn(8, 128)
    logits = torch.randn(8, 4)
    labels = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])
    loss, stats = criterion(embs, logits, labels)
    assert loss.item() >= 0.0
    assert {"loss_triplet", "loss_ce", "frac_active"} <= stats.keys()


def test_combined_reid_loss_is_differentiable() -> None:
    from tracking.train.loss import CombinedReIDLoss

    criterion = CombinedReIDLoss()
    embs = torch.randn(8, 128, requires_grad=True)
    logits = torch.randn(8, 4)
    labels = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])
    loss, _ = criterion(embs, logits, labels)
    loss.backward()
    assert embs.grad is not None


# ── compute_cmc ───────────────────────────────────────────────────────────────


def test_compute_cmc_perfect_retrieval() -> None:
    from tracking.train.evaluate_reid import compute_cmc

    q = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]], dtype=np.float32)
    g = np.array([
        [1.0, 0.0], [0.9, 0.1],
        [0.0, 1.0], [0.1, 0.9],
        [-1.0, 0.0], [-0.9, 0.0],
    ], dtype=np.float32)
    cmc, mAP = compute_cmc(q, g, np.array([0, 1, 2]), np.array([0, 0, 1, 1, 2, 2]), topk=5)
    assert cmc["rank_1"] == pytest.approx(1.0)
    assert mAP == pytest.approx(1.0)


def test_compute_cmc_worst_retrieval() -> None:
    from tracking.train.evaluate_reid import compute_cmc

    q = np.array([[1.0, 0.0]], dtype=np.float32)
    g = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]], dtype=np.float32)
    # Correct match (ID 0) is last gallery item (least similar)
    cmc, _ = compute_cmc(q, g, np.array([0]), np.array([1, 1, 0]), topk=3)
    assert cmc["rank_1"] == pytest.approx(0.0)
    assert cmc["rank_3"] == pytest.approx(1.0)


def test_compute_cmc_empty_gallery() -> None:
    from tracking.train.evaluate_reid import compute_cmc

    q = np.array([[1.0, 0.0]], dtype=np.float32)
    g = np.empty((0, 2), dtype=np.float32)
    cmc, mAP = compute_cmc(q, g, np.array([0]), np.array([]), topk=5)
    assert all(v == pytest.approx(0.0) for v in cmc.values())
    assert mAP == pytest.approx(0.0)


def test_compute_cmc_rank_monotone() -> None:
    from tracking.train.evaluate_reid import compute_cmc

    rng = np.random.default_rng(0)
    q = rng.standard_normal((10, 32)).astype(np.float32)
    g = rng.standard_normal((50, 32)).astype(np.float32)
    cmc, _ = compute_cmc(q, g, np.arange(10), np.repeat(np.arange(10), 5), topk=10)
    ranks = [cmc[f"rank_{k}"] for k in range(1, 11)]
    assert all(ranks[i] <= ranks[i + 1] for i in range(len(ranks) - 1))


# ── _build_val_query_gallery ──────────────────────────────────────────────────


def test_build_val_query_gallery_splits_correctly() -> None:
    from tracking.train.evaluate_reid import _build_val_query_gallery

    features = np.array([
        [1.0, 0.0], [0.9, 0.1],
        [0.0, 1.0], [0.1, 0.9],
    ], dtype=np.float32)
    labels = np.array([0, 0, 1, 1])
    qf, ql, gf, gl = _build_val_query_gallery(features, labels)
    assert qf.shape == (2, 2)
    assert gf.shape == (2, 2)
    assert set(ql.tolist()) == {0, 1}
    assert set(gl.tolist()) == {0, 1}


def test_build_val_query_gallery_skips_singletons() -> None:
    from tracking.train.evaluate_reid import _build_val_query_gallery

    features = np.array([
        [1.0, 0.0],               # ID 0 — only 1 image
        [0.0, 1.0], [0.1, 0.9],  # ID 1
    ], dtype=np.float32)
    labels = np.array([0, 1, 1])
    qf, ql, gf, gl = _build_val_query_gallery(features, labels)
    assert 0 not in ql.tolist()
    assert qf.shape[0] == 1


# ── VehicleReIDDataset ────────────────────────────────────────────────────────


def _make_dummy_reid_split(root: Path, split: str, n_ids: int, n_imgs: int) -> None:
    for pid in range(n_ids):
        pid_dir = root / split / f"{pid:03d}"
        pid_dir.mkdir(parents=True)
        for i in range(n_imgs):
            Image.new("RGB", (64, 64)).save(pid_dir / f"img_{i:03d}.jpg")


def test_vehicle_reid_dataset_loads() -> None:
    from tracking.train.dataloader import VehicleReIDDataset

    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        _make_dummy_reid_split(root, "train", n_ids=3, n_imgs=4)
        ds = VehicleReIDDataset(root, "train")
        assert len(ds) == 12
        assert ds.num_ids == 3
        img_tensor, label = ds[0]
        assert img_tensor.shape == (3, 256, 128)
        assert label in range(3)


def test_vehicle_reid_dataset_missing_split_raises() -> None:
    from tracking.train.dataloader import VehicleReIDDataset

    with tempfile.TemporaryDirectory() as tmpdir:
        with pytest.raises(FileNotFoundError):
            VehicleReIDDataset(Path(tmpdir), "nonexistent")


# ── FeatureExtractor ──────────────────────────────────────────────────────────


def test_feature_extractor_output_shape() -> None:
    from tracking.models.feature_extractor import FeatureExtractor
    from tracking.models.reid_net import VehicleReIDNet

    model = VehicleReIDNet(embedding_dim=128)
    extractor = FeatureExtractor(model, device="cpu")
    crops = [np.zeros((64, 64, 3), dtype=np.uint8) for _ in range(4)]
    out = extractor(crops)
    assert out.shape == (4, 128)


def test_feature_extractor_l2_normalised() -> None:
    from tracking.models.feature_extractor import FeatureExtractor
    from tracking.models.reid_net import VehicleReIDNet

    model = VehicleReIDNet(embedding_dim=128)
    extractor = FeatureExtractor(model, device="cpu")
    rng = np.random.default_rng(0)
    crops = [rng.integers(0, 255, (80, 60, 3), dtype=np.uint8) for _ in range(5)]
    out = extractor(crops)
    norms = np.linalg.norm(out, axis=1)
    np.testing.assert_allclose(norms, 1.0, atol=1e-5)


def test_feature_extractor_empty_input() -> None:
    from tracking.models.feature_extractor import FeatureExtractor
    from tracking.models.reid_net import VehicleReIDNet

    model = VehicleReIDNet(embedding_dim=128)
    extractor = FeatureExtractor(model, device="cpu")
    out = extractor([])
    assert out.shape == (0, 128)


def test_feature_extractor_variable_crop_sizes() -> None:
    from tracking.models.feature_extractor import FeatureExtractor
    from tracking.models.reid_net import VehicleReIDNet

    model = VehicleReIDNet(embedding_dim=128)
    extractor = FeatureExtractor(model, device="cpu")
    crops = [
        np.zeros((32, 16, 3), dtype=np.uint8),
        np.zeros((128, 64, 3), dtype=np.uint8),
        np.zeros((512, 256, 3), dtype=np.uint8),
    ]
    out = extractor(crops)
    assert out.shape == (3, 128)
