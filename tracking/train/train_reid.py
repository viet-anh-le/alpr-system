"""
tracking/train/train_reid.py — VehicleReIDNet training loop.

Strategy
--------
Loss:     BatchHardTripletLoss (online hard mining) + 0.5 × CrossEntropy
          with label smoothing. The CE head regularises the embedding space
          and speeds up early convergence.
Sampler:  PKSampler — P=16 identities × K=4 images per batch (batch_size=64).
          Ensures every batch has valid positive pairs for hard mining.
Optimiser: AdamW, linear LR warmup for first 10 epochs, then cosine decay.
Metric:   Rank-1 accuracy on val split (auto query/gallery split).
Checkpoint: saved on best Rank-1; final state dict exported for BoT-SORT.
Final eval: test split is evaluated once after training completes.

Usage
-----
  python -m tracking.train.train_reid \\
      --data  data/datasets/tracking \\
      --output weights/tracking \\
      --epochs 60

After training, BoT-SORT picks up:
  configs/tracking/botsort.yaml → reid_weights: weights/tracking/vehicle_reid.pt
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader

from tracking.models.reid_net import VehicleReIDNet
from tracking.train.dataloader import VehicleReIDDataset, build_query_gallery_loaders, build_reid_loader
from tracking.train.evaluate_reid import evaluate_reid_query_gallery, evaluate_reid_split
from tracking.train.loss import CombinedReIDLoss
from tracking.train.sampler import PKSampler

logger = logging.getLogger(__name__)


# ── argument parsing ──────────────────────────────────────────────────────────


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train VehicleReIDNet with batch-hard mining")
    p.add_argument("--data",            type=Path,  default=Path("data/datasets/tracking"),
                   help="Root of the Re-ID dataset (contains train/ val/ directories)")
    p.add_argument("--output",          type=Path,  default=Path("weights/tracking"),
                   help="Directory to save checkpoints and exported weights")
    p.add_argument("--epochs",          type=int,   default=60)
    p.add_argument("--P",               type=int,   default=16,
                   help="Number of identities per batch (PKSampler)")
    p.add_argument("--K",               type=int,   default=4,
                   help="Number of images per identity per batch (PKSampler)")
    p.add_argument("--lr",              type=float, default=3e-4)
    p.add_argument("--weight-decay",    type=float, default=1e-4)
    p.add_argument("--warmup-epochs",   type=int,   default=10,
                   help="Epochs for linear LR warmup before cosine decay")
    p.add_argument("--margin",          type=float, default=0.3,
                   help="Triplet loss margin")
    p.add_argument("--ce-weight",       type=float, default=0.5,
                   help="Weight for CE classification term")
    p.add_argument("--label-smoothing", type=float, default=0.1)
    p.add_argument("--embedding-dim",   type=int,   default=128)
    p.add_argument("--workers",         type=int,   default=4)
    p.add_argument("--device",          type=str,   default="cuda")
    p.add_argument("--resume",          type=Path,  default=None,
                   help="Resume training from a saved checkpoint (last/best .pt)")
    p.add_argument("--topk",            type=int,   default=10,
                   help="CMC top-k ranks to evaluate on val and test")
    return p.parse_args()


# ── data loaders ──────────────────────────────────────────────────────────────


def _build_train_loader(
    data: Path, P: int, K: int, num_workers: int
) -> DataLoader:
    ds = VehicleReIDDataset(data, "train")
    labels = [label for _, label in ds.samples]
    sampler = PKSampler(labels, P=P, K=K)
    return DataLoader(
        ds,
        batch_size=P * K,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )


# ── training loop ─────────────────────────────────────────────────────────────


def _run(args: argparse.Namespace) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    # ── data ──────────────────────────────────────────────────────────────────
    train_loader = _build_train_loader(args.data, args.P, args.K, args.workers)
    val_loader = build_reid_loader(
        args.data, "val",
        batch_size=args.P * args.K,
        num_workers=args.workers,
        triplet=False,
    )
    _test_dir = args.data / "test"
    _test_has_qg = (_test_dir / "query").exists() and (_test_dir / "gallery").exists()
    if _test_has_qg:
        _test_query_loader, _test_gallery_loader = build_query_gallery_loaders(
            args.data, "test", batch_size=args.P * args.K, num_workers=args.workers,
        )
        test_loader = None
    elif _test_dir.exists():
        test_loader = build_reid_loader(
            args.data, "test", batch_size=args.P * args.K, num_workers=args.workers, triplet=False,
        )
        _test_query_loader = _test_gallery_loader = None
    else:
        test_loader = _test_query_loader = _test_gallery_loader = None
        logger.warning("No test/ split found under %s — final test evaluation skipped", args.data)
    num_ids: int = train_loader.dataset.num_ids  # type: ignore[union-attr]
    logger.info("Train: %d identities", num_ids)

    # ── model ─────────────────────────────────────────────────────────────────
    model = VehicleReIDNet(embedding_dim=args.embedding_dim, num_ids=num_ids).to(device)

    # ── optimiser + scheduler ─────────────────────────────────────────────────
    criterion = CombinedReIDLoss(
        margin=args.margin,
        ce_weight=args.ce_weight,
        label_smoothing=args.label_smoothing,
    )
    optimiser = AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    warmup = LinearLR(
        optimiser, start_factor=0.1, end_factor=1.0,
        total_iters=args.warmup_epochs,
    )
    cosine = CosineAnnealingLR(
        optimiser, T_max=max(args.epochs - args.warmup_epochs, 1), eta_min=1e-6
    )
    scheduler = SequentialLR(
        optimiser, schedulers=[warmup, cosine], milestones=[args.warmup_epochs]
    )

    start_epoch = 1
    if args.resume and args.resume.exists():
        # weights_only=True prevents arbitrary code execution from pickle.
        ckpt = torch.load(args.resume, map_location=device, weights_only=True)
        model.load_state_dict(ckpt["model"])
        if "optimiser" in ckpt:
            optimiser.load_state_dict(ckpt["optimiser"])
        if "scheduler" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt.get("epoch", 0) + 1
        logger.info("Resumed from %s (epoch %d)", args.resume, start_epoch - 1)

    args.output.mkdir(parents=True, exist_ok=True)
    best_rank1 = 0.0

    for epoch in range(start_epoch, args.epochs + 1):
        # ── train ─────────────────────────────────────────────────────────────
        model.train()
        total_loss = total_tri = total_ce = total_active = 0.0
        n_batches = 0

        for imgs, ids in train_loader:
            imgs = imgs.to(device)
            ids = ids.to(device)

            embs, logits = model(imgs, return_logits=True)  # type: ignore[misc]
            loss, stats = criterion(embs, logits, ids)

            optimiser.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            optimiser.step()

            total_loss   += loss.item()
            total_tri    += stats["loss_triplet"]
            total_ce     += stats["loss_ce"]
            total_active += stats["frac_active"]
            n_batches    += 1

        scheduler.step()
        n = max(n_batches, 1)
        logger.info(
            "Epoch %3d/%d  loss=%.4f (tri=%.4f ce=%.4f)  "
            "active=%.1f%%  lr=%.2e",
            epoch, args.epochs,
            total_loss / n, total_tri / n, total_ce / n,
            total_active / n * 100,
            optimiser.param_groups[0]["lr"],
        )

        # ── val ───────────────────────────────────────────────────────────────
        model.eval()
        cmc, mAP = evaluate_reid_split(model, val_loader, device, topk=args.topk)
        logger.info(
            "        val  R1=%.4f  R5=%.4f  R10=%.4f  mAP=%.4f",
            cmc.get("rank_1", 0.0),
            cmc.get("rank_5", 0.0),
            cmc.get("rank_10", 0.0),
            mAP,
        )

        # ── checkpoint ────────────────────────────────────────────────────────
        rank1 = cmc.get("rank_1", 0.0)
        ckpt_data = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimiser": optimiser.state_dict(),
            "scheduler": scheduler.state_dict(),
            "rank1": rank1,
            "mAP": mAP,
        }

        if rank1 > best_rank1:
            best_rank1 = rank1
            best_path = args.output / "vehicle_reid_best.pt"
            torch.save(ckpt_data, best_path)
            logger.info("        New best R1=%.4f → %s", rank1, best_path)

        torch.save(ckpt_data, args.output / "vehicle_reid_last.pt")

    # ── export clean weights for BoT-SORT ─────────────────────────────────────
    best_ckpt_path = args.output / "vehicle_reid_best.pt"
    if best_ckpt_path.exists():
        best = torch.load(best_ckpt_path, map_location="cpu", weights_only=True)
        export_path = args.output / "vehicle_reid.pt"
        torch.save(best["model"], export_path)
        logger.info(
            "Exported → %s  (R1=%.4f  mAP=%.4f)",
            export_path, best["rank1"], best["mAP"],
        )

    # ── final evaluation on test split ────────────────────────────────────────
    _run_test_eval = _test_has_qg or test_loader is not None
    if _run_test_eval:
        logger.info("Evaluating best model on test split …")
        if best_ckpt_path.exists():
            best_state = torch.load(best_ckpt_path, map_location=device, weights_only=True)
            model.load_state_dict(best_state["model"])
            logger.info("Loaded best checkpoint for test evaluation: %s", best_ckpt_path)
        else:
            logger.warning(
                "Best checkpoint not found — test evaluation uses last-epoch weights"
            )
        model.eval()
        if _test_has_qg:
            test_cmc, test_mAP = evaluate_reid_query_gallery(
                model, _test_query_loader, _test_gallery_loader, device, topk=args.topk,
            )
        else:
            test_cmc, test_mAP = evaluate_reid_split(model, test_loader, device, topk=args.topk)
        logger.info(
            "Test  R1=%.4f  R5=%.4f  R10=%.4f  mAP=%.4f",
            test_cmc.get("rank_1", 0.0),
            test_cmc.get("rank_5", 0.0),
            test_cmc.get("rank_10", 0.0),
            test_mAP,
        )
        logger.info(
            "Done. Best val Rank-1=%.4f  Test Rank-1=%.4f",
            best_rank1, test_cmc.get("rank_1", 0.0),
        )
    else:
        logger.info("Done. Best val Rank-1=%.4f", best_rank1)


if __name__ == "__main__":
    _run(_parse_args())
