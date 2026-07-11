from __future__ import annotations

import argparse
import shutil
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATASET_DIR = ROOT / "data" / "raw" / "platesmania_vn"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass(frozen=True)
class RefreshStats:
    refreshed: int = 0
    skipped: int = 0
    missing_full_frame: int = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replace Platesmania pending-review crop previews with full-frame images."
    )
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument(
        "--reason-prefix",
        default="low_detector_confidence",
        help="Only refresh review entries whose .txt reason starts with this value.",
    )
    parser.add_argument("--all", action="store_true", help="Refresh every pending review image.")
    parser.add_argument("--dry-run", action="store_true", help="Print what would change without copying files.")
    return parser.parse_args()


def find_image_by_stem(directory: Path, stem: str) -> Path | None:
    for suffix in sorted(IMAGE_SUFFIXES):
        candidate = directory / f"{stem}{suffix}"
        if candidate.exists():
            return candidate
    return None


def remove_existing_review_images(review_dir: Path, stem: str, *, keep: Path | None = None) -> None:
    keep_resolved = keep.resolve() if keep is not None else None
    for suffix in IMAGE_SUFFIXES:
        candidate = review_dir / f"{stem}{suffix}"
        if not candidate.exists():
            continue
        if keep_resolved is not None and candidate.resolve() == keep_resolved:
            continue
        candidate.unlink()


def should_refresh(reason: str, *, reason_prefix: str, refresh_all: bool) -> bool:
    if refresh_all:
        return True
    return reason.startswith(reason_prefix)


def refresh_review_full_frames(
    dataset_dir: Path,
    *,
    reason_prefix: str = "low_detector_confidence",
    refresh_all: bool = False,
    dry_run: bool = False,
) -> RefreshStats:
    review_dir = dataset_dir / "review" / "pending_review"
    full_frame_dir = dataset_dir / "downloads" / "full_frames"
    if not review_dir.exists():
        raise FileNotFoundError(f"Review directory not found: {review_dir}")
    if not full_frame_dir.exists():
        raise FileNotFoundError(f"Full-frame directory not found: {full_frame_dir}")

    refreshed = 0
    skipped = 0
    missing = 0
    for reason_path in sorted(review_dir.glob("*.txt")):
        record_id = reason_path.stem
        reason = reason_path.read_text(encoding="utf-8").strip()
        if not should_refresh(reason, reason_prefix=reason_prefix, refresh_all=refresh_all):
            skipped += 1
            continue

        full_frame = find_image_by_stem(full_frame_dir, record_id)
        if full_frame is None:
            missing += 1
            continue

        target = review_dir / full_frame.name
        if dry_run:
            print(f"would_refresh {record_id}: {full_frame} -> {target}")
            refreshed += 1
            continue

        remove_existing_review_images(review_dir, record_id, keep=target)
        shutil.copy2(full_frame, target)
        refreshed += 1

    return RefreshStats(refreshed=refreshed, skipped=skipped, missing_full_frame=missing)


def main() -> None:
    args = parse_args()
    stats = refresh_review_full_frames(
        args.dataset_dir.expanduser().resolve(),
        reason_prefix=args.reason_prefix,
        refresh_all=args.all,
        dry_run=args.dry_run,
    )
    print(f"refreshed={stats.refreshed}")
    print(f"skipped={stats.skipped}")
    print(f"missing_full_frame={stats.missing_full_frame}")


if __name__ == "__main__":
    main()
