from __future__ import annotations

# ruff: noqa: E402 -- this executable script must add the repository to sys.path first.

import sys
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "LPRNet"))

from api.core.association import TrajectoryAssociator
from api.core.cascade_plate import detect_plate_tracks_cascade
from api.core.config import (
    ASSOCIATION_AGREEMENT_RATIO,
    ASSOCIATION_MATCH_FRAMES,
    CLUSTER_SIMILARITY_THRESHOLD,
    FRAME_STRIDE,
    MAX_CLUSTERS,
    TOP_K_FRAMES,
    VEHICLE_CLASSES,
)
from api.core.models import load_models, ocr_batch, preprocess_plate_for_model
from api.core.ocr_cluster import cluster_ocr_results
from api.core.ocr_ctm import fuse_ocr_outputs_ctm
from api.core.quality_router import PlateQualityRouter
from api.core.quality_scorer import quality_score
from api.core.route_ocr import consume_route_ocr_results, prepare_route_ocr_jobs
from api.core.track_ocr import _entries_with_deferred_ocr
from api.core.tracker import WebTrackletManager
from api.core.video_processor import crop_vehicle


@dataclass(frozen=True)
class SingleFrameCandidate:
    """Raw plate crop selected before either OCR or quality routing."""

    crop: np.ndarray
    score: float
    frame_idx: int


@dataclass(frozen=True)
class MultiFrameResult:
    """One CTM-voted result, possibly one of several plate clusters."""

    text: str
    char_probs: tuple[tuple[str, float], ...]
    frame_count: int


def select_best_single_frames(
    current: Mapping[int, SingleFrameCandidate],
    matched: list[tuple[int, np.ndarray, np.ndarray]],
    frame_idx: int,
    *,
    score_fn: Callable[[np.ndarray], float] = quality_score,
) -> dict[int, SingleFrameCandidate]:
    """Keep one best raw crop per vehicle track without invoking the router.

    The deterministic visual q-score is used only to rank raw crops. OCR output
    cannot influence selection because OCR is deliberately deferred until the
    video has been scanned.
    """

    selected = dict(current)
    for tid, plate_crop, _vehicle_crop in matched:
        score = float(score_fn(plate_crop))
        previous = selected.get(tid)
        if previous is not None and score <= previous.score:
            continue
        selected = {
            **selected,
            tid: SingleFrameCandidate(
                crop=plate_crop.copy(),
                score=score,
                frame_idx=frame_idx,
            ),
        }
    return selected


def ocr_best_single_frames(
    candidates: Mapping[int, SingleFrameCandidate],
    ocr_model: object,
    device: object,
    *,
    preprocess_fn: Callable[[object, np.ndarray], torch.Tensor] | None = None,
    ocr_batch_fn: Callable | None = None,
) -> dict[int, list[tuple[str, float]]]:
    """OCR exactly one preselected raw crop per track, with no router or vote."""

    if not candidates:
        return {}

    preprocess = preprocess_fn or preprocess_plate_for_model
    infer_batch = ocr_batch_fn or ocr_batch
    track_ids = sorted(candidates)
    tensors = torch.stack(
        [preprocess(ocr_model, candidates[tid].crop) for tid in track_ids]
    ).to(device)
    outputs = infer_batch(ocr_model, tensors, device)
    if len(outputs) != len(track_ids):
        raise RuntimeError(
            "Single-frame OCR returned a different number of results "
            f"({len(outputs)}) than selected tracks ({len(track_ids)})."
        )

    return {
        tid: list(char_probs)
        for tid, (char_probs, _all_confident) in zip(track_ids, outputs, strict=True)
    }


def fuse_multiframe_results(
    tracker: WebTrackletManager,
    models: object,
    *,
    top_k: int = TOP_K_FRAMES,
) -> dict[int, list[MultiFrameResult]]:
    """Run the same top-K clustering and CTM vote used by the full pipeline."""

    fused_tracks: dict[int, list[MultiFrameResult]] = {}
    for tid, buffer in sorted(tracker._buffers.items()):
        entries = buffer.top_k_entries(k=top_k)
        entries = _entries_with_deferred_ocr(entries, models)

        scored_entries = [
            (entry.char_probs, entry.combined_score)
            for entry in entries
            if entry.char_probs
        ]
        if not scored_entries:
            fused_tracks[tid] = []
            continue

        clusters = cluster_ocr_results(
            scored_entries,
            max_clusters=MAX_CLUSTERS,
            similarity_threshold=CLUSTER_SIMILARITY_THRESHOLD,
        )
        results: list[MultiFrameResult] = []
        for cluster in clusters:
            cluster_probs = [
                scored_entries[member.index][0]
                for member in cluster.members
            ]
            result = fuse_ocr_outputs_ctm(cluster_probs)
            if result.char_probs and result.is_valid:
                results.append(
                    MultiFrameResult(
                        text=result.text,
                        char_probs=tuple(result.char_probs),
                        frame_count=len(cluster.members),
                    )
                )

        if not results:
            fallback = fuse_ocr_outputs_ctm([char_probs for char_probs, _ in scored_entries])
            if fallback.char_probs and fallback.is_valid:
                results.append(
                    MultiFrameResult(
                        text=fallback.text,
                        char_probs=tuple(fallback.char_probs),
                        frame_count=len(scored_entries),
                    )
                )

        fused_tracks[tid] = sorted(
            results,
            key=lambda result: result.frame_count,
            reverse=True,
        )
    return fused_tracks


def _resolve_quality_router(models: object) -> PlateQualityRouter:
    model_router = getattr(models, "quality_router", None)
    return model_router if isinstance(model_router, PlateQualityRouter) else PlateQualityRouter()


def _build_vehicle_tracks(models: object, vehicle_tracker: object, frame: np.ndarray) -> list[dict]:
    prediction = models.vehicle.predict(
        frame,
        classes=VEHICLE_CLASSES,
        verbose=False,
    )[0]
    if prediction.boxes is not None and len(prediction.boxes) > 0:
        xyxy = prediction.boxes.xyxy.cpu().numpy()
        confidence = prediction.boxes.conf.cpu().numpy().reshape(-1, 1)
        classes = prediction.boxes.cls.cpu().numpy().reshape(-1, 1)
        detections = np.concatenate([xyxy, confidence, classes], axis=1).astype(np.float32)
    else:
        detections = np.zeros((0, 6), dtype=np.float32)

    boxes, ids, classes = vehicle_tracker.track(detections, frame)
    return [
        {"id": int(tid), "box": box.tolist(), "class_id": int(class_id)}
        for box, tid, class_id in zip(boxes, ids, classes)
    ]


def _associate_plate_crops(
    frame: np.ndarray,
    tracked: list[dict],
    plate_tracks: list[dict],
    associator: TrajectoryAssociator,
) -> list[tuple[int, np.ndarray, np.ndarray]]:
    matched: list[tuple[int, np.ndarray, np.ndarray]] = []
    for vehicle_tid, plate in associator.process_frame(plate_tracks, tracked):
        vehicle_box = associator.vehicle_cache.get(vehicle_tid)
        if vehicle_box is None:
            continue
        vehicle_image = crop_vehicle(frame, vehicle_box)
        if vehicle_image.size == 0:
            continue
        matched.append((vehicle_tid, plate["crop"], vehicle_image))
    return matched


def print_evaluation_table(
    single_candidates: Mapping[int, SingleFrameCandidate],
    single_results: Mapping[int, list[tuple[str, float]]],
    multiframe_results: Mapping[int, list[MultiFrameResult]],
) -> None:
    """Print the single-frame result and every valid CTM cluster per track."""

    print(f"\n{'Track':<12} | {'Best Single Frame':<20} | {'Multi-frame CTM':<20}")
    print("-" * 67)

    track_ids = sorted(set(single_candidates) | set(single_results) | set(multiframe_results))
    for tid in track_ids:
        single_text = "".join(char for char, _ in single_results.get(tid, []))
        voted_results = multiframe_results.get(tid, [])
        if not voted_results:
            print(f"{tid:<12} | {single_text:<20} | {'':<20}")
            continue

        for cluster_idx, result in enumerate(voted_results):
            row_tid = str(tid) if cluster_idx == 0 else f"{tid} (split)"
            row_single = single_text if cluster_idx == 0 else ""
            print(
                f"{row_tid:<12} | {row_single:<20} | {result.text:<20} "
                f"(Cluster {cluster_idx}, {result.frame_count} frames)"
            )


def run_eval(video_path: str) -> None:
    """Compare a strict single-frame baseline with the full multi-frame path.

    Shared front-end:
        vehicle crop -> plate crop -> plate/vehicle association
    Single-frame branch:
        best raw crop -> one OCR call
    Multi-frame branch:
        quality router -> per-frame OCR -> top-K clustering -> CTM voting
    """

    print(f"Processing {Path(video_path).name} ...")
    models = load_models()
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    multiframe_tracker = WebTrackletManager()
    associator = TrajectoryAssociator(
        match_frames=ASSOCIATION_MATCH_FRAMES,
        agreement_ratio=ASSOCIATION_AGREEMENT_RATIO,
    )
    vehicle_tracker = models.create_vehicle_tracker()
    quality_router = _resolve_quality_router(models)

    single_candidates: dict[int, SingleFrameCandidate] = {}
    frame_idx = 0

    def dummy_emit(_event: dict) -> None:
        return None

    try:
        while True:
            success, frame = cap.read()
            if not success:
                break
            frame_idx += 1

            tracked = _build_vehicle_tracks(models, vehicle_tracker, frame)
            if frame_idx % FRAME_STRIDE != 0:
                continue

            plate_tracks = detect_plate_tracks_cascade(
                frame,
                tracked,
                models.plate,
            )
            matched = _associate_plate_crops(frame, tracked, plate_tracks, associator)

            # Baseline branch stops here. It sees raw associated crops only.
            single_candidates = select_best_single_frames(
                single_candidates,
                matched,
                frame_idx,
            )

            # Full multi-frame branch: router -> OCR -> track buffer -> later CTM vote.
            ocr_jobs, _active_tids = prepare_route_ocr_jobs(
                matched,
                multiframe_tracker,
                quality_router,
                frame_idx,
            )
            if not ocr_jobs:
                continue

            tensors = torch.stack(
                [
                    preprocess_plate_for_model(models.ocr, job.candidate_crop)
                    for job in ocr_jobs
                ]
            ).to(models.device)
            ocr_results = ocr_batch(models.ocr, tensors, models.device)
            consume_route_ocr_results(
                ocr_jobs,
                ocr_results,
                multiframe_tracker,
                dummy_emit,
            )
    finally:
        cap.release()

    single_results = ocr_best_single_frames(
        single_candidates,
        models.ocr,
        models.device,
    )
    multiframe_results = fuse_multiframe_results(multiframe_tracker, models)
    print_evaluation_table(single_candidates, single_results, multiframe_results)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python scripts/eval_single_frame.py <video_path>")
        sys.exit(1)
    run_eval(sys.argv[1])
