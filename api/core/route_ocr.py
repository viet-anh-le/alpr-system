"""Route-aware per-frame OCR orchestration."""
from __future__ import annotations

import asyncio
from collections import defaultdict
from dataclasses import dataclass
from typing import Callable

import numpy as np

from .config import CONF_THRESHOLD
from .ocr_ambiguity import correct_ambiguous_chars
from .ocr_candidates import OcrCandidateResult, rerank_ocr_candidates
from .plate_format import mean_confidence
from .quality_router import PlateQualityResult, PlateQualityRouter
from .tracker import WebTrackletManager


@dataclass(frozen=True)
class RouteOcrJob:
    tid: int
    plate_crop: np.ndarray
    vehicle_crop: np.ndarray
    candidate_crop: np.ndarray
    candidate_method: str
    frame_idx: int
    quality: PlateQualityResult


def prepare_route_ocr_jobs(
    matched: list[tuple[int, np.ndarray, np.ndarray]],
    tracker: WebTrackletManager,
    router: PlateQualityRouter,
    frame_idx: int,
) -> tuple[list[RouteOcrJob], set[int]]:
    jobs: list[RouteOcrJob] = []
    active_tids: set[int] = set()

    for tid, plate_crop, vehicle_crop in matched:
        if not tracker.should_ocr(tid):
            continue

        quality = router.route(plate_crop)
        tracker.update_vehicle_img(tid, vehicle_crop, quality.quality_numeric)
        active_tids.add(tid)

        if quality.route == "unreadable_wait":
            tracker.buffer_crop(
                tid,
                plate_crop,
                quality.quality_numeric,
                0.10,
                [],
                frame_idx,
                candidate_method="unreadable",
                route=quality.route,
                router_result=quality.as_event_fields(),
            )
            continue

        jobs.append(RouteOcrJob(
            tid=tid,
            plate_crop=plate_crop,
            vehicle_crop=vehicle_crop,
            candidate_crop=plate_crop,
            candidate_method="original",
            frame_idx=frame_idx,
            quality=quality,
        ))

    return jobs, active_tids


def consume_route_ocr_results(
    jobs: list[RouteOcrJob],
    ocr_results: list[tuple[list[tuple[str, float]], bool]],
    tracker: WebTrackletManager,
    emit: Callable[[dict], None],
    *,
    session_id: str = "",
    loop: asyncio.AbstractEventLoop | None = None,
    record_save: Callable | None = None,
    user_id: str | None = None,
) -> None:
    grouped: dict[tuple[int, int], list[tuple[RouteOcrJob, OcrCandidateResult]]] = defaultdict(list)
    for job, (char_probs, _) in zip(jobs, ocr_results):
        correction = correct_ambiguous_chars(char_probs)
        grouped[(job.tid, job.frame_idx)].append((
            job,
            OcrCandidateResult(
                job.candidate_method,
                correction.char_probs,
                risk_penalty=correction.risk_penalty,
            ),
        ))

    for (tid, _frame_idx), items in grouped.items():
        candidates = [candidate for _, candidate in items]
        best = rerank_ocr_candidates(candidates)
        if best is None:
            continue
        best_job = next(job for job, candidate in items if candidate is best)
        quality = best_job.quality
        all_char_confident = _all_chars_confident(best.char_probs)

        if (
            quality.route == "direct"
            and best.method == "original"
            and best.is_valid
            and all_char_confident
        ):
            tracker.buffer_crop(
                tid,
                best_job.plate_crop,
                quality.quality_numeric,
                best.confidence,
                best.char_probs,
                best_job.frame_idx,
                candidate_method=best.method,
                route=quality.route,
                router_result=quality.as_event_fields(),
            )
            _accept_single_frame(
                tid,
                tracker,
                best_job,
                best,
                quality,
                emit,
                session_id=session_id,
                loop=loop,
                record_save=record_save,
                user_id=user_id,
            )
            continue

        route = "tracklet_fusion" if quality.route == "direct" else quality.route
        router_fields = {**quality.as_event_fields(), "route": route}
        for job, candidate in items:
            tracker.buffer_crop(
                tid,
                job.plate_crop,
                quality.quality_numeric,
                max(candidate.confidence, 0.10),
                candidate.char_probs,
                job.frame_idx,
                candidate_method=candidate.method,
                route=route,
                router_result=router_fields,
            )


def _all_chars_confident(char_probs: list[tuple[str, float]]) -> bool:
    return bool(char_probs) and all(float(conf) >= CONF_THRESHOLD for _, conf in char_probs)


def _accept_single_frame(
    tid: int,
    tracker: WebTrackletManager,
    job: RouteOcrJob,
    best: OcrCandidateResult,
    quality: PlateQualityResult,
    emit: Callable[[dict], None],
    *,
    session_id: str,
    loop: asyncio.AbstractEventLoop | None,
    record_save: Callable | None,
    user_id: str | None,
) -> None:
    tracker._best[tid] = best.char_probs
    buf = tracker._buffers.get(tid)
    buffered_ocr_frames = (
        sum(1 for char_probs in buf.char_prob_lists if char_probs)
        if buf is not None
        else 1
    )
    tracker._ocr_count[tid] = max(tracker._ocr_count.get(tid, 0), buffered_ocr_frames)
    tracker.update_plate_img(tid, job.plate_crop, best.char_probs)

    vote_summary = {best.text: 1} if best.text else {}
    if tracker.plate_changed(tid):
        emit(
            {
                "type": "vehicle",
                "id": tid,
                "cls": tracker._cls.get(tid, ""),
                "plate": tracker.display_text(tid),
                "chars": tracker.chars_json(tid),
                "done": False,
                "plate_b64": tracker.plate_b64(tid),
                "vehicle_b64": tracker.vehicle_b64(tid),
                "track_buffer": tracker.track_buffer_json(tid),
                "ocr_frames": tracker.ocr_frames(tid),
                "ocr_method": "single_frame_direct",
                "candidate_method": best.method,
                "vote_summary": vote_summary,
                "confidence": round(mean_confidence(best.char_probs), 4),
                **quality.as_event_fields(),
            }
        )
