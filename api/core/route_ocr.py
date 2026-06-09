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

        if quality.route != "direct":
            candidate_method = "unreadable" if quality.route == "unreadable_wait" else "tracklet_fusion"
            tracker.buffer_crop(
                tid,
                plate_crop,
                quality.quality_numeric,
                0.10,
                [],
                frame_idx,
                candidate_method=candidate_method,
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
        first_job = items[0][0]
        quality = first_job.quality

        if (
            quality.route == "direct"
            and best.method == "original"
            and best.is_valid
            and best.confidence >= CONF_THRESHOLD
        ):
            tracker.buffer_crop(
                tid,
                first_job.plate_crop,
                quality.quality_numeric,
                best.confidence,
                best.char_probs,
                first_job.frame_idx,
                candidate_method=best.method,
                route=quality.route,
                router_result=quality.as_event_fields(),
            )
            _accept_single_frame(
                tid,
                tracker,
                first_job.plate_crop,
                best,
                quality,
                emit,
                session_id=session_id,
                loop=loop,
                record_save=record_save,
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


def _accept_single_frame(
    tid: int,
    tracker: WebTrackletManager,
    plate_crop: np.ndarray,
    best: OcrCandidateResult,
    quality: PlateQualityResult,
    emit: Callable[[dict], None],
    *,
    session_id: str,
    loop: asyncio.AbstractEventLoop | None,
    record_save: Callable | None,
) -> None:
    tracker._best[tid] = best.char_probs
    tracker._ocr_count[tid] = max(tracker._ocr_count.get(tid, 0), 1)
    tracker._done[tid] = True
    tracker.update_plate_img(tid, plate_crop, best.char_probs)

    vote_summary = {best.text: 1} if best.text else {}
    if tracker.plate_changed(tid):
        emit(
            {
                "type": "vehicle",
                "id": tid,
                "cls": tracker._cls.get(tid, ""),
                "plate": tracker.display_text(tid),
                "chars": tracker.chars_json(tid),
                "done": True,
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

    if session_id and loop is not None and record_save is not None:
        record_save(session_id, tid, tracker, best.char_probs, "single_frame_direct", vote_summary, loop)
