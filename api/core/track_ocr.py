"""Track-level OCR finalization shared by sync and async pipelines.

Supports multi-cluster voting: when a track buffer contains OCR evidence for
2+ distinct licence plates, entries are clustered by text similarity and each
cluster is voted on independently.  All valid cluster results are stored in
``tracker._cluster_results[tid]`` and emitted as a ``clusters`` field in the
SSE ``vehicle`` event.  The largest valid cluster becomes the primary result
(``tracker._best[tid]``).
"""
from __future__ import annotations

import asyncio
import logging
from collections import Counter
from dataclasses import replace
from typing import Callable
import torch

import numpy as np

from .config import (
    CLUSTER_SIMILARITY_THRESHOLD,
    MAX_CLUSTERS,
    TOP_K_FRAMES
)
from .ocr_ambiguity import correct_ambiguous_chars
from .ocr_candidates import OcrCandidateResult, build_candidate_crops, rerank_ocr_candidates
from .ocr_cluster import cluster_ocr_results
from .ocr_ctm import CTMFusionResult, fuse_ocr_outputs_ctm

from .tracker import TrackBufferEntry, WebTrackletManager

logger = logging.getLogger(__name__)


def finalise_track_ocr(
    tid: int,
    tracker: WebTrackletManager,
    models: object,
    emit: Callable[[dict], None],
    session_id: str,
    loop: asyncio.AbstractEventLoop | None,
    record_save: Callable | None,
    ocr_backend: str = "default",
    user_id: str | None = None,
) -> None:
    if tracker._done.get(tid, False):
        return

    buf = tracker._buffers.get(tid)
    if buf is None:
        return

    entries = buf.top_k_entries(k=TOP_K_FRAMES)  
    if not entries:
        return

    entries = _entries_with_deferred_ocr(entries, models, ocr_backend)

    # ── Build scored entry list for clustering ─────────────────────────────
    # scored_entries aligns with entries that have non-empty char_probs.
    # Each element: (char_probs, combined_score)
    scored_entries: list[tuple[list[tuple[str, float]], float]] = []
    for entry in entries:
        if entry.char_probs:
            scored_entries.append((entry.char_probs, entry.combined_score))

    if not scored_entries:
        _emit_rejected(
            tid, tracker, entries,
            CTMFusionResult([], {}, [], [], [], None),
            emit, "no_ocr_evidence",
        )
        return

    # ── Cluster by text similarity ─────────────────────────────────────────
    clusters = cluster_ocr_results(
        scored_entries,
        max_clusters=MAX_CLUSTERS,
        similarity_threshold=CLUSTER_SIMILARITY_THRESHOLD,
    )

    # ── Vote per cluster ───────────────────────────────────────────────────
    cluster_results: list[tuple[CTMFusionResult, list[int]]] = []
    for cluster in clusters:
        cluster_probs = [scored_entries[m.index][0] for m in cluster.members]
        result = fuse_ocr_outputs_ctm(cluster_probs)
        if result.char_probs and result.is_valid:
            member_indices = [m.index for m in cluster.members]
            cluster_results.append((result, member_indices))

    # ── Fallback: merge all entries if no cluster is valid ─────────────────
    if not cluster_results:
        all_probs = [p for p, _ in scored_entries]
        result = fuse_ocr_outputs_ctm(all_probs)
        if result.char_probs and result.is_valid:
            cluster_results.append(
                (result, list(range(len(scored_entries))))
            )

    if not cluster_results:
        rejected_result = fuse_ocr_outputs_ctm(
            [p for p, _ in scored_entries]
        )
        _emit_rejected(
            tid, tracker, entries, rejected_result, emit,
            _unreadable_reason(rejected_result, [p for p, _ in scored_entries]),
        )
        return

    # ── Primary result = largest valid cluster ─────────────────────────────
    primary_result, primary_indices = max(
        cluster_results, key=lambda r: len(r[1])
    )

    # ── Store primary result in tracker ────────────────────────────────────
    tracker._best[tid] = primary_result.char_probs
    tracker._ocr_count[tid] = max(
        tracker._ocr_count.get(tid, 0),
        sum(1 for _, score in scored_entries),
    )
    tracker._done[tid] = True

    valid_entries = [e for e in entries if e.char_probs]
    primary_cluster_entries = [
        valid_entries[idx]
        for idx in primary_indices
        if idx < len(valid_entries)
    ]
    
    _store_best_plate_image(tid, tracker, primary_cluster_entries or entries, primary_result.char_probs)
    tracker.plate_changed(tid)

    # ── Store cluster plate images ─────────────────────────────────────────
    cluster_data = _build_cluster_data(
        tid, tracker, entries, scored_entries, cluster_results,
    )
    tracker.set_cluster_results(tid, cluster_data)

    # ── Emit vehicle event ─────────────────────────────────────────────────
    route_fields = _route_event_fields(entries)
    ocr_method = "ocr_output_ctm"
    event: dict = {
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
        "ocr_method": ocr_method,
        "candidate_method": _candidate_method_summary(entries),
        "ctm_support": primary_result.ctm_support,
        "unresolved_slots": primary_result.unresolved_slots,
        "vote_summary": primary_result.vote_summary,
        "final": True,
        **route_fields,
    }

    # Attach clusters when there are multiple valid results
    if len(cluster_data) > 1:
        event["clusters"] = cluster_data

    emit(event)

    if session_id and loop is not None and record_save is not None:
        record_save(
            session_id,
            tid,
            tracker,
            primary_result.char_probs,
            ocr_method,
            primary_result.vote_summary,
            loop,
            user_id,
        )

    # Event emitted + DB snapshot taken → free this track's heavy state.
    tracker.release_track(tid, recognized=True)


# ── Helpers ───────────────────────────────────────────────────────────────────


def _build_cluster_data(
    tid: int,
    tracker: WebTrackletManager,
    entries: list[TrackBufferEntry],
    scored_entries: list[tuple[list[tuple[str, float]], float]],
    cluster_results: list[tuple[CTMFusionResult, list[int]]],
) -> list[dict]:
    """Build the ``clusters`` list for the SSE event and tracker storage."""
    vehicle_b64 = tracker.vehicle_b64(tid)

    # valid_entries[i] corresponds to scored_entries[i]
    valid_entries = [e for e in entries if e.char_probs]

    data: list[dict] = []
    for result, member_indices in cluster_results:
        cluster_entries = [
            valid_entries[idx]
            for idx in member_indices
            if idx < len(valid_entries)
        ]
        best_entry = max(
            cluster_entries,
            key=lambda entry: entry.combined_score,
            default=None,
        )
        best_crop = best_entry.crop if best_entry is not None else _empty_crop()

        avg_conf = (
            sum(p for _, p in result.char_probs) / len(result.char_probs)
            if result.char_probs
            else 0.0
        )
        plate_b64 = tracker._encode(best_crop, max_w=None, quality=90)

        data.append(
            {
                "plate": result.text,
                "plate_text": result.text,
                "chars": [[c, round(p, 3)] for c, p in result.char_probs],
                "plate_b64": plate_b64,
                "vehicle_b64": vehicle_b64,
                "confidence": round(avg_conf, 4),
                "plate_text_confidence": round(avg_conf, 4),
                "frame_count": len(cluster_entries),
                "ocr_frames": len(cluster_entries),
                "ocr_method": "ocr_output_ctm",
                "vote_summary": result.vote_summary,
                "ocr_vote_summary": result.vote_summary,
                "track_buffer": _track_buffer_json_for_entries(tracker, cluster_entries),
                "template": result.template_name,
            }
        )

    # Sort by frame_count descending (largest cluster first)
    data.sort(key=lambda c: c["frame_count"], reverse=True)
    for cluster_index, cluster in enumerate(data):
        cluster["cluster_index"] = cluster_index
    return data


def _track_buffer_json_for_entries(
    tracker: WebTrackletManager,
    entries: list[TrackBufferEntry],
) -> list[dict]:
    frames: list[dict] = []
    for entry in sorted(
        entries,
        key=lambda item: (
            1 if item.router_result.get("legibility") in ("perfect", "good") else 0,
            item.combined_score,
        ),
        reverse=True,
    ):
        frames.append(
            {
                "frame_index": int(entry.frame_idx),
                "quality_score": round(float(entry.quality_score), 4),
                "ocr_confidence": round(float(entry.ocr_conf), 4),
                "combined_score": round(float(entry.combined_score), 4),
                "ocr_text": "".join(ch for ch, _ in entry.char_probs) if entry.char_probs else None,
                "candidate_method": entry.candidate_method,
                "route": entry.route,
                "image_b64": tracker._encode(entry.crop, max_w=None, quality=85),
                **entry.router_result,
            }
        )
    return frames


def _empty_crop() -> np.ndarray:
    """Return a minimal blank image as fallback."""
    return np.zeros((1, 1, 3), dtype=np.uint8)


def _emit_rejected(
    tid: int,
    tracker: WebTrackletManager,
    entries: list[TrackBufferEntry],
    result: CTMFusionResult,
    emit: Callable[[dict], None],
    reason: str,
) -> None:
    tracker._best.pop(tid, None)
    tracker._ocr_count[tid] = max(
        tracker._ocr_count.get(tid, 0),
        sum(1 for e in entries if e.char_probs),
    )
    _store_best_plate_image(tid, tracker, entries, result.char_probs)
    tracker._done[tid] = True
    emit(
        {
            "type": "rejected_vehicle",
            "id": tid,
            "done": True,
            "cls": tracker._cls.get(tid, ""),
            "plate": result.text,
            "chars": [[c, round(p, 3)] for c, p in result.char_probs],
            "plate_b64": tracker.plate_b64(tid),
            "vehicle_b64": tracker.vehicle_b64(tid),
            "track_buffer": tracker.track_buffer_json(tid),
            "ocr_frames": sum(1 for e in entries if e.char_probs),
            "ocr_method": "ocr_output_ctm",
            "candidate_method": _candidate_method_summary(entries),
            "ctm_support": result.ctm_support,
            "unresolved_slots": result.unresolved_slots,
            "vote_summary": result.vote_summary,
            "unreadable_reason": reason,
            **_route_event_fields(entries),
        }
    )
    # Event emitted → free heavy state (no DB save for rejected tracks).
    tracker.release_track(tid, recognized=False)


# ── Deferred OCR (unchanged from original) ────────────────────────────────────


def _entries_with_deferred_ocr(
    entries: list[TrackBufferEntry],
    models: object,
    ocr_backend: str = "default",
) -> list[TrackBufferEntry]:
    """OCR buffered degraded crops at track finalization.

    Direct high-confidence crops are accepted before this point. The remaining
    buffered evidence is mostly poor/illegible crops; illegible entries stay as
    evidence only, while poor entries get OCR candidates lazily here.
    """
    if not entries:
        return entries

    pending: list[tuple[int, str, object]] = []
    pending_entries: list[TrackBufferEntry] = []
    for entry in entries:
        if entry.char_probs or entry.route == "unreadable_wait":
            continue
        for method, crop in build_candidate_crops(entry.crop):
            pending.append((len(pending_entries), method, crop))
            pending_entries.append(entry)

    if not pending:
        return entries

    try:
        from .models import ocr_batch, preprocess_plate_for_model, select_ocr_model

        ocr_model = select_ocr_model(models, ocr_backend)
        device = getattr(models, "device")
        tensors = torch.stack(
            [
                preprocess_plate_for_model(ocr_model, crop)
                for _entry_idx, _method, crop in pending
            ]
        ).to(device)
        ocr_results = ocr_batch(ocr_model, tensors, device)
    except Exception:
        return entries

    grouped: dict[int, list[tuple[OcrCandidateResult, np.ndarray]]] = {}
    for (entry_idx, method, _crop), (char_probs, _all_confident) in zip(
        pending, ocr_results, strict=False
    ):
        correction = correct_ambiguous_chars(char_probs)
        grouped.setdefault(entry_idx, []).append(
            (
                OcrCandidateResult(
                    method,
                    correction.char_probs,
                    risk_penalty=correction.risk_penalty,
                ),
                _crop,
            )
        )

    replacements: dict[int, TrackBufferEntry] = {}
    for entry_idx, candidate_tuples in grouped.items():
        candidates = [c for c, _crop in candidate_tuples]
        best = rerank_ocr_candidates(candidates)
        if best is None:
            continue
            
        best_crop = next(_crop for c, _crop in candidate_tuples if c is best)
        
        entry = pending_entries[entry_idx]
        replacements[id(entry)] = replace(
            entry,
            crop=best_crop,
            char_probs=best.char_probs,
            ocr_conf=max(best.confidence, 0.10),
            candidate_method=best.method,
        )

    return [replacements.get(id(entry), entry) for entry in entries]


# ── Shared helpers (unchanged) ────────────────────────────────────────────────





def _store_best_plate_image(
    tid: int,
    tracker: WebTrackletManager,
    entries: list[TrackBufferEntry],
    char_probs: list[tuple[str, float]],
) -> None:
    if not entries:
        return
    best_entry = max(entries, key=lambda e: e.combined_score, default=entries[0])
    if char_probs:
        confidence = sum(p for _, p in char_probs) / len(char_probs)
        tracker.set_plate_img(tid, best_entry.crop, confidence)
    else:
        tracker.set_plate_img(tid, best_entry.crop, best_entry.combined_score)


def _unreadable_reason(
    result: CTMFusionResult,
    prob_lists: list[list[tuple[str, float]]],
) -> str:
    if not prob_lists:
        return "no_ocr_evidence"
    if result.unresolved_slots:
        return "unresolved_slots"
    if not result.is_valid:
        return "invalid_format"
    return "unreadable"


def _candidate_method_summary(entries: list[TrackBufferEntry]) -> str:
    methods = sorted(
        {entry.candidate_method for entry in entries if entry.candidate_method}
    )
    return "+".join(methods) if methods else "unknown"


def _route_event_fields(entries: list[TrackBufferEntry]) -> dict:
    routes = Counter(entry.route for entry in entries if entry.route)
    router_results = [entry.router_result for entry in entries if entry.router_result]
    tags: dict[str, bool] = {}
    for result in router_results:
        for key, value in result.get("degradation_tags", {}).items():
            tags[key] = bool(tags.get(key, False) or value)

    legibilities = Counter(
        result.get("legibility", "") for result in router_results if result.get("legibility")
    )
    bins = Counter(
        result.get("quality_bin", "") for result in router_results if result.get("quality_bin")
    )
    router_conf = max(
        (float(result.get("router_conf", 0.0)) for result in router_results),
        default=0.0,
    )

    return {
        "route": routes.most_common(1)[0][0] if routes else "",
        "legibility": legibilities.most_common(1)[0][0] if legibilities else "",
        "quality_bin": bins.most_common(1)[0][0] if bins else "",
        "degradation_tags": tags,
        "router_conf": round(router_conf, 4),
    }
