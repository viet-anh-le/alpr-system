"""Track-level OCR finalization shared by sync and async pipelines."""
from __future__ import annotations

import asyncio
from collections import Counter
from dataclasses import replace
from typing import Callable

from .config import TOP_K_FRAMES
from .ocr_ambiguity import correct_ambiguous_chars
from .ocr_candidates import OcrCandidateResult, build_candidate_crops, rerank_ocr_candidates
from .ocr_ctm import CTMFusionResult, fuse_ocr_outputs_ctm
from .quality_router import DegradationTags
from .tracker import TrackBufferEntry, WebTrackletManager


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
    prob_lists = [entry.char_probs for entry in entries if entry.char_probs]
    result = fuse_ocr_outputs_ctm(prob_lists)
    route_fields = _route_event_fields(entries)
    ocr_method = "ocr_output_ctm"

    if not result.char_probs or not result.is_valid:
        tracker._best.pop(tid, None)
        tracker._ocr_count[tid] = max(tracker._ocr_count.get(tid, 0), len(prob_lists))
        _store_best_plate_image(tid, tracker, entries, result.char_probs)
        emit(
            {
                "type": "rejected_vehicle",
                "id": tid,
                "cls": tracker._cls.get(tid, ""),
                "plate": result.text,
                "chars": [[c, round(p, 3)] for c, p in result.char_probs],
                "plate_b64": tracker.plate_b64(tid),
                "vehicle_b64": tracker.vehicle_b64(tid),
                "track_buffer": tracker.track_buffer_json(tid),
                "ocr_frames": len(prob_lists),
                "ocr_method": ocr_method,
                "candidate_method": _candidate_method_summary(entries),
                "ctm_support": result.ctm_support,
                "unresolved_slots": result.unresolved_slots,
                "vote_summary": result.vote_summary,
                "unreadable_reason": _unreadable_reason(result, prob_lists),
                **route_fields,
            }
        )
        if session_id and loop is not None and record_save is not None:
            record_save(
                session_id,
                tid,
                tracker,
                result.char_probs,
                ocr_method,
                result.vote_summary,
                loop,
                user_id,
            )
        return

    tracker._best[tid] = result.char_probs
    tracker._ocr_count[tid] = max(tracker._ocr_count.get(tid, 0), len(prob_lists))
    tracker._done[tid] = True
    _store_best_plate_image(tid, tracker, entries, result.char_probs)
    tracker.plate_changed(tid)

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
            "ocr_method": ocr_method,
            "candidate_method": _candidate_method_summary(entries),
            "ctm_support": result.ctm_support,
            "unresolved_slots": result.unresolved_slots,
            "vote_summary": result.vote_summary,
            **route_fields,
        }
    )

    if session_id and loop is not None and record_save is not None:
        record_save(
            session_id,
            tid,
            tracker,
            result.char_probs,
            ocr_method,
            result.vote_summary,
            loop,
            user_id,
        )


def _entries_with_deferred_ocr(
    entries: list[TrackBufferEntry],
    models: object,
    ocr_backend: str = "default",
) -> list[TrackBufferEntry]:
    """OCR buffered degraded crops at track finalization.

    Direct high-confidence crops are accepted before this point.  The remaining
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
        for method, crop in build_candidate_crops(entry.crop, _tags_from_entry(entry)):
            pending.append((len(pending_entries), method, crop))
        pending_entries.append(entry)

    if not pending:
        return entries

    try:
        from .models import ocr_batch, preprocess_plate_for_model, select_ocr_model
        import torch

        ocr_model = select_ocr_model(models, ocr_backend)
        device = getattr(models, "device")
        tensors = torch.stack([
            preprocess_plate_for_model(ocr_model, crop) for _entry_idx, _method, crop in pending
        ]).to(device)
        ocr_results = ocr_batch(ocr_model, tensors, device)
    except Exception:
        return entries

    grouped: dict[int, list[OcrCandidateResult]] = {}
    for (entry_idx, method, _crop), (char_probs, _all_confident) in zip(pending, ocr_results, strict=False):
        correction = correct_ambiguous_chars(char_probs)
        grouped.setdefault(entry_idx, []).append(
            OcrCandidateResult(
                method,
                correction.char_probs,
                risk_penalty=correction.risk_penalty,
            )
        )

    replacements: dict[int, TrackBufferEntry] = {}
    for entry_idx, candidates in grouped.items():
        best = rerank_ocr_candidates(candidates)
        if best is None:
            continue
        entry = pending_entries[entry_idx]
        replacements[id(entry)] = replace(
            entry,
            char_probs=best.char_probs,
            ocr_conf=max(best.confidence, 0.10),
            candidate_method=best.method,
        )

    return [replacements.get(id(entry), entry) for entry in entries]


def _tags_from_entry(entry: TrackBufferEntry) -> DegradationTags:
    raw_tags = entry.router_result.get("degradation_tags", {})
    if not isinstance(raw_tags, dict):
        raw_tags = {}
    return DegradationTags(**{
        key: bool(raw_tags.get(key, False))
        for key in DegradationTags.__dataclass_fields__
    })


def _store_best_plate_image(
    tid: int,
    tracker: WebTrackletManager,
    entries: list[TrackBufferEntry],
    char_probs: list[tuple[str, float]],
) -> None:
    if not entries:
        return
    best_entry = max(entries, key=lambda entry: entry.combined_score)
    if char_probs:
        tracker.update_plate_img(tid, best_entry.crop, char_probs)
    else:
        tracker._plate_img[tid] = best_entry.crop.copy()
        tracker._plate_img_conf[tid] = best_entry.combined_score


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
    methods = sorted({entry.candidate_method for entry in entries if entry.candidate_method})
    return "+".join(methods) if methods else "unknown"


def _route_event_fields(entries: list[TrackBufferEntry]) -> dict:
    routes = Counter(entry.route for entry in entries if entry.route)
    router_results = [entry.router_result for entry in entries if entry.router_result]
    tags: dict[str, bool] = {}
    for result in router_results:
        for key, value in result.get("degradation_tags", {}).items():
            tags[key] = bool(tags.get(key, False) or value)

    legibilities = Counter(
        result.get("legibility", "")
        for result in router_results
        if result.get("legibility")
    )
    bins = Counter(
        result.get("quality_bin", "")
        for result in router_results
        if result.get("quality_bin")
    )
    router_conf = max((float(result.get("router_conf", 0.0)) for result in router_results), default=0.0)

    return {
        "route": routes.most_common(1)[0][0] if routes else "",
        "legibility": legibilities.most_common(1)[0][0] if legibilities else "",
        "quality_bin": bins.most_common(1)[0][0] if bins else "",
        "degradation_tags": tags,
        "router_conf": round(router_conf, 4),
    }
