import re

with open("api/core/route_ocr.py", "r") as f:
    content = f.read()

# 1. Update RouteOcrJob to include source_vehicle_ids
job_def_block = """@dataclass(frozen=True)
class RouteOcrJob:
    tid: int
    plate_crop: np.ndarray
    vehicle_crop: np.ndarray
    candidate_crop: np.ndarray
    candidate_method: str
    frame_idx: int
    quality: PlateQualityResult"""

new_job_def_block = """@dataclass(frozen=True)
class RouteOcrJob:
    tid: int
    plate_crop: np.ndarray
    vehicle_crop: np.ndarray
    candidate_crop: np.ndarray
    candidate_method: str
    frame_idx: int
    quality: PlateQualityResult
    source_vehicle_ids: list[int]"""

content = content.replace(job_def_block, new_job_def_block)

# 2. Update prepare_route_ocr_jobs signature and loop
prepare_block = """def prepare_route_ocr_jobs(
    matched: list[tuple[int, np.ndarray, np.ndarray]],
    tracker: WebTrackletManager,
    router: PlateQualityRouter,
    frame_idx: int,
) -> tuple[list[RouteOcrJob], set[int]]:
    jobs: list[RouteOcrJob] = []
    active_tids: set[int] = set()

    for tid, plate_crop, vehicle_crop in matched:"""

new_prepare_block = """def prepare_route_ocr_jobs(
    matched: list[tuple[int, np.ndarray, np.ndarray, list[int]]],
    tracker: WebTrackletManager,
    router: PlateQualityRouter,
    frame_idx: int,
) -> tuple[list[RouteOcrJob], set[int]]:
    jobs: list[RouteOcrJob] = []
    active_tids: set[int] = set()

    for tid, plate_crop, vehicle_crop, source_vids in matched:"""

content = content.replace(prepare_block, new_prepare_block)


# 3. Update jobs.append to include source_vehicle_ids
append_block = """        jobs.append(RouteOcrJob(
            tid=tid,
            plate_crop=plate_crop,
            vehicle_crop=vehicle_crop,
            candidate_crop=plate_crop,
            candidate_method="original",
            frame_idx=frame_idx,
            quality=quality,
        ))"""

new_append_block = """        jobs.append(RouteOcrJob(
            tid=tid,
            plate_crop=plate_crop,
            vehicle_crop=vehicle_crop,
            candidate_crop=plate_crop,
            candidate_method="original",
            frame_idx=frame_idx,
            quality=quality,
            source_vehicle_ids=source_vids,
        ))"""

content = content.replace(append_block, new_append_block)


# 4. Update _accept_single_frame emit loop
accept_block = """    vote_summary = {best.text: 1} if best.text else {}
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
        )"""

new_accept_block = """    vote_summary = {best.text: 1} if best.text else {}
    if tracker.plate_changed(tid):
        for v_tid in job.source_vehicle_ids:
            emit(
                {
                    "type": "vehicle",
                    "id": v_tid,
                    "cls": tracker._cls.get(v_tid, ""),
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
            )"""

content = content.replace(accept_block, new_accept_block)

with open("api/core/route_ocr.py", "w") as f:
    f.write(content)

print("Patch applied to route_ocr.py")
