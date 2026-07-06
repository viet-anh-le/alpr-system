import re

with open("api/core/pipeline_async.py", "r") as f:
    content = f.read()

# 1. Imports
content = content.replace(
    "from .association import TrajectoryAssociator",
    "from .tracker_adapter import PlateTracker\nimport collections"
)

# 2. Worker Args
content = content.replace(
    "    tracker: WebTrackletManager,\n    associator: TrajectoryAssociator,",
    "    tracker: WebTrackletManager,\n    plate_tracker: PlateTracker,\n    plate_votes: dict[int, list[int]],"
)

# 3. Inside worker, replace Handle lost tracks (incremental finalization)
lost_tracks_block = """            # ── Handle lost tracks (tracks present before but missing now) ────
            # Only finalise on stride frames to keep timing consistent.
            if frame_idx % FRAME_STRIDE == 0:
                for tid in previously_tracked - currently_tracked:
                    if (
                        tracker.should_ocr(tid)
                        and tracker.mark_lost(tid)
                        and tracker.ready_for_track_ocr(tid)
                    ):
                        _finalise_track_ocr(
                            tid,
                            tracker,
                            models,
                            emit,
                            session_id,
                            loop,
                            record_save,
                            ocr_backend,
                            user_id,
                        )"""

content = content.replace(lost_tracks_block, """            # Incremental finalization is disabled for Deferred Frequency Matrix
            pass""")


# 4. Inside worker, replace Plate tracking and association
association_block = """            # ── Cascade plate detection ───────────────────────────────────────
            active_tids: set[int] = set()
            tracked_for_ocr = [v for v in tracked if tracker.should_ocr(int(v["id"]))]
            plate_tracks = detect_plate_tracks_cascade(
                frame, tracked_for_ocr, models.plate, timings=timings
            )

            stage_start = time.perf_counter()
            firm_matches = associator.process_frame(plate_tracks, tracked_for_ocr)
            _add_timing("association", stage_start)

            matched: list[tuple[int, np.ndarray, np.ndarray]] = []
            for v_tid, p in firm_matches:
                v_box = associator.vehicle_cache.get(v_tid)
                if v_box is not None:
                    vehicle_crop = _crop_vehicle(frame, v_box)
                    matched.append((v_tid, p["crop"], vehicle_crop))"""

new_association_block = """            # ── Cascade plate detection ───────────────────────────────────────
            active_tids: set[int] = set()
            tracked_for_ocr = [v for v in tracked if tracker.should_ocr(int(v["id"]))]
            plate_candidates = detect_plate_tracks_cascade(
                frame, tracked_for_ocr, models.plate, timings=timings
            )

            stage_start = time.perf_counter()
            
            # Convert plate_candidates to ByteTrack dets format: [x1, y1, x2, y2, conf, cls]
            if plate_candidates:
                dets = np.array([
                    [c["box"][0], c["box"][1], c["box"][2], c["box"][3], c["conf"], 0] 
                    for c in plate_candidates
                ], dtype=np.float32)
            else:
                dets = np.zeros((0, 6), dtype=np.float32)
                
            p_boxes, p_ids, p_classes = plate_tracker.track(dets, frame)
            _add_timing("association", stage_start)

            visible_vehicles = {int(v["id"]): tuple(int(c) for c in v["box"]) for v in tracked_for_ocr}
            
            matched: list[tuple[int, np.ndarray, np.ndarray]] = []
            
            # Map tracking results back to candidates to retrieve crops and source_vehicle_ids
            for p_box, p_tid in zip(p_boxes, p_ids):
                p_tid = int(p_tid)
                # Find the candidate that corresponds to this tracked box (via IoU or center distance)
                # Using simple center distance matching for speed
                cx, cy = (p_box[0] + p_box[2]) / 2, (p_box[1] + p_box[3]) / 2
                best_cand = None
                best_dist = float('inf')
                for cand in plate_candidates:
                    ccx, ccy = (cand["box"][0] + cand["box"][2]) / 2, (cand["box"][1] + cand["box"][3]) / 2
                    dist = (cx - ccx)**2 + (cy - ccy)**2
                    if dist < best_dist:
                        best_dist = dist
                        best_cand = cand
                
                if best_cand is not None and best_dist < 1000: # Threshold to avoid crazy matches
                    # Accumulate votes
                    source_vids = best_cand.get("source_vehicle_ids", [])
                    plate_votes[p_tid].extend(source_vids)
                    
                    # Prepare OCR matched job using p_tid instead of v_tid
                    # Pick the first available vehicle box for the vehicle crop (used for routing context)
                    v_box = visible_vehicles.get(source_vids[0]) if source_vids else None
                    if v_box is not None:
                        vehicle_crop = _crop_vehicle(frame, v_box)
                        # We also pass source_vids along in the matched tuple so route_ocr can emit direct routes immediately
                        matched.append((p_tid, best_cand["crop"], vehicle_crop, source_vids))
"""

content = content.replace(association_block, new_association_block)


# 5. We need to modify route_ocr prepare_route_ocr_jobs so it accepts source_vids.
# BUT wait, it's better to just pass the p_tid to prepare_route_ocr_jobs and then handle source_vids when emitting.
# prepare_route_ocr_jobs takes `matched: list[tuple[int, np.ndarray, np.ndarray]]`.
# We changed it to a 4-tuple. Let's patch prepare_route_ocr_jobs in api/core/route_ocr.py separately.
# Or we can just attach source_vids to the tracker for this p_tid, but that's messy.
# Let's see how consume_route_ocr_results works.


# 6. Public entry point
main_block = """    # Shared state
    tracker = WebTrackletManager()
    associator = TrajectoryAssociator(
        match_frames=ASSOCIATION_MATCH_FRAMES,
        agreement_ratio=ASSOCIATION_AGREEMENT_RATIO,
    )
    vehicle_tracker = models.create_vehicle_tracker()"""

new_main_block = """    # Shared state
    tracker = WebTrackletManager()
    plate_tracker = PlateTracker()
    plate_votes: dict[int, list[int]] = collections.defaultdict(list)
    vehicle_tracker = models.create_vehicle_tracker()"""

content = content.replace(main_block, new_main_block)

thread_args_block = """            crop_q,
            models,
            tracker,
            associator,
            emit,
            session_id,"""
            
new_thread_args_block = """            crop_q,
            models,
            tracker,
            plate_tracker,
            plate_votes,
            emit,
            session_id,"""
content = content.replace(thread_args_block, new_thread_args_block)


# 7. Finalization
finalization_block = """    for tid in list(tracker._buffers):
        if tracker.should_ocr(tid) and tracker.ready_for_track_ocr(tid):
            _finalise_track_ocr(
                tid, tracker, models, emit, session_id, loop, record_save, ocr_backend,
                user_id
            )"""

new_finalization_block = """    # Resolve Frequency Matrix
    plate_to_vehicle = {}
    for p_tid, v_tids in plate_votes.items():
        if v_tids:
            best_v_tid = collections.Counter(v_tids).most_common(1)[0][0]
            plate_to_vehicle[p_tid] = best_v_tid

    # Group plates by vehicle_id
    vehicle_to_plates = collections.defaultdict(list)
    for p_tid, v_tid in plate_to_vehicle.items():
        vehicle_to_plates[v_tid].append(p_tid)

    # Perform CTM vote across all plate tracks belonging to the same vehicle
    for v_tid, p_tids in vehicle_to_plates.items():
        if not tracker.should_ocr(v_tid):
            continue
            
        combined_entries = []
        for p_tid in p_tids:
            buf = tracker._buffers.get(p_tid)
            if buf:
                combined_entries.extend(buf.top_k_entries(buf.max_size))
                
        if not combined_entries:
            continue
            
        # Sort by combined score descending and take TOP_K_FRAMES
        from .config import TOP_K_FRAMES
        combined_entries = sorted(
            combined_entries,
            key=lambda e: (
                1 if e.router_result.get("legibility") in ("perfect", "good") else 0,
                e.combined_score
            ),
            reverse=True
        )[:TOP_K_FRAMES]
        
        # Merge best char_probs using CTM
        char_prob_lists = [e.char_probs for e in combined_entries if e.char_probs]
        if char_prob_lists:
            merged_chars = tracker._segment_vote(char_prob_lists)
            if merged_chars is None:
                merged_chars = tracker._prob_vote(char_prob_lists)
                
            if merged_chars:
                tracker._best[v_tid] = merged_chars
                tracker._done[v_tid] = True
                
                # Update evidence using the top entry
                top_entry = combined_entries[0]
                tracker.set_plate_img(v_tid, top_entry.crop, top_entry.ocr_conf)
                
                # Emit the final result
                event = {
                    "type": "plate",
                    "id": v_tid,
                    "plate": tracker.display_text(v_tid),
                    "chars": tracker.chars_json(v_tid),
                    "confidence": float(top_entry.ocr_conf),
                    "source": top_entry.candidate_method,
                    "route": "deferred_frequency_fusion",
                    **top_entry.router_result,
                }
                emit(event)"""

content = content.replace(finalization_block, new_finalization_block)


with open("api/core/pipeline_async.py", "w") as f:
    f.write(content)

print("Patch applied to pipeline_async.py")
