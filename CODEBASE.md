# Tài liệu mã nguồn — `api/core/` (runtime pipeline ALPR)

> Phạm vi: thư mục [api/core/](api/core/) — nơi chứa **pipeline chạy thật** (backend AI) mà hệ thống Web gọi tới. Các thư mục huấn luyện (`ocr/`, `LPRNet/`), model OCR đề xuất ([LPRNet/lprnet/small_lpr_line_ctc.py](LPRNet/lprnet/small_lpr_line_ctc.py)) và scripts nằm ngoài phạm vi file này (có thể lập tài liệu riêng nếu cần).

---

## 1. Kiến trúc tổng quan

Pipeline lõi là **3 tầng song song bằng threading** ([pipeline_async.py](api/core/pipeline_async.py)). PyTorch nhả GIL khi forward trên CUDA nên 2 luồng GPU chạy song song thật.

```
FrameSource ──▶ [Stage 1: Reader] ──frame_q(32)──▶ [Stage 2: Vehicle] ──crop_q(16)──▶ [Stage 3: Plate+OCR] ──▶ emit(event)
                 I/O đọc khung          YOLOv5 detect + BoT-SORT track       cascade OBB → quality router → OCR → CTM
```

Trạng thái dùng chung giữa các tầng (mỗi phiên khởi tạo mới): `WebTrackletManager` (bộ đệm theo vehicle track) và `vehicle_tracker` (BoT-SORT+ReID). Model (YOLOv5, YOLOv8-OBB, quality router, OCR) nạp **một lần**, dùng chung.

---

## 2. Các workflow và call-graph

### Workflow A — Phân tích toàn bộ video tải lên (Full-Video)

```
POST /upload  (hoặc /upload/chunk → /upload/complete cho file lớn)
  main.py: nhận file, kiểm tra, xin _job_semaphore (tối đa MAX_CONCURRENT_JOBS=2)
    └─ pipeline.py: run_job(...)                       # wrapper vào lõi
         └─ pipeline_async.py: process_frames_async(source=FileFrameSource, emit, models)
              ├─ Stage1 _reader_worker  → frame_q
              ├─ Stage2 _vehicle_worker → models.vehicle.predict → vehicle_tracker.track → crop_q
              └─ Stage3 _plate_ocr_worker
                    ├─ detect_plates_cascade()             # cascade_plate.py, gán trực tiếp vehicle ID
                    ├─ prepare_route_ocr_jobs()            # route_ocr.py → quality_router.route()
                    ├─ ocr_batch()                         # models.py
                    ├─ consume_route_ocr_results()         # route_ocr.py → tracker.buffer_crop / update
                    └─ (track mất dấu) finalise_track_ocr()# track_ocr.py
         → emit(...) → main.py đẩy SSE (/stream/{job_id}) + record_save() lưu Mongo/Supabase
```

### Workflow B — Phân tích trích đoạn từ luồng/monitor (Clip / mark→analysis)

```
Đăng ký nguồn RTSP  → live_session.LiveSession.start()
   └─ mediamtx_client.add_path(path, rtsp_url)          # MediaMTX kéo camera, transport TCP
   └─ _decoder_loop(): cv2.VideoCapture(rtsp://localhost:8554/path)
        → ring buffer 10s (deque)  +  hàng đợi MJPEG preview
Người dùng đánh dấu sự kiện (routes_monitor)
   └─ event_analyzer.run_event(...)                     # chạy trong thread-pool
        ├─ live_session.snapshot_window(seconds)         # lấy N giây cuối từ ring buffer
        └─ process_frames_async(source=LiveBufferFrameSource, ...)   # ĐÚNG lõi Workflow A
```

### Workflow C — Xem trực tiếp (Live preview)

```
LiveSession._decoder_loop → _enqueue_mjpeg (downscale+JPEG, drop khi consumer chậm)
   → main.py GET /stream/{job_id}/mjpeg  (hoặc MediaMTX WebRTC/HLS trực tiếp)
```

### Chi tiết `process_frames_async` (hàm lõi, [pipeline_async.py:420](api/core/pipeline_async.py#L420))

| Tầng     | Hàm                    | Gọi tới                                                                                                                                                                                                                                             |
| -------- | ---------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Khởi tạo | `process_frames_async` | `WebTrackletManager`, `models.create_vehicle_tracker`                                                                                                                                                                                               |
| 1        | `_reader_worker`       | `source.iter_frames()`                                                                                                                                                                                                                              |
| 2        | `_vehicle_worker`      | `apply_preprocessing`, `models.vehicle.predict`, `vehicle_tracker.track`, `tracker.reset_lost`                                                                                                                                                      |
| 3        | `_plate_ocr_worker`    | `detect_plates_cascade`, `_crop_vehicle`, `prepare_route_ocr_jobs`, `select_ocr_model`, `preprocess_plate_for_model`, `ocr_batch`, `consume_route_ocr_results`, `make_preview_frame_event`, `_finalise_track_ocr`                                  |
| Kết thúc | (sau join)             | `_finalise_track_ocr` cho track còn lại → emit "vehicle" `final`                                                                                                                                                                                    |

---

## 3. Tài liệu theo nhóm chức năng

### 3.1 Điều phối pipeline

**[pipeline_async.py](api/core/pipeline_async.py)** — Pipeline 3 tầng song song (lõi thật).

- `process_frames_async(source, emit, models, ...)` — điểm vào công khai; dựng trạng thái + 3 thread, join, finalize.
- `_reader_worker(source, frame_q, ...)` — Stage 1: đọc khung → `frame_q` (đo `s1_put_stall`).
- `_vehicle_worker(frame_q, crop_q, ...)` — Stage 2: detect xe (YOLOv5) + track (BoT-SORT) → `crop_q`. **Phải đơn luồng** (BoT-SORT cần thứ tự khung).
- `_plate_ocr_worker(crop_q, ...)` — Stage 3: cascade OBB → gán vehicle ID → quality routing → OCR → cập nhật buffer/emit; finalize track mất dấu.
- `_finalise_track_ocr(...)` — wrapper gọi [track_ocr.finalise_track_ocr](api/core/track_ocr.py).
- `_fps_stride`, `_safe_put` — tiện ích.

**[pipeline.py](api/core/pipeline.py)** — Job xử lý video (chạy trong thread-pool), tầng nối với DB.

- `run_job(video_path, job_id, queue, loop, models, ...)` — điểm vào "tải-và-xử-lý-cả-video"; wrapper mỏng quanh `process_frames_async`.
- `_session_create / _session_update` — tạo/cập nhật document phiên trong MongoDB (status, counters).
- `_record_save / _record_save_later` — dựng `RecognitionRecord` từ track đã chốt, lưu ảnh bằng chứng (fire-and-forget, không chặn worker).

**[event_analyzer.py](api/core/event_analyzer.py)** — Điều phối một job mark→analysis (Workflow B).

- `run_event()` — điểm vào thread-pool cho một sự kiện: lấy `snapshot_window` → `process_frames_async`.
- `_persist_event()` — dựng `MonitorEvent`, lưu bất đồng bộ.

**[frame_source.py](api/core/frame_source.py)** — Trừu tượng hoá nguồn khung.

- `FrameSource` (Protocol) — giao diện `iter_frames()` trả `(idx, frame, ts)`.
- `FileFrameSource` — đọc từ file video (Workflow A).
- `LiveBufferFrameSource` — bọc danh sách khung đã decode sẵn (Workflow B).
- `AdaptiveFrameSource` — **lấy mẫu (striding) + resize** trong khi giữ nguyên frame-id gốc. `_sample_stride(src_fps, target_fps)` = cơ chế bỏ khung.

### 3.2 Phát hiện phương tiện & theo vết

**[models.py](api/core/models.py)** — Nạp model, tiền xử lý, OCR batch. (Xem thêm 3.5 cho OCR.)

- `load_models()` — nạp toàn bộ trọng số một lần, trả `ModelBundle`.
- `ModelBundle.create_vehicle_tracker()` — tạo tracker BoT-SORT mới cho mỗi phiên (trạng thái độc lập).
- `load_yolov5_vehicle_detector(ckpt)` — nạp detector phương tiện (kế thừa Che et al.).

**[tracker.py](api/core/tracker.py)** — `WebTrackletManager` + `TrackBuffer` (bộ đệm bằng chứng theo track).

- `WebTrackletManager.should_ocr(tid)` — track này còn cần OCR không.
- `.buffer_crop(tid, crop, quality_score, ocr_conf, char_probs, frame_idx, route, ...)` — đẩy một quan sát biển vào bộ đệm track.
- `.ready_for_track_ocr(tid)` — đủ điều kiện chốt: có ≥1 khung `direct`, **hoặc** ≥ `MIN_FRAMES_FOR_OCR (=2)` khung phân biệt.
- `.update(tid, char_probs, all_confident)` — cập nhật kết quả tốt nhất của track.
- `.mark_lost / .reset_lost` — đếm khung mất dấu để quyết định finalize.
- `TrackBuffer.top_k / top_k_entries(k)` — xếp hạng theo `combined_score` (quality × ocr_conf).
- `_parse_plate_segments(text)` — tách vùng serial/số trong chuỗi biển.

**[tracker_adapter.py](api/core/tracker_adapter.py)** — Adapter khởi tạo BoT-SORT với ReID.

- Khởi tạo `AlwaysReIDBotSort(reid_model=vehicle_reid.onnx, proximity_thresh=0.5, appearance_thresh=0.25, ...)`; hàm `track(dets, img)` trả `(boxes, ids, classes)`.

**[botsort_reid.py](api/core/botsort_reid.py)** — Biến thể BoT-SORT: `AlwaysReIDBotSort`.

- `_update_impl` — dùng đặc trưng ReID ở **cả hai** lượt ghép cặp (mặc định BotSort chỉ lượt conf cao) để giữ ID khi conf tụt.
- `_confidence_masks / _split_detections / _second_association` — tách detection theo ngưỡng conf, ghép lượt hai bằng ReID.

### 3.3 Phát hiện biển & ghép cặp

**[cascade_plate.py](api/core/cascade_plate.py)** — Cascade: cắt xe rồi tìm biển trong crop.

- `detect_plates_cascade(frame, tracked, plate_model)` — điểm vào: chạy YOLOv8-OBB trên từng crop xe, khử trùng và trả ứng viên mang trực tiếp vehicle ID.
- `crop_vehicle_regions(frame, tracked)` / `expand_vehicle_box(...)` — cắt vùng xe, **nới biên 8%**.
- `map_crop_points_to_global(points, offset)` — ánh xạ OBB từ toạ độ crop về khung gốc.
- `deduplicate_plate_candidates(candidates, tracked)` — khử biển trùng do crop chồng lấp (IoU + ưu tiên chủ sở hữu nhất quán).
- `_best_containing_vehicle_id(box, tracked)` — chọn xe chứa tâm biển: ưu tiên **không sát mép, rồi diện tích nhỏ nhất** (tiebreak).
- `_extract_obb_candidates(...)` — lọc OBB theo `PLATE_DET_CONF`, `MIN_PLATE_W/H`; warp lấy crop.

Plate không còn được track độc lập. `deduplicate_plate_candidates` chọn vehicle chứa tâm phù hợp và đặt cả `source_vehicle_id` lẫn `id` bằng vehicle track ID ngay trong khung hiện tại.

**[video_processor.py](api/core/video_processor.py)** — Tiện ích ảnh: `crop_vehicle`, `warp_plate_crop` (nắn phối cảnh 4 điểm OBB).

### 3.4 Chất lượng & định tuyến

**[quality_router.py](api/core/quality_router.py)** — Định tuyến crop biển (Mục 3.2.2 đồ án).

- `PlateQualityRouter.route(crop_bgr)` — điểm vào: trả `PlateQualityResult(legibility, route, ...)`.
    - Ánh xạ nhánh: `perfect/good → direct`, `poor → tracklet_fusion`, `illegible|occluded → unreadable_wait`.
- `_predict_scores / _predict_ultralytics` — chạy YOLOv8n-cls (4 lớp).
- `diagnose_degradation(crop)` — gắn cờ suy giảm (mờ/tối/lóa/che...) bằng xử lý ảnh; fallback khi không có deep-net.
- `_heuristic_legibility(q, tags)` — suy luận nhãn từ q-score khi deep-net tắt.

**[quality_scorer.py](api/core/quality_scorer.py)** — `quality_score(crop)` ∈ [0,1] = 0.80·(Var Laplacian/`LAP_MAX=500`) + 0.20·(diện tích). Dùng để **xếp hạng buffer**, không quyết định định tuyến.

**[gates.py](api/core/gates.py)** — Bộ lọc thô trước OCR.

- `is_sharp(crop, threshold)` — cổng cứng loại crop nhoè/mất nét.
- `is_router_candidate(crop)` — crop có đáng đưa vào router không.

### 3.5 OCR (nhận dạng ký tự)

**[models.py](api/core/models.py)** — nhiều backend OCR sau một giao diện chung.

- `select_ocr_model(models, ocr_backend)` — chọn backend theo cấu hình.
- `ocr_batch(model, images, device)` — dispatch tới hàm decode đúng backend.
- Backend & wrapper: `SmallLprLineCtcOcrModel` + `small_lpr_line_ctc_ocr_batch` (**mô hình đề xuất**), `SmallLprCtcOcrModel`, `ParseqOcrModel` (`parseq_ocr_batch`), YOLOv5-char (xem 3.8).
- Tiền xử lý: `preprocess_plate_for_model`, `preprocess_plate_small_lpr` (chuẩn hoá [-1,1]), `preprocess_plate_parseq`.
- `_ctc_logits_to_char_probs(logits, chars)` — greedy CTC decode → danh sách `(ký tự, xác suất)`.

**[route_ocr.py](api/core/route_ocr.py)** — Điều phối OCR theo nhánh định tuyến, mỗi khung.

- `prepare_route_ocr_jobs(matched, tracker, router, frame_idx)` — gọi `router.route()` cho từng biển, dựng danh sách job OCR (chỉ nhánh cần OCR ngay).
- `consume_route_ocr_results(jobs, ocr_results, tracker, emit, ...)` — nhận kết quả OCR → `_accept_single_frame` (nếu direct + đủ tin cậy) hoặc `tracker.buffer_crop` (đưa vào bộ đệm chờ CTM).
- `_all_chars_confident(char_probs)` — mọi ký tự vượt ngưỡng?

**[ocr_candidates.py](api/core/ocr_candidates.py)** — Sinh & xếp hạng ứng viên OCR cho crop suy giảm.

- `build_candidate_crops(crop, tags)` — tạo các phiên bản tăng cường: `_unsharp` (khử mờ), `_clahe_luminance`/`_gamma`/`_stretch_luminance` (tương phản), `_gray_world` (cân bằng màu).
- `rerank_ocr_candidates(candidates)` — chọn ứng viên tốt nhất theo xác suất ký tự + chi phí sửa.

**[track_ocr.py](api/core/track_ocr.py)** — **Chốt kết quả cấp track** (nơi CTM chạy).

- `finalise_track_ocr(tid, tracker, models, emit, ...)` — điểm vào khi track kết thúc: OCR các crop suy giảm còn tồn (`_entries_with_deferred_ocr`) → **phân cụm** ([ocr_cluster](api/core/ocr_cluster.py)) → **hợp nhất CTM** ([ocr_ctm](api/core/ocr_ctm.py)) trên cụm lớn nhất → xác thực định dạng → emit "vehicle" hoặc `_emit_rejected`.
- `_build_cluster_data(...)` — dựng dữ liệu cụm cho SSE/UI (gỡ lỗi ID-switch).
- `_store_best_plate_image(...)` — chọn ảnh biển đại diện lưu bằng chứng.

### 3.6 Hậu xử lý: CTM, phân cụm, sửa lỗi, định dạng

**[ocr_ctm.py](api/core/ocr_ctm.py)** — Character Time-series Matching cấp chuỗi (Mục 3.3.2).

- `fuse_ocr_outputs_ctm(prob_lists, min_support_ratio=0.5, min_confidence=0.5, format_mode)` — điểm vào: căn từng chuỗi OCR vào template bằng DP → **bầu chọn per-slot** → trả `CTMFusionResult` (kèm slot vô định `?`).
- `PlateTemplate` — biểu diễn mẫu biển; `.render(slot_chars)` khôi phục dấu phân cách (conf cố định 0.90).
- `_choose_template / _dominant_template` — chọn mẫu khớp nhất theo chi phí căn.
- `_align_chars_to_template(chars, template)` — DP align (align/missing/skip) + truy vết.
- `_align_token_cost / _missing_template_token_cost / _skip_input_token_cost` — bộ chi phí căn chỉnh.

**[ocr_cluster.py](api/core/ocr_cluster.py)** — Tách nhiều biển trong một track (ID-switch).

- `cluster_ocr_results(scored_entries, max_clusters=3, similarity_threshold=0.6)` — gom theo tương đồng Levenshtein.
- `_levenshtein / _similarity / _text_fingerprint` — độ tương đồng chuỗi (bỏ dấu phân cách).

**[ocr_ambiguity.py](api/core/ocr_ambiguity.py)** — Sửa ký tự mơ hồ slot-aware, đơn khung (Mục 3.3.1).

- `correct_ambiguous_chars(char_probs, format_mode)` — căn chuỗi vào template, thay ký tự mơ hồ theo `DIGIT_TO_LETTER = {0/O,1/I,2/Z,5/S,6/G,8/B}` khi làm slot hợp lệ; giảm xác suất ký tự bị thay theo `CORRECTED_CONF_SCALE=0.92`.

**[plate_format.py](api/core/plate_format.py)** — Chuẩn hoá & xác thực định dạng biển VN.

- `VN_PLATE_TEMPLATE_PATTERNS` — ~37 mẫu định dạng.
- `is_vn_plate_text(text, format_mode)` — validate bằng regex ghép từ danh sách mẫu (`raw` giữ dấu / `alnum` chỉ chữ-số).
- `normalize_plate_text / chars_to_text / display_plate_text` — chuẩn hoá & hiển thị.

### 3.7 Streaming (MediaMTX / RTSP)

**[live_session.py](api/core/live_session.py)** — Một phiên giám sát: decoder RTSP + ring buffer.

- `LiveSession.start(rtsp_url, mjpeg_queue, on_error)` — đăng ký path MediaMTX + spawn thread decode.
- `_decoder_loop()` — `cv2.VideoCapture(rtsp://localhost:8554/path)` → `_push_frame` vào deque `maxlen=fps×10s`; có reconnect/backoff.
- `snapshot_window(seconds)` — trả N giây cuối (cấp cho Workflow B).
- `_enqueue_mjpeg(frame)` — downscale + JPEG cho preview, **bỏ khung khi consumer chậm** (giữ độ trễ chặn).
- `internal_mediamtx_path(rtsp_url)` — nhận diện path nội bộ.

**[mediamtx_client.py](api/core/mediamtx_client.py)** — Client HTTP điều khiển MediaMTX.

- `add_path(name, source)` — đăng ký path kéo từ RTSP nguồn, **`rtspTransport=tcp`**.
- `remove_path(name)` — gỡ path (idempotent, bỏ qua 404).

### 3.8 Backend OCR/pipeline thay thế (tham chiếu/so sánh)

- **[ocr_yolov5.py](api/core/ocr_yolov5.py)** — OCR kiểu **phát hiện ký tự** (YOLOv5 char) — tái hiện hướng Che et al. `find_chars_plate_probs`, `estimate_coef` (hồi quy góc sắp thứ tự).
- **[pipeline_yolov5_vietnamese.py](api/core/pipeline_yolov5_vietnamese.py)** — pipeline tham chiếu dùng char-detection + xoay khử nghiêng (`rotate_plate_crop`, `update_rotation_alpha`, `get_final_plate_text`). Backend tùy chọn này vẫn dùng ByteTrack riêng cho biển và có `plate_track_id`; việc bỏ plate tracker/association voting chỉ áp dụng pipeline mặc định.

### 3.9 Hạ tầng Web & tiện ích

| File                                                    | Vai trò                                 | Hàm/đối tượng chính                                                                                                              |
| ------------------------------------------------------- | --------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------- |
| [main.py](api/main.py)                                  | FastAPI app, route, SSE, semaphore GPU  | `/upload`, `/upload/chunk`, `/upload/complete`, `/stream/{job}`, `/stream/{job}/mjpeg`, `/sessions/...`; `MAX_CONCURRENT_JOBS=2` |
| [routes_monitor.py](api/routes_monitor.py)              | Route giám sát/streaming & mark sự kiện | quản lý phiên monitor, gọi `event_analyzer.run_event`                                                                            |
| [config.py](api/core/config.py)                         | Toàn bộ hằng số/đường dẫn/siêu tham số  | `FRAME_STRIDE`, `VEHICLE_CLASSES`, `LAP_MAX`, `MIN_PLATE_W/H`, `MIN_FRAMES_FOR_OCR`, `REID_MODEL_PATH`                           |
| [database.py](api/core/database.py)                     | MongoDB + Supabase                      | `get_supabase`, `upload_image(bucket, path, bytes)`                                                                              |
| [chunk_upload.py](api/core/chunk_upload.py)             | Upload phân mảnh (vượt giới hạn proxy)  | `ChunkUploadStore.begin_or_get / write_chunk / assemble_into`                                                                    |
| [progress.py](api/core/progress.py)                     | Sự kiện tiến độ SSE                     | `make_progress_event`                                                                                                            |
| [preview_frame.py](api/core/preview_frame.py)           | Khung preview + hộp bao cho UI          | `make_preview_frame_event(frame, boxes)`                                                                                         |
| [preprocessing.py](api/core/preprocessing.py)           | Tiền xử lý khung (đêm/mưa/lóa/sương...) | `apply_preprocessing(frame, mode)`                                                                                               |
| [preprocessed_video.py](api/core/preprocessed_video.py) | Ghi & phục vụ video đã tiền xử lý       | `RecordingFrameSource`, cleanup TTL                                                                                              |

---

## 4. Bảng tra nhanh "chức năng → file"

| Muốn xem...                  | Vào file                                                                                                              |
| ---------------------------- | --------------------------------------------------------------------------------------------------------------------- |
| Luồng 3 tầng chạy thật       | [pipeline_async.py](api/core/pipeline_async.py)                                                                       |
| Chốt track + CTM + cluster   | [track_ocr.py](api/core/track_ocr.py) → [ocr_ctm.py](api/core/ocr_ctm.py) + [ocr_cluster.py](api/core/ocr_cluster.py) |
| Gán biển vào vehicle track   | [cascade_plate.py](api/core/cascade_plate.py)                                                                         |
| Định tuyến chất lượng        | [quality_router.py](api/core/quality_router.py) + [quality_scorer.py](api/core/quality_scorer.py)                     |
| Xác thực/sửa định dạng biển  | [plate_format.py](api/core/plate_format.py) + [ocr_ambiguity.py](api/core/ocr_ambiguity.py)                           |
| Nạp model & OCR backend      | [models.py](api/core/models.py)                                                                                       |
| Theo vết BoT-SORT + ReID     | [tracker_adapter.py](api/core/tracker_adapter.py) + [botsort_reid.py](api/core/botsort_reid.py)                       |
| Bộ đệm bằng chứng theo track | [tracker.py](api/core/tracker.py)                                                                                     |
| Streaming RTSP/MediaMTX      | [live_session.py](api/core/live_session.py) + [mediamtx_client.py](api/core/mediamtx_client.py)                       |
| Route Web/SSE                | [main.py](api/main.py) + [routes_monitor.py](api/routes_monitor.py)                                                   |

---

_Sinh tự động từ đọc mã nguồn `api/core/` — kiểm lại khi code đổi. Model OCR đề xuất và training không thuộc phạm vi file này._
