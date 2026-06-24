# ALPR Vietnamese — Tài liệu Pipeline Kỹ thuật

> Tài liệu này mô tả toàn bộ pipeline xử lý biển số xe (ALPR) cho hai luồng đầu vào: **Upload video** và **Live RTSP stream**.

---

## 1. Tổng quan kiến trúc

```
┌─────────────────────────────────────────────────────────────────────┐
│                          ALPR Pipeline                              │
│                                                                     │
│  FrameSource                                                        │
│  (FileFrameSource │ LiveBufferFrameSource)                          │
│         │                                                           │
│         ▼                                                           │
│   ┌─────────────┐    ┌───────────────┐    ┌─────────────────────┐  │
│   │  Detection  │───▶│   Tracking    │───▶│  OCR + Voting       │  │
│   │             │    │               │    │                     │  │
│   │ vehicle YOLO│    │ BotSort+ReID  │    │ SmallLPR            │  │
│   │ plate OBB   │    │ ByteTrack OBB │    │ + _segment_vote /   │  │
│   │ YOLO        │    │ TrajectoryAss │    │   _prob_vote        │  │
│   └─────────────┘    └───────────────┘    │                     │  │
│                                           └─────────────────────┘  │
│                                                   │                 │
│                                                   ▼                 │
│                                          SSE Events → Frontend      │
└─────────────────────────────────────────────────────────────────────┘
```

**Điểm vào code:** [`api/core/pipeline_core.py:process_frames()`](../api/core/pipeline_core.py#L147)

---

## 2. Hai luồng đầu vào

### 2.1 Upload Video

```
Browser
  │
  ├─ POST /upload (multipart video file)
  │       │
  │       ▼  api/main.py:64
  │  Lưu file tạm → asyncio.Queue → run_in_executor(run_job)
  │
  ├─ GET /stream/{job_id}         ← SSE events (progress/vehicle/complete)
  └─ GET /stream/{job_id}/mjpeg   ← MJPEG annotated frames (realtime)
```

**Code thực thi:**
- Route: [`api/main.py:64-82`](../api/main.py#L64) — nhận file, tạo temp, gọi `run_job`
- Job runner: [`api/core/pipeline.py:run_job()`](../api/core/pipeline.py) — wrapper tạo `FileFrameSource` rồi gọi `process_frames_async`
- Frame source: [`api/core/frame_source.py:FileFrameSource`](../api/core/frame_source.py#L29) — mở video bằng `cv2.VideoCapture`, seek đến `t_start`, yield `(file_pos, frame, timestamp)`

### 2.2 Live RTSP Stream (Event Monitor)

```
Browser
  │
  ├─ POST /monitor/live/connect  { rtsp_url }
  │       │
  │       ▼  api/routes_monitor.py:107
  │  LiveSession.start()
  │   ├─ Đăng ký path với MediaMTX (HTTP API)
  │   └─ Spawn decoder thread (cv2.VideoCapture → RTSP qua MediaMTX)
  │         └─ Rolling buffer 10s (collections.deque, maxlen = fps×10)
  │
  ├─ GET /monitor/live/{sid}/mjpeg  ← MJPEG raw frames
  │   hoặc WebRTC (WHEP) qua MediaMTX
  │
  ├─ POST /monitor/{sid}/mark  { mode, t_start?, t_end? }
  │       │
  │       ▼  api/routes_monitor.py:235
  │  LiveBufferFrameSource(snapshot_window(10s))
  │   └─ _event_executor.submit(run_event) — single GPU worker
  │
  └─ GET /monitor/{sid}/events/stream  ← SSE events (event_*)
```

**Code thực thi:**
- Live connect: [`api/routes_monitor.py:107-135`](../api/routes_monitor.py#L107)
- LiveSession decoder: [`api/core/live_session.py:LiveSession._decoder_loop()`](../api/core/live_session.py#L80) — kết nối RTSP nội bộ qua MediaMTX, thử lại 3 lần nếu mất kết nối
- Mark dispatch: [`api/routes_monitor.py:189-232`](../api/routes_monitor.py#L189) — chụp snapshot buffer, tạo `LiveBufferFrameSource`
- Live frame source: [`api/core/frame_source.py:LiveBufferFrameSource`](../api/core/frame_source.py#L79) — wrap danh sách frame đã decode

---

## 3. Bước 1 — Detection (Phát hiện phương tiện và biển số)

### 3.1 Phát hiện phương tiện — YOLOv8 Custom

**Model:** `weights/detection/vehicle_best.pt` — YOLOv8 fine-tuned trên dữ liệu Việt Nam.

| Class ID | Tên | Nhãn hiển thị |
|----------|-----|---------------|
| 0 | car | Ô tô |
| 1 | bus | Xe buýt |
| 4 | truck | Xe tải |
| 5 | motorcycle | Xe máy |
| 15 | motorbike_rider | Xe máy |

> **Lưu ý quan trọng:** Class ID khác hoàn toàn với COCO chuẩn (car=2, motorcycle=3, bus=5, truck=7). Dùng sai model sẽ phát hiện nhầm lớp.

**Code thực thi:**
```python
# api/core/pipeline_core.py:176-184
v_pred = models.vehicle.predict(frame, classes=VEHICLE_CLASSES, verbose=False)[0]
if v_pred.boxes is not None and len(v_pred.boxes) > 0:
    xyxy = v_pred.boxes.xyxy.cpu().numpy()
    conf = v_pred.boxes.conf.cpu().numpy().reshape(-1, 1)
    cls  = v_pred.boxes.cls.cpu().numpy().reshape(-1, 1)
    dets = np.concatenate([xyxy, conf, cls], axis=1).astype(np.float32)
```
→ [`api/core/pipeline_core.py:176`](../api/core/pipeline_core.py#L176)

### 3.2 Phát hiện biển số — YOLOv8 OBB

**Model:** `weights/detection/best.pt` — YOLOv8 OBB (Oriented Bounding Box), phát hiện biển số với góc xoay.

Biển số được detect dưới dạng hộp nghiêng 4 điểm góc (OBB), sau đó chuyển về bounding box thẳng bằng `cv2.boundingRect`.

**Code thực thi:**
```python
# api/core/pipeline_core.py:217-248
p_res = models.plate.track(frame, persist=True, tracker=_PLATE_TRACKER_CFG, verbose=False)[0]

if p_res.obb is not None and p_res.obb.id is not None:
    obb_pts  = p_res.obb.xyxyxyxy.cpu().numpy().astype(int)   # 4 điểm góc OBB
    obb_conf = p_res.obb.conf.cpu().numpy()
    obb_ids  = p_res.obb.id.cpu().numpy().astype(int)

    for pts, det_conf, p_tid in zip(obb_pts, obb_conf, obb_ids):
        if float(det_conf) < PLATE_DET_CONF:   # Layer 1a: ngưỡng confidence ≥ 0.50
            continue
        raw_rx, raw_ry, raw_rw, raw_rh = cv2.boundingRect(pts)
        if raw_rw < MIN_PLATE_W or raw_rh < MIN_PLATE_H:  # Layer 1b: kích thước tối thiểu
            continue
        # Thêm padding PLATE_PAD=8px xung quanh
        plate_crop = frame[ry:ry+rh, rx:rx+rw]
        if not is_sharp(plate_crop):            # Layer 1c: độ sắc nét (Laplacian)
            continue
```
→ [`api/core/pipeline_core.py:217`](../api/core/pipeline_core.py#L217)

### 3.3 Ba lớp chống nhận diện sai (Anti-Hallucination — Layer 1)

| Lớp | Điều kiện | Ngưỡng | Code |
|-----|-----------|--------|------|
| 1a | YOLO OBB confidence | `PLATE_DET_CONF = 0.50` | `pipeline_core.py:229` |
| 1b | Kích thước biển tối thiểu | `MIN_PLATE_W=50px, MIN_PLATE_H=15px` | `pipeline_core.py:231` |
| 1c | Độ sắc nét (Laplacian variance) | `BLUR_THRESHOLD = 80.0` | [`gates.py:17`](../api/core/gates.py#L17) |

```python
# api/core/gates.py:17-28
def is_sharp(crop: np.ndarray, threshold: float = BLUR_THRESHOLD) -> bool:
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    if lap_var < threshold:
        return False
    return quality_score(crop) >= _HARD_GATE_MIN   # _HARD_GATE_MIN = 0.05
```

---

## 4. Bước 2 — Tracking (Theo dõi đối tượng)

Hệ thống sử dụng **hai tracker riêng biệt** cho phương tiện và biển số, sau đó dùng **TrajectoryAssociator** để ghép cặp chúng.

### 4.1 Vehicle Tracker — BotSort + Custom ReID

**Lý do không dùng Ultralytics BoT-SORT:** Ultralytics bọc mọi model ReID bằng `YOLO()` và áp dụng letterboxing 640×640, không tương thích với ReID tùy chỉnh 256×128 của dự án.

**Giải pháp:** Dùng `boxmot.BotSort` trực tiếp qua `VehicleTracker` adapter.

**ReID model:** `weights/tracking/vehicle_reid.onnx` — MobileNetV3-Small, input `(B, 3, 256, 128)`, được train trên dữ liệu phương tiện Việt Nam bằng metric/triplet loss.

**Hyperparameters BotSort:**

| Tham số | Giá trị | Ý nghĩa |
|---------|---------|---------|
| `track_high_thresh` | 0.6 | Ngưỡng detection cao cho track mới |
| `new_track_thresh` | 0.7 | Ngưỡng tạo track mới |
| `track_buffer` | 30 frames | Số frame giữ track khi mất |
| `match_thresh` | 0.8 | Ngưỡng match IoU+appearance |
| `appearance_thresh` | 0.25 | Ngưỡng khoảng cách ReID |

**Code thực thi:**
```python
# api/core/tracker_adapter.py:71-114
class VehicleTracker:
    def __init__(self, reid_weights, device, half):
        reid_model = ReID(path=self._reid_weights, device=self._device, half=half)
        self._tracker = BotSort(
            reid_model=reid_model.model,
            track_high_thresh=self._TRACK_HIGH_THRESH,
            ...
        )

    def track(self, dets: np.ndarray, frame: np.ndarray):
        # dets: (N, 6) = [x1, y1, x2, y2, conf, cls]
        result = self._tracker.update(dets.astype(np.float32), frame)
        # result: (M, 8) cols [x1,y1,x2,y2, id, conf, cls, det_ind]
        return boxes, ids, classes
```
→ [`api/core/tracker_adapter.py:125`](../api/core/tracker_adapter.py#L125)

**Gọi trong pipeline:**
```python
# api/core/pipeline_core.py:185
boxes, ids, classes = models.vehicle_tracker.track(dets, frame)
```

### 4.2 Plate Tracker — ByteTrack (Ultralytics)

Biển số được track bằng ByteTrack thông qua `model.plate.track(..., persist=True)`. Không cần ReID vì biển số có đặc trưng đủ ổn qua IoU.

**Config:** `configs/tracking/bytetrack_plate.yaml`

**Code thực thi:**
```python
# api/core/pipeline_core.py:217
p_res = models.plate.track(frame, persist=True, tracker=_PLATE_TRACKER_CFG, verbose=False)[0]
```

### 4.3 Trajectory Associator — Ghép cặp biển số với phương tiện

Thay vì match frame-by-frame (dễ bị nhiễu do occlusion), `TrajectoryAssociator` quan sát **5 frame liên tiếp** rồi vote để **khóa vĩnh viễn** (plate_track_id → vehicle_track_id).

**Thuật toán:**
1. Mỗi frame: Với mỗi biển số, tìm phương tiện nhỏ nhất chứa tâm biển số (Area Heuristic)
2. Tích lũy vote qua `match_frames=5` frame
3. Nếu ≥ 60% vote cùng một phương tiện → khóa association

```python
# api/core/association.py:33-93
def process_frame(self, plate_tracks, vehicle_tracks):
    for p in plate_tracks:
        if p_tid in self.plate_to_vehicle:   # Đã khóa → trả về luôn
            firm_matches.append((v_tid, p))
            continue

        # Area Heuristic: biển số nằm trong phương tiện nhỏ nhất
        cx, cy = center_of(p["box"])
        best_v_tid = smallest_vehicle_containing(cx, cy, vehicle_tracks)

        if best_v_tid:
            self.plate_votes[p_tid].append(best_v_tid)
            recent = plate_votes[-match_frames:]
            if count_most_common(recent) >= match_frames * agreement_ratio:
                self.plate_to_vehicle[p_tid] = most_common_v_tid  # Khóa!
```
→ [`api/core/association.py:33`](../api/core/association.py#L33)

**Gọi trong pipeline:**
```python
# api/core/pipeline_core.py:251
firm_matches = associator.process_frame(plate_tracks, tracked)
for v_tid, p in firm_matches:
    vehicle_crop = _crop_vehicle(frame, v_box)
    matched.append((v_tid, p["crop"], vehicle_crop))
```

---

## 5. Bước 3 — Track Buffer (Tích lũy frame chất lượng)

Trước khi chạy OCR, hệ thống tích lũy các crop biển số vào buffer theo track, evict frame tệ nhất khi đầy.

```python
# api/core/tracker.py:88-120
@dataclass
class TrackBuffer:
    crops:          list[np.ndarray]  # BGR crops
    quality_scores: list[float]       # điểm chất lượng [0,1]
    frame_indices:  list[int]

    def add(self, crop, quality_score, frame_idx):
        self.crops.append(crop)
        if len(self.crops) > MAX_BUFFER:     # MAX_BUFFER = 10
            worst = min_quality_index()
            del crops[worst]                  # Evict lowest quality
```

**Quality Score** ([`api/core/quality_scorer.py`](../api/core/quality_scorer.py)):  
Kết hợp Laplacian variance (độ sắc nét) với các đặc trưng hình thái học thành điểm [0, 1] liên tục.

**Code tích lũy trong pipeline:**
```python
# api/core/pipeline_core.py:258-264
for tid, plate_crop, vehicle_crop in matched:
    if not tracker.should_ocr(tid):
        continue
    q = quality_score(plate_crop)
    tracker.buffer_crop(tid, plate_crop, q, frame_idx)
    tracker.update_vehicle_img(tid, vehicle_crop, q)
```
→ [`api/core/pipeline_core.py:258`](../api/core/pipeline_core.py#L258)

---

## 6. Bước 4 — OCR Model: SmallLPR

### 6.1 Kiến trúc SmallLPR

**File:** [`LPRNet/lprnet/small_lpr.py`](../LPRNet/lprnet/small_lpr.py)  
**Checkpoint:** `weights/ocr/small_lpr-epoch=136-val_acc=0.914.ckpt`

SmallLPR là model OCR end-to-end gồm 4 module chính:

```
Input image (bất kỳ size)
        │
        ▼
  [1] smart_resize → (48, 96) BGR
        │
        ▼
  [2] _STNet (Spatial Transformer Network)
      Tự học affine transform để chuẩn hóa phối cảnh
      48×96 → localization conv → FC → theta (2×3) → affine_grid → grid_sample
        │
        ▼
  [3] SmallLPRBackbone (CNN)
      Input:  (B, 3, 48, 96)
      stem:   Conv3×3 → BN → Mish        → (B, 64, 48, 96)
      stage1: SmallBasicBlockCBAM(64→128) → MaxPool2 → (B, 128, 24, 48)
      stage2: SBCBAM(128→256) × 2        → MaxPool2 → (B, 256, 12, 24)
      stage3: SBCBAM(256→256) × 2        → MaxPool2 → (B, 256, 6, 12)
      Output: 2D spatial feature map, KHÔNG flatten → giữ thông tin vị trí 2D
        │
        ▼
  [4] 2D Positional Encoding + Projection
      Conv1×1: (B, 256, 6, 12) → (B, 384, 6, 12)
      LearnablePositional2D: row_pe + col_pe → mỗi token biết (row, col)
      Flatten: (B, 72, 384) — memory sequence cho decoder
        │
        ▼
  [5] MiniLMv2Decoder (Transformer Decoder, autoregressive)
      4 layers, 4 heads, d_model=384
      Teacher forcing khi train; autoregressive khi inference
      Decode: SOS → char₁ → char₂ → ... → EOS
      Output: [(char, probability), ...] mỗi character
```

**Điểm đổi mới so với LPRNet gốc:**
- **Không collapse feature map thành 1D** → decoder cross-attention tự học reading order cho biển 2 dòng
- **2D Positional Encoding tách biệt** row_pe + col_pe → token biết vị trí 2D trong feature map
- **CBAM attention** trong backbone → tập trung vào vùng ký tự quan trọng
- **STN đầu vào** → tự chuẩn hóa biển nghiêng trước khi extract features

### 6.2 SmallBasicBlockCBAM — Backbone building block

```python
# LPRNet/lprnet/small_lpr.py:64-91
class SmallBasicBlockCBAM(nn.Module):
    def __init__(self, ch_in, ch_out):
        mid = ch_out // 4
        self.block = nn.Sequential(
            MixConv2d(ch_in, mid, kernels=[3, 5]),  # Multi-scale convolution
            BN → Mish,
            Conv(mid, mid, (3,1)),  # Horizontal strip
            BN → Mish,
            Conv(mid, mid, (1,3)),  # Vertical strip
            BN → Mish,
            Conv(mid, ch_out, 1),
            BN,
        )
        self.cbam = CBAM(ch_out)          # Channel + Spatial attention
        self.residual = (ch_in == ch_out)

    def forward(self, x):
        out = self.cbam(self.block(x))
        return F.mish(out + x) if self.residual else F.mish(out)
```
→ [`LPRNet/lprnet/small_lpr.py:64`](../LPRNet/lprnet/small_lpr.py#L64)

### 6.3 Preprocessing

```python
# api/core/models.py:114-118
def preprocess_plate(bgr: np.ndarray) -> torch.Tensor:
    img = smart_resize(bgr, target_hw=(48, 96))   # Aspect-preserve + zero-pad
    img = (img.astype("float32") - 127.5) * 0.0078125   # Normalize → [-1, 1]
    return torch.from_numpy(img.transpose(2, 0, 1))      # HWC → CHW
```
→ [`api/core/models.py:114`](../api/core/models.py#L114)

`smart_resize`: scale to fit, giữ aspect ratio, pad bằng zeros về (48, 96).

### 6.4 Inference — Autoregressive Decode

```python
# api/core/models.py:121-160
@torch.no_grad()
def ocr_batch(model, images, device):
    memory = model.encode(images)           # CNN + 2D PE → (B, 72, 384)
    tokens = full((B, 1), SOS_IDX)
    per_chars = [[] for _ in range(B)]

    for _ in range(model.max_seq_len - 1):  # max 13 bước
        logits = model.decoder(tgt_tokens=tokens, memory_features=memory)
        probs  = softmax(logits[:, -1])
        next_tok = probs.argmax(-1)
        max_prob = probs.max(-1).values

        for b in range(B):
            if next_tok[b] == EOS_IDX: finished[b] = True
            else: per_chars[b].append((CHARS[next_tok[b]], float(max_prob[b])))

    # Layer 2 anti-hallucination: mỗi char phải có conf ≥ 0.90
    return [(chars, all(p >= CONF_THRESHOLD for _, p in chars)) for chars in per_chars]
```
→ [`api/core/models.py:121`](../api/core/models.py#L121)

**Vocabulary (38 ký tự):** `<PAD> <SOS> <EOS>` + `0-9` + `A-Z` (không có I,O,W,J) + `Đ` + `-` + `_`

---

## 7. Bước 5 — OCR Fusion & Voting

Mỗi crop được OCR bằng single-frame `SmallLPR` khi đưa vào buffer. Khi track bị mất (lost ≥ `LOST_THRESHOLD=5` stride) hoặc kết thúc video, hệ thống lấy `TOP_K_FRAMES=5` kết quả OCR chất lượng nhất để vote.

**Code trigger:**
```python
# api/core/pipeline_core.py:209-215
for tid in previously_tracked - currently_tracked:  # Track vừa mất
    if (
        tracker.should_ocr(tid)
        and tracker.mark_lost(tid)          # Đủ LOST_THRESHOLD = 5 lần mất
        and tracker.ready_for_track_ocr(tid)  # ≥ MIN_FRAMES_FOR_OCR = 3 crops
    ):
        _finalise_track_ocr(tid, tracker, models, emit, ...)
```
→ [`api/core/pipeline_core.py:209`](../api/core/pipeline_core.py#L209)

### 7.1 Segment Vote

**Bước 1:** Tái sử dụng kết quả `ocr_batch` đã cache cho từng crop độc lập.

**Bước 2:** `_segment_vote` — vote theo từng segment của biển:
```python
# api/core/tracker.py:304-365
@staticmethod
def _segment_vote(prob_lists):
    # 1. Parse mỗi kết quả OCR thành (province, serial, number)
    parsed = [(parse_segments(text), probs) for probs in prob_lists if valid]

    # 2. Chọn nhóm dominant theo (len(serial), len(number))
    target_serial_len, target_number_len = most_common(...)

    # 3. Vote độc lập từng segment qua _prob_vote
    prov_chars   = _prob_vote([probs[0:2] for _, probs in pool])
    serial_chars = _prob_vote([probs[serial_start:] for ...])
    number_chars = _prob_vote([probs[number_start:] for ...])

    return prov_chars + [("-", 0.9)] + serial_chars + [("-", 0.9)] + number_chars
```

**Bước 3:** `_prob_vote` — với mỗi vị trí ký tự, tính mean confidence và chọn ký tự có tổng confidence cao nhất:
```python
# api/core/tracker.py:274-302
@staticmethod
def _prob_vote(prob_lists):
    for pos in range(max_len):
        votes = {}  # char → [confidence_values]
        for probs in prob_lists:
            if pos < len(probs):
                char, conf = probs[pos]
                votes[char].append(conf)
        # Chọn char có tổng confidence cao nhất (frequency × confidence)
        best_char = max(votes, key=lambda c: sum(votes[c]))
        best_conf = sum(votes[best_char]) / len(prob_lists)  # ensemble conf
        result.append((best_char, best_conf))
```
→ [`api/core/tracker.py:274`](../api/core/tracker.py#L274)

### 7.3 Validation biển số Việt Nam (Layer 3)

Sau khi vote, kết quả phải match regex định dạng biển số Việt Nam:

```python
# api/core/pipeline_core.py:48-60
_VN_PLATE_RE = re.compile(
    r"^(?:"
    r"\d{2}[A-Z]{1,2}-\d{5}"    # 30A-12345, 50LD-12345
    r"|\d{2}-(?:[A-Z]\d|[A-Z]{2})-\d{5}"  # 29-X1-12345, 43-AA-01234
    r"|\d{2}[A-Z]-\d{4}"         # 31H-9999
    r"|\d{2}-[A-Z]\d-\d{4}"      # 29-F4-8888
    r")$"
)
```

Nếu **không** match → emit `rejected_vehicle` (hiển thị trong UI nhưng không lưu DB).

---

## 8. Lớp chống nhận diện sai — Tổng hợp

| Layer | Giai đoạn | Điều kiện | Ngưỡng | File |
|-------|-----------|-----------|--------|------|
| 1a | Pre-OCR | YOLO OBB detection confidence | ≥ 0.50 | `pipeline_core.py:229` |
| 1b | Pre-OCR | Kích thước biển tối thiểu | W≥50px, H≥15px | `pipeline_core.py:231` |
| 1c | Pre-OCR | Laplacian variance (độ sắc nét) | ≥ 80.0 | `gates.py:17` |
| 2 | Post-OCR | Confidence mỗi ký tự | ≥ 0.90 | `models.py:159` |
| 3 | Post-vote | Số frame OCR tối thiểu | ≥ 2 frame votes | `tracker.py:165` |
| 4 | Post-vote | Regex format biển VN | match pattern | `pipeline_core.py:58` |

---

## 9. SSE Events — Giao tiếp với Frontend

### Upload flow

| Event | Khi nào | Payload chính |
|-------|---------|---------------|
| `progress` | Mỗi 10 frame | `frame, total, pct` |
| `frame` | Mỗi frame (tùy config) | `base64 JPEG annotated` |
| `vehicle` | Khi có kết quả OCR | `id, cls, plate, chars, plate_b64, vehicle_b64` |
| `rejected_vehicle` | OCR không pass validation | `id, plate, chars, vote_summary` |
| `complete` | Pipeline xong | `total_vehicles, processed_frames` |
| `error` | Lỗi exception | `message` |

### Monitor (RTSP) flow

| Event | Ánh xạ từ | Payload thêm |
|-------|-----------|-------------|
| `event_started` | — | `event_id, window_start_sec, frames_count` |
| `event_progress` | `progress` | + `event_id` |
| `event_vehicle` | `vehicle` | + `event_id` |
| `event_rejected_vehicle` | `rejected_vehicle` | + `event_id` |
| `event_complete` | — | `total_vehicles, duration_ms` |
| `event_error` | — | `message` |

---

## 10. Khởi tạo model — Load một lần khi startup

```python
# api/main.py:41-48
@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.models = load_models()    # Tải tất cả model vào GPU một lần
    if MONGODB_URI:
        await init_db(MONGODB_URI, MONGODB_DB_NAME)
    yield
    await close_db()
```

```python
# api/core/models.py:60-111
def load_models() -> ModelBundle:
    vehicle  = YOLO("weights/detection/vehicle_best.pt")
    plate    = YOLO("weights/detection/best.pt")
    ocr      = SmallLPR(vocab_size=38, max_seq_len=14).to(device).eval()
    # Load checkpoint + strip "model." prefix từ Lightning checkpoint
    ocr.load_state_dict(...)

    vehicle_tracker = VehicleTracker(reid_weights="weights/tracking/vehicle_reid.onnx")
    return ModelBundle(device, vehicle, plate, ocr, vehicle_tracker)
```
→ [`api/core/models.py:60`](../api/core/models.py#L60)

---

## 11. Sơ đồ luồng đầy đủ

### Upload Video

```
POST /upload
    │ save temp file
    │ asyncio.Queue + mjpeg_queue
    └─ run_in_executor(run_job)
            │
            ▼ api/core/pipeline.py
       run_job() ─── FileFrameSource(tmp_path)
            │
            ▼ api/core/pipeline_core.py:process_frames()
       ┌────────────────────────────────────────────┐
       │ for each frame:                            │
       │   1. vehicle.predict() → dets              │
       │   2. vehicle_tracker.track(dets, frame)    │
       │      → boxes, ids, classes (BotSort+ReID)  │
       │   3. plate.track(frame, persist=True)      │
       │      → OBB detections (ByteTrack)          │
       │   4. Gate checks (conf, size, sharpness)   │
       │   5. associator.process_frame()            │
       │      → firm (vehicle_id, plate_crop) pairs │
       │   6. quality_score() → buffer_crop()       │
       │   7. draw_annotated_frame() → mjpeg_queue  │
       │                                            │
       │ on track lost (≥5 missing strides):        │
       │   8. top_k(5) crops from TrackBuffer       │
       │   9. _segment_vote or _prob_vote           │
       │  10. validate VN plate regex               │
       │  11. emit("vehicle" or "rejected_vehicle") │
       └────────────────────────────────────────────┘
            │
            ▼
       emit("complete")
            │
       MongoDB save (nếu MONGODB_URI configured)
```

### RTSP Stream (Event Monitor)

```
POST /monitor/live/connect { rtsp_url }
    │
    └─ LiveSession.start()
            │ MediaMTX: add_path(rtsp_url)
            └─ decoder_thread: cv2.VideoCapture(mediamtx_rtsp)
                    │ deque(maxlen=fps×10) rolling buffer
                    └─ mjpeg_queue → GET /monitor/live/{sid}/mjpeg
                                  hoặc WebRTC WHEP (trực tiếp qua MediaMTX)

POST /monitor/{sid}/mark { mode:"live" }
    │
    └─ snapshot_window(10s) → LiveBufferFrameSource(frames_list)
    │
    └─ _event_executor.submit(run_event)   ← single GPU worker
            │
            ▼ api/core/event_analyzer.py
       run_event()
            │
            ├─ emit("event_started")
            │
            ▼ api/core/pipeline_core.py:process_frames()
       [Toàn bộ pipeline giống Upload — xem trên]
            │
            ├─ emit("event_vehicle" / "event_rejected_vehicle")
            ├─ MongoDB: upsert_event() + upload evidence images
            └─ emit("event_complete")
```

---

## 12. Cấu hình quan trọng

Tất cả hằng số tập trung tại [`api/core/config.py`](../api/core/config.py):

```python
FRAME_STRIDE        = 1      # Chạy plate detection mỗi N frame
PLATE_DET_CONF      = 0.50   # Min YOLO OBB confidence
BLUR_THRESHOLD      = 80.0   # Laplacian variance ngưỡng sắc nét
CONF_THRESHOLD      = 0.90   # Min confidence mỗi ký tự OCR
MIN_FRAME_VOTES     = 2      # Min frame OCR trước khi kết thúc track
MAX_BUFFER          = 10     # Max crops giữ trong TrackBuffer
TOP_K_FRAMES        = 5      # Số frame tốt nhất đưa vào OCR
LOST_THRESHOLD      = 5      # Số stride mất liên tiếp → finalize track
MIN_FRAMES_FOR_OCR  = 3      # Min frames trong buffer mới finalize track OCR
IMG_W, IMG_H        = 96, 48 # Input size SmallLPR
```
