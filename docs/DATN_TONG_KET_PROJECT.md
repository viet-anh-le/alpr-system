# Tổng kết project ALPR Vietnamese và đối chiếu phiếu giao nhiệm vụ

Tài liệu này được viết theo hiện trạng repo ngày 14/06/2026, đối chiếu với phiếu giao nhiệm vụ `LeVietAnh_20225250_PGNV_20252.pdf`. Mục tiêu của em ở đây không phải là liệt kê file cho đủ, mà là nối lại toàn bộ quá trình làm đồ án: yêu cầu ban đầu là gì, em đã chọn hướng giải quyết nào, từng giai đoạn triển khai ra sao, vì sao lại triển khai như vậy, dữ liệu được xử lý thế nào, mô hình được tinh chỉnh thế nào, các kết quả định lượng hiện có là bao nhiêu, trong pipeline phần nào là preprocessing, phần nào là huấn luyện mô hình, phần nào là heuristic, phần nào là post-processing, và phần sản phẩm web gồm landing page, đăng nhập, dashboard được tổ chức như thế nào.

Một lưu ý quan trọng: em chỉ ghi số liệu khi repo có artifact rõ ràng như `results.csv`, `metrics.csv`, log huấn luyện hoặc file benchmark. Với các model có checkpoint nhưng không có file đánh giá đi kèm trong repo, em ghi rõ là chưa có artifact đánh giá độc lập để tránh đưa số liệu không kiểm chứng được.

## 1. Yêu cầu được giao trong phiếu nhiệm vụ

Theo phiếu giao nhiệm vụ, đề tài của em là:

> Hệ thống phát hiện và nhận dạng biển số xe chuyển động trong các nguồn video.

Thời gian thực hiện trong phiếu là từ ngày 23/02/2026 đến ngày 30/06/2026. Lĩnh vực chính là thị giác máy tính.

Các yêu cầu chính có thể tóm lại thành bốn nhóm.

### 1.1. Yêu cầu về kiến thức và công nghệ

Em cần nắm được cơ sở lý thuyết của ba bài toán chính:

- Object Detection: phát hiện phương tiện và biển số trong từng khung hình.
- Tracking: theo dõi phương tiện/biển số qua nhiều frame, gán ID ổn định.
- OCR: nhận dạng chuỗi ký tự biển số từ crop ảnh nhỏ, nhiều khi bị nghiêng, mờ hoặc thiếu sáng.

Phiếu giao nhiệm vụ cũng yêu cầu tìm hiểu các kiến trúc CNN, cơ chế chú ý như SE, CBAM, TripletAttention, STN, mô hình chuỗi thời gian bằng Transformer và kỹ thuật kết hợp nhiều khung hình. Về công nghệ, các nhóm được nêu gồm YOLOv5/YOLOv8, DeepSORT/ByteTrack/BoT-SORT, LPRNet/MFLPRNet/CRNN/MobileNetViT/MiniLM/PaddleOCR, cùng ReactJS, FastAPI và MongoDB.

Trong repo hiện tại, các yêu cầu này đã được phản ánh khá trực tiếp:

- YOLOv8 được dùng cho phát hiện phương tiện và YOLOv8 OBB cho phát hiện biển số có góc xoay.
- BoT-SORT kết hợp ReID được dùng cho tracking phương tiện.
- Pipeline OCR đã thử nhiều hướng: SmallLPR autoregressive, SmallLPR-CTC, SmallLPR-NAR, PARSeq và YOLOv5 character OCR.
- FastAPI, React/Vite, MongoDB và object storage được triển khai cho phần sản phẩm demo.

### 1.2. Yêu cầu về sản phẩm phần mềm

Sản phẩm kỳ vọng trong phiếu là:

- Có hệ thống phần mềm nhận dạng biển số xe ô tô đang chuyển động trong video.
- Có giao diện web cho phép người dùng tải video lên và xem kết quả nhận dạng trực tiếp trên trình duyệt.
- Có khả năng theo dõi và nhận diện biển số đối với các phương tiện trong video.

Trong repo, sản phẩm đã mở rộng hơn yêu cầu ban đầu ở ba điểm. Ngoài chế độ upload video, hệ thống còn có chế độ monitor/live để nhận RTSP hoặc video upload rồi đánh dấu một đoạn sự kiện cần phân tích. Kết quả nhận dạng được stream về frontend bằng SSE, còn khung hình annotate có thể truyền bằng MJPEG. Bên cạnh đó, phần web đã có landing page public để giới thiệu năng lực hệ thống, luồng đăng ký/đăng nhập tài khoản và dashboard riêng sau đăng nhập để upload video, xem tiến trình, xem kết quả nhận dạng và mở lịch sử theo người dùng.

### 1.3. Yêu cầu về vấn đề thực tiễn

Vấn đề thực tế của đồ án là ALPR trên video khó hơn ảnh tĩnh vì:

- Phương tiện chuyển động làm biển số nhỏ, rung, lệch góc hoặc bị motion blur.
- Chất lượng hình ảnh giữa các frame không đồng đều.
- Một frame đơn có thể không đủ tốt để OCR, trong khi vài frame liên tiếp có thể bổ sung cho nhau.

Vì vậy, giải pháp trong repo không dừng ở việc detect một crop rồi OCR ngay. Hệ thống thiết kế theo hướng video-first: detect, track, tích lũy crop theo track, đánh giá chất lượng, chọn frame tốt, hoặc hợp nhất nhiều kết quả OCR bằng CTM/voting trước khi chấp nhận biển số.

### 1.4. Kế hoạch triển khai trong phiếu và phần đã làm

Phiếu giao nhiệm vụ chia kế hoạch thành năm nội dung:

1. Tìm hiểu tổng quan bài toán ALPR.
2. Tìm hiểu công nghệ liên quan.
3. Phân tích thiết kế pipeline.
4. Xây dựng chương trình.
5. Thử nghiệm và đánh giá.

Repo hiện tại bám khá sát năm giai đoạn này. Phần khảo sát và thiết kế được thể hiện trong các tài liệu như `docs/PIPELINE.md`, `DATN_Muc_luc_de_xuat.md`, tài liệu thiết kế quality router theo LPLCv2 và kế hoạch synthetic dataset. Phần chương trình nằm trong các module `api/core`, `detection`, `ocr`, `tracking`, `scripts`, `web/src`. Phần đánh giá có artifact cho detector biển số, OCR, quality router và benchmark hiệu năng pipeline.

## 2. Đối chiếu yêu cầu với sản phẩm hiện tại

| Yêu cầu trong phiếu | Hiện trạng trong repo | Mức độ |
|---|---|---|
| Phát hiện phương tiện trong video | Có `vehicle_best.pt`, `VEHICLE_CLASSES`, `VehicleTracker`, YOLO inference trong `pipeline_core.py` và `pipeline_async.py` | Đã triển khai |
| Phát hiện biển số | Có YOLOv8 OBB, dataset OBB, checkpoint `runs/obb/.../weights/best.pt`, cascade theo crop phương tiện | Đã triển khai và có số liệu |
| Theo dõi phương tiện | Có BoT-SORT + custom ReID ONNX qua `VehicleTracker` | Đã triển khai |
| Theo dõi/ghép biển số với xe | Có `PlateTrackManager` và `TrajectoryAssociator` vote 5 frame, agreement 0.6 | Đã triển khai |
| OCR biển số | Có SmallLPR, SmallLPR-CTC, SmallLPR-NAR, PARSeq, YOLOv5 char backend | Đã triển khai và có số liệu OCR |
| Kết hợp đa khung hình | Có `TrackBuffer`, chọn top-k frame, CTM fusion theo template biển Việt Nam | Đã triển khai |
| Web upload video | Có React `DropZone`, FastAPI `/upload`, SSE `/stream/{job_id}` | Đã triển khai |
| Hiển thị kết quả trực tiếp | Có SSE event `progress`, `vehicle`, `rejected_vehicle`, `complete`; MJPEG preview nếu bật | Đã triển khai |
| Landing page giới thiệu sản phẩm | Có route `/`, React `LandingPage.jsx`, CTA chuyển đến đăng ký/đăng nhập hoặc dashboard theo trạng thái phiên | Đã triển khai |
| Đăng ký/đăng nhập người dùng | Có FastAPI router `/auth`, bcrypt password hash, JWT trong HttpOnly cookie, session MongoDB, CSRF token cho request ghi | Đã triển khai |
| Dashboard sau đăng nhập | Có route `/dashboard` được bảo vệ bởi `RequireAuth`, gồm upload video, monitor mode, OCR stats, danh sách xe, rejected vehicles, lịch sử | Đã triển khai |
| Live/video event monitor | Có `/monitor/upload`, `/monitor/live/connect`, `/monitor/{session_id}/mark` | Đã triển khai mở rộng |
| Lưu lịch sử nhận dạng | Có MongoDB models `RecognitionSession`, `RecognitionRecord`, `Event`, gắn `user_id`, lọc lịch sử theo tài khoản và upload ảnh chứng cứ | Đã triển khai, phụ thuộc env DB/storage |
| Thử nghiệm và đánh giá | Có kết quả OBB, OCR, router, benchmark sync/async; chưa có artifact chính thức cho vehicle detector và ReID | Đã có một phần, cần bổ sung cho báo cáo cuối |

## 3. Giải pháp tổng thể em đã đưa ra

Giải pháp của em là một pipeline ALPR theo hướng track-level, không phải single-image ALPR. Luồng chính có thể mô tả như sau:

```text
Video / RTSP / upload interval
  -> FrameSource
  -> optional whole-frame preprocessing
  -> YOLO vehicle detection
  -> BoT-SORT + ReID vehicle tracking
  -> crop từng phương tiện
  -> YOLOv8 OBB plate detection trên vehicle crop
  -> map OBB về frame gốc
  -> warp phối cảnh biển số
  -> quality router / quality score
  -> route OCR:
       direct single-frame nếu crop tốt
       tracklet fusion nếu crop kém
       unreadable wait nếu quá xấu/occluded
  -> OCR backend
  -> CTM fusion / format validation / reject nếu không đủ tin cậy
  -> SSE event + ảnh chứng cứ + lưu DB
```

Em chọn kiến trúc này vì yêu cầu của đề tài là video. Nếu chỉ detect biển số từng frame rồi OCR độc lập, hệ thống rất dễ bị sai ở các frame mờ hoặc thiếu sáng. Trong video, cùng một xe có thể xuất hiện trong nhiều frame; có frame biển số bị nghiêng, có frame rõ hơn, có frame OCR đúng vài ký tự nhưng sai ký tự khác. Vì vậy, việc gán track ID và gom bằng chứng theo track là phần quan trọng nhất để hệ thống bớt nhận dạng "ảo".

## 4. Các giai đoạn triển khai

### 4.1. Giai đoạn 1: Khảo sát bài toán và thiết kế hướng tiếp cận

Ban đầu, bài toán được tách thành ba khối học sâu:

- Detector phương tiện để biết trong frame có xe nào.
- Detector biển số để tìm vùng biển nhỏ trên phương tiện.
- OCR để đọc chuỗi ký tự.

Tuy nhiên, khi đưa vào video thực tế, ba khối này chưa đủ. Có hai vấn đề phát sinh:

1. Biển số nhỏ và dễ bị false positive nếu detect toàn frame.
2. OCR một frame dễ sai nhưng nhiều frame cùng xe có thể cho kết quả ổn hơn.

Vì vậy, thiết kế được chuyển thành vehicle-first cascade. Thay vì tìm biển số trên toàn frame ngay từ đầu, hệ thống phát hiện và track phương tiện trước, sau đó crop vùng phương tiện rồi mới detect biển số trong vùng đó. Cách này giảm không gian tìm kiếm, giúp detector biển tập trung vào vùng có khả năng chứa biển số.

Sau đó, hệ thống dùng `TrajectoryAssociator` để gán biển số với phương tiện. Việc ghép không làm cứng theo từng frame mà vote qua nhiều frame. Đây là lựa chọn quan trọng vì bounding box xe và biển số có thể dao động, đặc biệt khi có nhiều xe gần nhau.

### 4.2. Giai đoạn 2: Thu thập và xử lý dữ liệu

Repo có nhiều nguồn dữ liệu và script xử lý. Em tách dữ liệu theo từng nhiệm vụ thay vì dùng một dataset chung cho mọi model.

#### 4.2.1. Dữ liệu phát hiện biển số OBB

Dataset OBB nằm ở `data/datasets/lp_detection_obb`. Thống kê hiện tại:

- Tổng ảnh: `40.465`.
- Train: `33.333` ảnh.
- Val: `7.132` ảnh.
- Số file label: `40.467`.
- Hai lớp: `BSD` và `BSV`.

Trong đó:

- `BSD` có thể hiểu là biển số dạng dài.
- `BSV` có thể hiểu là biển số dạng vuông/hai dòng.

Script chính là `scripts/build_lp_detection_obb_dataset.py`. Script này gộp nhiều nguồn, chuẩn hóa nhãn về YOLOv8 OBB, kiểm tra coordinate nằm trong `[0,1]`, convert polygon nếu cần về minimum rotated rectangle và ghi ra cấu trúc:

```text
images/train
images/val
labels/train
labels/val
data.yaml
```

Lý do dùng OBB thay vì bounding box thường là biển số trong video thường nghiêng. Nếu chỉ crop bounding box thẳng, ảnh OCR sẽ có nhiều background ở góc, ký tự bị nghiêng và nhỏ hơn. OBB cho phép lấy bốn điểm góc rồi dùng perspective transform để đưa biển về crop chặt hơn.

#### 4.2.2. Dữ liệu OCR biển số

Dataset OCR chính nằm ở `data/datasets/ocr`:

- Train: `28.266` ảnh.
- Valid: `5.568` ảnh.

Nhãn được mã hóa trực tiếp trong filename, ví dụ có các định dạng chứa `-`, `.`, và `[SEP]`. Điều này rất quan trọng vì biển số Việt Nam có cả biển một dòng và hai dòng. `[SEP]` được dùng để biểu diễn điểm ngắt dòng trong crop OCR.

Ngoài ra, repo có dữ liệu Platesmania VN:

- `data/raw/platesmania_vn/ocr/images/train`: `29.651` ảnh.
- `data/raw/platesmania_vn/ocr/images/val`: `7.051` ảnh.
- `data/raw/platesmania_vn/detection/images/train`: `30.122` ảnh.
- `data/raw/platesmania_vn/detection/images/val`: `7.240` ảnh.

Các script liên quan:

- `scripts/collect_platesmania_vn_dataset.py`: parse HTML/gallery, lấy ảnh xe, chuẩn hóa plate text, tự detect biển bằng model ban đầu, tạo record.
- `scripts/crop_label_studio_platesmania.py`: đọc export Label Studio, lấy polygon đã review, warp crop biển số và ghi OCR sample.
- `scripts/generate_label_studio_annotations.py`: hỗ trợ tạo annotation để đưa vào Label Studio.
- `scripts/refresh_platesmania_review_full_frames.py`: làm mới frame/crop phục vụ review.

Lý do OCR dùng filename-label là crop biển số đã được tách ra, nhãn chuỗi có thể lấy trực tiếp từ text biển số. Với bài toán OCR end-to-end, dữ liệu dạng `image -> text` phù hợp hơn so với chỉ có bbox ký tự.

#### 4.2.3. Dữ liệu quality router từ LPLCv2

Để xử lý trường hợp biển số mờ, tối, thấp độ phân giải hoặc không đọc được, repo có hướng Plate Quality Router lấy cảm hứng từ LPLCv2.

Script chính là `scripts/prepare_lplcv2_quality_dataset.py`. Script đọc annotation LPLCv2, crop biển số theo bbox, normalize nhãn legibility và tạo hai dataset:

```text
data/lplcv2_quality/legibility4/{illegible,poor,good,perfect}
data/lplcv2_quality/binary/{suitable,unsuitable}
```

Thống kê từ `data/lplcv2_quality/summary.json`:

- `perfect`: `18.425`.
- `good`: `10.180`.
- `poor`: `7.520`.
- `illegible`: `5.362`.
- Tổng: `41.487` crop.

Mapping binary:

- `good + perfect -> suitable`.
- `poor + illegible -> unsuitable`.

Lý do có router là không phải crop nào cũng nên OCR trực tiếp. Với crop tốt, OCR ngay sẽ nhanh. Với crop kém, nên gom theo track hoặc tạo candidate ảnh tăng cường trước OCR. Với crop quá tệ, hệ thống nên chờ frame khác hoặc trả về unreadable, không cố đoán biển số.

#### 4.2.4. Dữ liệu ReID/tracking

Dataset tracking nằm ở `data/datasets/tracking`:

- Train: `47.619` ảnh.
- Val: `2.162` ảnh.
- Test: `18.841` ảnh.
- Test query: `1.678` ảnh.
- Test gallery: `17.163` ảnh.

ReID model được huấn luyện để tạo embedding cho phương tiện, hỗ trợ BoT-SORT giữ ID ổn định khi xe bị che, mất detection ngắn hoặc xuất hiện lại.

Trong repo có checkpoint:

- `weights/tracking/vehicle_reid.pt`.
- `weights/tracking/vehicle_reid_best.pt`.
- `weights/tracking/vehicle_reid.onnx`.

Tuy nhiên, hiện em chưa thấy file kết quả R1/mAP được lưu lại kèm checkpoint như các model OCR/detection, nên phần số liệu ReID cần chạy lại `tracking/train/evaluate_reid.py` hoặc lưu log huấn luyện trước khi đưa vào bảng kết quả chính thức.

#### 4.2.5. Synthetic dataset

Repo có tài liệu `SyntheticVietnameseLicensePlateAsLPLCV2.md` về pipeline synthetic VN-LPLC. Phần này có nhiều thiết kế và test, nhưng trạng thái tài liệu ghi rõ V4 chưa được promote production. Vì vậy, khi viết báo cáo chính thức, em nên trình bày phần synthetic như hướng nghiên cứu/bổ trợ dữ liệu, không nên coi nó là nguồn dataset final đã dùng để train model chính nếu chưa có artifact promote và benchmark đầy đủ.

### 4.3. Giai đoạn 3: Huấn luyện và tinh chỉnh detector biển số

Detector biển số dùng YOLOv8 OBB. File config huấn luyện được lưu ở:

```text
runs/obb/experiments/detection/lp_detection_obb_merged/args.yaml
```

Các tham số chính:

- Task: `obb`.
- Model khởi tạo: `weights/detection/best.pt`.
- Data: `data/datasets/lp_detection_obb/data.yaml`.
- Epoch khai báo: `50`, artifact hiện có `25` epoch.
- Batch size: `8`.
- Image size: `640`.
- Optimizer: `auto`.
- Learning rate `lr0`: `0.001`.
- Augmentation: mosaic `1.0`, hsv augment, flip ngang `0.5`.

Kết quả tốt nhất theo `mAP50-95(B)` trong `results.csv`:

| Model | Epoch tốt nhất | Precision | Recall | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|---:|
| YOLOv8 OBB biển số | 22 | 0.96776 | 0.94489 | 0.98292 | 0.95015 |

Dòng cuối epoch 25 cũng rất gần:

- Precision: `0.96818`.
- Recall: `0.94641`.
- mAP50: `0.98309`.
- mAP50-95: `0.94999`.

Lý do metric này tốt là bài toán detection biển số sau khi đã có dataset OBB được chuẩn hóa khá rõ. Tuy nhiên, trong pipeline thực tế, detector không chạy trực tiếp toàn frame mà chạy trên crop phương tiện. Điều này làm bài toán inference gần hơn với vùng chứa biển số, đồng thời giúp giảm false positive.

### 4.4. Giai đoạn 4: Phát hiện phương tiện và tracking

#### 4.4.1. Vehicle detector

Vehicle detector được load từ:

```text
weights/detection/vehicle_best.pt
```

Các class được dùng trong `api/core/config.py`:

```python
VEHICLE_CLASSES = [0, 1, 2, 3, 4]
```

Comment trong config mô tả:

- `0`: car.
- `1`: bus.
- `2`: truck.
- `3`: motorcycle.
- `4`: motorbike_rider.

Hiện trong repo có checkpoint `experiments/detection/vehicle_best.pt` và `weights/detection/vehicle_best.pt`, nhưng không có `results.csv` hoặc report đánh giá tương ứng. Vì vậy, em không nên tự ghi precision/recall/mAP của vehicle detector vào báo cáo nếu chưa chạy lại evaluation.

#### 4.4.2. Vehicle tracker

Tracking phương tiện được triển khai qua `api/core/tracker_adapter.py` bằng `boxmot.BotSort` và custom ReID. ReID model dùng:

```text
weights/tracking/vehicle_reid.onnx
```

Trong `tracking/train/train_reid.py`, chiến lược huấn luyện ReID là:

- Model: `VehicleReIDNet`.
- Embedding dimension: `128`.
- Loss: BatchHardTripletLoss + `0.5 * CrossEntropy`.
- Label smoothing: `0.1`.
- Sampler: PKSampler với `P=16` identities và `K=4` ảnh mỗi identity, batch size hiệu dụng `64`.
- Optimizer: AdamW.
- Scheduler: warmup tuyến tính 10 epoch rồi cosine decay.
- Metric: CMC Rank-1, Rank-5, Rank-10 và mAP.

Lý do dùng ReID là IoU tracking đơn thuần dễ đổi ID khi xe bị che hoặc hai xe gần nhau. Với embedding appearance, tracker có thêm tín hiệu hình ảnh để giữ track ổn định hơn.

#### 4.4.3. Plate tracking và association

Phiên bản pipeline hiện tại dùng cascade plate tracking trong `api/core/cascade_plate.py`, cụ thể là `PlateTrackManager`, thay vì chỉ gọi trực tiếp `model.plate.track()` trên toàn frame.

Các bước:

1. Crop từng phương tiện bằng `expand_vehicle_box`.
2. Detect biển số OBB trên crop phương tiện.
3. Map bốn điểm OBB từ tọa độ crop về tọa độ frame gốc.
4. Warp crop biển số bằng `warp_plate_crop`.
5. Deduplicate các detection trùng do các crop phương tiện chồng nhau.
6. Gán ID ổn định cho plate candidate bằng `PlateTrackManager`.

Sau đó, `TrajectoryAssociator` ghép plate track với vehicle track. Cơ chế này là heuristic có kiểm soát:

- Quan sát `match_frames = 5`.
- Cần agreement ratio `0.6`.
- Mỗi frame, plate vote cho xe chứa tâm biển số, ưu tiên `source_vehicle_id` nếu hợp lệ.
- Khi đủ số vote, association bị khóa.
- Association đã khóa vẫn được revalidate; nếu mâu thuẫn liên tiếp thì mở khóa và vote lại.

Lý do không ghép cứng theo frame là trong video thực tế bbox dao động. Nếu một frame detect lệch hoặc xe bị che, ghép cứng sẽ làm biển của xe này bị gán sang xe khác. Vote theo trajectory làm hệ thống ổn định hơn.

### 4.5. Giai đoạn 5: Crop phối cảnh và kiểm soát chất lượng

Sau khi có OBB, hệ thống không lấy bounding rect thẳng làm đầu vào OCR. Thay vào đó, `api/core/video_processor.py` dùng `warp_plate_crop`:

1. Sắp xếp bốn điểm thành top-left, top-right, bottom-right, bottom-left.
2. Tính width/height theo cạnh dài nhất.
3. Dùng `cv2.getPerspectiveTransform`.
4. Warp về ảnh crop phẳng.

Đây là preprocessing hình học. Nó xảy ra trước OCR và trước quality router. Mục tiêu là làm crop biển ít bị nghiêng hơn, giảm background và giúp OCR học/đọc ổn định hơn.

Sau crop, hệ thống tính `quality_score` trong `api/core/quality_scorer.py`:

```text
quality_score = 0.80 * sharpness + 0.20 * relative_size
```

Trong đó:

- `sharpness` dựa trên Laplacian variance, normalize bởi `LAP_MAX = 500.0`.
- `relative_size` dựa trên diện tích crop so với diện tích tối thiểu.

Hệ thống cố ý không dùng aspect ratio làm quality signal, vì biển Việt Nam có cả dạng dài và dạng vuông/hai dòng. Nếu dùng tỉ lệ khung hình làm ngưỡng cứng, rất dễ loại nhầm biển hợp lệ.

### 4.6. Giai đoạn 6: Quality router và route-aware OCR

Quality router nằm ở `api/core/quality_router.py`. Output gồm:

- `legibility`: `perfect`, `good`, `poor`, `illegible`.
- `quality_bin`: `suitable` hoặc `unsuitable`.
- `router_conf`.
- `degradation_tags`: `low_res`, `motion_blur`, `low_light`, `low_contrast`, `rain_or_haze`, `faulty_color`, `occluded`.
- `route`: `direct`, `tracklet_fusion`, hoặc `unreadable_wait`.
- `quality_numeric`.

Nếu có model classifier qua env `PLATE_QUALITY_ROUTER_MODEL`, router dùng YOLO classification. Nếu không có model, router fallback bằng heuristic từ brightness, contrast, Laplacian variance, saturation, color spread và kích thước crop.

Kết quả huấn luyện classifier:

| Model router | Dataset | Epoch tốt nhất | Accuracy top-1 | Accuracy top-5 | Val loss |
|---|---|---:|---:|---:|---:|
| Legibility 4 lớp | `data/lplcv2_quality/legibility4` | 35 | 0.85287 | 1.00000 | 0.34785 |
| Binary suitable/unsuitable | `data/lplcv2_quality/binary` | 46 | 0.95603 | 1.00000 | 0.10964 |
| Legibility fine-tuned VN | run `legibility_finetuned_vn` | 35 | 0.84705 | 1.00000 | 0.38505 |

Bản binary cao hơn bản 4 lớp là hợp lý. Phân biệt một crop có đủ đọc trực tiếp hay không dễ hơn phân biệt ranh giới giữa `perfect` và `good`, hoặc giữa `poor` và `illegible`.

Ở inference, route hoạt động như sau:

- `direct`: crop đủ tốt, OCR ngay. Nếu OCR hợp lệ và confidence đủ cao thì accept single frame.
- `tracklet_fusion`: crop kém, buffer lại theo track, sau đó OCR/candidate/fusion ở cuối track.
- `unreadable_wait`: crop quá tệ hoặc occluded, buffer như bằng chứng nhưng không ép OCR ngay.

Đây là phần kết hợp giữa model và heuristic. Classifier là model học sâu; còn mapping legibility sang route và degradation tags là logic quyết định thủ công.

### 4.7. Giai đoạn 7: OCR và tinh chỉnh các mô hình nhận dạng

OCR là phần em thử nhiều hướng nhất vì đây là khối quyết định biển số cuối cùng.

#### 4.7.1. SmallLPR autoregressive

SmallLPR nằm ở `LPRNet/lprnet/small_lpr.py`. Kiến trúc gồm:

```text
Input crop
  -> smart_resize 48x96
  -> STN
  -> CNN backbone với SmallBasicBlockCBAM
  -> 2D positional encoding
  -> MiniLMv2-style Transformer decoder autoregressive
  -> chuỗi ký tự
```

Các điểm chính:

- `smart_resize` giữ aspect ratio và pad về `48x96`.
- STN học affine transform để giảm lệch phối cảnh.
- Backbone giữ feature map 2D, không ép thành chuỗi 1D quá sớm.
- 2D positional encoding giúp decoder biết vị trí hàng/cột, phù hợp với biển hai dòng.
- Decoder autoregressive sinh ký tự từ trái sang phải.

Kết quả log:

| Run | Best epoch | Val acc |
|---|---:|---:|
| `small_lpr_20260607_190352` | 14 | 0.93528 |
| `small_lpr_20260608_041915` | 17 | 0.94337 |
| Legacy checkpoint `small_lpr-epoch=136-val_acc=0.914.ckpt` | 136 | 0.91400 theo tên file |

#### 4.7.2. SmallLPR-CTC

SmallLPR-CTC nằm ở `LPRNet/lprnet/small_lpr_ctc.py`. Model giữ STN, backbone và 2D positional encoding, nhưng bỏ decoder autoregressive, thay bằng CTC head.

Luồng:

```text
Input 48x96
  -> STN
  -> backbone
  -> projection
  -> 2D positional encoding
  -> reshape thành 72 time steps
  -> linear CTC head
  -> greedy CTC decode
```

Lý do thử CTC:

- Inference chỉ cần một forward pass, không có vòng lặp decode từng ký tự.
- CTC tự xử lý alignment giữa chuỗi feature và chuỗi ký tự.
- Phù hợp để tăng tốc trong pipeline video.

Config hiện tại trong `api/core/config.py` đặt `OCR_BACKEND` mặc định là `smalllpr_ctc`, checkpoint:

```text
weights/ocr/small_lpr_ctc/ctc_20260609_155238/small_lpr_ctc-epoch=055-val_acc=0.9358.ckpt
```

Kết quả tốt nhất theo log:

| Run | Best epoch | Val acc | Val loss |
|---|---:|---:|---:|
| `ctc_20260608_201842` | 54 | 0.93438 | 0.06917 |
| `ctc_20260609_155238` | 55 | 0.93579 | 0.06451 |
| `ctc_finetune_ep55_lr1e4` | 6 | 0.93561 | 0.06556 |

#### 4.7.3. SmallLPR-NAR

SmallLPR-NAR nằm ở `LPRNet/lprnet/small_lpr_nar.py`. Đây là hướng non-autoregressive:

- Mỗi vị trí output là một learned query.
- Decoder có self-attention giữa các position query.
- Cross-attention nhìn vào encoder memory.
- Output toàn bộ chuỗi trong một lần forward.

Lý do thử NAR là muốn giữ tốc độ gần CTC nhưng vẫn cho decoder học quan hệ giữa các vị trí ký tự. Bản NAR v2 thêm self-attention trước cross-attention để các position biết "hàng xóm" đang dự đoán gì.

Kết quả tốt nhất:

| Run | Best epoch | Val acc | Val loss |
|---|---:|---:|---:|
| `nar_20260608_061801` | 2 | 0.80672 | 0.53751 |
| `nar_20260608_065127` | 22 | 0.93545 | 0.43023 |
| `nar_20260608_123600` | 85 | 0.95811 | 0.41958 |

Đây là OCR checkpoint có val accuracy cao nhất trong các log hiện có: `95.81%`.

#### 4.7.4. PARSeq

PARSeq được fine-tune trong `ocr/train_parseq.py`, checkpoint:

```text
weights/ocr/parseq/parseq_vn_plate_best.pt
```

Config training:

- Image size: `128x32`.
- Batch size: `64`.
- LR: `3e-5`.
- Optimizer: AdamW.
- Scheduler: ReduceLROnPlateau.
- Metric chính: sequence accuracy và character accuracy.

Kết quả tốt nhất trong `parseq_vn_plate_log.csv`:

| Model | Best epoch | Val sequence accuracy | Val character accuracy |
|---|---:|---:|---:|
| PARSeq VN plate | 15 | 0.954586 | 0.982960 |

Epoch cuối 27 vẫn giữ mức cao:

- Val sequence accuracy: `0.952432`.
- Val character accuracy: `0.982233`.

#### 4.7.5. SlotLPR và YOLOv5 character OCR

Repo có checkpoint:

```text
weights/ocr/slot_lpr/slot-lpr-epoch=112-val_acc=0.8983.ckpt
```

Ngoài ra có YOLOv5 character/object model trong:

```text
references/Character-Time-series-Matching/Vietnamese/char.pt
references/Character-Time-series-Matching/Vietnamese/object.pt
```

Các hướng này được dùng như backend/baseline hoặc phục vụ CTM/diagnostic, nhưng không phải backend mặc định của pipeline hiện tại. OCR backend có thể chọn từ frontend gồm `default`, `smalllpr_ctc`, `parseq`, `yolov5_char`, `vietnamese_yolov5`.

### 4.8. Giai đoạn 8: Track buffer và hợp nhất đa khung hình

`TrackBuffer` nằm trong `api/core/tracker.py`. Mỗi track lưu:

- Crop biển số.
- `quality_score`.
- OCR confidence.
- Danh sách ký tự và xác suất.
- Frame index.
- Candidate method.
- Route.
- Kết quả router.

Kích thước buffer:

```text
MAX_BUFFER = 10
TOP_K_FRAMES = 5
MIN_FRAMES_FOR_OCR = 3
LOST_THRESHOLD = 5
```

Khi buffer đầy, hệ thống loại frame có score thấp nhất:

```text
combined_score = quality_score * max(ocr_conf, 0.10)
```

Lý do dùng cả quality và OCR confidence là crop sắc chưa chắc đọc đúng. Nếu chỉ dùng độ nét, một crop rõ nhưng OCR sai tự tin thấp vẫn có thể chiếm chỗ trong buffer. Ngược lại, crop hơi mềm nhưng OCR đúng và confidence tốt có thể có giá trị hơn.

#### 4.8.1. Direct single-frame accept

Nếu quality route là `direct`, OCR ra biển hợp lệ, confidence >= `CONF_THRESHOLD = 0.90`, hệ thống accept ngay bằng `single_frame_direct`. Điều này giúp xe có biển rõ không phải chờ đến khi track mất.

#### 4.8.2. Deferred OCR cho crop degraded

Nếu crop đi vào `tracklet_fusion`, pipeline có thể buffer crop trước. Khi track kết thúc, `_entries_with_deferred_ocr` trong `track_ocr.py` tạo candidate ảnh từ crop degraded rồi OCR batch. Candidate gồm:

- `original`.
- `sharpen`.
- `gamma`.
- `grayscale`.
- `clahe`.
- `contrast_stretch`.
- `denoise`.
- `haze_contrast`.
- `white_balance`.

Các candidate này không thay thế ảnh gốc một cách mù quáng. `original` luôn được giữ và còn có tie-break bonus nhẹ trong reranker.

#### 4.8.3. CTM fusion

Phần hậu xử lý quan trọng nhất hiện tại là `api/core/ocr_ctm.py`, tức OCR-output Character Time-series Matching.

Thay vì vote ký tự theo index thẳng, CTM làm các bước:

1. Normalize text OCR.
2. Chọn template biển số Việt Nam phù hợp từ `VN_PLATE_TEMPLATE_PATTERNS`.
3. Align chuỗi OCR vào các slot của template bằng dynamic programming.
4. Gom vote theo slot.
5. Chỉ chấp nhận ký tự nếu support > `0.5` và weighted confidence >= `0.50`.
6. Slot nào không đủ bằng chứng thì để `?`.
7. Nếu còn unresolved slot hoặc không match format thì reject.

Đây là post-processing, nhưng có yếu tố heuristic trong template scoring và ngưỡng majority. Điểm quan trọng là CTM không cố đoán cho đủ biển số. Nếu bằng chứng không đủ, hệ thống phát `rejected_vehicle` thay vì tạo một biển hợp lệ giả.

### 4.9. Giai đoạn 9: Hậu xử lý, chống nhận dạng sai và format validation

Hệ thống có nhiều lớp chống nhận dạng sai:

| Lớp | Vị trí | Điều kiện |
|---|---|---|
| Detection gate | Trước OCR | Plate confidence >= `PLATE_DET_CONF = 0.50` |
| Size gate | Trước OCR | Width >= `MIN_PLATE_W = 30`, height >= `MIN_PLATE_H = 15` |
| Router candidate gate | Trước router | Crop không rỗng, `quality_score >= 0.05` |
| OCR confidence | Sau OCR | `CONF_THRESHOLD = 0.90` cho direct accept |
| Track evidence | Trước final OCR | Tối thiểu `MIN_FRAMES_FOR_OCR = 3` frame khác nhau |
| CTM support | Hậu xử lý | Support ký tự theo slot > `0.5` |
| Format validation | Hậu xử lý cuối | Match regex/template biển số Việt Nam |

Định dạng biển số Việt Nam được gom trong `api/core/plate_format.py`. File này không chỉ có vài regex đơn giản mà có danh sách template, ví dụ các dạng có `-`, `.`, `[SEP]`, chữ/số và biển hai dòng. Khi hiển thị, `[SEP]` được đổi thành khoảng trắng.

Nếu kết quả không hợp lệ, pipeline emit:

```text
rejected_vehicle
```

Điều này giúp frontend vẫn thấy rằng có xe/crop đã được xử lý, nhưng hệ thống không lưu nó như kết quả nhận dạng đúng.

### 4.10. Giai đoạn 10: Backend, frontend và sản phẩm demo

#### 4.10.1. Backend upload video

FastAPI entrypoint là `api/main.py`. Luồng upload:

1. Frontend gửi file qua `POST /upload`.
2. Backend lưu file tạm.
3. Tạo `job_id`.
4. Chạy `run_job` trong executor.
5. Frontend nhận kết quả qua `GET /stream/{job_id}`.
6. Nếu bật preview, MJPEG frame qua `GET /stream/{job_id}/mjpeg`.

`run_job` hiện dùng `process_frames_async` làm pipeline mặc định. Nếu `ocr_backend == "vietnamese_yolov5"` thì dùng pipeline riêng.

Route `/upload` hiện không còn là endpoint public. Backend yêu cầu người dùng đã đăng nhập qua dependency `get_current_user_with_csrf`, sau đó gắn `job_id` với `user_id` trong `_job_owners`. Vì vậy, các stream kết quả như `/stream/{job_id}` và `/stream/{job_id}/mjpeg` chỉ trả dữ liệu khi người gọi là chủ của job đó. Đây là thay đổi quan trọng so với demo upload đơn giản, vì nó biến dashboard thành không gian làm việc theo phiên người dùng.

Backend cũng kiểm tra file upload ở boundary:

- Chỉ nhận các extension video như `.mp4`, `.avi`, `.webm`, `.mov`, `.mkv`.
- Kiểm tra `content_type`.
- Giới hạn dung lượng bằng `MAX_UPLOAD_MB`.
- Từ chối file rỗng.
- Chuẩn hóa `preprocess_mode` trước khi đưa vào pipeline.

#### 4.10.2. Đăng nhập, đăng ký và bảo vệ phiên

Phần đăng nhập nằm ở `api/auth.py` và `web/src/auth.jsx`. Backend cung cấp các endpoint:

- `POST /auth/register`: tạo tài khoản mới, normalize email, kiểm tra mật khẩu tối thiểu 8 ký tự, hash mật khẩu bằng `bcrypt`.
- `POST /auth/login`: xác thực email/mật khẩu và phát hành phiên đăng nhập.
- `POST /auth/logout`: revoke session server-side, xóa cookie đăng nhập và cookie CSRF.
- `GET /auth/me`: kiểm tra phiên hiện tại và trả thông tin user public.
- `GET /auth/csrf`: phát hành CSRF token mới cho các request ghi.

Phiên đăng nhập dùng hai lớp:

- JWT ngắn gọn chứa `sub`, `sid`, `exp`, lưu trong HttpOnly cookie `alpr_session`.
- Bản ghi server-side `AuthSession` trong MongoDB để có thể kiểm tra session còn tồn tại, hết hạn hay đã bị revoke.

CSRF token được lưu ở cookie riêng `alpr_csrf` và frontend gửi lại qua header `X-CSRF-Token` cho các request không phải `GET/HEAD/OPTIONS`. Cách này phù hợp với mô hình cookie auth: cookie đăng nhập được browser gửi tự động, còn thao tác thay đổi dữ liệu cần thêm token mà script frontend lấy chủ động từ `/auth/csrf`.

Các biến cấu hình chính nằm trong `api/core/config.py`:

- `AUTH_SECRET_KEY`: khóa ký JWT, nên cấu hình bằng biến môi trường khi triển khai thật.
- `AUTH_COOKIE_NAME`: mặc định `alpr_session`.
- `AUTH_COOKIE_SECURE`: bật secure cookie khi chạy HTTPS.
- `AUTH_SESSION_TTL_HOURS`: thời gian sống của session, mặc định 24 giờ.
- `CSRF_COOKIE_NAME`: mặc định `alpr_csrf`.
- `WEB_ORIGIN`: danh sách origin được CORS cho phép.

Frontend bọc toàn bộ app bằng `AuthProvider` trong `web/src/main.jsx`. Provider tự gọi `/auth/me` khi app khởi động để khôi phục phiên. Các hàm `login`, `register`, `logout` dùng chung `apiJson`, tự gửi cookie với `credentials: include` và reset CSRF cache sau khi trạng thái auth thay đổi.

#### 4.10.3. Dashboard sau đăng nhập

Dashboard chính nằm trong `DashboardPage` của `web/src/App.jsx`, route `/dashboard` được bảo vệ bằng `RequireAuth`. Nếu chưa có user, frontend tự chuyển về `/login`; nếu user đã đăng nhập mà vào `/login` hoặc `/register`, route public auth tự chuyển về `/dashboard`.

Dashboard có hai chế độ:

- `Xử lý video`: upload video, chọn preprocessing mode, chọn OCR backend, xem frame annotate realtime, tiến trình xử lý, thống kê OCR và danh sách xe nhận dạng.
- `Giám sát sự kiện`: dùng `MonitorPage` để xử lý nguồn live/upload theo luồng event monitor.

Trong chế độ xử lý video, dashboard hiển thị:

- Drop zone upload video khi trạng thái `idle`.
- `LiveFrame` để xem video gốc hoặc frame annotate từ SSE event `frame`.
- `OcrStatsPanel` để theo dõi số xe nhận dạng, xe bị reject và thông tin OCR.
- `VehiclePanel` để xem danh sách track đã hoàn tất, biển số, confidence và evidence.
- Nút `Lịch sử` mở `HistoryModal`.
- Nút `Đăng xuất` revoke session.

Dashboard truyền `preprocess_mode` và `ocr_backend` xuống backend qua `FormData`. Người dùng có thể chọn các chế độ tiền xử lý `none`, `night`, `low_contrast`, `fog`, `rain`, `glare`; đồng thời chọn backend OCR `default`, `smalllpr_ctc`, `parseq`, `yolov5_char`, `vietnamese_yolov5` để phục vụ demo và so sánh.

#### 4.10.4. Landing page public

Landing page nằm ở `web/src/pages/LandingPage.jsx` và được mount ở route `/`. Đây là trang public, không chạy nhận diện trực tiếp để tránh biến upload video thành thao tác ngoài phiên bảo vệ. Trang này có vai trò giới thiệu ngắn gọn năng lực đã có trong repo:

- Video ALPR từ upload.
- Tracking đa xe.
- OCR fusion đa khung hình.
- Các điều kiện khó như đêm, mưa, sương, lóa, tương phản thấp.
- Dashboard lịch sử với MongoDB.
- Triển khai nội bộ bằng FastAPI, React và MongoDB.

CTA của landing page thay đổi theo trạng thái đăng nhập:

- Nếu đang kiểm tra phiên: hiển thị trạng thái chờ.
- Nếu đã đăng nhập: dẫn vào `/dashboard`.
- Nếu chưa đăng nhập: dẫn đến `/register`, đồng thời header có link `/login`.

Trang cũng công bố các chỉ số đã có artifact nội bộ như YOLOv8 OBB mAP50, SmallLPR-NAR validation accuracy, PARSeq sequence accuracy, quality router binary accuracy và speedup của async pipeline. Các chỉ số này được trình bày thận trọng như kết quả validation/benchmark nội bộ, không mô tả như benchmark sản phẩm ngoài thực tế.

#### 4.10.5. Dashboard history và phân quyền dữ liệu

Lịch sử nhận dạng được mở từ `HistoryModal.jsx`. Frontend gọi:

- `GET /sessions?limit=50`: lấy các phiên xử lý gần nhất của user hiện tại.
- `GET /sessions/{session_id}/records`: lấy các record biển số trong một phiên.
- `GET /records/{job_id}/{track_id}`: lấy chi tiết một track/record cụ thể.

Các route này đều gọi `get_current_user` và truy vấn MongoDB bằng các hàm có hậu tố `_for_user`, ví dụ `list_sessions_for_user`, `get_session_for_user`, `get_records_for_session_for_user`, `get_record_by_track_for_user`. Nhờ đó, user chỉ thấy các session/record có `user_id` của chính mình. Nếu truy cập session của user khác, API trả `404` thay vì lộ sự tồn tại của tài nguyên.

MongoDB có thêm collections và index phục vụ auth/dashboard:

- `users`: unique index theo `email`.
- `auth_sessions`: unique index theo `session_id`, index theo `user_id` và `expires_at`.
- `recognition_sessions`: index theo `(user_id, created_at)`.
- `recognition_records`: index theo `(user_id, session_id)` và unique compound `(session_id, track_id)`.

#### 4.10.6. Monitor và live event

Monitor nằm trong `api/routes_monitor.py` và frontend `web/src/components/monitor`.

Có hai chế độ:

- Upload monitor: upload video, chọn đoạn `t_start -> t_end`, giới hạn tối đa `30s`.
- Live monitor: connect RTSP, MediaMTX tạo path, `LiveSession` decode frame vào rolling buffer `10s`, người dùng bấm mark để phân tích cửa sổ gần nhất.

`LiveSession` dùng một decoder thread, lưu frame vào deque theo FPS. Nếu stream mất, có retry 3 lần. Khi mark live, backend snapshot 10 giây buffer và đưa vào `LiveBufferFrameSource`.

Event analyzer chuyển các event thường thành event có prefix:

- `event_started`.
- `event_progress`.
- `event_vehicle`.
- `event_rejected_vehicle`.
- `event_complete`.
- `event_error`.

#### 4.10.7. Frontend

Frontend dùng React/Vite. Các phần chính:

- `App.jsx`: định nghĩa routing `/`, `/login`, `/register`, `/dashboard`; bảo vệ dashboard bằng `RequireAuth`; chuyển giữa chế độ xử lý video và monitor.
- `auth.jsx`: quản lý trạng thái user, login, register, logout và khôi phục phiên qua `/auth/me`.
- `apiClient.js`: wrapper fetch dùng `credentials: include`, tự lấy CSRF token cho request ghi.
- `LandingPage.jsx`: trang giới thiệu public và CTA vào đăng ký/dashboard.
- `DropZone.jsx`: chọn video, chọn preprocessing mode và OCR backend.
- `LiveFrame.jsx`: hiển thị video/frame annotate.
- `VehiclePanel.jsx`: danh sách xe đã nhận dạng.
- `OcrStatsPanel.jsx`: thống kê OCR và rejected vehicles.
- `HistoryModal.jsx`: danh sách session và record theo user.
- `MonitorPage.jsx`: chọn nguồn live/upload, mark event và xem panel sự kiện.

Người dùng có thể chọn preprocessing:

- `none`.
- `night`.
- `low_contrast`.
- `fog`.
- `rain`.
- `glare`.

Người dùng cũng có thể chọn OCR backend để so sánh:

- `default`.
- `smalllpr_ctc`.
- `parseq`.
- `yolov5_char`.
- `vietnamese_yolov5`.

Vite dev server proxy các route backend `/upload`, `/stream`, `/records`, `/auth`, `/sessions`, `/monitor`, `/events` về `http://localhost:8000`, nên khi phát triển local frontend vẫn gọi API bằng path tương đối.

#### 4.10.8. Lưu trữ

MongoDB models gồm:

- `User`: tài khoản ứng dụng, gồm email, tên, password hash, role và trạng thái active.
- `AuthSession`: session server-side cho HttpOnly auth cookie.
- `RecognitionSession`: một job xử lý video.
- `RecognitionRecord`: một xe/track đã nhận dạng.
- `Event`: một đoạn sự kiện được mark.
- `EventVehicle`: kết quả xe trong event.

`RecognitionSession` và `RecognitionRecord` có trường `user_id` để gắn kết quả với chủ tài khoản. Ảnh crop biển số và thumbnail phương tiện được upload sang object storage, MongoDB lưu metadata và URL thay vì nhét ảnh lớn vào document.

### 4.11. Giai đoạn 11: Pipeline bất đồng bộ và benchmark hiệu năng

`api/core/pipeline_async.py` chia pipeline thành ba stage:

```text
Stage 1: Reader thread
Stage 2: Vehicle detect + vehicle track
Stage 3: Plate cascade + OCR + finalization
```

Lý do dùng threading là PyTorch/OpenCV có phần native release GIL, đồng thời không phải serialize CUDA tensor như multiprocessing.

Benchmark mẫu trong `data/benchmark/results/pipeline_async/timing_ab_comparison.csv`:

| Video | Sync FPS | Async FPS | Sync wall time | Async wall time | Speedup |
|---|---:|---:|---:|---:|---:|
| `hcm_night_01.mp4` | 6.40 | 11.59 | 140.683s | 77.736s | 1.81x |
| `hcm_night_03.mp4` | 4.54 | 6.47 | 198.293s | 139.332s | 1.43x |

`timing_detail.csv` còn chia thời gian theo stage như vehicle detect, vehicle track, crop prep, plate cascade, plate postprocess, association và OCR. Đây là cơ sở để phân tích bottleneck trong Chương 5.

## 5. Bảng tổng hợp độ chính xác và kết quả hiện có

### 5.1. Detection biển số

| Hạng mục | Giá trị |
|---|---:|
| Dataset | `data/datasets/lp_detection_obb` |
| Tổng ảnh | 40.465 |
| Train | 33.333 |
| Val | 7.132 |
| Model | YOLOv8 OBB |
| Best epoch | 22 |
| Precision | 96.776% |
| Recall | 94.489% |
| mAP50 | 98.292% |
| mAP50-95 | 95.015% |

### 5.2. OCR

| OCR model | Best metric trong repo | Ghi chú |
|---|---:|---|
| SmallLPR autoregressive, run `20260608_041915` | Val acc 94.337% | STN + CBAM backbone + MiniLMv2 decoder |
| SmallLPR-CTC, run `ctc_20260609_155238` | Val acc 93.579% | Backend mặc định hiện tại |
| SmallLPR-NAR, run `nar_20260608_123600` | Val acc 95.811% | Cao nhất trong log SmallLPR |
| PARSeq VN plate | Val seq acc 95.459%, char acc 98.296% | Fine-tune từ PARSeq |
| SlotLPR checkpoint | Val acc 89.83% theo tên file | Không phải backend mặc định |

Dataset OCR chính:

- Train: `28.266`.
- Valid: `5.568`.

### 5.3. Quality router

| Model | Accuracy top-1 | Ghi chú |
|---|---:|---|
| Legibility 4 lớp | 85.287% | Phân loại `illegible/poor/good/perfect` |
| Binary quality | 95.603% | Phân loại `suitable/unsuitable` |
| Legibility fine-tuned VN | 84.705% | Run thử/fine-tune khác |

Inference summary trên Platesmania VN:

- Train source `data/raw/platesmania_vn/ocr/images/train`: `29.945` ảnh, mean router confidence `0.837883`.
- Val source `data/raw/platesmania_vn/ocr/images/val`: `7.219` ảnh, mean router confidence `0.8382`.

Lưu ý: hai file summary này là phân phối dự đoán của router trên nguồn Platesmania VN, không phải accuracy vì không có ground truth legibility trong file summary.

### 5.4. Vehicle detector và ReID

| Model | Artifact hiện có | Số liệu hiện có |
|---|---|---|
| Vehicle detector | `weights/detection/vehicle_best.pt` | Chưa thấy `results.csv`/report đánh giá trong repo |
| Vehicle ReID | `weights/tracking/vehicle_reid.pt`, `.onnx`, dataset tracking | Chưa thấy log R1/mAP được lưu kèm trong repo |

Phần này cần chạy lại đánh giá trước khi đưa vào bảng kết quả cuối:

- Vehicle detector: chạy `YOLO.val()` với dataset YAML tương ứng.
- ReID: chạy `tracking/train/evaluate_reid.py` để lấy Rank-1, Rank-5, Rank-10 và mAP.

### 5.5. Hiệu năng pipeline

| Video | Sync FPS | Async FPS | Speedup |
|---|---:|---:|---:|
| `hcm_night_01.mp4` | 6.40 | 11.59 | 1.81x |
| `hcm_night_03.mp4` | 4.54 | 6.47 | 1.43x |

Đây mới là benchmark mẫu trên hai video đêm. Để báo cáo cuối thuyết phục hơn, nên chạy thêm theo nhóm ngày/đêm/mưa/glare/nhiều xe và tính trung bình.

## 6. Phân loại rõ preprocessing, training, heuristic và post-processing

### 6.1. Preprocessing

Preprocessing là các bước biến đổi dữ liệu hoặc ảnh trước khi model chính đọc vào.

Trong project gồm:

1. Tiền xử lý dữ liệu trước huấn luyện:
   - Parse Platesmania HTML, chuẩn hóa text biển.
   - Crop polygon Label Studio thành OCR crop.
   - Chuẩn hóa YOLO OBB label.
   - Split train/val.
   - Tạo dataset quality từ annotation LPLCv2.

2. Tiền xử lý frame trước detection:
   - `PreprocessedFrameSource`.
   - Mode `night`: gamma + CLAHE + bilateral filter.
   - Mode `low_contrast`: contrast stretch + CLAHE + unsharp.
   - Mode `fog`: haze reduction nhẹ + CLAHE.
   - Mode `rain`: median/bilateral denoise + unsharp.
   - Mode `glare`: mask vùng chói, inpaint nhẹ, CLAHE.

3. Tiền xử lý hình học cho biển số:
   - Crop vehicle region.
   - Map OBB từ crop-local về global.
   - `warp_plate_crop` bằng perspective transform.

4. Tiền xử lý OCR input:
   - `smart_resize` giữ aspect ratio và pad về `48x96`.
   - Normalize pixel về `[-1,1]` cho SmallLPR.
   - PARSeq dùng transform riêng, RGB và ImageNet-style normalization.

5. Candidate preprocessing trước OCR cho crop kém:
   - sharpen.
   - gamma.
   - grayscale.
   - CLAHE.
   - contrast stretch.
   - denoise.
   - white balance.

### 6.2. Huấn luyện mô hình

Các phần thuộc training/model learning:

1. YOLO vehicle detector:
   - Fine-tune YOLO trên dữ liệu phương tiện.
   - Artifact có checkpoint, nhưng thiếu report định lượng trong repo.

2. YOLOv8 OBB plate detector:
   - Train/fine-tune trên dataset OBB `BSD/BSV`.
   - Có precision, recall, mAP50, mAP50-95.

3. OCR models:
   - SmallLPR autoregressive.
   - SmallLPR-CTC.
   - SmallLPR-NAR.
   - PARSeq fine-tune.
   - SlotLPR/YOLOv5 char như baseline/phụ trợ.

4. Quality router:
   - YOLOv8 classification 4 lớp.
   - YOLOv8 classification binary.

5. ReID model:
   - VehicleReIDNet với triplet loss + CE.
   - Export sang `.pt` và `.onnx` để dùng trong BoT-SORT.

### 6.3. Heuristic

Heuristic là các luật quyết định thủ công, không phải model học trực tiếp.

Trong project gồm:

1. Ngưỡng detection:
   - `PLATE_DET_CONF = 0.50`.
   - `MIN_PLATE_W = 30`.
   - `MIN_PLATE_H = 15`.

2. Quality score:
   - `0.80 * sharpness + 0.20 * size`.
   - `_HARD_GATE_MIN = 0.05`.

3. Degradation diagnosis fallback:
   - Brightness < 70 -> low light.
   - Contrast < 24 -> low contrast.
   - Laplacian < 80 -> motion blur.
   - Color spread > 55 -> faulty color.
   - Các rule khác cho haze/rain/occlusion.

4. Route mapping:
   - `perfect/good -> direct`.
   - `poor -> tracklet_fusion`.
   - `illegible/occluded -> unreadable_wait`.

5. Association:
   - Chọn xe nhỏ nhất chứa tâm biển.
   - Vote 5 frame, cần 60% agreement.
   - Revalidate locked association.

6. Candidate reranking:
   - Bonus nếu format hợp lệ.
   - Penalty nếu invalid hoặc transform rủi ro.
   - Bonus nhẹ cho original crop.
   - Temporal agreement bonus nếu có.

7. OCR ambiguity correction:
   - Sửa/rerank các ký tự dễ nhầm trong ngữ cảnh biển số, ví dụ nhóm số/chữ.

### 6.4. Post-processing

Post-processing là xử lý sau khi model đã trả prediction.

Trong project gồm:

1. CTC greedy decode:
   - Argmax theo time step.
   - Bỏ blank.
   - Gộp ký tự lặp liên tiếp.

2. NAR decode:
   - Argmax từng output slot.
   - Bỏ pad.

3. PARSeq decode:
   - Tokenizer decode từ probability.
   - Convert label thành list ký tự và confidence.

4. CTM fusion:
   - Chọn template biển.
   - Align OCR outputs vào slot.
   - Vote theo slot.
   - Chỉ nhận ký tự đủ majority support.

5. Format validation:
   - Normalize text.
   - Match `VN_PLATE_RE`.
   - Nếu không hợp lệ thì reject.

6. Track finalization:
   - Khi track mất `LOST_THRESHOLD` stride hoặc hết video.
   - Chọn top-k frame theo combined score.
   - Emit `vehicle` hoặc `rejected_vehicle`.

7. Evidence packaging:
   - Chọn crop biển tốt nhất.
   - Chọn thumbnail xe tốt nhất.
   - Encode base64 cho UI hoặc upload storage khi lưu DB.

## 7. Vì sao các quyết định triển khai này hợp lý với đề tài

### 7.1. Vì sao dùng vehicle-first cascade

Biển số thường nhỏ hơn rất nhiều so với frame. Nếu detect toàn frame, model phải tìm một object nhỏ giữa nhiều chi tiết nền, biển quảng cáo, chữ trên xe, đèn, biển đường. Khi detect phương tiện trước, hệ thống giảm vùng tìm kiếm xuống vùng có xác suất chứa biển số cao. Điều này phù hợp với video giao thông vì hầu hết biển số cần đọc đều gắn với một phương tiện đã detect được.

### 7.2. Vì sao dùng OBB và warp phối cảnh

Biển số trong video không luôn song song với camera. Xe có thể đi chéo, camera đặt cao hoặc lệch. YOLOv8 OBB trả bốn điểm góc giúp crop biển số theo đúng mặt phẳng hơn. Sau khi warp, OCR nhận ảnh ít background hơn và ký tự ít bị nghiêng hơn.

### 7.3. Vì sao cần tracking trước OCR final

Một frame đơn có thể OCR sai do mờ hoặc che khuất. Nhưng nếu cùng một biển xuất hiện trong 10 frame, có thể 3-5 frame đủ tốt. Tracking cho phép hệ thống:

- Gom crop cùng xe.
- Chọn frame tốt hơn.
- So sánh nhiều kết quả OCR.
- Không emit kết quả sai quá sớm.

Đây chính là phần bám sát yêu cầu "dùng thông tin thời gian từ nhiều khung hình liên tiếp" trong phiếu nhiệm vụ.

### 7.4. Vì sao cần quality router

Không phải crop nào cũng nên xử lý giống nhau. Nếu crop tốt, OCR trực tiếp tiết kiệm thời gian. Nếu crop xấu, OCR trực tiếp dễ sinh kết quả sai nhưng trông có vẻ hợp lệ. Quality router giúp hệ thống quyết định:

- Đọc ngay.
- Buffer/fusion.
- Chờ frame khác hoặc reject.

Đây là lớp kiểm soát rủi ro, đặc biệt quan trọng khi demo cho dữ liệu video thực tế.

### 7.5. Vì sao dùng CTM thay vì vote index đơn giản

Vote index đơn giản giả định mọi OCR output có cùng độ dài và cùng alignment. Điều này không đúng với biển số vì OCR có thể thiếu dấu `-`, thiếu `[SEP]`, dư ký tự hoặc lệch vị trí. CTM chọn template rồi align vào slot trước khi vote, nên phù hợp hơn với biển Việt Nam nhiều định dạng.

### 7.6. Vì sao reject là một kết quả hợp lệ

Trong ALPR, nhận sai một biển số hợp lệ còn nguy hiểm hơn là báo không đọc được. Vì vậy, hệ thống có `rejected_vehicle`. Đây không phải lỗi pipeline mà là một quyết định an toàn: nếu bằng chứng không đủ, không ghi nhận biển số như kết quả đúng.

## 8. Kiểm thử và chất lượng phần mềm

Repo có `41` file test Python, bao phủ các nhóm:

- Association và tracking: `test_association.py`, `test_tracker_adapter.py`, `test_track_buffer.py`.
- Plate cascade và crop: `test_cascade_plate.py`, `test_warp_plate_crop.py`.
- OCR hậu xử lý: `test_ocr_ctm.py`, `test_ocr_candidates.py`, `test_ocr_ambiguity.py`, `test_plate_format.py`.
- Quality router: `test_quality_router.py`, `test_evaluate_quality_router.py`, `test_prepare_lplcv2_quality_dataset.py`.
- Pipeline: `test_pipeline_async.py`, `test_pipeline_core_parity.py`, `test_pipeline_progress.py`.
- Monitor/live: `test_monitor_routes.py`, `test_live_session.py`, `test_event_analyzer.py`, `test_event_crud.py`, `test_mediamtx_client.py`.
- Auth và dashboard API: `test_auth_and_dashboard_routes.py`, kiểm tra đăng ký, đăng nhập, `/auth/me`, logout, route yêu cầu đăng nhập và việc lọc session/record theo `user_id`.
- Data scripts: tests cho collect/crop/generate/refresh.
- Synthetic pipeline: test V3/V4.

Trong môi trường hiện tại, lệnh `pytest` không có trong PATH và `python3 -m pytest` báo `No module named pytest`, nên em chưa chạy lại được toàn bộ test tại thời điểm viết tài liệu này. Tuy nhiên, cấu trúc test đã có đủ để chứng minh các module quan trọng được thiết kế theo hướng có thể kiểm thử.

## 9. Những phần đã đạt được so với yêu cầu

Em đã hoàn thành được một hệ thống ALPR dạng end-to-end:

- Có landing page public giới thiệu năng lực hệ thống và dẫn người dùng vào đăng ký/đăng nhập/dashboard.
- Có đăng ký, đăng nhập, đăng xuất, khôi phục phiên và CSRF protection cho request ghi.
- Có dashboard riêng sau đăng nhập, không cho upload/xem stream khi chưa có phiên hợp lệ.
- Có backend FastAPI nhận video.
- Có frontend React để upload và xem kết quả.
- Có live/event monitor mở rộng cho RTSP hoặc video upload.
- Có detector phương tiện.
- Có detector biển số OBB với metric tốt.
- Có crop phối cảnh biển số.
- Có tracking phương tiện bằng BoT-SORT + ReID.
- Có tracking biển số và association theo nhiều frame.
- Có nhiều OCR backend đã huấn luyện/đánh giá.
- Có quality router để phân luồng crop tốt/xấu.
- Có track buffer và CTM fusion để tận dụng nhiều frame.
- Có cơ chế reject thay vì nhận dạng bừa.
- Có lưu session/record/event và ảnh chứng cứ, đồng thời lọc lịch sử nhận dạng theo tài khoản.
- Có benchmark hiệu năng sync/async.

Nếu đối chiếu với phiếu giao nhiệm vụ, phần sản phẩm kỳ vọng đã đạt và còn mở rộng thêm live monitor. Phần nghiên cứu mô hình cũng vượt mức "dùng YOLO + OCR" cơ bản vì đã có các biến thể OCR, quality router và CTM fusion.

## 10. Những điểm còn thiếu hoặc nên bổ sung trước báo cáo cuối

Để báo cáo tốt nghiệp chặt chẽ hơn, em nên bổ sung các phần sau:

1. Đánh giá chính thức vehicle detector:
   - Precision, recall, mAP50, mAP50-95 trên dataset phương tiện.
   - Lưu `results.csv` hoặc report vào repo.

2. Đánh giá chính thức ReID:
   - Rank-1, Rank-5, Rank-10, mAP trên query/gallery.
   - Nếu có thể, thêm ID switch/fragmentation ở mức video.

3. Đánh giá end-to-end ALPR:
   - Theo từng video: số xe ground truth, số xe detect, số biển đúng, sai, reject.
   - Plate exact-match accuracy ở vehicle-track level.
   - False positive rate của biển số qua regex.

4. Ablation study:
   - Không cascade vs cascade.
   - Single-frame OCR vs CTM fusion.
   - Không quality router vs có quality router.
   - Không preprocessing vs các mode `night`, `low_contrast`, `fog`, `rain`, `glare`.
   - SmallLPR-CTC vs NAR vs PARSeq trên cùng tập crop.

5. Bổ sung môi trường thực nghiệm:
   - CPU/GPU/RAM.
   - Version PyTorch, CUDA, Ultralytics.
   - Thời gian load model, VRAM nếu đo được.

6. Chạy lại test:
   - Cài `pytest`.
   - Chạy unit/integration test.
   - Ghi số test pass/fail vào báo cáo.

7. Hoàn thiện checklist triển khai web:
   - Cấu hình `AUTH_SECRET_KEY` cố định bằng biến môi trường khi deploy, không dùng secret tạm.
   - Bật `AUTH_COOKIE_SECURE=true` khi chạy HTTPS.
   - Chụp screenshot landing page, trang đăng nhập và dashboard để đưa vào báo cáo.
   - Bổ sung một E2E test ngắn cho luồng landing page -> đăng ký/đăng nhập -> dashboard -> lịch sử.

## 11. Kết luận theo giọng đồ án

Nhìn lại toàn bộ project, em thấy phần quan trọng nhất của đồ án không chỉ là train một model OCR hay một model YOLO, mà là ghép các model đó thành một hệ thống đủ cẩn thận cho video thực tế. Video có nhiều frame, nhưng cũng có nhiều nhiễu. Nếu tận dụng frame tốt và loại frame xấu, kết quả ổn hơn. Nếu cứ OCR mọi crop và ép ra biển số hợp lệ, hệ thống có thể trông "nhiều kết quả" hơn nhưng độ tin cậy thấp hơn.

Vì vậy, hướng em chọn là pipeline nhiều tầng: phát hiện xe, phát hiện biển theo crop xe, tracking, quality routing, OCR theo route, buffer đa khung hình, CTM fusion và validation định dạng. Các mô hình học sâu đảm nhiệm phần nhận biết phức tạp như detect/OCR/classify quality, còn heuristic được dùng ở những chỗ cần ràng buộc nghiệp vụ rõ ràng như ngưỡng confidence, luật định dạng biển Việt Nam, ghép biển với xe và quyết định reject.

Kết quả hiện có cho thấy các khối chính đã đạt mức khả quan: detector biển số OBB đạt mAP50-95 khoảng `95.0%`, OCR tốt nhất trong log đạt khoảng `95.8%` exact/val accuracy với SmallLPR-NAR, PARSeq đạt `95.46%` sequence accuracy và `98.30%` character accuracy, quality router binary đạt `95.6%`. Pipeline async cũng cải thiện tốc độ khoảng `1.43x` đến `1.81x` trên benchmark mẫu.

Phần còn thiếu chủ yếu không phải là code pipeline, mà là đánh giá cuối cùng ở mức hệ thống: vehicle detector, ReID, tracking metrics và end-to-end accuracy trên tập video có ground truth đầy đủ. Đây là phần em nên hoàn thiện tiếp để báo cáo cuối có đủ bằng chứng định lượng, không chỉ mô tả kỹ thuật.
