# Mục lục đề xuất cho ĐATN ver.0

Ghi chú: mục lục dưới đây được viết lại từ `DATN_Muc_luc.docx` và bám theo codebase hiện tại của project `ALPR_Vietnamese`. Cấu trúc giữ tối đa 3 cấp: Chương, mục `x.y`, tiểu mục `x.y.z`; các chi tiết nhỏ hơn nên đưa vào thân bài, không đưa vào mục lục.

## Định hướng chỉnh sửa chính

- Tách rõ Chương 3 là phương pháp/giải pháp đề xuất: kiến trúc, thuật toán, pipeline, các cải tiến so với việc chỉ dùng mô hình có sẵn.
- Thêm Chương 4 cho triển khai cài đặt phần mềm: backend, frontend, streaming, cơ sở dữ liệu, cấu hình, demo và kiểm thử.
- Đẩy đánh giá thực nghiệm sang Chương 5: đánh giá chất lượng mô hình, chất lượng đầu cuối, hiệu năng hệ thống và so sánh với baseline/sản phẩm tương tự.
- Chương 6 kết luận: nêu đúng đóng góp, hạn chế, hướng phát triển.

## Mục lục chi tiết

### Chương 1. Giới thiệu đề tài

**1.1. Đặt vấn đề**

1.1.1. Nhu cầu nhận dạng biển số xe trong giám sát giao thông và quản lý sự kiện  
1.1.2. Đặc thù biển số xe Việt Nam trong ảnh và video thực tế  
1.1.3. Các thách thức: ban đêm, mờ nhòe, góc nghiêng, nhiều phương tiện, biển số nhỏ

**1.2. Các giải pháp hiện tại và hạn chế**

1.2.1. Nhóm giải pháp thương mại ALPR dựa trên camera/IP camera  
1.2.2. Nhóm giải pháp nhận dạng biển số bằng pipeline học sâu  
1.2.3. Hạn chế khi áp dụng trực tiếp vào dữ liệu xe máy và biển số Việt Nam

**1.3. Mục tiêu và phạm vi đề tài**

1.3.1. Mục tiêu nhận dạng biển số từ video upload và luồng RTSP/live  
1.3.2. Phạm vi phương tiện: ô tô, xe buýt, xe tải, xe máy và người đi xe máy  
1.3.3. Phạm vi đầu ra: biển số, ảnh chứng cứ, lịch sử nhận dạng và sự kiện đánh dấu

**1.4. Định hướng giải pháp**

1.4.1. Pipeline phát hiện phương tiện, phát hiện biển số, theo vết và OCR  
1.4.2. Tận dụng thông tin đa khung hình để giảm sai số OCR  
1.4.3. Kết hợp kiểm tra chất lượng ảnh và ràng buộc định dạng biển số Việt Nam

**1.5. Đóng góp của đồ án**

1.5.1. Bộ pipeline ALPR tích hợp cho video và live monitor  
1.5.2. Bộ xử lý biển số nghiêng bằng YOLOv8 OBB và crop phối cảnh  
1.5.3. Cơ chế buffer đa khung hình, voting OCR và loại bỏ kết quả không hợp lệ  
1.5.4. Demo phần mềm gồm giao diện web, backend API, stream kết quả và lưu trữ chứng cứ

**1.6. Bố cục đồ án**

1.6.1. Nội dung Chương 1 và Chương 2  
1.6.2. Nội dung Chương 3 và Chương 4  
1.6.3. Nội dung Chương 5 và Chương 6

### Chương 2. Cơ sở lý thuyết và công nghệ nền tảng

**2.1. Bài toán nhận dạng biển số xe tự động**

2.1.1. Định nghĩa bài toán ALPR/ANPR  
2.1.2. Các bước xử lý phổ biến trong hệ thống ALPR  
2.1.3. Các độ đo thường dùng cho phát hiện, OCR, tracking và hệ thống thời gian thực

**2.2. Mô hình phát hiện đối tượng**

2.2.1. Tổng quan kiến trúc YOLO  
2.2.2. YOLOv8 cho phát hiện phương tiện  
2.2.3. YOLOv8 OBB cho phát hiện biển số có góc xoay

**2.3. Theo vết đối tượng trong video**

2.3.1. Bài toán multi-object tracking  
2.3.2. BoT-SORT và ByteTrack  
2.3.3. Re-identification và các độ đo CMC, mAP cho ReID

**2.4. Nhận dạng ký tự biển số**

2.4.1. OCR theo hướng end-to-end cho chuỗi ký tự  
2.4.2. LPRNet/SmallLPR và CTC/decoder tuần tự  
2.4.3. OCR đa khung hình và voting xác suất

**2.5. Tiền xử lý và đánh giá chất lượng ảnh**

2.5.1. CLAHE, gamma correction và tăng tương phản  
2.5.2. Phát hiện ảnh mờ bằng Laplacian variance  
2.5.3. Xử lý các tình huống đêm, mưa, sương mù, glare và low contrast

**2.6. Công nghệ sử dụng**

2.6.1. Python, PyTorch, Ultralytics YOLO và OpenCV  
2.6.2. FastAPI, SSE, MJPEG/WebRTC và MediaMTX  
2.6.3. React/Vite, MongoDB và Supabase Storage

### Chương 3. Phương pháp đề xuất

**3.1. Tổng quan kiến trúc hệ thống**

3.1.1. Sơ đồ pipeline tổng thể từ video/RTSP đến kết quả nhận dạng  
3.1.2. Hai luồng xử lý: upload video và live event monitor  
3.1.3. Thiết kế `FrameSource` để thống nhất nguồn dữ liệu file và live buffer

**3.2. Khối tiền xử lý đầu vào**

3.2.1. Chuẩn hóa frame từ `FileFrameSource` và `LiveBufferFrameSource`  
3.2.2. Các chế độ tiền xử lý: `night`, `low_contrast`, `fog`, `rain`, `glare`  
3.2.3. Điều kiện áp dụng và tác động dự kiến đến phát hiện/OCR

**3.3. Khối phát hiện phương tiện**

3.3.1. Mô hình `vehicle_best.pt` và các lớp phương tiện trong hệ thống  
3.3.2. Lọc lớp phương tiện bằng `VEHICLE_CLASSES`  
3.3.3. Vai trò của phát hiện phương tiện trong cascade biển số

**3.4. Khối phát hiện biển số bằng YOLOv8 OBB**

3.4.1. Mô hình `best.pt` cho biển số dạng OBB  
3.4.2. Cascade vehicle-first: phát hiện biển số trên crop phương tiện  
3.4.3. Chuyển tọa độ crop-local về tọa độ frame toàn cục  
3.4.4. Khử trùng lặp ứng viên biển số từ các crop phương tiện chồng lấn

**3.5. Khối crop phối cảnh và kiểm soát chất lượng**

3.5.1. Crop biển số từ bốn điểm OBB bằng `warp_plate_crop`  
3.5.2. Lọc theo confidence, kích thước tối thiểu và độ sắc nét  
3.5.3. Tính `quality_score` cho chọn frame đại diện

**3.6. Khối theo vết và ghép biển số với phương tiện**

3.6.1. BoT-SORT kết hợp ReID cho theo vết phương tiện  
3.6.2. `PlateTrackManager` cho theo vết biển số sau cascade  
3.6.3. `TrajectoryAssociator` để khóa quan hệ biển số - phương tiện theo trajectory vote

**3.7. Khối OCR biển số**

3.7.1. Tiền xử lý crop biển số về kích thước `96x48`  
3.7.2. SmallLPR/LPRNet cho nhận dạng chuỗi ký tự  
3.7.3. PaddleOCR là hướng baseline/thử nghiệm trong `main_paddle.py`

**3.8. Tổng hợp đa khung hình**

3.8.1. `TrackBuffer` lưu tối đa `MAX_BUFFER` crop cho mỗi track  
3.8.2. Chính sách loại frame kém bằng điểm kết hợp `quality_score * ocr_conf`  
3.8.3. Chọn `TOP_K_FRAMES` và voting bằng `segment_vote`/`prob_vote`  
3.8.4. Tùy chọn `MultiFrameSmallLPR` khi có checkpoint đa khung hình

**3.9. Xác thực và hậu xử lý kết quả**

3.9.1. Regex định dạng biển số Việt Nam trong pipeline  
3.9.2. Cơ chế `rejected_vehicle` để giảm false positive OCR  
3.9.3. Lưu ảnh biển số tốt nhất, crop buffer và thumbnail phương tiện làm chứng cứ

**3.10. Pipeline bất đồng bộ cho tăng tốc**

3.10.1. Thiết kế ba stage: reader, vehicle detect/track, plate/OCR  
3.10.2. Back-pressure bằng queue và các chỉ số stall  
3.10.3. So sánh pipeline sync và async bằng benchmark A/B

### Chương 4. Triển khai cài đặt phần mềm demo

**4.1. Cấu trúc mã nguồn**

4.1.1. Nhóm module `api/core`, `detection`, `ocr`, `tracking`, `pipeline`  
4.1.2. Nhóm script huấn luyện, tạo dữ liệu và benchmark trong `scripts`  
4.1.3. Nhóm giao diện web trong `web/src`

**4.2. Huấn luyện và chuẩn bị dữ liệu**

4.2.1. Dữ liệu phương tiện và biển số trong `data/datasets` và `data/raw`  
4.2.2. Tạo dữ liệu fine-tune biển số bằng `generate_lp_finetune_dataset.py`  
4.2.3. Sinh nhãn Label Studio bằng `generate_label_studio_annotations.py`  
4.2.4. Gộp dữ liệu YOLOv8 OBB bằng `build_lp_detection_obb_dataset.py`

**4.3. Triển khai backend FastAPI**

4.3.1. API upload video: `/upload`, `/stream/{job_id}`, `/stream/{job_id}/mjpeg`  
4.3.2. API monitor: `/monitor/upload`, `/monitor/live/connect`, `/monitor/{session_id}/mark`  
4.3.3. Quản lý vòng đời model bằng `lifespan` và `ModelBundle`

**4.4. Triển khai live monitor và đánh dấu sự kiện**

4.4.1. Nhận RTSP bằng `LiveSession` và MediaMTX  
4.4.2. Rolling buffer 10 giây cho chế độ live  
4.4.3. Phân tích cửa sổ sự kiện bằng `event_analyzer.py`

**4.5. Triển khai lưu trữ kết quả**

4.5.1. Mô hình dữ liệu `RecognitionSession` và `RecognitionRecord`  
4.5.2. Mô hình dữ liệu `Event` và `EventVehicle`  
4.5.3. Lưu metadata trong MongoDB và ảnh chứng cứ trong object storage

**4.6. Triển khai giao diện người dùng**

4.6.1. Giao diện xử lý video upload trong `App.jsx`  
4.6.2. Giao diện giám sát sự kiện trong `MonitorPage.jsx`  
4.6.3. Các thành phần hiển thị video, danh sách xe, lịch sử và chi tiết event

**4.7. Kiểm thử phần mềm**

4.7.1. Unit test cho buffer, voting, preprocessing và association  
4.7.2. Integration test cho API monitor, stream và event flow  
4.7.3. Regression test bằng golden event cho pipeline

**4.8. Hướng dẫn vận hành demo**

4.8.1. Khởi động backend và frontend  
4.8.2. Demo xử lý video upload  
4.8.3. Demo live/event monitor với RTSP hoặc video giả lập  
4.8.4. Truy xuất lịch sử, ảnh chứng cứ và kết quả OCR

### Chương 5. Đánh giá thực nghiệm

**5.1. Môi trường thực nghiệm**

5.1.1. Cấu hình phần cứng, hệ điều hành, GPU/CPU và phiên bản thư viện  
5.1.2. Cấu hình model, checkpoint và tham số pipeline  
5.1.3. Quy trình chạy thí nghiệm và cách lưu kết quả

**5.2. Dữ liệu thực nghiệm**

5.2.1. Tập dữ liệu phát hiện phương tiện và biển số  
5.2.2. Tập dữ liệu OCR biển số  
5.2.3. Tập video thực tế trong `data/realworld-videos/chunks`  
5.2.4. Tập benchmark trong `data/benchmark/videos`

**5.3. Đánh giá phát hiện biển số**

5.3.1. Precision, recall, mAP50 và mAP50-95 cho YOLOv8 OBB  
5.3.2. Đánh giá theo dạng biển số dài/vuông và theo kích thước biển số  
5.3.3. Phân tích lỗi phát hiện sai, bỏ sót và OBB lệch góc

**5.4. Đánh giá nhận dạng ký tự**

5.4.1. Plate exact-match accuracy  
5.4.2. Character accuracy, CER và edit distance trung bình  
5.4.3. Tỷ lệ kết quả bị loại bởi kiểm tra định dạng biển số Việt Nam

**5.5. Đánh giá tracking và ghép biển số - phương tiện**

5.5.1. ID switch, track fragmentation và duplicate track rate  
5.5.2. Association accuracy giữa biển số và phương tiện  
5.5.3. CMC rank-1 và mAP cho ReID nếu dùng tập query/gallery

**5.6. Đánh giá đầu cuối của hệ thống ALPR**

5.6.1. Vehicle-level plate recognition accuracy  
5.6.2. Recall phương tiện có biển số và false positive rate  
5.6.3. So sánh single-frame OCR, segment voting, probability voting và multi-frame OCR  
5.6.4. Phân tích các trường hợp `rejected_vehicle`

**5.7. Đánh giá hiệu năng hệ thống**

5.7.1. FPS, wall time và latency trên từng video  
5.7.2. Thời gian từng stage: vehicle detection, tracking, plate cascade, OCR, association  
5.7.3. So sánh pipeline sync và async bằng `run_benchmark_async.py`  
5.7.4. Độ trễ SSE/MJPEG và thời gian phản hồi khi đánh dấu sự kiện live

**5.8. Đánh giá theo tình huống thực tế**

5.8.1. Ban ngày, ban đêm, ánh đèn pha và glare  
5.8.2. Mưa, sương mù, low contrast và dữ liệu đã tiền xử lý  
5.8.3. Xe máy đông, ô tô/xe tải/xe buýt, nhiều xe trong cùng khung hình  
5.8.4. Biển số nghiêng, nhỏ, xa camera, mờ chuyển động và bị che khuất một phần  
5.8.5. Luồng live RTSP, buffer chưa đủ frame, mất kết nối và đánh dấu nhiều sự kiện

**5.9. So sánh với baseline và sản phẩm tương tự**

5.9.1. So sánh với PaddleOCR baseline trong codebase  
5.9.2. So sánh kiến trúc với OpenALPR/Rekor Scout và Plate Recognizer  
5.9.3. So sánh pipeline và chỉ số FPS/latency với demo OpenVINO Security Barrier Camera  
5.9.4. Phân tích điểm mạnh, điểm yếu của hệ thống đề xuất

### Chương 6. Kết luận và hướng phát triển

**6.1. Kết luận**

6.1.1. Kết quả đạt được về mô hình và pipeline  
6.1.2. Kết quả đạt được về phần mềm demo  
6.1.3. Ý nghĩa của các đánh giá thực nghiệm

**6.2. Hạn chế**

6.2.1. Hạn chế về dữ liệu và độ bao phủ tình huống  
6.2.2. Hạn chế về tốc độ xử lý, OCR và tracking  
6.2.3. Hạn chế khi triển khai live nhiều camera

**6.3. Hướng phát triển**

6.3.1. Mở rộng dữ liệu và chuẩn hóa ground truth cho video  
6.3.2. Tối ưu model và triển khai TensorRT/ONNX/OpenVINO  
6.3.3. Bổ sung dashboard, cảnh báo, tìm kiếm biển số và phân quyền người dùng  
6.3.4. Đánh giá trên nhiều camera và hạ tầng thực tế

## Sản phẩm/hệ thống nên tham chiếu

**1. Plate Recognizer Snapshot/Stream/ParkPow**

- Phù hợp để tham chiếu vì project hiện tại cũng có upload video, live/stream, dashboard kết quả, ảnh chứng cứ và lịch sử. Plate Recognizer công bố hai chế độ Snapshot cho ảnh và Stream cho live/video, có API/webhook/dashboard, cloud/on-prem và nêu các tình huống như ảnh mờ, tối, góc nghiêng, xe chạy nhanh, low-res, nhiều xe, biển số hai dòng.
- Nên dùng trong đồ án: so sánh tính năng và nếu có điều kiện thì thử API Snapshot trên một tập crop/ảnh đại diện.
- Nguồn: https://platerecognizer.com/

**2. Rekor Scout / OpenALPR**

- Phù hợp để tham chiếu vì đây là sản phẩm ALPR chạy với IP/traffic/security camera, có dashboard tìm kiếm, alert và hỗ trợ triển khai trên camera sẵn có. Điều này gần với phần live monitor và lịch sử nhận dạng của project.
- Nên dùng trong đồ án: so sánh ở mức sản phẩm/demo, đặc biệt là khả năng live camera, dashboard, cảnh báo, lưu trữ và tìm kiếm.
- Nguồn: https://www.openalpr.com/software/scout và https://docs.rekor.ai/scout

**3. OpenVINO Security Barrier Camera Demo**

- Phù hợp để tham chiếu kỹ thuật vì demo này cũng ghép vehicle/license plate detection, vehicle attributes và license plate recognition trên ảnh/video/camera; tài liệu cũng nêu FPS và latency là chỉ số đánh giá application-level performance.
- Nên dùng trong đồ án: so sánh kiến trúc pipeline và cách báo cáo hiệu năng FPS/latency.
- Nguồn: https://docs.openvino.ai/2023.3/omz_demos_security_barrier_camera_demo_cpp.html

**4. PaddleOCR**

- Phù hợp làm baseline OCR vì codebase đã có `api/main_paddle.py`, `api/core/models_paddle.py`, `api/core/pipeline_paddle.py` và một test/diagnostic dùng PaddleOCR. Đây là baseline thực nghiệm sát codebase nhất.
- Nên dùng trong đồ án: so sánh SmallLPR/LPRNet của đề tài với PaddleOCR trên cùng crop biển số. Trước khi chạy chính thức cần kiểm tra lại tương thích `pipeline_paddle.py` với chữ ký hiện tại của `TrackBuffer.top_k()`.
- Nguồn: https://www.paddleocr.ai/main/en/index/index.html

**5. Viettel AI License Plate Recognition**

- Phù hợp để tham chiếu trong nước vì là giải pháp nhận diện biển số xe tự động cho giao thông, an ninh và ứng dụng doanh nghiệp tại Việt Nam. Không nên hứa so sánh định lượng nếu không có API/tập test chung.
- Nên dùng trong đồ án: so sánh bối cảnh ứng dụng và nhóm chức năng PM cần có trong sản phẩm Việt Nam.
- Nguồn: https://viettelai.vn/en/tin-tuc/phan-mem-nhan-dien-bien-so-xe-tu-dong-chinh-xac-cua-viettel-ai

## Các đánh giá hiệu năng/chất lượng cần thực hiện

**Nhóm chất lượng mô hình**

- Phát hiện biển số OBB: precision, recall, mAP50, mAP50-95; có thể dùng kết quả `runs/obb/experiments/detection/lp_detection_obb_merged/results.csv` làm bảng huấn luyện/validation ban đầu.
- OCR: plate exact-match accuracy, character accuracy, CER, edit distance trung bình, tỷ lệ chuỗi hợp lệ theo regex biển số Việt Nam.
- Tracking/ReID: ID switch, duplicate track rate, track fragmentation; nếu có dữ liệu ReID đầy đủ thì thêm CMC rank-1, rank-5 và mAP theo `tracking/train/evaluate_reid.py`.
- End-to-end ALPR: tỷ lệ nhận đúng biển số trên mỗi vehicle track, tỷ lệ bỏ sót xe có biển số, tỷ lệ nhận sai nhưng vẫn qua regex, tỷ lệ `rejected_vehicle`.

**Nhóm hiệu năng hệ thống**

- FPS, wall time, số frame xử lý và số phương tiện tìm thấy theo từng video.
- Thời gian theo stage từ `timings`: `vehicle_detect`, `vehicle_track`, `crop_prep`, `plate_cascade`, `plate_postprocess`, `association`, `ocr`.
- So sánh sync/async bằng `data/benchmark/results/pipeline_async/timing_ab_comparison.csv`; hiện đã có kết quả mẫu cho `hcm_night_01.mp4` và `hcm_night_03.mp4`.
- Độ trễ từ lúc upload/mark đến event đầu tiên, đến kết quả vehicle đầu tiên và đến `complete`.
- Tài nguyên: VRAM/RAM/CPU/GPU utilization, thời gian load model, dung lượng ảnh chứng cứ và số bản ghi lưu trong MongoDB/Supabase.

**Nhóm chất lượng phần mềm demo**

- Upload video: xử lý file hợp lệ, file lỗi, video ngắn/dài, nhiều xe.
- Live monitor: kết nối RTSP, buffer warm-up, mark event, ngắt kết nối, reconnect.
- SSE/MJPEG/WebRTC: stream không treo, có ping/timeout, giao diện cập nhật tiến độ.
- Lưu trữ: tạo session/record/event đúng, có ảnh chứng cứ, xem lịch sử được.
- Kiểm thử: bổ sung/ghi nhận unit, integration, E2E cho các luồng demo quan trọng.

## Các tình huống nên đưa vào thực nghiệm

**Theo điều kiện môi trường**

- Ban ngày rõ nét.
- Ban đêm Hà Nội/TP.HCM, nhiều đèn pha.
- Low contrast, glare, mưa, sương mù; chạy cả `preprocess_mode=none` và mode tương ứng.
- Biển số bị motion blur hoặc out-of-focus để kiểm tra `is_sharp` và `quality_score`.

**Theo bố cục giao thông**

- Một xe rõ biển số.
- Nhiều xe trong cùng khung hình, xe che khuất nhau.
- Xe máy đông, biển số nhỏ và nhiều xe chạy sát nhau.
- Ô tô, xe tải, xe buýt và người đi xe máy để kiểm tra đủ class trong `VEHICLE_CLASSES`.
- Xe đi nhanh, xe đi chéo, biển số nghiêng mạnh để kiểm tra lợi ích của YOLOv8 OBB và `warp_plate_crop`.

**Theo định dạng biển số**

- Biển số ô tô 5 số dạng `30G-51827`.
- Biển số xe máy hai cụm dạng `29-X1-12345`.
- Biển số 4 số cũ dạng `31H-9999`.
- Biển vuông/biển dài nếu tập BSD/BSV có nhãn tương ứng.
- Ảnh/track gây OCR sai định dạng để đo `rejected_vehicle`.

**Theo luồng demo**

- Upload video toàn bộ.
- Upload video rồi mark một đoạn tối đa 30 giây trong monitor mode.
- Live RTSP: chờ buffer 1-2 giây, mark 10 giây, xem event result.
- Live RTSP mất kết nối hoặc camera không truy cập được.
- Cùng một xe xuất hiện nhiều frame để kiểm tra chống duplicate và multi-frame voting.

**Theo so sánh/ablation**

- YOLOv8 OBB toàn frame so với cascade vehicle-first.
- Không tiền xử lý so với `night`, `low_contrast`, `fog`, `rain`, `glare`.
- Single-frame OCR so với `segment_vote`/`prob_vote`.
- SmallLPR/LPRNet so với PaddleOCR baseline.
- Pipeline sync so với async.

## Gợi ý bảng biểu nên có trong quyển

- Bảng dữ liệu: số video, số frame, số xe, số biển số, phân bố ngày/đêm/tỉnh/thành/loại xe.
- Bảng thông số model: checkpoint, input size, số lớp, ngưỡng confidence, stride, buffer.
- Bảng kết quả detection: precision/recall/mAP theo tập validation.
- Bảng kết quả OCR: exact-match, char accuracy, CER theo từng tình huống.
- Bảng end-to-end: đúng/sai/bỏ sót/rejected theo từng video.
- Bảng hiệu năng: FPS, latency, stage timing, sync/async speedup.
- Bảng so sánh sản phẩm: đầu vào hỗ trợ, live camera, dashboard, API, lưu lịch sử, cảnh báo, deployment, ưu/nhược điểm.

## Mapping nhanh từ codebase sang nội dung đồ án

- `api/core/pipeline_core.py`: pipeline ALPR lõi, emit event, multi-frame finalization.
- `api/core/pipeline_async.py`: pipeline 3 stage và benchmark hiệu năng.
- `api/core/cascade_plate.py`: cascade vehicle-first, map OBB, dedup và plate tracking.
- `api/core/tracker.py`: `TrackBuffer`, voting, định dạng biển số, ảnh chứng cứ.
- `api/core/preprocessing.py`: các chế độ tiền xử lý theo tình huống.
- `api/routes_monitor.py` và `api/core/live_session.py`: live monitor, RTSP, mark event.
- `api/database/models.py`: mô hình dữ liệu session, record và event.
- `web/src/App.jsx` và `web/src/components/monitor/MonitorPage.jsx`: giao diện demo.
- `scripts/run_benchmark.py` và `scripts/run_benchmark_async.py`: đánh giá đầu cuối và hiệu năng.
- `scripts/build_lp_detection_obb_dataset.py`: xây dữ liệu YOLOv8 OBB.
- `runs/obb/experiments/detection/lp_detection_obb_merged/results.csv`: kết quả huấn luyện/validation detector OBB.
