# Kế hoạch sửa đổi Chương 3 và Chương 4 theo hướng dẫn giáo viên

## Context

Giáo viên yêu cầu phân tách rõ ràng:

- **Chương 2**: "Từ điển các công cụ có sẵn" — kiến thức thuần túy từ bên ngoài (lý thuyết YOLO, BoT-SORT, LPRNet, CTC, v.v.)
- **Chương 3**: "Bản vẽ thiết kế của bạn" — chỉ trình bày sơ đồ tổng thể + biện luận lựa chọn. Khi đụng mô hình đã nói ở Chương 2, chỉ nhắc nhẹ tên, không giải thích lại lý thuyết.
- **Chương 4**: "Công sức triển khai thực tế" — cài đặt, huấn luyện, tích hợp, kết quả

**Nguyên tắc vàng Chương 3**: Không lặp lại lý thuyết của người khác. Chỉ nói về QUYẾT ĐỊNH và LÝ DO của bạn.

**Lưu ý**:

- Không dùng TikZ. Tất cả flowchart sẽ được cung cấp dưới dạng **PlantUML code**. Tôi sẽ xem qua PlantUML, sau đó tự vẽ lại bằng draw.io.
- Diễn đạt bằng lời, hạn chế tối đa dùng tên biến, tên hàm, vì như vậy người đọc sẽ phải đối chiếu code để hiểu ý nghĩa.

---

## Các thay đổi chính

### 1. Chương 3 (3_Cong_nghe.tex) — "PHƯƠNG PHÁP ĐỀ XUẤT"

#### 1.2. Section 3.1 — Tổng quan giải pháp

- **GIỮ**: Mô tả ngắn gọn pipeline theo hướng xử lý theo quỹ đạo phương tiện (track-based thay vì frame-independent)
- Ngay sau phần vẽ sơ đồ, bạn cần đi vào giải thích từng khối, nhưng phải tập trung nhấn mạnh vào các mô hình deep learning, thuật toán BoT-SORT, thuật toán CTM + Voting.

#### 1.3. Section 3.2 — Thu thập và xử lí dữ liệu

- **GIỮ NGUYÊN** toàn bộ — đây là nội dung về cách bạn THU THẬP và XỬ LÝ dữ liệu riêng, không phải lý thuyết người khác
- Bao gồm: 3.2.1 (dữ liệu phát hiện OBB), 3.2.2 (dữ liệu OCR), 3.2.3 (dữ liệu chất lượng)
- Lý do: phần này nói về TẬP DỮ LIỆU CỦA BẠN, không phải lý thuyết thuật toán

#### 1.4. Section 3.3 — Các mô hình và thuật toán (CẬP NHẬT LỚN)

_Đổi tên từ "Huấn luyện mô hình" → "Các mô hình và thuật toán"_

**Nguyên tắc chung cho cả section 3.3**:

- Mỗi mô hình: chỉ nhắc tên → giải thích TẠI SAO chọn → giải thích BẠN ĐIỀU CHỈNH gì
- Không giải thích lại kiến trúc YOLO, BoT-SORT, LPRNet, CTC — đã có ở Chương 2
- Không đưa hyperparameters training — thuộc Chương 4

**3.3.1 Mô hình phát hiện phương tiện**

- **GIỮ**: Tại sao chọn YOLOv5m (6 lớp VN, pre-trained trên data VN), lý do dùng pre-trained weights của Che et al. thay vì train từ đầu
- **LOẠI**: Giải thích kiến trúc YOLO — thuộc Chương 2
- **LOẠI**: "bộ trọng số pretrain Che et al. được sử dụng trực tiếp mà không fine-tuning" chi tiết → Chương 4
- **LOẠI**: Class ID cụ thể — thuộc cài đặt Chương 4

**3.3.2 Mô hình phát hiện biển số**

- **GIỮ**: Tại sao chọn YOLOv8-OBB (biển số nghiêng, góc camera), lý do dùng OBB thay BB thường
- **GIỮ**: Các cải tiến/cấu hình đặc thù cho bài toán VN (nếu có)
- **GIỮ**: Bảng cấu hình kiến trúc mô hình (nếu có)
- **LOẠI**: Toàn bộ Bảng `tab:lp_detection_obb_train_config` (hyperparameters huấn luyện) → Chương 4
- **LOẠI**: Data augmentation chi tiết → Chương 4
- **LOẠI**: Giải thích lại YOLO hoạt động thế nào — thuộc Chương 2

**3.3.3 Thuật toán theo dõi phương tiện**

- **GIỮ**: Tại sao chọn BoT-SORT (ổn định track_id, phù hợp video), tại sao track phương tiện thay vì track biển số trực tiếp
- **GIỮ**: Logic bỏ phiếu đa khung cho liên kết biển số-phương tiện
- **THAY TEXT BẰNG FLOWCHART PlantUML**: Thuật toán tracking (phát hiện → gán track → crop → liên kết → bỏ phiếu → chốt)
- **LOẠI**: Giải thích BoT-SORT hoạt động thế nào — thuộc Chương 2
- **LOẠI**: Chi tiết cài đặt BoxMOT, ONNX ReID — thuộc Chương 4
- **File PlantUML**: `docs/plantuml/ch3_tracking_algorithm.puml`

association diagram:

┌──────────────────────────────────────────────┐
│ Plate track p_tid có locked vehicle v_tid? │
└──────────────┬───────────────────────────────┘
│
┌────────┴────────┐
│ YES │ NO
▼ ▼
┌──────────────┐ ┌─────────────────────────┐
│ Revalidate: │ │ Plate center inside │
│ - Plate in │ │ vehicle box? │
│ vehicle │ │ (with margin 8%) │
│ box? │ └──────────┬──────────────┘
│ - Source │ │
│ vehicle │ ┌──────────┴──────────┐
│ matches? │ │ YES → vote for v_tid │
└──────┬───────┘ └──────────┬──────────────┘
│ │
┌────┴────┐ ┌──────┴──────┐
│ Valid │ │ ≥ 5 votes │
│ → keep │ │ + ≥ 60% │
└─────────┘ │ agreement? │
└──────┬──────┘
│
┌──────┴──────┐
│ YES → LOCK │
│ plate→vehicle│
└─────────────┘

**3.3.4 Mô hình phân loại chất lượng**

- **GIỮ**: Tại sao cần Quality Router (tiết kiệm tài nguyên OCR, giảm ảo), lý do chọn 4 lớp + 3 hướng xử lý
- **GIỮ**: Công thức điểm chất lượng q và ý nghĩa các hệ số
- **GIỮ**: Bảng mô tả kiến trúc YOLOv8n-cls (nếu có)
- **LOẠI**: Chi tiết huấn luyện (epoch, batch size, optimizer) → Chương 4
- **LOẠI**: Giải thích YOLO classification — thuộc Chương 2

**3.3.5 Kiến trúc mô hình OCR (SmallLPR-Line-CTC)**

- **GIỮ TẤT CẢ PHẦN KIẾN TRÚC** (vì đây là ĐÓNG GÓP MỚI của luận văn):
    - Hạn chế LPRNet gốc với biển số VN (4 điểm) — đây là PHÂN TÍCH của bạn
    - Tổng quan mô hình đề xuất
    - Flowchart kiến trúc (chuyển sang PlantUML)
    - Bảng STN, SmallBasicBlockCBAM, Backbone CNN, các head — GIỮ
    - Feature map 2D và positional encoding
    - Multi-head CTC theo bố cục
    - Hàm mất mát (công thức + trọng số λ) và lý do chọn trọng số
- **LOẠI TOÀN BỘ PHẦN HUẤN LUYỆN** (dòng 513-526):
    - Tập dữ liệu, tiền xử lý, augmentation, optimizer, scheduler → Chương 4
    - Lý do: training là CÔNG SỨC TRIỂN KHAI, thuộc Chương 4

**Lưu ý đặc biệt**: Phần kiến trúc SmallLPR-Line-CTC được GIỮ lại vì đây là ĐÓNG GÓP MỚI (novel contribution) của luận văn, không phải lý thuyết thuần túy của người khác. Giáo viên muốn thấy thiết kế của bạn ở Chương 3.

#### 1.5. Section 3.4 — Xử lí kết quả sau mô hình (CẬP NHẬT)

- **GIỮ**: Mô tả bài toán hậu xử lý, 3 lý do cần hậu xử lý
- **GIỮ**: Nguyên lý slot-aware correction và CTM voting
- **THAY TEXT BẰNG FLOWCHART PlantUML**:
    - 3.4.1: Flowchart thuật toán slot-aware correction (quy hoạch động align/skip/missing)
        - **File**: `docs/plantuml/ch3_slot_correction.puml`
    - 3.4.2: Flowchart thuật toán CTM voting
        - **File**: `docs/plantuml/ch3_ctm_voting.puml`
- **LOẠI**: Chi tiết cài đặt cụ thể (7 kỹ thuật enhancement code) → Chương 4

---

### 2. Chương 4 (4_new.tex) — "XÂY DỰNG VÀ TRIỂN KHAI HỆ THỐNG"

#### 2.1. Nguyên tắc biên tập Chương 4

- Chương 4 trả lời ba câu hỏi theo đúng thứ tự: **khối nhận dạng được cài đặt ra sao → các mô hình được huấn luyện/fine-tune thế nào → khối nhận dạng được tích hợp vào backend như thế nào**.
- Dành tối thiểu 80% nội dung chương cho pipeline nhận dạng, huấn luyện và backend xử lý AI.
- Không mở đầu bằng use case, giao diện hoặc kiến trúc ba lớp. Loại các mục độc lập về Frontend, xác thực và thiết kế cơ sở dữ liệu.
- Không lặp lại lý thuyết, kiến trúc mô hình hoặc lưu đồ thuật toán đã trình bày ở Chương 2–3. Khi mô tả cài đặt, dẫn chiếu trực tiếp đến mục/hình tương ứng của Chương 3.
- Chương 4 chỉ trình bày quy trình huấn luyện, cấu hình và artifact đầu ra. Bảng metric, confusion matrix, ablation và phân tích chất lượng thuộc Chương 5.
- Diễn đạt theo hành vi xử lý, hạn chế tên lớp/hàm. Tên module chỉ dùng khi cần chứng minh vị trí cài đặt trong mã nguồn.

Mở đầu chương bằng bảng phân định nguồn gốc và mức độ can thiệp:

| Nhóm | Thành phần | Cách trình bày trong Chương 4 |
| --- | --- | --- |
| Kế thừa | YOLOv5m và trọng số phát hiện phương tiện của Che et al.; framework YOLO, BoT-SORT/BoxMOT | Nêu nguồn, vai trò và cấu hình tích hợp; không nhận là đóng góp tự xây dựng |
| Tự huấn luyện/fine-tune | YOLOv8-OBB phát hiện biển số; YOLOv8n-cls Quality Router; SmallLPR-Line-CTC | Mô tả dữ liệu, thiết lập huấn luyện, checkpoint và cách đưa trọng số vào pipeline |
| Tự cài đặt/điều chỉnh | Cascade detection; liên kết biển số–phương tiện; định tuyến chất lượng; xử lý ảnh; sửa ký tự theo slot; bộ đệm và CTM voting; điều phối pipeline | Nhấn mạnh quyết định cài đặt, luồng dữ liệu và liên hệ với thiết kế ở Chương 3 |
| Thành phần hỗ trợ | ReID cho BoT-SORT | Chỉ mô tả ở mức tích hợp tracking; không tạo mục huấn luyện hoặc công bố metric ReID |

#### 2.2. Section 4.1 — Cài đặt khối nhận dạng độc lập

_Đây là phần trọng tâm lớn nhất của Chương 4. Khối nhận dạng được trình bày như một thành phần độc lập trước khi nói đến Web API hoặc cơ sở dữ liệu._

**4.1.1 Đầu vào, đầu ra và tổ chức khối nhận dạng**

- Xác định đầu vào là chuỗi khung hình kèm chỉ số frame/thời gian; đầu ra là kết quả theo phương tiện gồm lớp xe, biển số tổng hợp, độ tin cậy và ảnh minh chứng.
- Mô tả sáu khối xử lý theo Hình~\ref{fig:alpr_pipeline_flowchart} ở Chương 3, nhưng tập trung vào cách chúng được nối và trao đổi dữ liệu trong mã Python.
- Thêm sơ đồ kiến trúc cài đặt khối nhận dạng độc lập; phân biệt rõ mô hình kế thừa, mô hình tự huấn luyện và thuật toán tự cài đặt.
- **File PlantUML**: `docs/plantuml/ch4_recognition_architecture.puml`.

**4.1.2 Cài đặt phát hiện và theo vết phương tiện**

- Mô tả cách nạp trọng số YOLOv5m kế thừa, lọc các lớp phương tiện Việt Nam và chuẩn hóa detection cho BoT-SORT.
- Trình bày BoT-SORT và ReID ở mức cài đặt/tích hợp: tracker có trạng thái riêng theo phiên, nhận detection và trả về định danh phương tiện ổn định.
- Nêu rõ phần kế thừa từ thư viện và phần tự điều chỉnh để ReID tham gia ghép cặp; không mở mục huấn luyện ReID.
- Dẫn chiếu Mục~\ref{subsection:3.3.tracking}; không giải thích lại Kalman, Hungarian hoặc lý thuyết ReID.

**4.1.3 Cài đặt phát hiện và nắn chỉnh biển số**

- Mô tả cascade detection: mở rộng crop phương tiện, chạy YOLOv8-OBB theo batch, lọc confidence/kích thước và ánh xạ bốn điểm OBB về hệ tọa độ khung hình.
- Trình bày bước loại detection trùng, duy trì track biển số ngắn hạn và liên kết biển số với phương tiện.
- Mô tả phép biến đổi phối cảnh từ bốn điểm OBB để tạo crop thẳng trước Quality Router/OCR; phân biệt bước này với STN nằm bên trong mô hình OCR.
- Dẫn chiếu thiết kế tracking/cascade ở Mục~\ref{subsection:3.3.tracking}.

**4.1.4 Cài đặt Quality Router**

- Mô tả cách nạp YOLOv8n-cls, ánh xạ bốn lớp `perfect/good/poor/illegible` sang ba nhánh `direct/tracklet_fusion/unreadable_wait`.
- Giải thích điểm chất lượng số dùng để xếp hạng crop và cơ chế dự phòng khi classifier không khả dụng.
- Dẫn chiếu Hình~\ref{fig:quality_router_flow}; không tạo lại flowchart Quality Router trong Chương 4.

**4.1.5 Cài đặt suy luận SmallLPR-Line-CTC**

- Mô tả nạp checkpoint, chuẩn hóa ảnh đầu vào, chạy suy luận theo batch và chuyển logits CTC thành chuỗi ký tự kèm độ tin cậy.
- Giải thích cách xác suất bố cục điều khiển việc chọn nhánh một dòng hoặc hai dòng khi decode.
- Dẫn chiếu Hình~\ref{fig:ocr_model_architecture} và các bảng kiến trúc ở Mục~\ref{subsection:3.3.4}; không chép lại cấu trúc STN, backbone hoặc các head.

**4.1.6 Cài đặt hậu xử lý và tổng hợp đa khung hình**

- Mô tả các biến thể tăng cường ảnh được tạo cho crop suy giảm, cách chấm điểm và chọn ứng viên OCR.
- Trình bày việc cài đặt căn chỉnh chuỗi theo slot bằng quy hoạch động, sửa cặp ký tự mơ hồ và phạt độ tin cậy.
- Mô tả bộ đệm theo track, chọn top-k crop, gom cụm kết quả và kích hoạt CTM khi track kết thúc.
- Dẫn chiếu Mục~\ref{subsection:3.4.1}, Mục~\ref{subsection:3.4.2} và Hình~\ref{fig:ctm_voting_flowchart}; không tạo lại lưu đồ slot correction hoặc CTM.

#### 2.3. Section 4.2 — Huấn luyện và fine-tuning mô hình

_Phần này chứng minh quá trình tự chuẩn bị dữ liệu, chạy huấn luyện và quản lý trọng số; không lặp phần đánh giá định lượng của Chương 5._

**4.2.1 Môi trường và khả năng tái lập**

- Máy huấn luyện cục bộ: NVIDIA GeForce RTX 4050.
- Phần mềm huấn luyện: Python 3.10, PyTorch 2.6.0, CUDA 12.4; ghi rõ AMP, seed, số worker và cách chọn thiết bị.
- Mô tả cấu trúc file cấu hình, thư mục log, checkpoint tốt nhất/cuối cùng và quy tắc resume hoặc khởi tạo từ checkpoint.
- Tách rõ môi trường huấn luyện RTX 4050 khỏi môi trường suy luận triển khai RTX 3090 ở Mục 4.4.

**4.2.2 Fine-tune mô hình phát hiện biển số YOLOv8-OBB**

- Dẫn lại tập dữ liệu OBB đã trình bày ở Mục~\ref{subsection:3.2.1}; không lặp toàn bộ quá trình thu thập dữ liệu.
- Trình bày định dạng nhãn bốn điểm, train/validation split, trọng số khởi tạo, kích thước ảnh, batch size, augmentation, AMP và cơ chế lưu checkpoint.
- Phân biệt số epoch cấu hình với số epoch thực sự có trong log; không mặc định ghi "đã huấn luyện đủ 50 epoch" nếu artifact không chứng minh điều đó.
- Kết thúc bằng dẫn chiếu sang Mục~\ref{subsection:5.2.2} để xem Precision, Recall và mAP.

**4.2.3 Huấn luyện và fine-tune Quality Router**

- Dẫn lại dữ liệu chất lượng ở Mục~\ref{subsection:3.2.3}.
- Mô tả hai giai đoạn: huấn luyện khái niệm legibility trên LPLCv2, sau đó fine-tune trên crop biển số Việt Nam cùng hệ bốn nhãn.
- Nêu biến thể nhị phân như thí nghiệm phụ, nhưng xác định mô hình bốn lớp là mô hình phục vụ ánh xạ ba nhánh trong pipeline.
- Trình bày input size, batch size, augmentation, early stopping/resume và checkpoint; chuyển toàn bộ accuracy/confusion matrix sang Mục~\ref{subsection:5.2.3}.

**4.2.4 Huấn luyện SmallLPR-Line-CTC**

- Dẫn lại tập OCR ở Mục~\ref{subsection:3.2.2}; mô tả cách đọc nhãn từ tên file và token `[SEP]` cho biển hai dòng.
- Trình bày resize, lọc ảnh lỗi, augmentation mô phỏng phối cảnh/nhòe/ánh sáng và cách mã hóa target cho các head CTC.
- Mô tả loss đa nhiệm, AdamW, weight decay, cosine annealing, batch size, seed, checkpoint và quy trình resume/khởi tạo trọng số.
- Chỉ nhắc rằng các cấu hình ablation dùng cùng dữ liệu và seed; bảng so sánh một head/multi-head cùng toàn bộ metric thuộc Mục~\ref{subsection:5.2.4} và Mục~\ref{subsection:5.2.5}.

#### 2.4. Section 4.3 — Tích hợp khối nhận dạng vào backend

_Phần này làm rõ backend là nơi vận hành pipeline AI có trạng thái, không chỉ là lớp CRUD._

**4.3.1 Kiến trúc tích hợp và vòng đời thành phần**

- Các mô hình học sâu không trạng thái được nạp một lần khi ứng dụng khởi động và dùng chung để tránh nhân bản VRAM.
- Tracker phương tiện, tracker biển số, bộ liên kết, Quality Router fallback và bộ đệm OCR được khởi tạo riêng cho từng phiên xử lý.
- Thể hiện ranh giới giữa nguồn frame, lõi suy luận, bộ phát sự kiện và lớp lưu kết quả.
- **File PlantUML**: `docs/plantuml/ch4_backend_integration.puml`.

**4.3.2 Luồng xử lý trong backend**

- Mô tả đầy đủ: nguồn frame → phát hiện phương tiện → tracking → plate OBB → association → Quality Router → OCR → buffer/CTM → kết quả.
- Giải thích điều kiện kết thúc track, chốt OCR, trả kết quả hợp lệ hoặc kết quả bị từ chối.
- Trình bày các lớp kiểm soát sai lệch: cổng chất lượng trước OCR, ngưỡng tin cậy ký tự, số frame tối thiểu và kiểm tra định dạng.

**4.3.3 Hai luồng đầu vào dùng chung lõi pipeline**

- Video tải lên và trích đoạn từ phiên giám sát đều được chuẩn hóa thành cùng giao diện nguồn frame rồi gọi chung lõi xử lý.
- Worker, semaphore, hàng đợi bất đồng bộ, SSE và MJPEG chỉ là lớp vận chuyển tiến độ/preview/kết quả, không chứa logic nhận dạng riêng.
- Frontend chỉ được nhắc ở ranh giới: gửi nguồn video, nhận tiến độ và hiển thị kết quả; tối đa 1–2 đoạn.

**4.3.4 Lưu kết quả và xử lý lỗi**

- Cơ sở dữ liệu và object storage chỉ được mô tả như đích nhận metadata và ảnh evidence từ callback lưu kết quả.
- Không trình bày collection, index, CRUD, JWT, CSRF hoặc luồng đăng nhập.
- Nêu cách cô lập lỗi một phiên, giải phóng tài nguyên và không để lỗi lưu trữ làm hỏng toàn bộ kết quả suy luận.

#### 2.5. Section 4.4 — Đóng gói và triển khai

- Giữ một bảng ngắn gồm FastAPI, Docker, RunPod RTX 3090, MediaMTX, MongoDB và object storage; loại bảng URL công cụ dài.
- Mô tả cách đóng gói backend cùng model weights, truyền đường dẫn model bằng biến môi trường và nạp model trên GPU khi khởi động.
- Phân biệt rõ: **RTX 4050 + PyTorch 2.6.0/CUDA 12.4 dùng cho huấn luyện cục bộ**; **RTX 3090 và image triển khai dùng cho suy luận**.
- Chỉ giữ 1–2 đoạn về giao diện Web gửi video và nhận SSE/MJPEG; không tạo mục Frontend riêng.
- Không tạo mục cơ sở dữ liệu riêng và không giữ bảng collection/index.

---

## Các file thay đổi

| File                                              | Loại thay đổi | Mô tả                                                                                                       |
| ------------------------------------------------- | ------------- | ----------------------------------------------------------------------------------------------------------- |
| `Chuong/3_Cong_nghe.tex`                          | Sửa lớn       | Cắt training details; thay text bằng flowchart PlantUML cho thuật toán; thêm flowchart tổng quan đầu chương |
| `Chuong/4_new.tex`                                | Viết lại lớn  | Tổ chức pipeline-first: cài đặt khối nhận dạng → huấn luyện → tích hợp backend → triển khai rút gọn          |
| `docs/plantuml/ch3_pipeline_overview.puml`        | Mới           | Flowchart tổng quan pipeline đầu Chương 3                                                                   |
| `docs/plantuml/ch3_tracking_algorithm.puml`       | Mới           | Flowchart thuật toán tracking                                                                               |
| `docs/plantuml/ch3_slot_correction.puml`          | Mới           | Flowchart slot-aware correction                                                                             |
| `docs/plantuml/ch3_ctm_voting.puml`               | Mới           | Flowchart CTM voting                                                                                        |
| `docs/plantuml/ch4_recognition_architecture.puml` | Mới           | Kiến trúc cài đặt khối nhận dạng độc lập và ranh giới đóng góp                                              |
| `docs/plantuml/ch4_backend_integration.puml`      | Mới           | Sơ đồ tích hợp khối nhận dạng vào backend và các luồng đầu vào/đầu ra                                      |

---

## Thứ tự thực hiện

1. **Kiểm tra artifact**: đối chiếu dataset, cấu hình, log, checkpoint và môi trường huấn luyện; lập danh sách số liệu được phép sử dụng.
2. **Lập ma trận dẫn chiếu Chương 3 → Chương 4**: mỗi thuật toán ở Chương 4 phải trỏ về đúng mục/hình thiết kế, tránh giải thích lại.
3. **Tạo hai sơ đồ Chương 4**: kiến trúc khối nhận dạng độc lập và tích hợp pipeline–backend; không tạo lại flowchart Quality Router/slot correction/CTM.
4. **Viết lại Chương 4** theo thứ tự 4.1 cài đặt khối nhận dạng → 4.2 huấn luyện → 4.3 tích hợp backend → 4.4 đóng gói/triển khai.
5. **Kiểm tra chéo Chương 5**: chuyển mọi metric, confusion matrix, ablation và phân tích kết quả sang Chương 5; loại nội dung trùng lặp.

---

## Cách kiểm tra

- Mở hai file PlantUML Chương 4 để kiểm tra sơ đồ có phân biệt phần kế thừa, phần tự huấn luyện và phần tự cài đặt.
- Biên dịch LaTeX thành công, không có label/reference thiếu hoặc trùng.
- Kiểm tra tỷ trọng: tối thiểu 80% Chương 4 dành cho pipeline, training và backend AI; Web App/cơ sở dữ liệu/triển khai không vượt quá 20%.
- Kiểm tra ranh giới chương: Chương 3 = thiết kế và lý do lựa chọn; Chương 4 = cài đặt, huấn luyện và tích hợp; Chương 5 = số liệu và đánh giá.
- Kiểm tra mọi thuật toán triển khai ở Chương 4 đều dẫn chiếu Chương 3 và không lặp lại lưu đồ/lý thuyết.
- Kiểm tra mọi con số training có nguồn từ artifact thực tế. Các điểm đang cần đối chiếu trước khi viết gồm:
    - Số ảnh validation OBB trong tài liệu và số file hiện có chưa hoàn toàn khớp.
    - Cấu hình OBB đặt tối đa 50 epoch nhưng `results.csv` hiện chỉ có log của 25 epoch; checkpoint tốt nhất nằm ở một epoch sớm hơn.
    - Phiên bản PyTorch/CUDA của môi trường huấn luyện phải thống nhất là PyTorch 2.6.0/CUDA 12.4; không trộn với image suy luận hoặc bảng công nghệ cũ.
- Không đưa bảng metric, confusion matrix hoặc ablation vào Chương 4; chỉ dẫn chiếu sang Chương 5.
- Không mô tả YOLOv5m, BoT-SORT/BoxMOT hoặc framework có sẵn như đóng góp tự xây dựng.
