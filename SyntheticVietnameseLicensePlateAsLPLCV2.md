# Kế Hoạch Sinh Dataset VN Biển Số Theo Style LPLC

> **Trạng thái ngày 2026-06-05:** pipeline `vn_lplc_reference_v3` bị chặn
> production vì `vn_blank_bank` và `lplc_reference_bank` chưa bảo đảm xóa hết
> chữ nguồn. Không được dùng các bank/manifests V3 hiện có để sinh dataset final.
>
> Phiên bản thay thế trong kế hoạch này là `vn_lplc_reference_v4`. V4 loại bỏ
> dependency production vào VN blank được tạo bằng inpainting tổng quát, dùng
> template blank bảo đảm không chữ làm nguồn hình học, và chỉ nhận style/residual
> từ ảnh thật sau khi qua text-removal gate độc lập.

## Trạng Thái Triển Khai V4 Ngày 2026-06-05

V4 đã được triển khai thành code path cô lập và chạy được end-to-end ở mức
smoke, nhưng **chưa được promote production**:

- V3 production kill switch đã hoạt động; V3 chỉ chạy khi có cờ
  `--allow-legacy-v3-baseline` và output nằm dưới ablation root.
- Đã có appearance registry, appearance-aware grammar/renderer, token masks,
  template bank bảo đảm không chữ, mask ensemble đa cực tính, deterministic
  surface reconstruction và LPLC reference cleanup.
- Generator V4 dùng template làm plate base; optional accepted VN surface
  residual và LPLC style chỉ truyền zero-mean residual. Final gate kiểm tra cả
  `source_vn_label` lẫn `source_lplc_label`. Generated sample chỉ accepted khi:
  - PARSeq đọc đúng target label;
  - SlotLPR source-text branch không đọc source label/substring và đủ confidence;
  - local-contrast validator không thấy stroke ngoài target glyph và semantic
    border;
  - appearance/aspect và token-level color contract pass.
- SlotLPR không được dùng làm target exact-match gate vì smoke thật cho thấy
  checkpoint hiện tại đọc sai target renderer dù PARSeq đọc đúng. SlotLPR được
  giữ đúng vai trò independent source-text detector; confidence thấp chuyển
  `manual_review`, không loại oan thành `rejected`.
- VN surface residual và LPLC reference builders đã nối SlotLPR +
  local-contrast validation thật; thiếu validator thì fail closed.
- Paired text-removal data được xuất trực tiếp theo contract TMIM
  `text_rmv/VNPlate/<split>/{all_images,all_labels,mask}`. Launcher TMIM mặc
  định dry-run, yêu cầu custom config cho Uformer-T/S và chặn Uformer-B nếu chưa
  có explicit resource override.
- Generator và paired-data builder audit hai nguồn VN, loại toàn bộ label thật
  cùng source labels của residual bank trước khi sinh label mới.
- Đã có metadata/artifact audit filters và pilot gate cân bằng theo
  `appearance_class/layout`. Audit không bao giờ promote record đã rejected hoặc
  manual-review.

Evidence hiện tại:

- `36` V4 unit/integration tests pass; tổng synthetic regression suite
  `90` tests pass.
- Smoke generator thật trên 10 template xanh: `7 accepted`, `3 manual_review`
  do SlotLPR confidence thấp, `0 rejected`; `6.835` label VN thật được exclude.
- Smoke LPLC reference với validator thật: `1 accepted`.
- Smoke VN surface residual: sample đầu bị reject sớm vì
  `appearance_confidence_low`, đúng fail-closed contract.

Các blocker trước production promotion:

- chưa xác minh grammar/source slice cho `military_red_light`; renderer,
  template và mask đã hỗ trợ nhưng generator production vẫn cố ý chặn;
- chưa chạy manual benchmark 600 blank/reference và 300 generated;
- chưa fine-tune/evaluate TMIM Uformer-T/S/B và các ablation bắt buộc;
- chưa hiệu chuẩn threshold trên held-out real benchmark;
- các gate DeltaE và seam/blob/over-smoothing vẫn là deliverable chưa hoàn
  thành.

## Quyết Định Kiến Trúc

- **Dừng production V3.** Giữ V3 và LaMa chỉ để tái hiện lỗi/baseline ablation.
- **Vô hiệu hóa toàn bộ `data/synthetic/vn_lplc_reference_v3/vn_blank_bank`.**
  Bank hiện có 13.577 record nhưng không có quality gate trước khi ghi manifest.
- **Không dùng ảnh VN đã inpaint làm plate base final.** Plate base final phải là
  blank vector/template theo đúng layout và lớp màu; theo construction không thể
  chứa số cũ.
- **Ảnh VN thật chỉ dùng để:**
  - đo palette, độ dày viền, bo góc, phản quang, blur và phân phối degradation;
  - trích surface residual tùy chọn sau khi xóa chữ và qua hard gate;
  - làm dữ liệu validation/error analysis.
- **Thay `dark_glyph_mask` production bằng text localization đa nguồn, đa cực
  tính.** Phải xử lý cả chữ tối trên nền sáng và chữ sáng trên nền xanh/đỏ/tối.
- **Thay LaMa production bằng hai tầng:**
  - primary: deterministic plate-surface reconstruction/template composition;
  - learned fallback: plate-specific scene-text-removal, ưu tiên
    `Uformer-B + TMIM`, chỉ promote nếu vượt primary theo từng slice.
- **Áp dụng cùng contract text-removal và quality gate cho cả VN residual bank
  và LPLC style/reference bank.**

## Kết Quả Audit Pipeline Hiện Tại

### Bằng Chứng Trên Bank Hiện Có

- `vn_blank_bank.jsonl` hiện có:
  - 13.577 record tổng;
  - 11.462 record, tương đương 84,4%, từ `filename_ocr_train` và dùng trực tiếp
    `dark_glyph_mask`;
  - 2.115 record từ `raw_ocr_char_yolo_train` và dùng box ký tự YOLO.
- Visual audit trên các cặp `source -> mask -> blank` cho thấy ba lỗi độc lập:
  - biển chữ sáng trên nền tối gần như giữ nguyên chữ vì mask sai cực tính;
  - biển chữ tối dù box phủ rộng vẫn có bóng/hình dạng số do LaMa tái tạo text;
  - dấu `-`, `.`, halo và nét mờ ngoài box còn nguyên.
- Một sample audit ngẫu nhiên 1.200 record cho thấy mask được ghi và inpaint
  chạy, nhưng pipeline không có metric nào chứng minh chữ nguồn đã biến mất.
  Việc mask có diện tích lớn không đồng nghĩa mask đúng nét chữ hoặc blank sạch.

### Root Cause Trong Mã

1. `synthetic_vn_lplc/blank_bank.py:112` dùng grayscale black-hat và Otsu.
   Phép này chỉ hợp với chữ tối trên nền sáng; chữ trắng trên nền xanh/đỏ không
   thể được bắt ổn định.
2. `synthetic_vn_lplc/blank_bank.py:207` dùng `dark_glyph_mask` làm nguồn mask
   duy nhất cho `filename_ocr_train`; không có OCR/text detector hoặc
   polarity-aware fallback.
3. `yolo_char_mask()` chỉ tô rectangle từ annotation. `data/raw/OCR/data.yaml`
   có 30 class chữ/số nhưng không có `-` và `.`, nên separator chắc chắn có thể
   sót. Box cũng không mô tả chính xác stroke, shadow và blur halo.
4. Mask chỉ được dilate bằng một radius cố định theo chiều cao canonical. Radius
   này không phụ thuộc stroke width, blur, resolution hay uncertainty của mask.
5. `inpaint_masked_region()` chỉ copy kết quả model tại pixel mask nhị phân.
   Halo/chữ nằm ngoài mask được giữ nguyên tuyệt đối; biên binary dễ tạo seam.
6. LaMa là general image inpainting. Nó được huấn luyện để hoàn thành nội dung
   hợp lý và có thể sinh lại texture giống chữ/số ở vùng vốn chứa text.
7. `build_vn_blank_bank()` ghi mọi candidate đã inpaint vào manifest; record
   không có `blank_quality`, accepted/rejected status hay independent text gate.
8. `reference_bank.py:194` tiếp tục dùng `dark_glyph_mask` cho LPLC. Metric
   `style_residual_fraction` lại được đo bằng cùng detector đã sinh mask, nên
   không phát hiện được phần chữ mà detector ban đầu bỏ sót. Threshold hiện tại
   còn cho phép residual đến `0.45`.
9. `reference_synthesis.py:268` bảo vệ các gradient cao khi surface transfer.
   Bóng chữ cũ là gradient cao nên có thể được giữ và truyền tiếp sang ảnh final.
10. `render_vn_plate()` hiện mặc định duy nhất nền trắng/chữ đen/viền đen, chưa
    có appearance contract cho nền xanh/chữ trắng, nền vàng/chữ đen, nền đỏ/chữ
    sáng hoặc các lớp đặc thù khác.

### Vì Sao Chỉ Nới Mask Hoặc Đổi Sang LaMa Lớn Hơn Không Đủ

- Mask sai cực tính vẫn sai dù inpainting model mạnh hơn.
- Box ký tự không có separator/halo vẫn để lộ thông tin nguồn.
- General inpainting có objective tái tạo nội dung hợp lý, không có objective
  bắt buộc xóa text. TMIM chỉ ra trực tiếp rằng inpainting tổng quát có thể sinh
  texture giống chữ và kém mô hình được huấn luyện riêng cho text removal.
- Không có independent gate thì bất kỳ model nào cũng có thể đưa lỗi vào bank.

## Mục Tiêu Và Invariant V4

- Mục tiêu duy nhất vẫn là sinh **OCR crop biển số Việt Nam**; không sinh
  full-frame và không tối ưu detector.
- Không một sample production nào được phụ thuộc vào VN blank có khả năng chứa
  glyph nguồn.
- Mỗi plate phải có `plate_appearance_class` rõ ràng và hợp lệ với grammar:
  `civil_white_black`, `state_blue_white`, `commercial_yellow_black`,
  `diplomatic_white_red_black`, `military_red_light`, hoặc lớp được xác minh
  riêng.
- Hỗ trợ đúng `state_blue_white` và lớp biển nền đỏ Việt Nam là exit criterion
  bắt buộc của V4. Trong giai đoạn xây dựng, sample đỏ chưa xác minh được chuyển
  manual review/reject; nhưng V4 không được tuyên bố hoàn thành bằng cách loại
  bỏ toàn bộ blue/red slice.
- Với `military_red_light`, phải thu thập/xác minh grammar, palette, border và
  validation slice từ nguồn phù hợp trước promotion; không ép về lớp gần nhất.
- Cả glyph tối và glyph sáng phải đi qua cùng quality contract.
- Bank builder được phép reject mạnh. Yield thấp nhưng sạch tốt hơn bank lớn có
  label leakage.
- Mọi model/threshold phải được version hóa; V4 không đọc manifest V3.

## Pipeline V4

### 1. Appearance Và Grammar Contract

Tạo `PlateAppearance` bất biến gồm:

```text
appearance_class
allowed_layouts
allowed_grammar/serials
background_palette_lab
foreground_palette_lab
border_palette_lab
token_style_rules
glyph_polarity
border_geometry
reflective_profile
source/evidence
```

- Renderer nhận `label + layout + appearance_class`, không nhận màu rời rạc.
- Grammar generator chỉ sinh tổ hợp label/màu có bằng chứng. Ví dụ, nền xanh
  chữ trắng phải tuân theo serial được hỗ trợ cho lớp đó.
- `token_style_rules` hỗ trợ plate có nhiều màu glyph trong cùng một biển, thay
  vì ép toàn bộ label dùng một foreground color.
- Renderer phải xuất `token_masks` và `expected_token_styles` để gate riêng từng
  token, ví dụ chữ đỏ nhưng số đen trên cùng một biển.
- Palette được lấy từ quy chuẩn/ảnh train sạch và lưu dưới dạng distribution,
  không hardcode một RGB duy nhất.
- `unknown` hoặc mismatch giữa label, layout và appearance phải bị reject.

### 2. VN Template Blank Bank Là Nguồn Hình Học Primary

Tạo `vn_template_bank` từ vector/template, không từ ảnh VN đã inpaint:

- Ba physical template:
  - ô tô dài: `520x110 mm`, canonical `1040x220`;
  - ô tô ngắn: `330x165 mm`, canonical `660x330`;
  - xe máy: `190x140 mm`, canonical `380x280`.
- Mỗi template có:
  - blank RGB/Lab bảo đảm không chữ;
  - interior mask;
  - border/protected-detail mask;
  - material/reflective mask;
  - appearance metadata.
- Border, góc, khoảng trống và màu được render theo appearance contract.
- Template-only là fallback luôn khả dụng và là baseline bắt buộc phải khó vượt.

### 3. VN Surface Residual Bank Chỉ Là Tùy Chọn

Ảnh VN thật không còn được dùng nguyên ảnh blank. Nếu muốn giữ texture thật:

1. Xác định plate quad/interior rồi rectify trước; không resize tự do toàn crop.
2. Ước lượng appearance và polarity.
3. Tạo high-recall text mask đa nguồn.
4. Phân rã ảnh thành:
   - low-frequency illumination/material field;
   - band-limited non-text residual;
   - border/detail riêng.
5. Zero residual trong text mask đã expand và trong vùng uncertainty.
6. Clamp amplitude để residual không thể tái tạo stroke đậm.
7. Chỉ ghi residual vào accepted bank nếu independent detector/OCR gate không
   phát hiện text.
8. Nếu fail, dùng template-only; không dùng LaMa output làm fallback mặc định.

### 4. Text Localization Đa Nguồn, Đa Cực Tính

`dark_glyph_mask` được giữ với tên/contract legacy cho ablation, không được gọi
trong production V4. Mask production là union có kiểm soát của:

- character boxes/polygons hiện có, chỉ dùng như coarse prompt;
- character/text probability từ detector segmentation như CRAFT/DBNet hoặc
  char detector đã fine-tune cho plate;
- dark-on-light evidence bằng black-hat/local contrast;
- light-on-dark evidence bằng top-hat/local contrast;
- Lab color distance so với robust local background;
- edge/stroke evidence ở nhiều scale;
- expected separator/slot regions từ label và layout để bắt `-`, `.`;
- uncertainty ring để bắt blur halo, shadow và compression ringing.

Quy tắc fusion:

- ưu tiên recall trong interior, nhưng không được chạm border/protected mask;
- dilation dựa trên estimated stroke width và blur sigma, không dùng radius cố
  định;
- lưu riêng `coarse_text_roi`, `stroke_probability`, `erase_mask`,
  `uncertainty_mask`, `border_protect_mask`;
- mask confidence thấp phải reject hoặc chuyển manual review, không tự động
  inpaint.

### 5. Blank Reconstruction Và Learned Fallback

#### Primary: Deterministic Plate-Surface Reconstruction

- Fit smooth illumination/material field trong Lab từ pixel non-text.
- Nội suy vùng text bằng robust polynomial/thin-plate surface hoặc patch transfer
  chỉ từ patch không giao text mask.
- Compose lại border/protected detail từ template hoặc nguồn đã xác minh.
- Không có generative text prior nên không thể chủ động hallucinate số.

#### Learned Fallback: Plate-Specific Text Removal

- Candidate ưu tiên về chất lượng: TMIM với Uformer, fine-tune trên plate crops.
- Lý do chọn:
  - TMIM huấn luyện background modeling và text erasing riêng;
  - dùng được text detection labels/pseudo labels;
  - paper báo cáo tốt hơn LaMa ngay cả khi LaMa có ground-truth text mask.
- TMIM là training framework, không bắt buộc Uformer-B. Benchmark
  `Uformer-T/S/B` theo quality, VRAM và throughput; paper dùng 8 RTX 3090, nên
  với mặc định 2 GPU 16 GB phải bắt đầu từ T/S và chỉ promote B nếu resource
  gate pass.
- FETNet/SAEN/PSSTRNet là ablation alternatives nếu TMIM không phù hợp tài
  nguyên hoặc license.
- Learned output không tự động accepted. Nó phải cạnh tranh với deterministic
  candidate và qua cùng independent gates.
- LaMa/Telea chỉ là baseline/debug, không là production backend V4.

### 6. LPLC Style/Reference Bank Cũng Phải Được Sửa

- Không dùng `dark_glyph_mask` đơn nguồn cho LPLC.
- Dùng cùng mask ensemble và text-erasing candidate selection như VN residual
  bank.
- `style_residual_fraction` phải được tính bởi detector độc lập với detector đã
  sinh mask.
- Giữ source context, quad, blur, noise, light và plate pose; loại bỏ hoàn toàn
  OCR Brazil trước khi dùng làm style reference.
- Có accepted/rejected manifest riêng và reject reason rõ ràng.

### 7. Reference-Conditioned Synthesis

Contract mới:

```text
vn_template_blank
+ optional_accepted_vn_surface_residual
+ accepted_lplc_style/reference
+ rendered_vn_glyph
+ geometry/homography
-> generated OCR crop
```

- `make_surface_transfer_mask()` không được bảo vệ mọi gradient cao.
- Chỉ semantic border/protected-detail mask được bảo vệ trước khi render glyph.
- Bất kỳ residual gradient nào trùng text probability/uncertainty phải bị xóa.
- LPLC donor chỉ truyền zero-mean illumination/residual, blur, noise và artifact
  đã xác minh; không được overwrite hue, palette hoặc glyph polarity của
  `appearance_class`. Không dùng trực tiếp absolute Lab mean như V3.
- IP-Adapter reference phải được color-normalize hoặc condition theo
  `appearance_class`; final crop đổi lớp màu phải bị reject.
- New VN glyph chỉ được bảo vệ sau khi composite và phải theo
  `appearance_class`.
- IP-Adapter/ControlNet harmonization vẫn là candidate final, nhưng chỉ chạy sau
  khi template/reference banks pass gate.

### 8. Independent Validation Và Generated-Crop Gate

`independent` là contract bắt buộc, không chỉ là tên metric:

- checkpoint/model dùng để sinh `stroke_probability`, mask, reconstruction hoặc
  chọn candidate không được dùng làm validation-only text detector;
- validation-only detector phải là model family/checkpoint khác, frozen trước
  khi hiệu chuẩn gate và không nhận gradient/pseudo-label từ output V4;
- OCR validator là nhánh độc lập thứ hai; dùng source label để phát hiện exact
  match và substring còn sót. Nhánh này không thay target exact-match gate;
  confidence thấp phải chuyển manual review;
- threshold được khóa trên held-out real benchmark, không trên bank đang build;
- nếu không có validator độc lập khả dụng, record phải manual review/reject.

Bank sạch chưa đủ. Mỗi generated crop phải qua gate trước khi ghi accepted
manifest:

- OCR đọc đúng **new target label** với confidence yêu cầu;
- validation-only detector/OCR không đọc **source VN label**, **source LPLC
  label**, hoặc substring nguồn dài từ 3 ký tự;
- không có text probability ngoài `target_glyph_mask` sau khi trừ uncertainty
  ring của glyph mới;
- `appearance_class`, glyph polarity, border và token-level styles khớp contract;
- geometry, seam, blob, over-smoothing, style và color-drift gates pass;
- record fail đi thẳng vào generated rejected/manual-review manifest.

## Data Và Annotation Plan

- Không dùng V3 blank output làm target huấn luyện.
- Sinh ít nhất 50.000 paired synthetic sample:
  `clean_blank + rendered_text/degradation -> text_image`, kèm exact stroke mask.
- Tạo real mask benchmark tối thiểu 600 crop, stratified theo:
  - appearance/polarity: white-dark, blue-light, yellow-dark, red/light hoặc
    lớp quan sát hợp lệ khác;
  - layout: long, short, motor;
  - source kind: filename và char-YOLO;
  - quality: clean, blur, low-resolution, glare, shadow, compression.
- Tạo tối thiểu 150 real crop có blank target được retouch thủ công để đo PSNR,
  SSIM, LPIPS, DeltaE và border preservation.
- Split theo source identity/camera; không để cùng plate/source vào train và
  validation/test.
- Mọi annotation có `annotator`, `review_status`, `appearance_class`,
  `text_mask`, `border_mask`, `uncertainty_mask`.

## Hard Gates Trước Khi Ghi Bank

Mỗi VN residual hoặc LPLC style blank phải pass toàn bộ:

- không có text/char detection trong plate interior ở multi-scale;
- OCR không đọc lại source label hoặc substring source dài từ 3 ký tự với
  confidence đã hiệu chuẩn;
- `residual_stroke_energy_ratio <= 0.05`;
- `border_preservation_ssim >= 0.995`;
- `outside_erase_mask_ssim >= 0.99`;
- `appearance_class` và polarity không đổi;
- từng `token_mask` đạt expected token color/polarity và không bleed sang token
  khác;
- color drift ngoài erase mask `DeltaE2000 <= 3`;
- không có seam/blob/over-smoothing vượt threshold;
- mọi artifact path và metric đều tồn tại trước khi record được accepted.

`residual_stroke_energy_ratio`:

```text
sum(independent_text_probability(blank) * coarse_text_roi)
/
sum(independent_text_probability(source) * coarse_text_roi)
```

Nếu denominator quá thấp vì detector không nhìn thấy text nguồn, sample phải
reject/manual review; không được coi tỷ lệ bằng `0`.

Threshold ban đầu được khóa như trên, sau đó chỉ được điều chỉnh bằng held-out
real benchmark. Không hạ gate chỉ để tăng yield.

## Manifest V4

### Bank Record Bắt Buộc

```json
{
  "pipeline_version": "vn_lplc_reference_v4",
  "record_id": "...",
  "status": "accepted",
  "reject_reasons": [],
  "appearance_class": "state_blue_white",
  "layout": "long",
  "source_image": "...",
  "template_blank_path": "...",
  "surface_residual_path": "...",
  "coarse_text_roi_path": "...",
  "stroke_probability_path": "...",
  "erase_mask_path": "...",
  "uncertainty_mask_path": "...",
  "border_protect_mask_path": "...",
  "quality_metrics": {
    "residual_stroke_energy_ratio": 0.01,
    "border_preservation_ssim": 0.998,
    "outside_erase_mask_ssim": 0.997,
    "delta_e_2000": 1.2
  },
  "models": {
    "text_detector": "...",
    "text_eraser": "...",
    "appearance_classifier": "..."
  }
}
```

- Accepted, rejected và manual-review records phải nằm ở manifest riêng.
- Generator V4 chỉ đọc `status=accepted` và đúng `pipeline_version`.
- V4 phải fail fast nếu gặp manifest V2/V3 hoặc record thiếu quality metrics.
- Generated manifest cũng phải tách `accepted/rejected/manual_review`; sample chỉ
  geometry-valid chưa được phép coi là accepted.
- Generated record bắt buộc propagate `target_label`, `source_vn_label`,
  `source_lplc_label`, source IDs, `target_glyph_mask_path`, appearance/token
  styles và validation-only metrics để gate ghost source text có thể audit lại.

## Entrypoints Đã Triển Khai

- `scripts/synthetic_vn_lplc_build_template_bank.py`
- `scripts/synthetic_vn_lplc_build_surface_residual_bank.py`
- `scripts/synthetic_vn_lplc_build_reference_bank_v4.py`
- `scripts/synthetic_vn_lplc_filter_blank_bank.py`
- `scripts/synthetic_vn_lplc_prepare_text_removal_dataset.py`
- `scripts/synthetic_vn_lplc_train_text_eraser.py`
- `scripts/synthetic_vn_lplc_generate_reference_v4.py`
- `scripts/synthetic_vn_lplc_filter_reference_v4.py`
- `scripts/synthetic_vn_lplc_pilot_gate_v4.py`

Các builder ghi riêng accepted/rejected/manual-review manifests. Hai filter
entrypoint là conservative metadata/artifact audit, không tự promote record
manual/rejected. `synthetic_vn_lplc_train_text_eraser.py` chỉ là staging/launcher;
không có nghĩa TMIM đã được train hoặc pass promotion gate. Cho đến khi hoàn
thành Phase 4 bên dưới, không chạy lại command production V3.

## Implementation Phases

### Phase 0: Freeze Và Regression Corpus

**Trạng thái: code freeze/kill switch hoàn thành; manual regression corpus chưa
hoàn tất.**

- Đánh dấu V3 manifests/banks là `blocked_ghost_text`.
- Thêm repo-level kill switch: config mặc định đặt V3 `production_enabled:
  false`; mọi V3 production entrypoint fail fast trừ khi truyền cờ explicit
  `--allow-legacy-v3-baseline`. Cờ này chỉ được ghi output vào thư mục ablation,
  không được ghi đè bank/manifests production.
- Chọn các lỗi đã quan sát, gồm hai ảnh người dùng cung cấp và contact sheet
  audit, làm regression corpus bắt buộc.
- Thêm test chứng minh V3 fail trên dark/light polarity, separator và LaMa
  hallucination.

### Phase 1: Appearance-Aware Template Baseline

**Trạng thái: hoàn thành cho các appearance có grammar đã xác minh; red
production grammar còn bị chặn có chủ đích.**

- Thêm `PlateAppearance` và grammar mapping.
- Mở rộng renderer cho các appearance được xác minh.
- Tạo template bank và chuyển generator sang template-only.
- Xóa logic bảo vệ gradient chung; chỉ bảo vệ semantic border/detail.

### Phase 2: Mask Ensemble Và Bank Gates

**Trạng thái: implementation và smoke hoàn thành; threshold calibration trên
held-out real benchmark còn thiếu.**

- Xây text probability/mask ensemble đa polarity.
- Thêm separator/halo recovery và adaptive expansion.
- Thêm accepted/rejected/manual-review manifests.
- Build `candidate_bank` cô lập để phát triển/đo gate; candidate records không
  được generator production đọc.
- Áp dụng independent detector/OCR/text-energy gates cho VN và LPLC banks.

### Phase 3: Surface Residual Và Text-Eraser Ablation

**Trạng thái: deterministic residual, paired-data export và TMIM launcher hoàn
thành; learned training/evaluation và ablation chưa chạy.**

- Xây deterministic surface reconstruction/residual extractor.
- Chuẩn bị paired synthetic + real benchmark.
- Fine-tune/evaluate Uformer-T/S trước; chạy Uformer-B khi resource gate pass.
- So sánh template-only, deterministic residual, TMIM, FETNet và LaMa.

### Phase 4: Pilot Và Promotion

**Trạng thái: pilot/report tooling đã triển khai; review, ablation và production
promotion chưa chạy. Pilot phải fail cho đến khi red slice và mọi gate bắt buộc
được đáp ứng.**

- Chỉ promote/công bố lại **production accepted bank** khi Phase 1-3 pass;
  experimental candidate bank được phép build trong Phase 2-3.
- Chạy pilot cân bằng theo layout và appearance, không chỉ theo layout.
- Review thủ công tối thiểu 600 blank/reference và 300 generated crops.
- Chỉ promote V4 nếu tất cả bank gate, visual gate và OCR ablation cùng pass.

## Command Runbook Theo Phase

Các lệnh dưới đây dùng config mặc định `configs/synthetic/vn_lplc.yaml`. Chạy
từ repo root:

```bash
export PY=/home/vietanh/anaconda3/envs/myenv/bin/python
export CFG=configs/synthetic/vn_lplc.yaml
export DEVICE=auto
```

### Phase 0: Freeze V3 Và Baseline Ablation

V3 production phải fail fast nếu không có cờ legacy:

```bash
$PY scripts/synthetic_vn_lplc_build_vn_blank_bank.py --config $CFG --limit 1
$PY scripts/synthetic_vn_lplc_build_reference_bank.py --config $CFG --limit 1
$PY scripts/synthetic_vn_lplc_generate_reference.py --config $CFG --long-count 1
```

Chỉ chạy V3 trong ablation root để tái hiện lỗi:

```bash
$PY scripts/synthetic_vn_lplc_build_vn_blank_bank.py \
  --config $CFG \
  --output-dir data/synthetic/vn_lplc_reference_v3_ablation/vn_blank_bank_smoke \
  --manifest data/synthetic/vn_lplc_reference_v3_ablation/vn_blank_bank_smoke.jsonl \
  --inpaint-backend telea \
  --limit 50 \
  --allow-legacy-v3-baseline

$PY scripts/synthetic_vn_lplc_build_reference_bank.py \
  --config $CFG \
  --output-dir data/synthetic/vn_lplc_reference_v3_ablation/lplc_reference_bank_smoke \
  --manifest data/synthetic/vn_lplc_reference_v3_ablation/lplc_reference_bank_smoke.jsonl \
  --inpaint-backend telea \
  --disable-obb \
  --limit 50 \
  --allow-legacy-v3-baseline

$PY scripts/synthetic_vn_lplc_generate_reference.py \
  --config $CFG \
  --vn-blank-manifest data/synthetic/vn_lplc_reference_v3_ablation/vn_blank_bank_smoke.jsonl \
  --reference-manifest data/synthetic/vn_lplc_reference_v3_ablation/lplc_reference_bank_smoke.jsonl \
  --output-dir data/synthetic/vn_lplc_reference_v3_ablation/generated_smoke \
  --manifest data/synthetic/vn_lplc_reference_v3_ablation/generated_smoke.jsonl \
  --backend deterministic \
  --long-count 10 \
  --short-count 10 \
  --motor-count 10 \
  --allow-legacy-v3-baseline

$PY scripts/synthetic_vn_lplc_filter_reference.py \
  --config $CFG \
  --manifest data/synthetic/vn_lplc_reference_v3_ablation/generated_smoke.jsonl \
  --accepted data/synthetic/vn_lplc_reference_v3_ablation/generated_smoke_accepted.jsonl \
  --hard-pool data/synthetic/vn_lplc_reference_v3_ablation/generated_smoke_hard_pool.jsonl \
  --rejected data/synthetic/vn_lplc_reference_v3_ablation/generated_smoke_rejected.jsonl \
  --device $DEVICE \
  --disable-style-models \
  --allow-legacy-v3-baseline
```

Tạo review/promotion report cho ablation V3:

```bash
$PY scripts/synthetic_vn_lplc_pilot_gate.py \
  --config $CFG \
  --accepted-manifest data/synthetic/vn_lplc_reference_v3_ablation/generated_smoke_accepted.jsonl \
  --review-manifest data/synthetic/vn_lplc_reference_v3_ablation/generated_smoke_review.jsonl \
  --promotion-report data/synthetic/vn_lplc_reference_v3_ablation/generated_smoke_promotion_report.json \
  --review-count 300 \
  --target-per-layout 100 \
  --allow-legacy-v3-baseline
```

### Phase 1: Template Bank Và Template-Only Generator

Build toàn bộ template bank V4:

```bash
$PY scripts/synthetic_vn_lplc_build_template_bank.py --config $CFG
```

Smoke một appearance/layout:

```bash
$PY scripts/synthetic_vn_lplc_build_template_bank.py \
  --config $CFG \
  --output-dir /tmp/vn_lplc_v4_smoke/templates \
  --accepted-manifest /tmp/vn_lplc_v4_smoke/templates.jsonl \
  --layouts long \
  --appearances state_blue_white
```

Generate template-only V4 với validators thật:

```bash
$PY scripts/synthetic_vn_lplc_generate_reference_v4.py \
  --config $CFG \
  --template-manifest data/synthetic/vn_lplc_reference_v4/template_bank_accepted.jsonl \
  --output-dir data/synthetic/vn_lplc_reference_v4/generated_template_only \
  --accepted-manifest data/synthetic/vn_lplc_reference_v4/generated_template_only_accepted.jsonl \
  --rejected-manifest data/synthetic/vn_lplc_reference_v4/generated_template_only_rejected.jsonl \
  --manual-review-manifest data/synthetic/vn_lplc_reference_v4/generated_template_only_manual_review.jsonl \
  --split train \
  --long-count 1000 \
  --short-count 1000 \
  --motor-count 1000 \
  --device $DEVICE
```

Debug generator không validator, chỉ để kiểm tra geometry/manifest; mọi record
phải vào `manual_review`:

```bash
$PY scripts/synthetic_vn_lplc_generate_reference_v4.py \
  --config $CFG \
  --template-manifest /tmp/vn_lplc_v4_smoke/templates.jsonl \
  --output-dir /tmp/vn_lplc_v4_smoke/generated_debug \
  --accepted-manifest /tmp/vn_lplc_v4_smoke/generated_debug_accepted.jsonl \
  --rejected-manifest /tmp/vn_lplc_v4_smoke/generated_debug_rejected.jsonl \
  --manual-review-manifest /tmp/vn_lplc_v4_smoke/generated_debug_manual.jsonl \
  --long-count 10 \
  --disable-independent-validators
```

### Phase 2: Surface Residual, LPLC Reference Bank Và Bank Audit

Build VN surface residual candidate bank:

```bash
$PY scripts/synthetic_vn_lplc_build_surface_residual_bank.py \
  --config $CFG \
  --output-dir data/synthetic/vn_lplc_reference_v4/surface_residual_bank \
  --accepted-manifest data/synthetic/vn_lplc_reference_v4/surface_residual_accepted.jsonl \
  --rejected-manifest data/synthetic/vn_lplc_reference_v4/surface_residual_rejected.jsonl \
  --manual-review-manifest data/synthetic/vn_lplc_reference_v4/surface_residual_manual_review.jsonl \
  --device $DEVICE
```

Smoke VN surface residual nhanh:

```bash
$PY scripts/synthetic_vn_lplc_build_surface_residual_bank.py \
  --config $CFG \
  --output-dir /tmp/vn_lplc_v4_smoke/surface \
  --accepted-manifest /tmp/vn_lplc_v4_smoke/surface_accepted.jsonl \
  --rejected-manifest /tmp/vn_lplc_v4_smoke/surface_rejected.jsonl \
  --manual-review-manifest /tmp/vn_lplc_v4_smoke/surface_manual.jsonl \
  --exclude-char-yolo-train \
  --limit 10 \
  --device cpu
```

Build LPLC reference/style bank:

```bash
$PY scripts/synthetic_vn_lplc_build_reference_bank_v4.py \
  --config $CFG \
  --output-dir data/synthetic/vn_lplc_reference_v4/lplc_reference_bank \
  --accepted-manifest data/synthetic/vn_lplc_reference_v4/lplc_reference_accepted.jsonl \
  --rejected-manifest data/synthetic/vn_lplc_reference_v4/lplc_reference_rejected.jsonl \
  --manual-review-manifest data/synthetic/vn_lplc_reference_v4/lplc_reference_manual_review.jsonl \
  --scene-inpaint-backend telea \
  --device $DEVICE
```

Smoke LPLC reference không OBB fallback:

```bash
$PY scripts/synthetic_vn_lplc_build_reference_bank_v4.py \
  --config $CFG \
  --output-dir /tmp/vn_lplc_v4_smoke/reference \
  --accepted-manifest /tmp/vn_lplc_v4_smoke/reference_accepted.jsonl \
  --rejected-manifest /tmp/vn_lplc_v4_smoke/reference_rejected.jsonl \
  --manual-review-manifest /tmp/vn_lplc_v4_smoke/reference_manual.jsonl \
  --disable-obb \
  --scene-inpaint-backend telea \
  --limit 10 \
  --device cpu
```

Audit/repartition các bank V4 sau khi build:

```bash
$PY scripts/synthetic_vn_lplc_filter_blank_bank.py \
  --config $CFG \
  --manifest data/synthetic/vn_lplc_reference_v4/template_bank_accepted.jsonl \
  --manifest data/synthetic/vn_lplc_reference_v4/surface_residual_accepted.jsonl \
  --manifest data/synthetic/vn_lplc_reference_v4/surface_residual_rejected.jsonl \
  --manifest data/synthetic/vn_lplc_reference_v4/surface_residual_manual_review.jsonl \
  --manifest data/synthetic/vn_lplc_reference_v4/lplc_reference_accepted.jsonl \
  --manifest data/synthetic/vn_lplc_reference_v4/lplc_reference_rejected.jsonl \
  --manifest data/synthetic/vn_lplc_reference_v4/lplc_reference_manual_review.jsonl \
  --accepted data/synthetic/vn_lplc_reference_v4/bank_audit_accepted.jsonl \
  --rejected data/synthetic/vn_lplc_reference_v4/bank_audit_rejected.jsonl \
  --manual-review data/synthetic/vn_lplc_reference_v4/bank_audit_manual_review.jsonl
```

### Phase 3: Generated Dataset, Text-Removal Pairs Và Ablation

Generate V4 dùng template + accepted VN surface residual + accepted LPLC style:

```bash
$PY scripts/synthetic_vn_lplc_generate_reference_v4.py \
  --config $CFG \
  --template-manifest data/synthetic/vn_lplc_reference_v4/template_bank_accepted.jsonl \
  --surface-manifest data/synthetic/vn_lplc_reference_v4/surface_residual_accepted.jsonl \
  --reference-manifest data/synthetic/vn_lplc_reference_v4/lplc_reference_accepted.jsonl \
  --output-dir data/synthetic/vn_lplc_reference_v4/generated \
  --accepted-manifest data/synthetic/vn_lplc_reference_v4/generated_accepted.jsonl \
  --rejected-manifest data/synthetic/vn_lplc_reference_v4/generated_rejected.jsonl \
  --manual-review-manifest data/synthetic/vn_lplc_reference_v4/generated_manual_review.jsonl \
  --split train \
  --long-count 20000 \
  --short-count 20000 \
  --motor-count 20000 \
  --seed 42 \
  --device $DEVICE
```

Audit/repartition generated V4:

```bash
$PY scripts/synthetic_vn_lplc_filter_reference_v4.py \
  --config $CFG \
  --manifest data/synthetic/vn_lplc_reference_v4/generated_accepted.jsonl \
  --manifest data/synthetic/vn_lplc_reference_v4/generated_rejected.jsonl \
  --manifest data/synthetic/vn_lplc_reference_v4/generated_manual_review.jsonl \
  --accepted data/synthetic/vn_lplc_reference_v4/generated_filtered_accepted.jsonl \
  --rejected data/synthetic/vn_lplc_reference_v4/generated_filtered_rejected.jsonl \
  --manual-review data/synthetic/vn_lplc_reference_v4/generated_filtered_manual_review.jsonl
```

Sinh paired data cho text eraser/TMIM:

```bash
$PY scripts/synthetic_vn_lplc_prepare_text_removal_dataset.py \
  --config $CFG \
  --template-manifest data/synthetic/vn_lplc_reference_v4/template_bank_accepted.jsonl \
  --output-dir data/synthetic/vn_lplc_reference_v4/text_removal_pairs \
  --manifest data/synthetic/vn_lplc_reference_v4/text_removal_pairs_train.jsonl \
  --count 50000 \
  --split train \
  --seed 42

$PY scripts/synthetic_vn_lplc_prepare_text_removal_dataset.py \
  --config $CFG \
  --template-manifest data/synthetic/vn_lplc_reference_v4/template_bank_accepted.jsonl \
  --output-dir data/synthetic/vn_lplc_reference_v4/text_removal_pairs \
  --manifest data/synthetic/vn_lplc_reference_v4/text_removal_pairs_valid.jsonl \
  --count 5000 \
  --split valid \
  --seed 1042

$PY scripts/synthetic_vn_lplc_prepare_text_removal_dataset.py \
  --config $CFG \
  --template-manifest data/synthetic/vn_lplc_reference_v4/template_bank_accepted.jsonl \
  --output-dir data/synthetic/vn_lplc_reference_v4/text_removal_pairs \
  --manifest data/synthetic/vn_lplc_reference_v4/text_removal_pairs_test.jsonl \
  --count 5000 \
  --split test \
  --seed 2042
```

TMIM dry-run; command chỉ validate dataset/config và in lệnh train:

```bash
$PY scripts/synthetic_vn_lplc_train_text_eraser.py \
  --tmim-root /path/to/TMIM \
  --tmim-config configs/uformer_t_vn_plate.py \
  --dataset-root data/synthetic/vn_lplc_reference_v4/text_removal_pairs/text_rmv/VNPlate \
  --model-size T \
  --checkpoint-name vn_plate_uformer_t_tmim \
  --nproc-per-node 2
```

Chạy train thật chỉ khi đã kiểm tra config custom Uformer-T/S:

```bash
$PY scripts/synthetic_vn_lplc_train_text_eraser.py \
  --tmim-root /path/to/TMIM \
  --tmim-config configs/uformer_t_vn_plate.py \
  --dataset-root data/synthetic/vn_lplc_reference_v4/text_removal_pairs/text_rmv/VNPlate \
  --model-size T \
  --checkpoint-name vn_plate_uformer_t_tmim \
  --nproc-per-node 2 \
  --execute
```

Uformer-B bị chặn theo mặc định; chỉ chạy khi đã có tài nguyên phù hợp và muốn
override:

```bash
$PY scripts/synthetic_vn_lplc_train_text_eraser.py \
  --tmim-root /path/to/TMIM \
  --tmim-config configs/uformer_b_vn_plate.py \
  --dataset-root data/synthetic/vn_lplc_reference_v4/text_removal_pairs/text_rmv/VNPlate \
  --model-size B \
  --checkpoint-name vn_plate_uformer_b_tmim \
  --nproc-per-node 2 \
  --allow-uformer-b-resource-override \
  --execute
```

Các ablation chính:

```bash
# 1. Template-only
$PY scripts/synthetic_vn_lplc_generate_reference_v4.py \
  --config $CFG \
  --template-manifest data/synthetic/vn_lplc_reference_v4/template_bank_accepted.jsonl \
  --output-dir data/synthetic/vn_lplc_reference_v4/ablation_template_only \
  --accepted-manifest data/synthetic/vn_lplc_reference_v4/ablation_template_only_accepted.jsonl \
  --rejected-manifest data/synthetic/vn_lplc_reference_v4/ablation_template_only_rejected.jsonl \
  --manual-review-manifest data/synthetic/vn_lplc_reference_v4/ablation_template_only_manual_review.jsonl \
  --long-count 1000 --short-count 1000 --motor-count 1000 \
  --device $DEVICE

# 2. Template + LPLC accepted style
$PY scripts/synthetic_vn_lplc_generate_reference_v4.py \
  --config $CFG \
  --template-manifest data/synthetic/vn_lplc_reference_v4/template_bank_accepted.jsonl \
  --reference-manifest data/synthetic/vn_lplc_reference_v4/lplc_reference_accepted.jsonl \
  --output-dir data/synthetic/vn_lplc_reference_v4/ablation_lplc_style \
  --accepted-manifest data/synthetic/vn_lplc_reference_v4/ablation_lplc_style_accepted.jsonl \
  --rejected-manifest data/synthetic/vn_lplc_reference_v4/ablation_lplc_style_rejected.jsonl \
  --manual-review-manifest data/synthetic/vn_lplc_reference_v4/ablation_lplc_style_manual_review.jsonl \
  --long-count 1000 --short-count 1000 --motor-count 1000 \
  --device $DEVICE

# 3. Template + VN surface residual + LPLC style
$PY scripts/synthetic_vn_lplc_generate_reference_v4.py \
  --config $CFG \
  --template-manifest data/synthetic/vn_lplc_reference_v4/template_bank_accepted.jsonl \
  --surface-manifest data/synthetic/vn_lplc_reference_v4/surface_residual_accepted.jsonl \
  --reference-manifest data/synthetic/vn_lplc_reference_v4/lplc_reference_accepted.jsonl \
  --output-dir data/synthetic/vn_lplc_reference_v4/ablation_surface_lplc_style \
  --accepted-manifest data/synthetic/vn_lplc_reference_v4/ablation_surface_lplc_style_accepted.jsonl \
  --rejected-manifest data/synthetic/vn_lplc_reference_v4/ablation_surface_lplc_style_rejected.jsonl \
  --manual-review-manifest data/synthetic/vn_lplc_reference_v4/ablation_surface_lplc_style_manual_review.jsonl \
  --long-count 1000 --short-count 1000 --motor-count 1000 \
  --device $DEVICE
```

### Phase 4: Pilot, Manual Review Và Promotion Gate

Tạo generated review set và promotion report. Nếu chưa có `metrics-json`, report
phải ghi `promote=false`:

```bash
$PY scripts/synthetic_vn_lplc_pilot_gate_v4.py \
  --config $CFG \
  --accepted-manifest data/synthetic/vn_lplc_reference_v4/generated_filtered_accepted.jsonl \
  --target-per-slice 100 \
  --review-count 300 \
  --review-manifest data/synthetic/vn_lplc_reference_v4/generated_review_300.jsonl \
  --report data/synthetic/vn_lplc_reference_v4/v4_promotion_report.json
```

Template cho metrics JSON sau manual review/eval:

```bash
cat > data/synthetic/vn_lplc_reference_v4/v4_metrics.json <<'JSON'
{
  "exact_ocr_match_rate": 0.0,
  "ghost_source_text_rate": 0.0,
  "visual_pass_rate": 0.0,
  "real_vn_hard_validation_delta": 0.0,
  "clean_validation_delta": 0.0,
  "manual_blank_ghost_count": 0,
  "manual_generated_ghost_count": 0
}
JSON
```

Chạy promotion gate thật; lệnh này được phép fail nếu bất kỳ gate nào chưa đạt:

```bash
$PY scripts/synthetic_vn_lplc_pilot_gate_v4.py \
  --config $CFG \
  --accepted-manifest data/synthetic/vn_lplc_reference_v4/generated_filtered_accepted.jsonl \
  --target-per-slice 100 \
  --review-count 300 \
  --review-manifest data/synthetic/vn_lplc_reference_v4/generated_review_300.jsonl \
  --metrics-json data/synthetic/vn_lplc_reference_v4/v4_metrics.json \
  --report data/synthetic/vn_lplc_reference_v4/v4_promotion_report.json \
  --fail-on-promotion-fail
```

### Verification Commands

Synthetic regression suite:

```bash
$PY -m pytest -q \
  tests/test_synthetic_vn_lplc.py \
  tests/test_synthetic_vn_lplc_v3.py \
  tests/test_synthetic_vn_lplc_v3_pilot.py \
  tests/test_synthetic_vn_lplc_v4.py \
  --disable-warnings --maxfail=1
```

Static checks cho phần V4:

```bash
$PY -m ruff check \
  synthetic_vn_lplc/v4_synthesis.py \
  synthetic_vn_lplc/validators.py \
  synthetic_vn_lplc/v4_filtering.py \
  synthetic_vn_lplc/v4_pilot.py \
  synthetic_vn_lplc/text_eraser_training.py \
  synthetic_vn_lplc/v4_reference_bank.py \
  synthetic_vn_lplc/surface_residual_bank.py \
  synthetic_vn_lplc/text_mask.py \
  synthetic_vn_lplc/text_removal_dataset.py \
  synthetic_vn_lplc/config.py \
  scripts/synthetic_vn_lplc_build_template_bank.py \
  scripts/synthetic_vn_lplc_build_surface_residual_bank.py \
  scripts/synthetic_vn_lplc_build_reference_bank_v4.py \
  scripts/synthetic_vn_lplc_prepare_text_removal_dataset.py \
  scripts/synthetic_vn_lplc_generate_reference_v4.py \
  scripts/synthetic_vn_lplc_filter_blank_bank.py \
  scripts/synthetic_vn_lplc_filter_reference_v4.py \
  scripts/synthetic_vn_lplc_pilot_gate_v4.py \
  scripts/synthetic_vn_lplc_train_text_eraser.py \
  tests/test_synthetic_vn_lplc_v4.py \
  --ignore E501

$PY -m compileall -q synthetic_vn_lplc scripts
git diff --check -- SyntheticVietnameseLicensePlateAsLPLCV2.md configs/synthetic/vn_lplc.yaml
```

Full repo tests có thể chạy để phát hiện regression ngoài phạm vi synthetic:

```bash
$PY -m pytest -q --disable-warnings --maxfail=1
```

Nếu full suite dừng ở `tests/test_incident_crud.py` với async fixture warning thì
đó là lỗi môi trường/plugin test hiện có, không phải gate V4.

## Test Plan Và Promotion Gates

### Unit/Integration

- `dark_on_light`, `light_on_blue`, `light_on_red`, `dark_on_yellow`.
- separator `-` và `.`, blur halo, glare, low contrast, compression.
- mask fusion không chạm border/protected detail.
- adaptive expansion tăng theo blur/stroke width.
- template blank luôn có zero glyph pixels.
- V4 reject V3 manifests và record thiếu quality metrics.
- generator không bảo vệ residual text gradient.
- validation-only checkpoint không trùng checkpoint/model family dùng để sinh
  mask/candidate.
- `state_blue_white` và `military_red_light` có positive test end-to-end, không
  được biến thành toàn bộ rejected slice.
- mixed-color appearance có token-level color/polarity assertions.

### Mask/Text-Removal Benchmark

- Synthetic paired test:
  - stroke-mask recall `>= 99,5%`;
  - text-mask precision `>= 95%`;
  - residual stroke energy ratio `<= 0,03`;
  - outside-mask SSIM `>= 0,99`.
- Real manually masked/retouched test:
  - ghost original text: `0/600` trong manual blank review;
  - border damage rate `<= 1%`;
  - từng appearance/layout slice phải pass, không chỉ aggregate.

### Generated Dataset

- exact OCR match rate `>= 95%`;
- canonical aspect error `<= 0,5%`;
- ghost source text rate `<= 0,1%` và `0` trong manual review sample;
- visual pass rate `>= 90%`;
- accepted count tối thiểu theo từng appearance/layout slice đã khai báo, gồm
  blue và red; thiếu slice là promotion fail;
- real VN hard-validation seq accuracy tăng so với real-only;
- clean validation không giảm quá `1%`.

### Ablation Bắt Buộc

1. real-only;
2. renderer/template-only;
3. template + LPLC accepted style;
4. template + deterministic VN residual + LPLC style;
5. template + TMIM residual/reference cleanup + LPLC style;
6. V3 LaMa baseline, chỉ để chứng minh lỗi đã được loại bỏ.

Chọn pipeline đơn giản nhất vượt toàn bộ gate. Learned eraser không được promote
nếu không tạo lợi ích rõ ràng so với deterministic/template-only.

## Assumptions Và Non-Goals

- Mục tiêu khóa là OCR crop cho PARSeq/SlotLPR.
- Không tạo biển trùng label thật trong VN raw datasets.
- Không cố giữ mọi real VN crop; reject là hành vi đúng khi uncertainty cao.
- Appearance/grammar chưa được xác minh sẽ không được sinh production.
- Tài nguyên mặc định: local 6 GB cho smoke; server 2 GPU 16 GB cho training và
  pilot.

## Nguồn Kỹ Thuật

- LaMa, general large-mask inpainting:
  [WACV 2022](https://openaccess.thecvf.com/content/WACV2022/html/Suvorov_Resolution-Robust_Large_Mask_Inpainting_With_Fourier_Convolutions_WACV_2022_paper.html)
- TMIM, text-aware background modeling/text erasing và so sánh với LaMa:
  [ECCV 2024 paper](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/08360.pdf),
  [official code](https://github.com/wzx99/TMIM)
- Stroke-aware text erasing:
  [SAEN, WACV 2023](https://openaccess.thecvf.com/content/WACV2023/papers/Du_Modeling_Stroke_Mask_for_End-to-End_Text_Erasing_WACV_2023_paper.pdf)
- Character/text localization:
  [CRAFT, CVPR 2019](https://openaccess.thecvf.com/content_CVPR_2019/papers/Baek_Character_Region_Awareness_for_Text_Detection_CVPR_2019_paper.pdf),
  [DBNet, AAAI 2020](https://ojs.aaai.org/index.php/AAAI/article/download/6812/6666)
- FETNet alternative:
  [paper](https://arxiv.org/abs/2306.09593),
  [official code](https://github.com/GuangtaoLyu/FETNet)
- Màu/seri biển số Việt Nam áp dụng từ 2025:
  [Thông tư 79/2024/TT-BCA, Bộ Công an](https://bocongan.gov.vn/chinh-sach-phap-luat/bai-viet/nhan-dien-mau-sac-seri-ky-hieu-bien-so-xe-cua-co-quan-to-chuc-ca-nhan-tu-01012025-d1-t1617),
  [QCVN 08:2024/BCA](https://mps.gov.vn/chinh-sach-phap-luat/bai-viet/quy-chuan-ky-thuat-quoc-gia-ve-bien-so-xe-d1-t1592)
