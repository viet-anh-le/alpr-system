# Thiết Kế Pipeline ALPR Có Quality Router Theo LPLCv2

## Summary

Thiết kế pipeline mới sẽ giữ nền tảng hiện tại: YOLOv8 OBB phát hiện biển `BSD/BSV`, tracking theo phương tiện bằng BoT-SORT/ReID, OCR single-frame hiện có, `_segment_vote`, `_prob_vote`, template matching và format validation. Điểm nâng cấp chính là thêm **Plate Quality Router** trước quyết định OCR/fusion, lấy cảm hứng từ LPLCv2: `Perfect`, `Good`, `Poor`, `Illegible`, hoặc nhị phân `Suitable/Unsuitable`.

Bổ sung sau trao đổi: `_segment_vote` và `_prob_vote` có thể được thay bằng **CTM-first fusion** làm module hợp nhất chính. Trong thiết kế mới, các triết lí voting/tổng hợp prob vẫn giữ, nhưng được đưa vào bên trong CTM thay vì tồn tại như các nhánh vote rời rạc.

Luồng quyết định:

```text
frame/video
  -> vehicle detect + vehicle track
  -> plate OBB detect BSD/BSV
  -> rectify plate crop
  -> quality router
  -> route-specific OCR/enhancement/fusion
  -> plate format validation + confidence rerank + temporal consistency
  -> emit recognized / rejected / unreadable
```

Cơ sở nghiên cứu:
[LPLCv2](https://arxiv.org/html/2604.08741) đề xuất legibility classification để bỏ mẫu unusable, OCR trực tiếp mẫu high-quality, hoặc xử lí thêm mẫu degraded. [LPLC](https://arxiv.org/html/2508.18425) cảnh báo SR/enhancement có thể làm OCR tệ hơn nếu áp dụng mù quáng. [ICPR 2026 LRLPR](https://arxiv.org/html/2604.22506v1) cho thấy multi-frame fusion, character voting, logit aggregation là hướng mạnh cho biển low-resolution. [CTM](https://github.com/chequanghuy/Character-Time-series-Matching) là triết lí phù hợp để gom bằng chứng ký tự theo tracklet.

## Key Changes

### 1. Plate Quality Router

Train thêm một model classifier từ LPLCv2 local:

```text
Input: rectified plate crop
Output:
  legibility: perfect | good | poor | illegible
  quality_bin: suitable | unsuitable
  degradation_tags: optional heuristic/proxy tags
```

Mapping chính:

```text
perfect, good -> suitable
poor           -> recoverable_degraded
illegible      -> unreadable_candidate
```

Dữ liệu LPLCv2 local có thể dùng trực tiếp:
`37,099` ảnh, `41,487` biển, legibility `{0: illegible, 1: poor, 2: good, 3: perfect}`. Metadata có `rain`, `faulty`, `time`, `occ`, nhưng `motion blur/low-res/low contrast` không phải nhãn trực tiếp, nên chỉ dùng làm proxy hoặc rule phụ.

Model đề xuất:
- V1 thesis-strong: YOLO-cls hoặc ResNet/ViT theo reference `references/LPLCv2-Dataset/`.
- Train 2 head hoặc 2 config:
  - 4-class: `illegible/poor/good/perfect`.
  - binary quality filter: `suitable = good+perfect`, `unsuitable = poor+illegible`.
- Inference dùng crop đã rectify, không dùng layout head vì detector đã có `BSD/BSV`.

Bổ sung quan trọng: `suitable/unsuitable` chỉ là **gate tầng 1**, không đủ để phân biệt route B/C/D/E. Cần thêm tầng **Degradation Diagnosis**:

```text
Quality Filter:
  suitable / unsuitable

Degradation Diagnosis:
  poor_or_low_res
  motion_blur
  low_light_or_low_contrast
  rain_or_haze
  faulty_color
  occluded_or_illegible
```

Router output cuối cùng nên là:

```text
{
  legibility: perfect | good | poor | illegible,
  quality_bin: suitable | unsuitable,
  router_conf: float,
  tags: {
    low_res,
    motion_blur,
    low_light,
    low_contrast,
    rain_or_haze,
    faulty_color,
    occluded
  }
}
```

### 2. Routing Logic

Route A: `suitable/perfect/good`

```text
plate crop -> OCR single-frame -> format validation -> accept nếu confidence đủ cao
```

Không chờ tracklet nếu OCR đạt:
- plate regex hợp lệ,
- mean char confidence cao,
- không có ký tự nghi ngờ theo confusion/template check.

Nếu confidence thấp dù router báo suitable, fallback sang route poor để không bỏ lỡ biển khó.

Route B: `poor/low-res`

```text
buffer tracklet -> OCR từng crop -> OCR-output CTM/prob fusion/template rerank -> final plate
```

Không train char detector. CTM được chuyển hóa thành **OCR-output CTM**:
- mỗi frame cung cấp string, per-char confidence, về sau mở rộng top-k/logits;
- align chuỗi qua các frame bằng edit distance/position-aware matching;
- gom xác suất theo vị trí ký tự;
- áp dụng regex Việt Nam và layout `BSD/BSV`;
- dùng template matching chỉ để rerank ký tự nhập nhằng như `0/O/D`, `1/I`, `5/S`, `8/B`.

Bổ sung CTM-first:
- Không vote theo raw index vì OCR có thể thiếu/thừa dấu `-` hoặc lệch ký tự.
- CTM phải align chuỗi trước, sau đó mới vote theo slot đã align.
- Chỉ chấp nhận ký tự nếu support vượt `50%` số frame hợp lệ tại slot đó.
- Nếu không có ký tự vượt `50%`, slot đó là unresolved; không bịa ký tự.

Rule:

```text
min_frames = 3
accept char at slot_i iff:
  support(char, slot_i) / valid_frames(slot_i) > 0.5
  and weighted_confidence(slot_i, char) >= threshold
```

Cascade mới:

```text
OCR-output CTM
  -> nếu tie/ambiguous: template matching rerank
  -> nếu vẫn unresolved: rejected_vehicle hoặc unreadable
  -> validate VN plate format
```

Route C: `motion blur`

Không dùng LPDGAN vì không train mô hình khử mờ riêng.

```text
original crop
  -> OCR gốc
  -> deblur/enhancement candidate bằng classical filters nhẹ
  -> OCR candidate
  -> chọn bằng confidence + regex + temporal consistency
```

Candidate không được thay thế mù quáng. Luôn giữ OCR gốc làm baseline. Deblur classical có thể gồm sharpen kernel, Wiener-like/USM, hoặc multi-frame chọn frame sắc nhất thay vì generate chi tiết mới.

Route D: `night / low contrast / rain / faulty camera`

```text
original crop
  -> candidate set:
       grayscale
       white balance/color constancy
       CLAHE
       gamma correction
       denoise
       dehaze/contrast normalization nếu cần
  -> OCR từng candidate
  -> rerank bằng OCR confidence + VN plate format + temporal consistency
```

`faulty camera` trong LPLCv2 là metadata ảnh, nên có thể train classifier/proxy riêng hoặc dùng rule màu: magenta/red cast, saturation lệch, low contrast.

Route E: `occluded/illegible`

```text
không hallucinate
  -> nếu đang trong video: chờ frame tốt hơn cùng vehicle track
  -> nếu hết track: emit unreadable
```

Không ép OCR ra biển hợp lệ nếu router `illegible`, OCR confidence thấp, hoặc temporal votes mâu thuẫn.

Bổ sung: B/C/D/E không nhất thiết loại trừ nhau. Một crop có thể vừa `poor`, vừa `motion_blur`, vừa `low_light`; khi đó pipeline tạo nhiều OCR candidate tương ứng, rồi đưa toàn bộ evidence vào CTM/reranker.

### 3. Future Multi-Frame OCR Model

Thiết kế future branch lấy cảm hứng từ team Việt Nam tại ICPR 2026:

```text
T crops per vehicle track
  -> STN/alignment
  -> frame encoder CNN/ViT
  -> temporal fusion/Transformer attention
  -> CTC hoặc autoregressive decoder
  -> layout/format-aware decoding
```

Tính khả thi với UFPR/RodoSol:

- **UFPR-ALPR**: khả thi hơn cho multi-frame vì là video-based, có track/frame sequence. Có thể tạo input `T x C x H x W` theo cùng biển, train OCR sequence hoặc train fusion trên logits. Nhưng cần chuyển nhãn Brazil sang charset/model riêng, không trực tiếp là tiếng Việt.
- **RodoSol-ALPR**: tốt để bổ sung đa dạng detection/OCR Brazil, nhưng nếu không có tracklet thật thì kém phù hợp hơn UFPR cho temporal fusion.
- **Cho đồ án hiện tại**: không nên đặt multi-frame OCR model là deliverable chính. Nên đặt là “future extension/ablation prototype”. Deliverable chính nên là router + OCR-output CTM/fusion, vì không cần data video VN và tận dụng được OCR hiện có.

## Implementation Plan

1. Add `PlateQualityRouter`
   - Load model quality classifier.
   - Return `legibility`, `quality_bin`, `router_conf`, and optional diagnostic scores.
   - Keep current `quality_score()` as numeric signal, but không dùng nó thay classifier.
   - Add `DegradationDiagnosis` after quality filter to produce route tags for B/C/D/E.

2. Build LPLCv2 training dataset
   - Convert LPLCv2 annotations into crop-class folders.
   - Train 4-class and binary quality filter.
   - Report macro-F1, per-class recall, confusion matrix.
   - Important metric: recall của `poor/illegible`, vì false suitable sẽ làm OCR sai tự tin.

3. Refactor inference decision
   - Current `process_frames()` buffers every matched crop.
   - New behavior:
     - suitable: OCR immediately and emit if valid/high confidence.
     - poor/low-res/blur/night/rain: buffer into tracklet.
     - illegible/occluded: buffer only as evidence, wait for better frame.
   - Track-level finalization remains at lost/end-of-video.
   - If multiple route tags are active, run all corresponding candidate transforms and send all OCR evidence to CTM.

4. Add candidate OCR runner
   - Function receives crop and route tags.
   - Produces candidates:
     ```text
     original
     grayscale/CLAHE
     white_balance
     denoise
     sharpen
     dehaze/contrast if selected
     ```
   - Each candidate runs OCR.
   - Reranker scores:
     ```text
     score = OCR confidence
           + format validity bonus
           + temporal agreement bonus
           + template matching bonus for ambiguous chars
           - hallucination/risk penalty
     ```

5. Add OCR-output CTM
   - Input: list of per-frame OCR outputs from a vehicle track.
   - Use no char detector.
   - Align strings by Vietnamese plate segments and edit distance.
   - Aggregate char probabilities per position.
   - Accept character only when support exceeds `50%` of valid aligned frames for that slot.
   - Use template matching only for ambiguous/tie cases.
   - Later extend `ocr_batch()` to expose top-k/logits per decoded step.

6. Add route-aware events/logging
   - Emit fields:
     ```text
     route
     legibility
     quality_bin
     degradation_tags
     router_conf
     ocr_method
     candidate_method
     ctm_support
     unresolved_slots
     vote_summary
     unreadable_reason
     ```
   - This makes ablation and thesis tables easy.

## Test Plan

Core tests:
- Router maps `perfect/good` to direct OCR.
- Router maps `poor` to tracklet fusion path.
- Router maps `illegible/occluded` to wait/unreadable, not forced OCR.
- Candidate reranker never selects invalid format over valid format when confidence is comparable.
- Candidate reranker keeps original OCR if enhancement lowers confidence or breaks format.
- OCR-output CTM improves over single-frame when frames disagree by one ambiguous char.
- CTM does not accept a character unless it appears in the same aligned slot in `>50%` valid frames.
- CTM marks unresolved slots instead of hallucinating when no character passes majority support.
- Template matching only changes/tie-breaks ambiguous characters and cannot override strong CTM majority.
- `_segment_vote`, `_prob_vote`, template rerank remain backward compatible until CTM-first replacement is verified.

Evaluation:
- LPLCv2 router:
  - macro-F1 4-class,
  - binary suitable/unsuitable F1,
  - recall for `poor` and `illegible`.
- Degradation diagnosis:
  - heuristic/proxy quality report for blur, low-res, low-light, contrast, faulty color, occlusion.
- ALPR pipeline:
  - single-frame baseline,
  - baseline + quality router,
  - router + candidate OCR,
  - router + OCR-output CTM,
  - router + template rerank.
- Video datasets:
  - UFPR for tracklet/temporal fusion evaluation.
  - RodoSol mainly for OCR/detection robustness unless usable temporal sequences are available.

## Assumptions

- Không dùng LPDGAN và không train deblur model riêng.
- Không train character detector.
- Không thêm layout head vì YOLO OBB đã có `BSD/BSV`.
- Không dùng synthetic VN plates làm trụ chính.
- Tracking vẫn theo phương tiện, không thêm plate embedding model.
- Multi-frame OCR model kiểu ICPR 2026 là hướng future, không phải deliverable chính.
- `suitable/unsuitable` là quality gate tầng 1, không phải full route classifier.
- B/C/D/E được quyết định bởi degradation tags tầng 2 và có thể đồng thời active.
- CTM-first có thể thay `_segment_vote/_prob_vote` sau khi test chứng minh không giảm accuracy.
- Deliverable chính đủ sức nặng: LPLCv2-style quality router + degradation diagnosis + route-specific OCR/enhancement + OCR-output CTM majority support + template-aware reranking + ablation rõ ràng.
