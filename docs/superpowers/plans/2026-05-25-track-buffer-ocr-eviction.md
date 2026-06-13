# TrackBuffer OCR-Confidence Eviction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix TrackBuffer eviction so that physically-sharper but OCR-garbled crops cannot displace lower-quality but correctly-read crops.

**Architecture:** Each plate crop is OCR-inferred immediately when buffered (batched across all matched crops per frame for efficiency). The resulting per-character average confidence (`ocr_conf`) is stored alongside the visual `quality_score`. Eviction and `top_k` ranking both use `combined = quality_score × ocr_conf`. At finalisation the cached per-frame OCR `char_probs` are reused directly in `_segment_vote` / `_prob_vote`, eliminating the redundant re-inference pass for the non-multiframe path.

**Tech Stack:** Python 3.10+, PyTorch, pytest. Core files: `api/core/tracker.py`, `api/core/pipeline_core.py`.

---

## Root Cause Summary

`TrackBuffer` evicts by lowest `quality_score` (Laplacian sharpness + crop area). When a 2-row plate moves closer to the camera the later crops become sharper (`q ≈ 0.93–0.94`) but the changed viewing angle garbles the OCR (`avg char conf ≈ 0.30–0.45`). These sharp-but-wrong crops evict the earlier correct crops (`q ≈ 0.90–0.93`, `avg char conf ≈ 0.93`). `top_k(5)` then selects only wrong crops, and the vote produces garbage.

**Concrete numbers from `hn_oto_18.mp4` Track 6 (`30G-51827`):**
- 25 correct frames, quality 0.890–0.929, avg OCR conf ≈ 0.929 → combined ≈ 0.846–0.864
- 12 wrong frames, quality 0.913–0.943, avg OCR conf ≈ 0.30–0.45 → combined ≈ 0.28–0.42

With combined scoring the 25 correct crops all outrank the 12 wrong ones.

---

## File Map

| File | Action | Purpose |
|------|--------|---------|
| `api/core/tracker.py` | **Modify** | Add `ocr_confs`, `char_prob_lists` to `TrackBuffer`; change eviction + `top_k` to use `combined`; update `buffer_crop()` signature |
| `api/core/pipeline_core.py` | **Modify** | Batch OCR all matched crops before buffering; pass `ocr_conf` + `char_probs` to `buffer_crop()`; use cached `prob_lists` from `top_k()` at finalisation |
| `tests/test_track_buffer.py` | **Create** | Unit tests for combined-score eviction and `top_k` ordering |
| `scripts/diag_ocr_frames.py` | **Modify** | Update `buffer_crop` call to new signature (minor) |

---

## Task 1: Write failing tests for TrackBuffer combined-score eviction

**Files:**
- Create: `tests/test_track_buffer.py`

- [ ] **Step 1: Write the test file**

```python
# tests/test_track_buffer.py
from __future__ import annotations

import numpy as np
import pytest

from api.core.tracker import TrackBuffer


def _crop(h: int = 24, w: int = 80) -> np.ndarray:
    return np.zeros((h, w, 3), dtype=np.uint8)


def _chars(avg_conf: float, n: int = 5) -> list[tuple[str, float]]:
    return [("A", avg_conf)] * n


class TestTrackBufferEviction:
    def test_high_visual_low_ocr_evicted_before_lower_visual_high_ocr(self):
        """
        Bug regression: a crop with high visual quality but low OCR confidence
        must NOT displace a crop with lower visual quality but high OCR confidence.

        combined(correct)  = 0.91 × 0.93 = 0.846
        combined(wrong)    = 0.94 × 0.35 = 0.329  ← should be evicted
        """
        buf = TrackBuffer(max_size=2)
        correct_chars = _chars(avg_conf=0.93)
        wrong_chars = _chars(avg_conf=0.35)

        buf.add(_crop(), quality_score=0.91, ocr_conf=0.93, char_probs=correct_chars, frame_idx=1)
        buf.add(_crop(), quality_score=0.92, ocr_conf=0.93, char_probs=correct_chars, frame_idx=2)
        # Buffer full. Add a visually sharper but OCR-garbled crop.
        buf.add(_crop(), quality_score=0.94, ocr_conf=0.35, char_probs=wrong_chars, frame_idx=3)

        assert len(buf.crops) == 2
        crops, scores, prob_lists = buf.top_k(k=2)
        # Both retained entries must be the correct-OCR ones (avg_conf=0.93)
        for pl in prob_lists:
            assert all(abs(p - 0.93) < 1e-6 for _, p in pl)

    def test_eviction_removes_worst_combined_not_worst_visual(self):
        """
        When the buffer is full and a new crop arrives, the crop with the lowest
        combined score (quality × ocr_conf) is evicted — even if it has a
        higher visual quality_score than the new crop.
        """
        buf = TrackBuffer(max_size=1)
        # Existing crop: high visual, low OCR → combined = 0.95 × 0.20 = 0.19
        buf.add(_crop(), quality_score=0.95, ocr_conf=0.20, char_probs=_chars(0.20), frame_idx=1)

        # New crop: lower visual, high OCR → combined = 0.80 × 0.93 = 0.744 > 0.19
        buf.add(_crop(), quality_score=0.80, ocr_conf=0.93, char_probs=_chars(0.93), frame_idx=2)

        # New crop should have replaced the existing one
        assert len(buf.crops) == 1
        _, _, prob_lists = buf.top_k(k=1)
        assert abs(prob_lists[0][0][1] - 0.93) < 1e-6

    def test_new_crop_not_added_when_combined_score_below_all_existing(self):
        """
        A new crop whose combined score is lower than every existing entry is
        rejected immediately (never added then evicted — eviction still removes
        the worst, which in this case is the new crop itself).
        """
        buf = TrackBuffer(max_size=2)
        buf.add(_crop(), quality_score=0.91, ocr_conf=0.93, char_probs=_chars(0.93), frame_idx=1)
        buf.add(_crop(), quality_score=0.92, ocr_conf=0.93, char_probs=_chars(0.93), frame_idx=2)

        # New crop: combined = 0.99 × 0.10 = 0.099 < min existing 0.846 → rejected
        buf.add(_crop(), quality_score=0.99, ocr_conf=0.10, char_probs=_chars(0.10), frame_idx=3)

        assert len(buf.crops) == 2
        _, _, prob_lists = buf.top_k(k=2)
        for pl in prob_lists:
            assert all(abs(p - 0.93) < 1e-6 for _, p in pl)

    def test_top_k_ordered_by_combined_score(self):
        """top_k returns crops ranked by combined score descending."""
        buf = TrackBuffer(max_size=5)
        # Insert in reverse order of expected rank
        data = [
            (0.90, 0.93, 3),   # combined 0.837 — rank 2
            (0.95, 0.20, 1),   # combined 0.190 — rank 3 (worst)
            (0.91, 0.95, 4),   # combined 0.865 — rank 1 (best)
        ]
        for q, ocr_c, fidx in data:
            buf.add(_crop(), quality_score=q, ocr_conf=ocr_c, char_probs=_chars(ocr_c), frame_idx=fidx)

        crops, scores, prob_lists = buf.top_k(k=3)
        # Scores should be sorted descending
        assert scores[0] >= scores[1] >= scores[2]
        # Best entry: q=0.91 × ocr=0.95 = 0.865
        assert abs(scores[0] - 0.91 * 0.95) < 1e-4
        # Worst entry: q=0.95 × ocr=0.20 = 0.190
        assert abs(scores[2] - 0.95 * 0.20) < 1e-4

    def test_top_k_returns_prob_lists(self):
        """top_k third return value contains the cached char_probs for each crop."""
        buf = TrackBuffer(max_size=3)
        chars_a = [("3", 0.93), ("0", 0.93), ("G", 0.91)]
        chars_b = [("6", 0.45), ("0", 0.38)]
        buf.add(_crop(), quality_score=0.92, ocr_conf=0.92, char_probs=chars_a, frame_idx=1)
        buf.add(_crop(), quality_score=0.90, ocr_conf=0.41, char_probs=chars_b, frame_idx=2)

        _, _, prob_lists = buf.top_k(k=2)
        # First (highest combined) should be chars_a
        assert prob_lists[0] == chars_a
        assert prob_lists[1] == chars_b

    def test_empty_char_probs_treated_as_low_conf(self):
        """A crop that produces no OCR output gets ocr_conf=0.1 (minimum penalty)."""
        buf = TrackBuffer(max_size=2)
        buf.add(_crop(), quality_score=0.92, ocr_conf=0.93, char_probs=_chars(0.93), frame_idx=1)
        # Empty char_probs → caller passes ocr_conf=0.1
        buf.add(_crop(), quality_score=0.99, ocr_conf=0.10, char_probs=[], frame_idx=2)

        assert len(buf.crops) == 2
        _, scores, _ = buf.top_k(k=2)
        # Good crop (0.92×0.93=0.856) must rank above empty crop (0.99×0.10=0.099)
        assert scores[0] > scores[1]
        assert abs(scores[0] - 0.92 * 0.93) < 1e-4
```

- [ ] **Step 2: Run tests — verify they all fail**

```bash
pytest tests/test_track_buffer.py -v
```

Expected: `FAILED` on every test (TrackBuffer.add doesn't accept `ocr_conf`/`char_probs` yet).

---

## Task 2: Implement combined-score eviction in TrackBuffer

**Files:**
- Modify: `api/core/tracker.py:88-120`

- [ ] **Step 1: Replace the TrackBuffer dataclass**

In `api/core/tracker.py`, replace the entire `TrackBuffer` class (lines 88–120) with:

```python
@dataclass
class TrackBuffer:
    """
    Per-track ring buffer of plate crops.

    Eviction policy: when full, the crop with the lowest
    *combined* score (visual quality × OCR confidence) is dropped.
    This prevents sharp-but-garbled crops from displacing softer-but-
    correctly-read ones — the root cause of the 30G-51827 mis-read.
    """

    crops:          list[np.ndarray]              = field(default_factory=list)
    quality_scores: list[float]                   = field(default_factory=list)
    ocr_confs:      list[float]                   = field(default_factory=list)
    char_prob_lists: list[list[tuple[str, float]]] = field(default_factory=list)
    frame_indices:  list[int]                     = field(default_factory=list)

    @staticmethod
    def _combined(quality: float, ocr_conf: float) -> float:
        return quality * max(ocr_conf, 0.10)

    def add(
        self,
        crop: np.ndarray,
        quality_score: float,
        ocr_conf: float,
        char_probs: list[tuple[str, float]],
        frame_idx: int,
    ) -> None:
        self.crops.append(crop)
        self.quality_scores.append(quality_score)
        self.ocr_confs.append(ocr_conf)
        self.char_prob_lists.append(char_probs)
        self.frame_indices.append(frame_idx)
        if len(self.crops) > MAX_BUFFER:
            worst = min(
                range(len(self.crops)),
                key=lambda i: self._combined(self.quality_scores[i], self.ocr_confs[i]),
            )
            del self.crops[worst]
            del self.quality_scores[worst]
            del self.ocr_confs[worst]
            del self.char_prob_lists[worst]
            del self.frame_indices[worst]

    def top_k(
        self, k: int = TOP_K_FRAMES
    ) -> tuple[list[np.ndarray], list[float], list[list[tuple[str, float]]]]:
        """Return up to k entries ranked by combined score descending."""
        if not self.crops:
            return [], [], []
        combined = [
            self._combined(q, c)
            for q, c in zip(self.quality_scores, self.ocr_confs)
        ]
        triples = sorted(
            zip(combined, self.crops, self.char_prob_lists),
            key=lambda x: x[0],
            reverse=True,
        )[:k]
        scores, crops, prob_lists = zip(*triples)
        return list(crops), list(scores), list(prob_lists)
```

- [ ] **Step 2: Update `WebTrackletManager.buffer_crop()` signature**

In `api/core/tracker.py`, replace the `buffer_crop` method (lines 195–204):

```python
def buffer_crop(
    self,
    tid: int,
    crop: np.ndarray,
    quality_score: float,
    ocr_conf: float,
    char_probs: list[tuple[str, float]],
    frame_idx: int,
) -> None:
    if tid not in self._buffers:
        self._buffers[tid] = TrackBuffer()
    self._buffers[tid].add(crop, quality_score, ocr_conf, char_probs, frame_idx)
```

- [ ] **Step 3: Run the tests — verify they all pass**

```bash
pytest tests/test_track_buffer.py -v
```

Expected: all 6 tests `PASSED`.

- [ ] **Step 4: Commit**

```bash
git add api/core/tracker.py tests/test_track_buffer.py
git commit -m "feat(tracker): evict by quality×ocr_conf instead of quality alone

High-visual-quality but OCR-garbled crops were displacing correct crops
from TrackBuffer, causing the final vote to run on wrong frames.
TrackBuffer now stores ocr_conf + char_probs per crop and uses
combined = quality × max(ocr_conf, 0.10) for eviction and top_k ranking.
"
```

---

## Task 3: Update pipeline_core — batch OCR at buffer time, cache for finalisation

**Files:**
- Modify: `api/core/pipeline_core.py:258-264` (buffering loop), `api/core/pipeline_core.py:77-101` (`_run_multiframe_ocr`)

The buffering loop must run OCR on all matched crops in a single batched call before writing to the buffer. `_run_multiframe_ocr` must then read the cached `prob_lists` from `top_k()` instead of re-running OCR (for the non-multiframe path).

**No new test is written here.** The failure signal is structural: after Task 2 renames `buffer_crop` to require 6 arguments, the old 4-argument call in `pipeline_core.py` is a `TypeError` at runtime. The unit tests in `test_track_buffer.py` (Task 1/2) cover the eviction logic; the end-to-end regression is validated by re-running the video diagnostic in Step 5.

The parity test (`tests/test_pipeline_core_parity.py`) runs against `tests/fixtures/short_clip.mp4` using real model weights (`weights/`). Because the eviction policy changes the buffer contents, the golden JSON (`tests/fixtures/golden_run_job_events.json`) may legitimately differ — better eviction may produce different (correct) plates. The golden file must be re-captured after the implementation and committed as the new baseline.

- [ ] **Step 1: Confirm the API mismatch causes a TypeError**

After Task 2 is committed, verify the old call site is broken:

```bash
python -c "
from api.core.frame_source import FileFrameSource
from api.core.models import load_models
from api.core.pipeline_core import process_frames
models = load_models()
src = FileFrameSource('tests/fixtures/short_clip.mp4')
process_frames(src, lambda e: None, models)
" 2>&1 | grep -E "TypeError|Error"
```

Expected: `TypeError: buffer_crop() missing 2 required positional arguments: 'ocr_conf' and 'char_probs'`

- [ ] **Step 2: Update the buffering loop in `process_frames`**

In `api/core/pipeline_core.py`, replace lines 258–264:

```python
        for tid, plate_crop, vehicle_crop in matched:
            if not tracker.should_ocr(tid):
                continue
            q = quality_score(plate_crop)
            tracker.buffer_crop(tid, plate_crop, q, frame_idx)
            tracker.update_vehicle_img(tid, vehicle_crop, q)
            active_tids.add(tid)
```

with:

```python
        to_buffer = [
            (tid, plate_crop, vehicle_crop)
            for tid, plate_crop, vehicle_crop in matched
            if tracker.should_ocr(tid)
        ]

        if to_buffer:
            _tensors = torch.stack(
                [preprocess_plate(pc) for _, pc, _ in to_buffer]
            ).to(models.device)
            _ocr_results = ocr_batch(models.ocr, _tensors, models.device)

            for (tid, plate_crop, vehicle_crop), (char_probs, _) in zip(to_buffer, _ocr_results):
                ocr_conf = (
                    sum(p for _, p in char_probs) / len(char_probs)
                    if char_probs else 0.10
                )
                q = quality_score(plate_crop)
                tracker.buffer_crop(tid, plate_crop, q, ocr_conf, char_probs, frame_idx)
                tracker.update_vehicle_img(tid, vehicle_crop, q)
                active_tids.add(tid)
```

- [ ] **Step 4: Update `_run_multiframe_ocr` to use cached prob_lists**

In `api/core/pipeline_core.py`, replace the `_run_multiframe_ocr` function body:

```python
def _run_multiframe_ocr(
    tid: int,
    tracker: WebTrackletManager,
    models: ModelBundle,
    emit: Callable[[dict], None],
    session_id: str,
    loop: asyncio.AbstractEventLoop | None,
    record_save: Callable | None,
) -> None:
    crops, scores, cached_prob_lists = tracker._buffers[tid].top_k(k=TOP_K_FRAMES)
    if not crops:
        return

    vote_summary: dict[str, int] = {}

    if models.multiframe_ocr is not None:
        # Multiframe model has its own cross-frame attention — re-inference needed.
        tensors = torch.stack([preprocess_plate(c) for c in crops]).unsqueeze(0)
        quality = torch.tensor(scores, dtype=torch.float32).unsqueeze(0)
        char_probs = multiframe_ocr_infer(models.multiframe_ocr, tensors, quality, models.device)
        ocr_method = "multiframe"
    else:
        # Use cached per-frame OCR results stored at buffer time — no re-inference.
        prob_lists = cached_prob_lists
        for pl in prob_lists:
            text = "".join(c for c, _ in pl)
            if text:
                vote_summary[text] = vote_summary.get(text, 0) + 1
        char_probs = WebTrackletManager._segment_vote(prob_lists)
        if char_probs is not None:
            ocr_method = "segment_vote"
        else:
            char_probs = WebTrackletManager._prob_vote(prob_lists)
            ocr_method = "prob_vote"

    if not _plate_valid(char_probs):
        if crops and scores:
            best_idx = scores.index(max(scores))
            tracker.update_plate_img(tid, crops[best_idx], char_probs)

        rejected_plate = "".join(c for c, _ in char_probs)
        rejected_chars = [[c, round(p, 3)] for c, p in char_probs]
        emit({
            "type": "rejected_vehicle",
            "id": tid,
            "cls": tracker._cls.get(tid, ""),
            "plate": rejected_plate,
            "chars": rejected_chars,
            "plate_b64": tracker.plate_b64(tid),
            "vehicle_b64": tracker.vehicle_b64(tid),
            "ocr_frames": len(crops),
            "vote_summary": vote_summary,
        })
        return

    tracker.update(tid, char_probs, all_confident=True)
    tracker._done[tid] = True

    if crops and scores:
        best_idx = scores.index(max(scores))
        tracker.update_plate_img(tid, crops[best_idx], char_probs)

    if tracker.plate_changed(tid):
        emit({
            "type": "vehicle",
            "id": tid,
            "cls": tracker._cls.get(tid, ""),
            "plate": tracker.display_text(tid),
            "chars": tracker.chars_json(tid),
            "done": True,
            "plate_b64": tracker.plate_b64(tid),
            "vehicle_b64": tracker.vehicle_b64(tid),
            "ocr_frames": tracker.ocr_frames(tid),
        })

    if session_id and loop is not None and record_save is not None:
        record_save(session_id, tid, tracker, char_probs, ocr_method, vote_summary, loop)
```

- [ ] **Step 5: Re-capture the golden file and verify the parity test**

The parity test compares events against `tests/fixtures/golden_run_job_events.json`. Because the new eviction policy may produce different (improved) final plate text for some tracks in `short_clip.mp4`, the golden must be refreshed before the test can pass.

Re-capture the golden:

```bash
python tests/_capture_golden.py
```

Then run the parity test to confirm the new run is self-consistent:

```bash
pytest tests/test_pipeline_core_parity.py tests/test_track_buffer.py -v
```

Expected: `test_pipeline_core_parity.py::test_process_frames_matches_run_job_golden` PASSED (against the freshly captured golden) + all 6 TrackBuffer tests PASSED.

Inspect the diff between old and new golden to confirm any changes are plate-text improvements, not regressions:

```bash
git diff tests/fixtures/golden_run_job_events.json
```

- [ ] **Step 6: Run the diagnostic on the original video to confirm the fix**

```bash
python scripts/diag_ocr_frames.py data/realworld-videos/chunks/hn_oto_18.mp4 2>/dev/null | grep -A 20 "FINALISE track 6"
```

Expected output (Track 6 should now vote on correct crops):
```
── FINALISE track 6 ──
   Method: segment_vote
   Top-5 crops passed to vote (frames: [...]):
     [0] '30G-51827'  min_char_conf=0.xxx
     [1] '30G-51827'  min_char_conf=0.xxx
     ...
   Voted result: '30G-51827'  valid=True
```

- [ ] **Step 7: Commit**

```bash
git add api/core/pipeline_core.py tests/fixtures/golden_run_job_events.json
git commit -m "feat(pipeline): batch-OCR at buffer time; cache prob_lists for vote

Per-crop OCR inference is now run in a single batched call when crops are
buffered. The resulting char_probs are cached in TrackBuffer so that
_run_multiframe_ocr (non-multiframe path) can reuse them directly without
a second round of inference, reducing latency and ensuring the vote runs
on the same OCR outputs that determined eviction priority.

Updates golden_run_job_events.json: eviction policy change may produce
different (improved) plate text on short_clip.mp4 tracks.
"
```

---

## Task 4: Update the diagnostic script to new signatures

**Files:**
- Modify: `scripts/diag_ocr_frames.py`

- [ ] **Step 1: Fix the `buffer_crop` call in the diagnostic script**

In `scripts/diag_ocr_frames.py`, the loop that calls `tracker.buffer_crop` currently uses the old 4-argument signature. Replace it to match the new 6-argument signature.

Find the block:

```python
            # Run OCR on this single crop right now (diagnostic only)
            tensor = preprocess_plate(plate_crop).unsqueeze(0).to(models.device)
            results = ocr_batch(models.ocr, tensor, models.device)
            char_probs, all_conf = results[0]
            text = "".join(c for c, _ in char_probs)
            crop_logs[v_tid].add(frame_idx, q, text, char_probs, all_conf)
            gate_counts["buffered"] += 1
```

Replace with:

```python
            # Run per-frame OCR (same path as the fixed pipeline_core)
            tensor = preprocess_plate(plate_crop).unsqueeze(0).to(models.device)
            results = ocr_batch(models.ocr, tensor, models.device)
            char_probs, all_conf = results[0]
            ocr_conf = (
                sum(p for _, p in char_probs) / len(char_probs)
                if char_probs else 0.10
            )
            tracker.buffer_crop(v_tid, plate_crop, q, ocr_conf, char_probs, frame_idx)
            if v_box is not None:
                tracker.update_vehicle_img(v_tid, crop_vehicle(frame, v_box), q)
            text = "".join(c for c, _ in char_probs)
            crop_logs[v_tid].add(frame_idx, q, text, char_probs, all_conf)
            gate_counts["buffered"] += 1
```

Also update `top_k` calls in `_finalise()` — find:

```python
    crops, scores = tracker._buffers[tid].top_k(k=TOP_K_FRAMES)
```

Replace with:

```python
    crops, scores, prob_lists_cached = tracker._buffers[tid].top_k(k=TOP_K_FRAMES)
```

And remove the re-inference inside `_finalise()`:

```python
    tensors = torch.stack([preprocess_plate(c) for c in crops]).to(models.device)
    ocr_results = ocr_batch(models.ocr, tensors, models.device)
    prob_lists = [chars for chars, _ in ocr_results]
```

Replace with:

```python
    prob_lists = prob_lists_cached
```

- [ ] **Step 2: Run the diagnostic end-to-end and capture output**

```bash
python scripts/diag_ocr_frames.py data/realworld-videos/chunks/hn_oto_18.mp4 2>/dev/null \
  | tee /tmp/diag_after_fix.txt \
  | grep -E "FINALISE|Voted result|Method"
```

Expected: Track 6 voted result is `'30G-51827'  valid=True`.

- [ ] **Step 3: Commit**

```bash
git add scripts/diag_ocr_frames.py
git commit -m "fix(diag): update diag_ocr_frames to new buffer_crop/top_k signatures"
```

---

## Task 5: Run the full test suite and confirm coverage

- [ ] **Step 1: Run full test suite**

```bash
pytest tests/ -v --tb=short 2>&1 | tail -40
```

Expected: all existing tests pass; 6 new TrackBuffer tests pass.

- [ ] **Step 2: Check coverage on modified modules**

`api/core/tracker.py` is covered by the pure unit tests (no model weights or video needed).
`api/core/pipeline_core.py` is covered by the integration parity test, which requires real model weights in `weights/` and reads `tests/fixtures/short_clip.mp4`.

```bash
# tracker.py — pure unit tests, no external data
pytest tests/test_track_buffer.py \
  --cov=api/core/tracker \
  --cov-report=term-missing

# pipeline_core.py — integration test; needs weights/ + tests/fixtures/short_clip.mp4
pytest tests/test_pipeline_core_parity.py \
  --cov=api/core/pipeline_core \
  --cov-report=term-missing
```

Expected: `api/core/tracker` ≥ 80%; `api/core/pipeline_core` ≥ 80%.

- [ ] **Step 3: Final commit if needed**

If coverage was below 80% for either module, add targeted tests, then:

```bash
git add tests/
git commit -m "test(tracker): increase coverage to ≥80% on TrackBuffer and pipeline_core"
```

---

## Self-Review Checklist

**Spec coverage:**
- [x] Combined eviction score `quality × ocr_conf` — Task 2
- [x] `ocr_conf` + `char_probs` stored in TrackBuffer — Task 2
- [x] Per-frame OCR batched at buffer time in pipeline — Task 3, Step 3
- [x] Cached `prob_lists` reused at finalisation (no double inference) — Task 3, Step 4
- [x] Diagnostic script updated — Task 4
- [x] Tests for eviction logic — Task 1
- [x] Regression verification on `hn_oto_18.mp4` Track 6 — Task 3, Step 6

**Placeholder scan:** No TBD/TODO/placeholder found.

**Type consistency:**
- `TrackBuffer.add(crop, quality_score, ocr_conf, char_probs, frame_idx)` — matches all 3 call sites (Task 2, Task 3, Task 4)
- `TrackBuffer.top_k()` → `(list[np.ndarray], list[float], list[list[tuple[str, float]]])` — all 3 callers unpacked correctly (`_run_multiframe_ocr`, diagnostic `_finalise`, test assertions)
- `WebTrackletManager.buffer_crop(tid, crop, quality_score, ocr_conf, char_probs, frame_idx)` — matches the 1 call site in `pipeline_core.py` and the spy fixture in the integration test
