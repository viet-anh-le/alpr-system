"""Unit tests for api/core/tracker.py — WebTrackletManager and helpers."""
from __future__ import annotations

import numpy as np
import pytest

from api.core.config import (
    CONF_THRESHOLD,
    LOST_THRESHOLD,
    MIN_FRAME_VOTES,
    MIN_FRAMES_FOR_OCR,
)
from api.core.tracker import (
    TrackBuffer,
    WebTrackletManager,
    _PlateSegments,
    _parse_plate_segments,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _crop(h: int = 20, w: int = 94) -> np.ndarray:
    return np.zeros((h, w, 3), dtype=np.uint8)


def _probs(text: str, conf: float = 0.95) -> list[tuple[str, float]]:
    return [(c, conf) for c in text]


# ── _PlateSegments ────────────────────────────────────────────────────────────

class TestPlateSegments:
    def test_serial_start_fmt0(self):
        """fmt=0 (e.g. 51G-12345): serial begins at index 2."""
        seg = _PlateSegments(province="51", serial="G", number="12345", fmt=0)
        assert seg.serial_start == 2

    def test_serial_start_fmt1(self):
        """fmt=1 (e.g. 29-A1-12345): serial begins at index 3."""
        seg = _PlateSegments(province="29", serial="A1", number="12345", fmt=1)
        assert seg.serial_start == 3

    def test_serial_start_fmt2(self):
        """fmt=2 (e.g. 31H-9999): serial begins at index 2."""
        seg = _PlateSegments(province="31", serial="H", number="9999", fmt=2)
        assert seg.serial_start == 2

    def test_serial_start_fmt3(self):
        """fmt=3 (e.g. 29-F4-8888): serial begins at index 3."""
        seg = _PlateSegments(province="29", serial="F4", number="8888", fmt=3)
        assert seg.serial_start == 3

    def test_number_start_fmt0(self):
        """fmt=0 e.g. '51G-12345': number at 2 + 1(serial) + 1(hyphen) = 4."""
        seg = _PlateSegments(province="51", serial="G", number="12345", fmt=0)
        assert seg.number_start == 4

    def test_number_start_fmt1(self):
        """fmt=1 e.g. '29-A1-12345': number at 3 + 2(serial) + 1(hyphen) = 6."""
        seg = _PlateSegments(province="29", serial="A1", number="12345", fmt=1)
        assert seg.number_start == 6

    def test_number_start_two_char_serial_fmt0(self):
        """fmt=0 two-char serial e.g. '50LD-12345': number at 2 + 2 + 1 = 5."""
        seg = _PlateSegments(province="50", serial="LD", number="12345", fmt=0)
        assert seg.number_start == 5


# ── _parse_plate_segments ─────────────────────────────────────────────────────

class TestParsePlateSegments:
    def test_fmt0_standard(self):
        seg = _parse_plate_segments("51G-12345")
        assert seg is not None
        assert seg.province == "51"
        assert seg.serial == "G"
        assert seg.number == "12345"
        assert seg.fmt == 0

    def test_fmt0_two_char_serial(self):
        seg = _parse_plate_segments("50LD-12345")
        assert seg is not None
        assert seg.serial == "LD"
        assert seg.fmt == 0

    def test_fmt1_two_segment(self):
        seg = _parse_plate_segments("29-A1-12345")
        assert seg is not None
        assert seg.province == "29"
        assert seg.serial == "A1"
        assert seg.fmt == 1

    def test_fmt2_four_digit_number(self):
        seg = _parse_plate_segments("31H-9999")
        assert seg is not None
        assert seg.number == "9999"
        assert seg.fmt == 2

    def test_fmt3_four_digit_number(self):
        seg = _parse_plate_segments("29-F4-8888")
        assert seg is not None
        assert seg.fmt == 3

    def test_valid_30g(self):
        seg = _parse_plate_segments("30G-51827")
        assert seg is not None
        assert seg.province == "30"
        assert seg.serial == "G"
        assert seg.number == "51827"

    def test_invalid_returns_none(self):
        assert _parse_plate_segments("INVALID") is None

    def test_empty_string_returns_none(self):
        assert _parse_plate_segments("") is None

    def test_partial_plate_returns_none(self):
        assert _parse_plate_segments("51G-123") is None


# ── WebTrackletManager — init ─────────────────────────────────────────────────

class TestWebTrackletManagerInit:
    def test_init_empty_dicts(self):
        mgr = WebTrackletManager()
        assert mgr._done == {}
        assert mgr._best == {}
        assert mgr._buffers == {}
        assert mgr._lost_count == {}
        assert mgr._ocr_count == {}


# ── should_ocr ────────────────────────────────────────────────────────────────

class TestShouldOcr:
    def test_unknown_tid_returns_true(self):
        mgr = WebTrackletManager()
        assert mgr.should_ocr(99) is True

    def test_done_tid_returns_false(self):
        mgr = WebTrackletManager()
        mgr._done[1] = True
        assert mgr.should_ocr(1) is False

    def test_undone_tid_returns_true(self):
        mgr = WebTrackletManager()
        mgr._done[1] = False
        assert mgr.should_ocr(1) is True


# ── update ────────────────────────────────────────────────────────────────────

class TestUpdate:
    def test_increments_ocr_count(self):
        mgr = WebTrackletManager()
        mgr.update(1, [("3", 0.95), ("0", 0.95)], all_confident=False)
        assert mgr._ocr_count[1] == 1
        mgr.update(1, [("3", 0.95), ("0", 0.95)], all_confident=False)
        assert mgr._ocr_count[1] == 2

    def test_sets_best_on_first_call(self):
        mgr = WebTrackletManager()
        probs = [("A", 0.95), ("B", 0.92)]
        mgr.update(1, probs, all_confident=False)
        assert mgr._best[1] == probs

    def test_empty_char_probs_noop(self):
        mgr = WebTrackletManager()
        mgr.update(1, [], all_confident=False)
        assert 1 not in mgr._ocr_count
        assert 1 not in mgr._best

    def test_marks_done_when_all_confident_and_enough_frames(self):
        mgr = WebTrackletManager()
        probs = [("A", 0.95)] * 5
        # Call MIN_FRAME_VOTES times with all_confident=True
        for _ in range(MIN_FRAME_VOTES):
            mgr.update(1, probs, all_confident=True)
        assert mgr._done.get(1) is True

    def test_not_done_before_min_frame_votes(self):
        mgr = WebTrackletManager()
        probs = [("A", 0.95)] * 5
        # Call fewer times than required
        for _ in range(MIN_FRAME_VOTES - 1):
            mgr.update(1, probs, all_confident=True)
        assert mgr._done.get(1) is not True

    def test_marks_done_when_chars_fully_confident(self):
        mgr = WebTrackletManager()
        # All probs above CONF_THRESHOLD
        probs = [("A", CONF_THRESHOLD + 0.01)] * 3
        for _ in range(MIN_FRAME_VOTES):
            mgr.update(1, probs, all_confident=False)
        assert mgr._done.get(1) is True


# ── update_plate_img ──────────────────────────────────────────────────────────

class TestUpdatePlateImg:
    def test_stores_first_crop(self):
        mgr = WebTrackletManager()
        crop = _crop()
        mgr.update_plate_img(1, crop, [("A", 0.9)])
        assert 1 in mgr._plate_img

    def test_replaces_with_higher_confidence(self):
        mgr = WebTrackletManager()
        mgr.update_plate_img(1, _crop(), [("A", 0.7)])
        mgr.update_plate_img(1, _crop(), [("A", 0.95)])
        assert abs(mgr._plate_img_conf[1] - 0.95) < 1e-6

    def test_ignores_lower_confidence(self):
        mgr = WebTrackletManager()
        mgr.update_plate_img(1, _crop(), [("A", 0.9)])
        mgr.update_plate_img(1, _crop(), [("A", 0.5)])
        assert abs(mgr._plate_img_conf[1] - 0.9) < 1e-6

    def test_empty_char_probs_noop(self):
        mgr = WebTrackletManager()
        mgr.update_plate_img(1, _crop(), [])
        assert 1 not in mgr._plate_img


# ── update_vehicle_img ────────────────────────────────────────────────────────

class TestUpdateVehicleImg:
    def test_stores_first_crop(self):
        mgr = WebTrackletManager()
        mgr.update_vehicle_img(1, _crop(), 0.8)
        assert 1 in mgr._vehicle_img

    def test_replaces_with_higher_confidence(self):
        mgr = WebTrackletManager()
        mgr.update_vehicle_img(1, _crop(), 0.7)
        mgr.update_vehicle_img(1, _crop(), 0.95)
        assert abs(mgr._vehicle_img_conf[1] - 0.95) < 1e-6

    def test_ignores_lower_confidence(self):
        mgr = WebTrackletManager()
        mgr.update_vehicle_img(1, _crop(), 0.9)
        mgr.update_vehicle_img(1, _crop(), 0.5)
        assert abs(mgr._vehicle_img_conf[1] - 0.9) < 1e-6

    def test_ignores_empty_crop(self):
        mgr = WebTrackletManager()
        empty = np.zeros((0, 0, 3), dtype=np.uint8)
        mgr.update_vehicle_img(1, empty, 1.0)
        assert 1 not in mgr._vehicle_img


# ── buffer_crop ───────────────────────────────────────────────────────────────

class TestBufferCrop:
    def test_creates_buffer_on_first_call(self):
        mgr = WebTrackletManager()
        mgr.buffer_crop(1, _crop(), 0.9, 0.93, _probs("30G"), 0)
        assert 1 in mgr._buffers
        assert isinstance(mgr._buffers[1], TrackBuffer)

    def test_second_call_appends_to_same_buffer(self):
        mgr = WebTrackletManager()
        mgr.buffer_crop(1, _crop(), 0.9, 0.93, _probs("30G"), 0)
        mgr.buffer_crop(1, _crop(), 0.8, 0.91, _probs("30G"), 1)
        assert len(mgr._buffers[1].crops) == 2

    def test_different_tids_get_separate_buffers(self):
        mgr = WebTrackletManager()
        mgr.buffer_crop(1, _crop(), 0.9, 0.93, _probs("30G"), 0)
        mgr.buffer_crop(2, _crop(), 0.85, 0.91, _probs("51G"), 0)
        assert 1 in mgr._buffers
        assert 2 in mgr._buffers
        assert len(mgr._buffers[1].crops) == 1
        assert len(mgr._buffers[2].crops) == 1


# ── mark_lost and reset_lost ──────────────────────────────────────────────────

class TestMarkLostResetLost:
    def test_returns_false_before_threshold(self):
        mgr = WebTrackletManager()
        for _ in range(LOST_THRESHOLD - 1):
            result = mgr.mark_lost(1)
        assert result is False

    def test_returns_true_at_threshold(self):
        mgr = WebTrackletManager()
        for _ in range(LOST_THRESHOLD):
            result = mgr.mark_lost(1)
        assert result is True

    def test_reset_lost_clears_counter(self):
        mgr = WebTrackletManager()
        for _ in range(LOST_THRESHOLD - 1):
            mgr.mark_lost(1)
        mgr.reset_lost(1)
        # After reset the counter is gone — next call starts from 1
        assert not mgr.mark_lost(1) or LOST_THRESHOLD == 1

    def test_reset_lost_on_unknown_tid_noop(self):
        mgr = WebTrackletManager()
        mgr.reset_lost(999)  # should not raise

    def test_lost_count_increments(self):
        mgr = WebTrackletManager()
        mgr.mark_lost(1)
        mgr.mark_lost(1)
        assert mgr._lost_count[1] == 2


# ── ready_for_track_ocr ───────────────────────────────────────────────────────

class TestReadyForTrackOcr:
    def test_returns_false_when_no_buffer(self):
        mgr = WebTrackletManager()
        assert mgr.ready_for_track_ocr(1) is False

    def test_returns_false_when_below_min_frames(self):
        mgr = WebTrackletManager()
        for i in range(MIN_FRAMES_FOR_OCR - 1):
            mgr.buffer_crop(1, _crop(), 0.9, 0.93, _probs("30G"), i)
        assert mgr.ready_for_track_ocr(1) is False

    def test_returns_true_when_at_min_frames(self):
        mgr = WebTrackletManager()
        for i in range(MIN_FRAMES_FOR_OCR):
            mgr.buffer_crop(1, _crop(), 0.9, 0.93, _probs("30G"), i)
        assert mgr.ready_for_track_ocr(1) is True

    def test_returns_true_when_above_min_frames(self):
        mgr = WebTrackletManager()
        for i in range(MIN_FRAMES_FOR_OCR + 2):
            mgr.buffer_crop(1, _crop(), 0.9, 0.93, _probs("30G"), i)
        assert mgr.ready_for_track_ocr(1) is True

    def test_candidate_variants_from_one_frame_do_not_count_as_multiple_frames(self):
        mgr = WebTrackletManager()
        for method in ("original", "clahe", "sharpen"):
            mgr.buffer_crop(
                1,
                _crop(),
                0.9,
                0.93,
                _probs("30G"),
                42,
                candidate_method=method,
            )
        assert mgr.ready_for_track_ocr(1) is False


# ── Accessors ─────────────────────────────────────────────────────────────────

class TestAccessors:
    def test_display_text_empty_when_no_best(self):
        mgr = WebTrackletManager()
        assert mgr.display_text(99) == ""

    def test_display_text_returns_plate_string(self):
        mgr = WebTrackletManager()
        mgr._best[1] = [("3", 0.95), ("0", 0.93), ("G", 0.91)]
        assert mgr.display_text(1) == "30G"

    def test_display_text_renders_sep_as_space(self):
        mgr = WebTrackletManager()
        mgr._best[1] = [
            ("7", 0.95),
            ("1", 0.95),
            ("-", 0.95),
            ("C", 0.95),
            ("1", 0.95),
            ("[SEP]", 0.95),
            ("5", 0.95),
            ("1", 0.95),
            ("3", 0.95),
            ("1", 0.95),
        ]

        assert mgr.display_text(1) == "71-C1 5131"
        assert mgr.chars_json(1)[5][0] == "[SEP]"

    def test_chars_json_returns_list(self):
        mgr = WebTrackletManager()
        mgr._best[1] = [("3", 0.95123), ("0", 0.93)]
        result = mgr.chars_json(1)
        assert result == [["3", 0.951], ["0", 0.93]]

    def test_ocr_frames_returns_zero_for_unknown(self):
        mgr = WebTrackletManager()
        assert mgr.ocr_frames(99) == 0

    def test_ocr_frames_increments_with_update(self):
        mgr = WebTrackletManager()
        mgr.update(1, [("A", 0.95)], all_confident=False)
        mgr.update(1, [("A", 0.95)], all_confident=False)
        assert mgr.ocr_frames(1) == 2

    def test_plate_changed_first_call_returns_true(self):
        mgr = WebTrackletManager()
        mgr._best[1] = [("3", 0.95)]
        assert mgr.plate_changed(1) is True

    def test_plate_changed_second_call_returns_false(self):
        mgr = WebTrackletManager()
        mgr._best[1] = [("3", 0.95)]
        mgr.plate_changed(1)
        assert mgr.plate_changed(1) is False

    def test_plate_changed_true_when_text_changes(self):
        mgr = WebTrackletManager()
        mgr._best[1] = [("3", 0.95)]
        mgr.plate_changed(1)
        mgr._best[1] = [("4", 0.95)]
        assert mgr.plate_changed(1) is True


# ── _encode / plate_b64 / vehicle_b64 ────────────────────────────────────────

class TestEncode:
    def test_plate_b64_returns_none_for_missing_tid(self):
        mgr = WebTrackletManager()
        assert mgr.plate_b64(99) is None

    def test_vehicle_b64_returns_none_for_missing_tid(self):
        mgr = WebTrackletManager()
        assert mgr.vehicle_b64(99) is None

    def test_plate_b64_returns_nonempty_string_for_stored_image(self):
        mgr = WebTrackletManager()
        img = np.zeros((20, 94, 3), dtype=np.uint8)
        mgr._plate_img[1] = img
        result = mgr.plate_b64(1)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_vehicle_b64_returns_nonempty_string_for_stored_image(self):
        mgr = WebTrackletManager()
        img = np.zeros((60, 120, 3), dtype=np.uint8)
        mgr._vehicle_img[1] = img
        result = mgr.vehicle_b64(1)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_track_buffer_json_returns_buffered_plate_frames(self):
        mgr = WebTrackletManager()
        img = np.zeros((20, 94, 3), dtype=np.uint8)
        mgr.buffer_crop(
            1,
            img,
            quality_score=0.82,
            ocr_conf=0.76,
            char_probs=[("3", 0.95), ("0", 0.92)],
            frame_idx=12,
            candidate_method="original",
            route="direct",
            router_result={"quality_bin": "good"},
        )

        frames = mgr.track_buffer_json(1)

        assert len(frames) == 1
        assert frames[0]["frame_index"] == 12
        assert frames[0]["quality_score"] == 0.82
        assert frames[0]["ocr_confidence"] == 0.76
        assert frames[0]["candidate_method"] == "original"
        assert frames[0]["route"] == "direct"
        assert frames[0]["quality_bin"] == "good"
        assert frames[0]["image_b64"]

    def test_encode_none_returns_none(self):
        result = WebTrackletManager._encode(None, max_w=None, quality=90)
        assert result is None

    def test_encode_resizes_wide_image(self):
        # Image wider than max_w should be resized
        img = np.zeros((100, 400, 3), dtype=np.uint8)
        result = WebTrackletManager._encode(img, max_w=320, quality=85)
        assert isinstance(result, str)
        assert len(result) > 0


# ── _prob_vote ────────────────────────────────────────────────────────────────

class TestProbVote:
    def test_single_list_returns_same(self):
        probs = [("3", 0.9), ("0", 0.85)]
        result = WebTrackletManager._prob_vote([probs])
        assert len(result) == 2
        assert result[0][0] == "3"
        assert result[1][0] == "0"

    def test_picks_char_with_highest_total_confidence(self):
        # Position 0: "3" appears twice with 0.9, "6" once with 0.8
        # "3" total = 1.8, "6" total = 0.8 → pick "3"
        prob_lists = [
            [("3", 0.9), ("0", 0.85)],
            [("3", 0.9), ("0", 0.85)],
            [("6", 0.8), ("0", 0.85)],
        ]
        result = WebTrackletManager._prob_vote(prob_lists)
        assert result[0][0] == "3"

    def test_empty_list_returns_empty(self):
        result = WebTrackletManager._prob_vote([])
        assert result == []

    def test_handles_variable_length_sequences(self):
        prob_lists = [
            [("A", 0.9), ("B", 0.9), ("C", 0.9)],
            [("A", 0.9)],
        ]
        result = WebTrackletManager._prob_vote(prob_lists)
        # Position 0: both have "A" — should stay "A"
        assert result[0][0] == "A"
        # Positions 1 and 2: only first list contributes
        assert len(result) == 3

    def test_confidence_is_ensemble_averaged(self):
        prob_lists = [
            [("3", 0.9)],
            [("3", 0.8)],
        ]
        result = WebTrackletManager._prob_vote(prob_lists)
        # best_conf = (0.9 + 0.8) / 2 frames = 0.85
        assert abs(result[0][1] - 0.85) < 1e-6


# ── _segment_vote ─────────────────────────────────────────────────────────────

class TestSegmentVote:
    def _make_prob_lists_for_plate(self, plate: str, conf: float = 0.95) -> list[list[tuple[str, float]]]:
        return [_probs(plate, conf) for _ in range(3)]

    def test_valid_plate_returns_result(self):
        prob_lists = self._make_prob_lists_for_plate("30G-51827")
        result = WebTrackletManager._segment_vote(prob_lists)
        assert result is not None
        plate_text = "".join(c for c, _ in result)
        assert plate_text == "30G-51827"

    def test_invalid_plate_returns_none(self):
        # All sequences are non-matching garbage
        prob_lists = [_probs("XXXXXXXXX") for _ in range(3)]
        result = WebTrackletManager._segment_vote(prob_lists)
        assert result is None

    def test_empty_list_returns_none(self):
        result = WebTrackletManager._segment_vote([])
        assert result is None

    def test_fmt1_plate_with_leading_hyphen(self):
        """fmt=1 plates like 29-A1-12345 use leading hyphen."""
        prob_lists = self._make_prob_lists_for_plate("29-A1-12345")
        result = WebTrackletManager._segment_vote(prob_lists)
        assert result is not None
        plate_text = "".join(c for c, _ in result)
        assert plate_text == "29-A1-12345"


# ── _fuse ─────────────────────────────────────────────────────────────────────

class TestFuse:
    def test_identical_sequences_keeps_higher_conf(self):
        mgr = WebTrackletManager()
        seq1 = [("A", 0.9), ("B", 0.8)]
        seq2 = [("A", 0.7), ("B", 0.95)]
        result = mgr._fuse(seq1, seq2)
        # For "A": 0.9 > 0.7 → keep seq1's "A"
        # For "B": 0.95 > 0.8 → keep seq2's "B"
        assert result[0] == ("A", 0.9)
        assert result[1] == ("B", 0.95)

    def test_handles_substitution(self):
        mgr = WebTrackletManager()
        seq1 = [("A", 0.9)]
        seq2 = [("B", 0.95)]
        result = mgr._fuse(seq1, seq2)
        # Substitution: cost=1, but both chars are retained (substituted)
        assert len(result) == 1
        # Higher conf wins: seq2's B at 0.95
        assert result[0] == ("B", 0.95)

    def test_empty_seq1_returns_seq2_above_threshold(self):
        mgr = WebTrackletManager()
        seq2 = [("A", CONF_THRESHOLD + 0.01)]
        result = mgr._fuse([], seq2)
        assert len(result) == 1
        assert result[0][0] == "A"

    def test_empty_seq2_returns_seq1(self):
        mgr = WebTrackletManager()
        seq1 = [("A", 0.9)]
        result = mgr._fuse(seq1, [])
        assert result == seq1

    def test_seq2_below_threshold_not_added_on_insertion(self):
        mgr = WebTrackletManager()
        # seq2 char below CONF_THRESHOLD: should not appear in output if inserted
        seq2_char = ("Z", CONF_THRESHOLD - 0.05)
        result = mgr._fuse([], [seq2_char])
        # Since it's below threshold it should not be added
        assert all(p >= CONF_THRESHOLD for _, p in result) or result == []
