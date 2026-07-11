"""Unit tests for api/core/ocr_cluster.py — multi-cluster OCR disambiguation."""
from __future__ import annotations

import pytest

from api.core.ocr_cluster import (
    ClusterMember,
    OcrCluster,
    _levenshtein,
    _similarity,
    _text_fingerprint,
    cluster_indices,
    cluster_ocr_results,
)


# ── Helpers ───────────────────────────────────────────────────────────────────


def _probs(text: str, conf: float = 0.92) -> list[tuple[str, float]]:
    """Build a char_probs list from a plain string."""
    return [(c, conf) for c in text]


def _scored(text: str, score: float, conf: float = 0.92) -> tuple[list[tuple[str, float]], float]:
    return (_probs(text, conf), score)


# ── Fingerprint tests ─────────────────────────────────────────────────────────


class TestTextFingerprint:
    def test_simple_plate(self) -> None:
        assert _text_fingerprint(_probs("30A-12345")) == "30A12345"

    def test_strips_separator_token(self) -> None:
        assert _text_fingerprint([("3", 0.9), ("0", 0.9), ("[SEP]", 0.9), ("A", 0.9)]) == "30A"

    def test_strips_dots(self) -> None:
        assert _text_fingerprint(_probs("30.A.12345")) == "30A12345"

    def test_uppercases(self) -> None:
        assert _text_fingerprint(_probs("30a-12bcd")) == "30A12BCD"

    def test_empty_input(self) -> None:
        assert _text_fingerprint([]) == ""

    def test_only_literals(self) -> None:
        assert _text_fingerprint([("-", 0.9), (".", 0.9)]) == ""


# ── Similarity tests ──────────────────────────────────────────────────────────


class TestSimilarity:
    def test_identical_strings(self) -> None:
        assert _similarity("30A12345", "30A12345") == 1.0

    def test_completely_different(self) -> None:
        # "30A12345" vs "99Z98765": 8 chars, 7 substitutions → sim = 1/8 = 0.125
        sim = _similarity("30A12345", "99Z98765")
        assert abs(sim - 1 / 8) < 1e-9

    def test_one_char_diff(self) -> None:
        sim = _similarity("30A12345", "30B12345")
        assert abs(sim - 7 / 8) < 1e-9

    def test_empty_strings(self) -> None:
        assert _similarity("", "") == 1.0

    def test_one_empty(self) -> None:
        assert _similarity("30A12345", "") == 0.0
        assert _similarity("", "30A12345") == 0.0

    def test_prefix_diff(self) -> None:
        # "30A12345" vs "30A1234": 7 chars vs 8 chars, 1 deletion → sim = 7/8 = 0.875
        sim = _similarity("30A12345", "30A1234")
        assert abs(sim - 7 / 8) < 1e-9


# ── Levenshtein tests ─────────────────────────────────────────────────────────


class TestLevenshtein:
    def test_identical(self) -> None:
        assert _levenshtein("abc", "abc") == 0

    def test_empty(self) -> None:
        assert _levenshtein("", "abc") == 3
        assert _levenshtein("abc", "") == 3

    def test_single_substitution(self) -> None:
        assert _levenshtein("abc", "axc") == 1

    def test_insert_delete(self) -> None:
        assert _levenshtein("abc", "abcd") == 1
        assert _levenshtein("abcd", "abc") == 1


# ── Clustering tests ──────────────────────────────────────────────────────────


class TestClusterOcrResults:
    def test_empty_input_returns_empty(self) -> None:
        assert cluster_ocr_results([]) == []

    def test_single_entry_single_cluster(self) -> None:
        entries = [_scored("30A-12345", 0.9)]
        clusters = cluster_ocr_results(entries)
        assert len(clusters) == 1
        assert clusters[0].size == 1
        assert clusters[0].members[0].text == "30A12345"

    def test_identical_entries_single_cluster(self) -> None:
        """Same plate text with different scores → 1 cluster."""
        entries = [
            _scored("30A-12345", 0.9),
            _scored("30A-12345", 0.8),
            _scored("30A-12345", 0.7),
        ]
        clusters = cluster_ocr_results(entries)
        assert len(clusters) == 1
        assert clusters[0].size == 3

    def test_completely_different_entries_max_clusters(self) -> None:
        """Totally different plates → each gets its own cluster, capped at max_clusters."""
        entries = [
            _scored("30A-12345", 0.9),
            _scored("51B-67890", 0.85),
            _scored("29C-11111", 0.8),
            _scored("43D-22222", 0.75),  # 4th → merged into most similar
        ]
        clusters = cluster_ocr_results(entries, max_clusters=3)
        # 4 distinct plates, max 3 clusters → 3 clusters (4th merged into best match)
        assert len(clusters) == 3

    def test_two_distinct_plates_two_clusters(self) -> None:
        """Two clearly different plates → 2 clusters."""
        entries = [
            _scored("30A-12345", 0.9),
            _scored("30A-12345", 0.85),
            _scored("51B-67890", 0.8),
            _scored("51B-67890", 0.75),
        ]
        clusters = cluster_ocr_results(entries)
        assert len(clusters) == 2
        # First cluster (higher total score) should be "30A12345"
        assert clusters[0].centroid_text == "30A12345"
        assert clusters[0].size == 2
        assert clusters[1].centroid_text == "51B67890"
        assert clusters[1].size == 2

    def test_similar_but_different_plates_merge(self) -> None:
        """Plates that differ by 1 char should merge (similarity > 0.6)."""
        entries = [
            _scored("30A-12345", 0.9),
            _scored("30B-12345", 0.85),  # 1 char diff → sim = 7/8 = 0.875
        ]
        clusters = cluster_ocr_results(entries, similarity_threshold=0.6)
        assert len(clusters) == 1

    def test_threshold_controls_merge(self) -> None:
        """High threshold prevents merge of slightly different plates."""
        entries = [
            _scored("30A-12345", 0.9),
            _scored("30B-12345", 0.85),
        ]
        clusters_strict = cluster_ocr_results(entries, similarity_threshold=0.9)
        assert len(clusters_strict) == 2  # sim=0.875 < 0.9

        clusters_loose = cluster_ocr_results(entries, similarity_threshold=0.5)
        assert len(clusters_loose) == 1  # sim=0.875 > 0.5

    def test_clusters_sorted_by_total_score(self) -> None:
        """Clusters should be sorted by total score descending."""
        entries = [
            _scored("30A-12345", 0.9),
            _scored("30A-12345", 0.8),
            _scored("51B-67890", 0.95),  # higher individual score but fewer members
        ]
        clusters = cluster_ocr_results(entries)
        # Cluster 0: "30A12345" total = 1.7
        # Cluster 1: "51B67890" total = 0.95
        assert clusters[0].total_score >= clusters[1].total_score

    def test_members_sorted_by_score_within_cluster(self) -> None:
        """Within a cluster, members should be sorted by combined_score descending."""
        entries = [
            _scored("30A-12345", 0.5),
            _scored("30A-12345", 0.9),
            _scored("30A-12345", 0.7),
        ]
        clusters = cluster_ocr_results(entries)
        assert clusters[0].members[0].combined_score == 0.9
        assert clusters[0].members[1].combined_score == 0.7
        assert clusters[0].members[2].combined_score == 0.5

    def test_centroid_is_highest_scoring_member(self) -> None:
        """centroid_index should point to the highest-scoring member."""
        entries = [
            _scored("30A-12345", 0.5),
            _scored("30A-12345", 0.9),
        ]
        clusters = cluster_ocr_results(entries)
        assert clusters[0].centroid_index == 1  # index of the 0.9 score entry

    def test_three_distinct_plates(self) -> None:
        """Three clearly different plates → 3 clusters."""
        entries = [
            _scored("30A-12345", 0.9),
            _scored("51B-67890", 0.85),
            _scored("29C-11111", 0.8),
        ]
        clusters = cluster_ocr_results(entries, max_clusters=3)
        assert len(clusters) == 3

    def test_entries_without_char_probs_skipped(self) -> None:
        """Only entries with non-empty char_probs should be clustered."""
        entries = [
            (_probs("30A-12345"), 0.9),
            ([], 0.8),  # empty char_probs — fingerprint is "", forms its own cluster
            (_probs("30A-12345"), 0.7),
        ]
        clusters = cluster_ocr_results(entries)
        # Empty fingerprint "" is very different from "30A12345" (sim=0)
        # so it forms its own cluster. 2 clusters total.
        assert len(clusters) == 2
        # The non-empty cluster should have 2 members
        non_empty = [c for c in clusters if c.centroid_text == "30A12345"]
        assert len(non_empty) == 1
        assert non_empty[0].size == 2

    def test_mixed_quality_same_plate(self) -> None:
        """Same plate with varying OCR confidence → single cluster."""
        entries = [
            _scored("30A-12345", 0.95),
            _scored("30A-12345", 0.30),  # low quality OCR
            _scored("30A-12345", 0.60),
        ]
        clusters = cluster_ocr_results(entries)
        assert len(clusters) == 1

    def test_cluster_indices_wrapper(self) -> None:
        """cluster_indices should return list of index lists."""
        entries = [
            _scored("30A-12345", 0.9),
            _scored("30A-12345", 0.8),
            _scored("51B-67890", 0.85),
        ]
        indices = cluster_indices(entries)
        assert len(indices) == 2
        # First cluster (higher score) should have indices [0, 1]
        assert set(indices[0]) == {0, 1}
        assert indices[1] == [2]


# ── Integration: realistic multi-plate scenario ───────────────────────────────


class TestRealisticScenario:
    def test_track_id_reuse_scenario(self) -> None:
        """Simulate track ID reuse: frames from plate A then plate B."""
        # First 5 frames: plate "30A-12345" (varying quality)
        entries = [
            _scored("30A-12345", 0.95),
            _scored("30A-12345", 0.90),
            _scored("30A-12345", 0.85),
            _scored("30A-12345", 0.80),
            _scored("30A-12345", 0.75),
            # Then track ID reused for different vehicle: plate "51B-67890"
            _scored("51B-67890", 0.92),
            _scored("51B-67890", 0.88),
            _scored("51B-67890", 0.82),
        ]
        clusters = cluster_ocr_results(entries, max_clusters=3)
        assert len(clusters) == 2
        # First cluster should be "30A12345" (higher total score)
        assert clusters[0].centroid_text == "30A12345"
        assert clusters[0].size == 5
        assert clusters[1].centroid_text == "51B67890"
        assert clusters[1].size == 3

    def test_three_plate_reuse_scenario(self) -> None:
        """Track ID reused 3 times for different vehicles."""
        entries = (
            [_scored("30A-12345", 0.9 + i * 0.01) for i in range(4)]
            + [_scored("51B-67890", 0.85 + i * 0.01) for i in range(3)]
            + [_scored("29C-11111", 0.8 + i * 0.01) for i in range(2)]
        )
        clusters = cluster_ocr_results(entries, max_clusters=3)
        assert len(clusters) == 3
        sizes = [c.size for c in clusters]
        assert sizes == [4, 3, 2]

    def test_ocr_noise_same_plate(self) -> None:
        """OCR noise (1-2 char errors) on the same plate should still cluster together."""
        entries = [
            _scored("30A-12345", 0.9),
            _scored("30A-1234S", 0.85),  # last char wrong
            _scored("30A-12345", 0.8),
            _scored("30B-12345", 0.75),  # 1 char diff
        ]
        clusters = cluster_ocr_results(entries, similarity_threshold=0.7)
        # All should be in one cluster (similarity >= 7/8 = 0.875 for most pairs)
        assert len(clusters) == 1
