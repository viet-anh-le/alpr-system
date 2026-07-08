"""OCR-output clustering for multi-plate track disambiguation.

When a track buffer accumulates OCR results from 2-3 different licence plates
(e.g. after track ID reuse), this module separates them into independent
clusters so each cluster can be voted on separately.

Algorithm
---------
1. Build a text fingerprint for each buffered entry.
2. Compute pairwise Levenshtein-based similarity.
3. Greedy agglomerative clustering: sort by combined score desc, merge into
   the best-matching existing cluster if similarity >= threshold, else start
   a new cluster. Cap at ``max_clusters``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Sequence

logger = logging.getLogger(__name__)


# ── Data types ────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ClusterMember:
    """One buffered entry assigned to a cluster."""

    index: int  # position in the original prob_lists / entries list
    text: str  # fingerprint text
    combined_score: float


@dataclass
class OcrCluster:
    """A group of mutually-similar OCR results."""

    members: list[ClusterMember] = field(default_factory=list)
    centroid_index: int = 0  # index of the highest-scoring member

    @property
    def size(self) -> int:
        return len(self.members)

    @property
    def centroid_text(self) -> str:
        if self.members:
            return self.members[0].text
        return ""

    @property
    def total_score(self) -> float:
        return sum(m.combined_score for m in self.members)


# ── Fingerprint & similarity ──────────────────────────────────────────────────


def _text_fingerprint(char_probs: list[tuple[str, float]]) -> str:
    """Normalise a char_probs list to a plain uppercase string.

    Drops separator tokens and literal punctuation so that
    ``30A-12345`` and ``30A 12345`` compare as equal.
    """
    chars: list[str] = []
    for ch, _conf in char_probs:
        ch = ch.strip().upper()
        if not ch:
            continue
        if ch in ("-", ".", "[SEP]"):
            continue
        chars.append(ch)
    return "".join(chars)


def _levenshtein(a: str, b: str) -> int:
    """Classic DP Levenshtein distance."""
    n, m = len(a), len(b)
    if n == 0:
        return m
    if m == 0:
        return n
    prev = list(range(m + 1))
    curr = [0] * (m + 1)
    for i in range(1, n + 1):
        curr[0] = i
        ai = a[i - 1]
        for j in range(1, m + 1):
            cost = 0 if ai == b[j - 1] else 1
            curr[j] = min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + cost)
        prev, curr = curr, prev
    return prev[m]


def _similarity(a: str, b: str) -> float:
    """Normalised similarity in [0, 1] based on Levenshtein distance."""
    if not a and not b:
        return 1.0
    dist = _levenshtein(a, b)
    longer = max(len(a), len(b))
    if longer == 0:
        return 1.0
    return 1.0 - dist / longer


# ── Clustering ────────────────────────────────────────────────────────────────


def cluster_ocr_results(
    scored_entries: Sequence[tuple[list[tuple[str, float]], float]],
    *,
    max_clusters: int = 3,
    similarity_threshold: float = 0.6,
) -> list[OcrCluster]:
    """Partition scored OCR entries into similarity-based clusters.

    Parameters
    ----------
    scored_entries:
        Sequence of ``(char_probs, combined_score)`` tuples.  Only entries
        with non-empty *char_probs* should be passed in.
    max_clusters:
        Hard cap on the number of clusters produced (default 3).
    similarity_threshold:
        Minimum normalised Levenshtein similarity to merge an entry into an
        existing cluster (default 0.6).

    Returns
    -------
    list[OcrCluster]
        Clusters sorted by descending total score.  Each cluster's members
        are sorted by descending combined score.
    """
    if not scored_entries:
        return []

    # Build members sorted by combined_score descending
    members = [
        ClusterMember(
            index=i,
            text=_text_fingerprint(char_probs),
            combined_score=score,
        )
        for i, (char_probs, score) in enumerate(scored_entries)
    ]
    members.sort(key=lambda m: m.combined_score, reverse=True)

    clusters: list[OcrCluster] = []

    for member in members:
        best_cluster: OcrCluster | None = None
        best_sim = -1.0

        for cluster in clusters:
            sim = _similarity(member.text, cluster.centroid_text)
            if sim > best_sim:
                best_sim = sim
                best_cluster = cluster

        if best_cluster is not None and best_sim >= similarity_threshold:
            best_cluster.members.append(member)
            best_cluster.members.sort(key=lambda m: m.combined_score, reverse=True)
        elif len(clusters) < max_clusters:
            new_cluster = OcrCluster(members=[member], centroid_index=member.index)
            clusters.append(new_cluster)
        else:
            # At capacity and no good match → merge into best-matching cluster
            # (even if below threshold) to avoid dropping data.
            if best_cluster is not None:
                best_cluster.members.append(member)
                best_cluster.members.sort(key=lambda m: m.combined_score, reverse=True)
            else:
                # Should not happen (clusters is non-empty at this point), but
                # defensively create a new cluster anyway.
                new_cluster = OcrCluster(members=[member], centroid_index=member.index)
                clusters.append(new_cluster)

    # Sort clusters by total score descending
    clusters.sort(key=lambda c: c.total_score, reverse=True)
    # Re-index centroid_index after sort (members already sorted internally)
    for cluster in clusters:
        if cluster.members:
            cluster.centroid_index = cluster.members[0].index

    return clusters


def cluster_indices(
    scored_entries: Sequence[tuple[list[tuple[str, float]], float]],
    *,
    max_clusters: int = 3,
    similarity_threshold: float = 0.6,
) -> list[list[int]]:
    """Convenience wrapper: return list of index-lists per cluster.

    Each inner list contains the *original* indices (into *scored_entries*)
    of the members in that cluster, sorted by descending combined score.
    """
    clusters = cluster_ocr_results(
        scored_entries,
        max_clusters=max_clusters,
        similarity_threshold=similarity_threshold,
    )
    return [[m.index for m in cluster.members] for cluster in clusters]
