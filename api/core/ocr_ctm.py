"""OCR-output Character Time-series Matching (CTM) fusion.

This module aligns decoded OCR strings to Vietnamese plate slots before voting.
It intentionally leaves unresolved slots as ``?`` when no character has strict
majority support.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from .plate_format import (
    VN_PLATE_TEMPLATE_PATTERNS,
    chars_to_text,
    is_vn_plate_text,
    normalize_plate_text,
)

SLOT_CLASSES = {"D", "L", "A"}
LITERAL_TOKENS = {"-", ".", "[SEP]"}
LITERAL_CONFIDENCE = 0.90


def _tokenize_template_pattern(pattern: str) -> list[str]:
    tokens: list[str] = []
    i = 0
    while i < len(pattern):
        if pattern.startswith("[SEP]", i):
            tokens.append("[SEP]")
            i += len("[SEP]")
        else:
            tokens.append(pattern[i])
            i += 1
    return tokens


def _template_name(pattern: str) -> str:
    return (
        pattern.lower()
        .replace("[sep]", "_sep_")
        .replace("-", "_dash_")
        .replace(".", "_dot_")
        .strip("_")
    )


@dataclass(frozen=True)
class PlateTemplate:
    name: str
    pattern: tuple[str, ...]

    @classmethod
    def from_pattern(cls, pattern: str) -> "PlateTemplate":
        return cls(name=_template_name(pattern), pattern=tuple(_tokenize_template_pattern(pattern)))

    @property
    def slots(self) -> tuple[str, ...]:
        return tuple(token for token in self.pattern if token in SLOT_CLASSES)

    def render(self, slot_chars: list[tuple[str, float]]) -> list[tuple[str, float]]:
        rendered: list[tuple[str, float]] = []
        slot_idx = 0
        for token in self.pattern:
            if token in SLOT_CLASSES:
                rendered.append(slot_chars[slot_idx])
                slot_idx += 1
            else:
                rendered.append((token, LITERAL_CONFIDENCE))
        return rendered

    @property
    def hyphen_positions(self) -> tuple[int, ...]:
        return tuple(i for i, token in enumerate(self.pattern) if token == "-")

    @property
    def literal_positions(self) -> tuple[int, ...]:
        return tuple(i for i, token in enumerate(self.pattern) if token not in SLOT_CLASSES)


OCR_DATASET_TEMPLATE_PATTERNS: tuple[str, ...] = VN_PLATE_TEMPLATE_PATTERNS

TEMPLATES: tuple[PlateTemplate, ...] = tuple(
    PlateTemplate.from_pattern(pattern) for pattern in OCR_DATASET_TEMPLATE_PATTERNS
)


@dataclass(frozen=True)
class CTMFusionResult:
    char_probs: list[tuple[str, float]]
    vote_summary: dict[str, int]
    unresolved_slots: list[int]
    ctm_support: list[float]
    support_by_slot: list[dict[str, int]]
    template_name: str | None

    @property
    def text(self) -> str:
        return chars_to_text(self.char_probs)

    @property
    def is_valid(self) -> bool:
        return not self.unresolved_slots and is_vn_plate_text(self.text)


def fuse_ocr_outputs_ctm(
    prob_lists: list[list[tuple[str, float]]],
    *,
    min_support_ratio: float = 0.5,
    min_confidence: float = 0.50,
) -> CTMFusionResult:
    vote_summary: dict[str, int] = {}
    frame_choices: list[tuple[PlateTemplate, list[tuple[str, float]], float]] = []

    for probs in prob_lists:
        text = normalize_plate_text(chars_to_text(probs))
        if text:
            vote_summary[text] = vote_summary.get(text, 0) + 1
        tokens = _plate_token_probs(probs)
        chars = _slot_token_probs(tokens)
        if not chars and not tokens:
            continue
        template, score = _choose_template(tokens)
        frame_choices.append((template, tokens, score))

    if not frame_choices:
        return CTMFusionResult([], vote_summary, [], [], [], None)

    template = _dominant_template(frame_choices)
    aligned_frames = [_align_chars_to_template(chars, template) for _, chars, _ in frame_choices]

    slot_votes: list[dict[str, list[float]]] = [dict() for _ in template.slots]
    valid_frames_by_slot = [0 for _ in template.slots]
    for aligned in aligned_frames:
        for slot_idx, char_prob in enumerate(aligned):
            if char_prob is None:
                continue
            char, conf = char_prob
            valid_frames_by_slot[slot_idx] += 1
            if char not in slot_votes[slot_idx]:
                slot_votes[slot_idx][char] = [0.0, 0.0]
            slot_votes[slot_idx][char][0] += 1.0
            slot_votes[slot_idx][char][1] += float(conf)

    slot_chars: list[tuple[str, float]] = []
    unresolved_slots: list[int] = []
    ctm_support: list[float] = []
    support_by_slot: list[dict[str, int]] = []

    for slot_idx, votes in enumerate(slot_votes):
        support_counts = {char: int(vals[0]) for char, vals in votes.items()}
        support_by_slot.append(support_counts)
        valid_frames = valid_frames_by_slot[slot_idx]
        if not votes or valid_frames == 0:
            unresolved_slots.append(slot_idx)
            ctm_support.append(0.0)
            slot_chars.append(("?", 0.0))
            continue

        best_char, (count, conf_sum) = max(
            votes.items(),
            key=lambda item: (item[1][0], item[1][1]),
        )
        support = count / valid_frames
        weighted_conf = conf_sum / valid_frames
        ctm_support.append(round(float(support), 4))
        if support > min_support_ratio and weighted_conf >= min_confidence:
            slot_chars.append((best_char, round(float(weighted_conf), 4)))
        else:
            unresolved_slots.append(slot_idx)
            slot_chars.append(("?", 0.0))

    return CTMFusionResult(
        char_probs=template.render(slot_chars),
        vote_summary=vote_summary,
        unresolved_slots=unresolved_slots,
        ctm_support=ctm_support,
        support_by_slot=support_by_slot,
        template_name=template.name,
    )


def _dominant_template(
    frame_choices: list[tuple[PlateTemplate, list[tuple[str, float]], float]],
) -> PlateTemplate:
    counts = Counter(template.name for template, _, _ in frame_choices)
    score_by_name: dict[str, float] = {}
    by_name: dict[str, PlateTemplate] = {}
    for template, _, score in frame_choices:
        by_name[template.name] = template
        score_by_name[template.name] = score_by_name.get(template.name, 0.0) + score
    name = max(counts, key=lambda key: (counts[key], score_by_name.get(key, 0.0)))
    return by_name[name]


def _choose_template(
    tokens: list[tuple[str, float]],
) -> tuple[PlateTemplate, float]:
    scored = [(_template_score(tokens, template), template) for template in TEMPLATES]
    score, template = max(scored, key=lambda item: item[0])
    return template, score


def _template_score(
    tokens: list[tuple[str, float]],
    template: PlateTemplate,
) -> float:
    cost = _template_alignment_cost(tokens, template)
    slot_count = sum(1 for token, _ in tokens if _is_slot_token(token))
    literal_count = sum(1 for token, _ in tokens if token in LITERAL_TOKENS)
    score = -cost - 2.0 * abs(slot_count - len(template.slots))
    if literal_count == 0:
        score -= 0.2 * len(template.literal_positions)
    return score


def _align_chars_to_template(
    chars: list[tuple[str, float]],
    template: PlateTemplate,
) -> list[tuple[str, float] | None]:
    n = len(template.pattern)
    m = len(chars)

    dp = [[0.0] * (m + 1) for _ in range(n + 1)]
    back: list[list[str]] = [[""] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        dp[i][0] = dp[i - 1][0] + _missing_template_token_cost(template.pattern[i - 1])
        back[i][0] = "missing"
    for j in range(1, m + 1):
        dp[0][j] = dp[0][j - 1] + _skip_input_token_cost(chars[j - 1][0])
        back[0][j] = "skip"

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            char, _ = chars[j - 1]
            pattern_token = template.pattern[i - 1]
            align_cost = _align_token_cost(char, pattern_token)
            options = [
                (dp[i - 1][j - 1] + align_cost, "align"),
                (dp[i - 1][j] + _missing_template_token_cost(pattern_token), "missing"),
                (dp[i][j - 1] + _skip_input_token_cost(char), "skip"),
            ]
            dp[i][j], back[i][j] = min(options, key=lambda item: item[0])

    aligned_pattern: list[tuple[str, float] | None] = [None for _ in range(n)]
    i, j = n, m
    while i > 0 or j > 0:
        action = back[i][j]
        if action == "align":
            char, conf = chars[j - 1]
            aligned_pattern[i - 1] = (_normalize_ocr_token(char), conf)
            i -= 1
            j -= 1
        elif action == "missing":
            i -= 1
        else:
            j -= 1

    aligned_slots: list[tuple[str, float] | None] = []
    for pattern_token, char_prob in zip(template.pattern, aligned_pattern):
        if pattern_token not in SLOT_CLASSES:
            continue
        if char_prob is None:
            aligned_slots.append(None)
            continue
        char, conf = char_prob
        aligned_slots.append((char, conf) if _matches_slot_class(char, pattern_token) else None)
    return aligned_slots


def _plate_token_probs(probs: list[tuple[str, float]]) -> list[tuple[str, float]]:
    chars: list[tuple[str, float]] = []
    for char, conf in probs:
        normalized = _normalize_ocr_token(char)
        if not normalized:
            continue
        for token in _tokenize_ocr_text(normalized):
            if _is_slot_token(token) or token in LITERAL_TOKENS:
                chars.append((token, float(conf)))
    return chars


def _slot_token_probs(probs: list[tuple[str, float]]) -> list[tuple[str, float]]:
    return [(char, conf) for char, conf in probs if _is_slot_token(char)]


def _matches_slot_class(char: str, slot_class: str) -> bool:
    if slot_class == "D":
        return char.isdigit()
    if slot_class == "L":
        return char.isalpha()
    return char.isalnum() or char == "Đ"


def _template_alignment_cost(tokens: list[tuple[str, float]], template: PlateTemplate) -> float:
    n = len(template.pattern)
    m = len(tokens)
    dp = [[0.0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        dp[i][0] = dp[i - 1][0] + _missing_template_token_cost(template.pattern[i - 1])
    for j in range(1, m + 1):
        dp[0][j] = dp[0][j - 1] + _skip_input_token_cost(tokens[j - 1][0])
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            token = tokens[j - 1][0]
            pattern_token = template.pattern[i - 1]
            dp[i][j] = min(
                dp[i - 1][j - 1] + _align_token_cost(token, pattern_token),
                dp[i - 1][j] + _missing_template_token_cost(pattern_token),
                dp[i][j - 1] + _skip_input_token_cost(token),
            )
    return dp[n][m]


def _align_token_cost(token: str, pattern_token: str) -> float:
    if pattern_token in SLOT_CLASSES:
        if _matches_slot_class(token, pattern_token):
            return 0.0 if pattern_token != "A" else 0.1
        if _is_slot_token(token):
            return 0.45
        return 2.0
    return 0.0 if token == pattern_token else 2.0


def _missing_template_token_cost(pattern_token: str) -> float:
    if pattern_token in SLOT_CLASSES:
        return 1.2
    if pattern_token == "[SEP]":
        return 1.0
    return 0.75


def _skip_input_token_cost(token: str) -> float:
    return 1.0 if _is_slot_token(token) else 0.65


def _is_slot_token(token: str) -> bool:
    return len(token) == 1 and (token.isalnum() or token == "Đ")


def _normalize_ocr_token(token: str) -> str:
    return token.strip().upper()


def _tokenize_ocr_text(text: str) -> list[str]:
    tokens: list[str] = []
    i = 0
    while i < len(text):
        if text.startswith("[SEP]", i):
            tokens.append("[SEP]")
            i += len("[SEP]")
        else:
            tokens.append(text[i])
            i += 1
    return tokens
