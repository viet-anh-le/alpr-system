"""Slot-aware OCR ambiguity correction for Vietnamese plate strings."""
from __future__ import annotations

from dataclasses import dataclass

from .ocr_ctm import ALNUM_TEMPLATES, LITERAL_TOKENS, SLOT_CLASSES, TEMPLATES, PlateTemplate

DIGIT_TO_LETTER = {
    "0": "O",
    "1": "I",
    "2": "Z",
    "5": "S",
    "6": "G",
    "8": "B",
}
LETTER_TO_DIGIT = {value: key for key, value in DIGIT_TO_LETTER.items()}
CORRECTED_CONF_SCALE = 0.92


@dataclass(frozen=True)
class AmbiguousCorrectionResult:
    char_probs: list[tuple[str, float]]
    changed_positions: list[int]
    risk_penalty: float
    template_name: str | None

    @property
    def changed(self) -> bool:
        return bool(self.changed_positions)


def correct_ambiguous_chars(
    char_probs: list[tuple[str, float]],
    *,
    format_mode: str = "raw",
) -> AmbiguousCorrectionResult:
    templates = _templates_for_mode(format_mode)
    tokens = _flatten_ocr_tokens(char_probs, format_mode=format_mode)
    if not tokens:
        return AmbiguousCorrectionResult([], [], 0.0, None)

    candidates = [_correct_with_template(tokens, template) for template in templates]
    best = min(
        candidates,
        key=lambda candidate: (
            candidate.alignment_cost,
            len(candidate.result.changed_positions),
            candidate.result.risk_penalty,
        ),
    )
    return best.result


@dataclass(frozen=True)
class _TemplateCorrection:
    result: AmbiguousCorrectionResult
    alignment_cost: float


def _correct_with_template(
    tokens: list[tuple[str, float, int]],
    template: PlateTemplate,
) -> _TemplateCorrection:
    aligned, cost = _align_to_template(tokens, template)
    output: list[tuple[str, float, int]] = []
    changed_positions: list[int] = []
    risk_penalty = 0.0

    for pattern_token, token_prob in zip(template.pattern, aligned):
        if token_prob is None:
            continue
        token, conf, original_pos = token_prob
        corrected = _correct_token_for_pattern(token, pattern_token)
        if corrected != token:
            changed_positions.append(original_pos)
            risk_penalty += 0.055
            output.append((corrected, round(float(conf) * CORRECTED_CONF_SCALE, 6), original_pos))
        else:
            output.append((token, conf, original_pos))

    output.sort(key=lambda item: item[2])
    char_probs = [(token, conf) for token, conf, _ in output]
    return _TemplateCorrection(
        result=AmbiguousCorrectionResult(
            char_probs=char_probs,
            changed_positions=changed_positions,
            risk_penalty=round(risk_penalty, 6),
            template_name=template.name,
        ),
        alignment_cost=cost + risk_penalty,
    )


def _align_to_template(
    tokens: list[tuple[str, float, int]],
    template: PlateTemplate,
) -> tuple[list[tuple[str, float, int] | None], float]:
    n = len(template.pattern)
    m = len(tokens)
    dp = [[0.0] * (m + 1) for _ in range(n + 1)]
    back: list[list[str]] = [[""] * (m + 1) for _ in range(n + 1)]

    for i in range(1, n + 1):
        dp[i][0] = dp[i - 1][0] + _missing_template_token_cost(template.pattern[i - 1])
        back[i][0] = "missing"
    for j in range(1, m + 1):
        dp[0][j] = dp[0][j - 1] + _skip_input_token_cost(tokens[j - 1][0])
        back[0][j] = "skip"

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            token = tokens[j - 1][0]
            pattern_token = template.pattern[i - 1]
            options = [
                (dp[i - 1][j - 1] + _align_token_cost(token, pattern_token), "align"),
                (dp[i - 1][j] + _missing_template_token_cost(pattern_token), "missing"),
                (dp[i][j - 1] + _skip_input_token_cost(token), "skip"),
            ]
            dp[i][j], back[i][j] = min(options, key=lambda item: item[0])

    aligned: list[tuple[str, float, int] | None] = [None for _ in range(n)]
    i, j = n, m
    while i > 0 or j > 0:
        action = back[i][j]
        if action == "align":
            aligned[i - 1] = tokens[j - 1]
            i -= 1
            j -= 1
        elif action == "missing":
            i -= 1
        else:
            j -= 1

    return aligned, dp[n][m]


def _align_token_cost(token: str, pattern_token: str) -> float:
    if pattern_token in SLOT_CLASSES:
        if _matches_slot(token, pattern_token):
            return 0.0
        if _correct_token_for_pattern(token, pattern_token) != token:
            return 0.12
        if _is_slot_token(token):
            return 2.0
        return 2.4
    return 0.0 if token == pattern_token else 2.0


def _correct_token_for_pattern(token: str, pattern_token: str) -> str:
    if pattern_token == "D":
        return LETTER_TO_DIGIT.get(token, token)
    if pattern_token == "L":
        return DIGIT_TO_LETTER.get(token, token)
    return token


def _matches_slot(token: str, slot_class: str) -> bool:
    if slot_class == "D":
        return token.isdigit()
    if slot_class == "L":
        return token.isalpha()
    return _is_slot_token(token)


def _missing_template_token_cost(pattern_token: str) -> float:
    if pattern_token in SLOT_CLASSES:
        return 1.4
    if pattern_token == "[SEP]":
        return 1.0
    return 0.75


def _skip_input_token_cost(token: str) -> float:
    return 1.1 if _is_slot_token(token) else 0.7


def _templates_for_mode(format_mode: str) -> tuple[PlateTemplate, ...]:
    if format_mode == "alnum":
        return ALNUM_TEMPLATES
    if format_mode != "raw":
        raise ValueError("format_mode must be either 'raw' or 'alnum'")
    return TEMPLATES


def _flatten_ocr_tokens(
    char_probs: list[tuple[str, float]],
    *,
    format_mode: str = "raw",
) -> list[tuple[str, float, int]]:
    tokens: list[tuple[str, float, int]] = []
    for original_pos, (raw_token, conf) in enumerate(char_probs):
        token = raw_token.strip().upper()
        if not token:
            continue
        if format_mode != "alnum" and token == "[SEP]":
            tokens.append((token, float(conf), original_pos))
            continue
        for char in token:
            if _is_slot_token(char) or (format_mode != "alnum" and char in LITERAL_TOKENS):
                tokens.append((char, float(conf), original_pos))
    return tokens


def _is_slot_token(token: str) -> bool:
    return len(token) == 1 and (token.isalnum() or token == "Đ")
