from __future__ import annotations

import re
from dataclasses import dataclass

from gia_evidence_finder.text import tokenize

_WORD_RE = re.compile(r"[a-zA-Z0-9_./:-]+")
_BOUNDARY_MARKER = "__polarity_boundary__"
_BOUNDARY_TOKEN = "polarityboundary"
_CONTRACTION_REPLACEMENTS = (
    ("can't", "can not"),
    ("cannot", "can not"),
    ("won't", "will not"),
    ("don't", "do not"),
    ("doesn't", "does not"),
    ("didn't", "did not"),
    ("isn't", "is not"),
    ("aren't", "are not"),
    ("wasn't", "was not"),
    ("weren't", "were not"),
    ("shouldn't", "should not"),
    ("mustn't", "must not"),
    ("wouldn't", "would not"),
    ("couldn't", "could not"),
)

_ANCHOR_STOP_TOKENS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "claim",
        "direct",
        "document",
        "evidence",
        "find",
        "for",
        "from",
        "in",
        "intent",
        "is",
        "it",
        "of",
        "on",
        "or",
        "span",
        "that",
        "the",
        "this",
        "to",
        "with",
        "can",
        "could",
        "may",
        "might",
        "must",
        "shall",
        "should",
        "will",
        "would",
    }
)
_NEGATION_CUES = frozenset(
    {
        "absent",
        "avoid",
        "disallow",
        "disallowed",
        "exclude",
        "excluded",
        "excluding",
        "forbid",
        "forbidden",
        "instead",
        "lack",
        "lacks",
        "never",
        "no",
        "not",
        "rather",
        "reject",
        "rejected",
        "without",
        "unsupported",
    }
)
_NEGATION_BOUNDARIES = frozenset(
    {
        "although",
        "because",
        "but",
        "however",
        _BOUNDARY_TOKEN,
        "so",
        "therefore",
        "then",
        "though",
        "unless",
        "whereas",
        "while",
        "yet",
    }
)
_NON_NEGATING_NOT_FOLLOWERS = frozenset({"just", "merely", "only", "simply"})
_PRECEDING_SCOPE = 10
_FOLLOWING_SCOPE = 10
_POSTPOSED_NEGATED_PREDICATES = frozenset(
    {
        "accept",
        "accepted",
        "allow",
        "allowed",
        "available",
        "enable",
        "enabled",
        "implement",
        "implemented",
        "include",
        "included",
        "includ",
        "permit",
        "permitted",
        "present",
        "provide",
        "provided",
        "provid",
        "proof",
        "require",
        "required",
        "requir",
        "support",
        "supported",
        "use",
        "used",
    }
)
_POSTPOSED_NEGATION_CUES = frozenset(
    {
        "absent",
        "disallowed",
        "excluded",
        "forbidden",
        "reject",
        "rejected",
        "unsupported",
    }
)


@dataclass(frozen=True)
class PolarityScore:
    alignment: float
    mismatch: float
    contradiction: float
    intent_negated_anchor_coverage: float
    span_negated_anchor_coverage: float
    shared_anchor_count: int


@dataclass(frozen=True)
class _TokenOccurrence:
    token: str
    index: int


@dataclass(frozen=True)
class _TokenPolarity:
    affirmed: int = 0
    negated: int = 0

    @property
    def is_negated(self) -> bool:
        return self.negated > 0 and self.negated >= self.affirmed

    @property
    def mentions(self) -> int:
        return self.affirmed + self.negated


def polarity_score(intent_texts: tuple[str, ...], span_text: str) -> PolarityScore:
    """Compare assertion polarity for shared content anchors.

    The score intentionally stays lexical and deterministic. It does not try to
    prove entailment; it catches the common false-support pattern where a span
    contains the right anchors but explicitly negates them.
    """

    intent_polarity = _token_polarity(intent_texts)
    span_polarity = _token_polarity((span_text,))
    shared_tokens = tuple(
        sorted(
            token
            for token in set(intent_polarity) & set(span_polarity)
            if _is_anchor_token(token)
        )
    )
    if not shared_tokens:
        return PolarityScore(
            alignment=0.0,
            mismatch=0.0,
            contradiction=0.0,
            intent_negated_anchor_coverage=0.0,
            span_negated_anchor_coverage=0.0,
            shared_anchor_count=0,
        )

    mismatches = 0
    alignments = 0
    intent_negated = 0
    span_negated = 0
    polarity_touched = 0
    for token in shared_tokens:
        intent_token = intent_polarity[token]
        span_token = span_polarity[token]
        intent_is_negated = intent_token.is_negated
        span_is_negated = span_token.is_negated
        if intent_is_negated:
            intent_negated += 1
        if span_is_negated:
            span_negated += 1
        if intent_is_negated or span_is_negated:
            polarity_touched += 1
        if intent_is_negated == span_is_negated:
            alignments += 1
        else:
            mismatches += 1

    shared_count = len(shared_tokens)
    mismatch = mismatches / shared_count
    polarity_touch_rate = polarity_touched / shared_count
    contradiction = mismatch * min(1.0, 0.35 + polarity_touch_rate)
    return PolarityScore(
        alignment=alignments / shared_count,
        mismatch=mismatch,
        contradiction=contradiction,
        intent_negated_anchor_coverage=intent_negated / shared_count,
        span_negated_anchor_coverage=span_negated / shared_count,
        shared_anchor_count=shared_count,
    )


def _token_polarity(texts: tuple[str, ...]) -> dict[str, _TokenPolarity]:
    combined: dict[str, _TokenPolarity] = {}
    for text in texts:
        occurrences = _token_occurrences(text)
        for occurrence in occurrences:
            if not _is_anchor_token(occurrence.token):
                continue
            current = combined.get(occurrence.token, _TokenPolarity())
            if _is_negated(occurrences, occurrence.index):
                combined[occurrence.token] = _TokenPolarity(
                    affirmed=current.affirmed,
                    negated=current.negated + 1,
                )
            else:
                combined[occurrence.token] = _TokenPolarity(
                    affirmed=current.affirmed + 1,
                    negated=current.negated,
                )
    return combined


def _token_occurrences(text: str) -> tuple[_TokenOccurrence, ...]:
    normalized = text.lower()
    for source, replacement in _CONTRACTION_REPLACEMENTS:
        normalized = normalized.replace(source, replacement)
    normalized = re.sub(r"[.;]", f" {_BOUNDARY_MARKER} ", normalized)
    occurrences: list[_TokenOccurrence] = []
    for raw_token in _WORD_RE.findall(normalized):
        if raw_token == _BOUNDARY_MARKER:
            occurrences.append(_TokenOccurrence(token=_BOUNDARY_TOKEN, index=len(occurrences)))
            continue
        for token in tokenize(raw_token):
            occurrences.append(_TokenOccurrence(token=token, index=len(occurrences)))
    return tuple(occurrences)


def _is_anchor_token(token: str) -> bool:
    return (
        len(token) > 1
        and token not in _ANCHOR_STOP_TOKENS
        and token not in _NEGATION_CUES
        and token not in _NEGATION_BOUNDARIES
    )


def _is_negated(occurrences: tuple[_TokenOccurrence, ...], index: int) -> bool:
    return _has_preceding_negation(occurrences, index) or _has_following_negation(
        occurrences,
        index,
    )


def _has_preceding_negation(occurrences: tuple[_TokenOccurrence, ...], index: int) -> bool:
    lower_bound = max(0, index - _PRECEDING_SCOPE)
    for cue_index in range(index - 1, lower_bound - 1, -1):
        cue = occurrences[cue_index].token
        if cue in _NEGATION_BOUNDARIES:
            return False
        if (
            cue == "not"
            and cue_index + 1 == index
            and occurrences[index].token in _NON_NEGATING_NOT_FOLLOWERS
        ):
            return True
        if _is_active_negation_cue(occurrences, cue_index) and (
            index - cue_index <= _preceding_scope_for_cue(cue)
        ):
            return True
    return False


def _has_following_negation(occurrences: tuple[_TokenOccurrence, ...], index: int) -> bool:
    upper_bound = min(len(occurrences), index + _FOLLOWING_SCOPE + 1)
    for cue_index in range(index + 1, upper_bound):
        cue = occurrences[cue_index].token
        if cue in _NEGATION_BOUNDARIES:
            return False
        if _is_postposed_negation_cue(occurrences, cue_index):
            return True
    return False


def _is_active_negation_cue(
    occurrences: tuple[_TokenOccurrence, ...],
    cue_index: int,
) -> bool:
    cue = occurrences[cue_index].token
    if cue not in _NEGATION_CUES:
        return False
    next_index = cue_index + 1
    return not (
        cue == "not"
        and next_index < len(occurrences)
        and occurrences[next_index].token in _NON_NEGATING_NOT_FOLLOWERS
    )


def _is_postposed_negation_cue(
    occurrences: tuple[_TokenOccurrence, ...],
    cue_index: int,
) -> bool:
    cue = occurrences[cue_index].token
    if cue in _POSTPOSED_NEGATION_CUES:
        return True
    next_index = cue_index + 1
    return (
        cue == "not"
        and next_index < len(occurrences)
        and occurrences[next_index].token in _POSTPOSED_NEGATED_PREDICATES
    )


def _preceding_scope_for_cue(cue: str) -> int:
    if cue == "avoid":
        return 2
    if cue == "no":
        return 4
    if cue == "without":
        return 5
    if cue in {"instead", "rather"}:
        return 8
    if cue in {"reject", "rejected"}:
        return 8
    return _PRECEDING_SCOPE
