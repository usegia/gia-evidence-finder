from __future__ import annotations

import re

from gia_evidence_finder.contracts import EvidenceFacet, IntentSpec
from gia_evidence_finder.quantifiers import requirements_from_text
from gia_evidence_finder.text import tokenize

_CAPITALIZED_PHRASE_RE = re.compile(
    r"\b(?:[A-Z][A-Za-z0-9]*(?:[- ][A-Z][A-Za-z0-9]*){0,4})\b"
)
_TECHNICAL_TOKEN_RE = re.compile(r"\b[a-zA-Z0-9_./-]*[A-Z0-9_./-][a-zA-Z0-9_./-]*\b")
_RAW_WORD_RE = re.compile(r"\b[a-zA-Z][a-zA-Z0-9-]*\b")
_STOP_PHRASES = {
    "Find",
    "Evidence",
    "It",
    "The",
    "This",
}
_INDIRECTION_MARKERS = (
    "based on",
    "derived from",
    "distribution of",
    "fork of",
    "released under",
    "wrapper around",
)
_LEXICAL_FACET_STOP_TOKENS = frozenset(
    {
        "claim",
        "direct",
        "evidence",
        "find",
        "has",
        "have",
        "include",
        "includ",
        "itself",
        "this",
    }
)
_RAW_LEXICAL_STOP_WORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "be",
        "claim",
        "direct",
        "evidence",
        "find",
        "for",
        "from",
        "has",
        "have",
        "include",
        "includes",
        "included",
        "including",
        "in",
        "is",
        "itself",
        "of",
        "on",
        "or",
        "supports",
        "support",
        "supported",
        "supporting",
        "that",
        "the",
        "this",
        "to",
        "with",
    }
)
_NEGATION_TOKENS = frozenset(
    {
        "no",
        "not",
        "without",
        "unsupported",
    }
)


def compile_intent(
    claim: str | None = None,
    *,
    intent_id: str = "claim",
    positive_examples: tuple[str, ...] = (),
    negative_examples: tuple[str, ...] = (),
    min_support_score: float = 0.55,
) -> IntentSpec:
    """Compile a raw claim into a deterministic typed intent.

    This is intentionally conservative. It extracts stable technical anchors and
    obvious negation scopes, but it does not attempt broad natural-language
    understanding. More complex claim typing should become explicit data, not a
    hidden prompt.
    """

    if claim is None:
        raise TypeError("compile_intent() missing required argument: 'claim'")
    claim = claim.strip()
    required_facets = _required_facets(claim)
    excluded_facets = _excluded_facets(claim)
    quantifier_requirements = requirements_from_text(claim)
    return IntentSpec(
        id=intent_id,
        label=claim,
        description=f"Find direct evidence for this claim: {claim}",
        positive_examples=(claim, *positive_examples),
        negative_examples=negative_examples,
        required_facets=required_facets,
        excluded_facets=excluded_facets,
        quantifier_requirements=quantifier_requirements,
        min_support_score=min_support_score,
    )


def _required_facets(claim: str) -> tuple[EvidenceFacet, ...]:
    phrases: list[str] = []
    covered_tokens: set[str] = set()
    for match in _CAPITALIZED_PHRASE_RE.finditer(claim):
        phrase = match.group(0).strip()
        if phrase not in _STOP_PHRASES:
            _append_phrase(phrases, covered_tokens, phrase)
    for match in _TECHNICAL_TOKEN_RE.finditer(claim):
        token = match.group(0).strip()
        if len(token) > 2 and token not in _STOP_PHRASES:
            _append_phrase(phrases, covered_tokens, token)
    phrases.extend(_lexical_phrase_facets(claim, covered_tokens=covered_tokens))
    phrases.extend(_lexical_anchor_facets(claim, covered_tokens=covered_tokens))
    return tuple(
        EvidenceFacet(name=_facet_name(index, phrase), phrases=(phrase,))
        for index, phrase in enumerate(_dedupe(phrases), start=1)
    )


def _append_phrase(phrases: list[str], covered_tokens: set[str], phrase: str) -> None:
    tokens = set(tokenize(phrase))
    if not tokens or tokens <= covered_tokens:
        return
    phrases.append(phrase)
    covered_tokens.update(tokens)


def _lexical_phrase_facets(claim: str, *, covered_tokens: set[str]) -> tuple[str, ...]:
    phrases: list[str] = []
    chunk: list[str] = []
    for match in _RAW_WORD_RE.finditer(claim):
        word = match.group(0)
        if _is_raw_phrase_word(word, covered_tokens):
            chunk.append(word)
            continue
        _append_lexical_phrase_chunk(phrases, covered_tokens, chunk)
        chunk = []
    _append_lexical_phrase_chunk(phrases, covered_tokens, chunk)
    return tuple(phrases)


def _append_lexical_phrase_chunk(
    phrases: list[str],
    covered_tokens: set[str],
    chunk: list[str],
) -> None:
    if len(chunk) == 2:
        _append_phrase(phrases, covered_tokens, " ".join(chunk))


def _is_raw_phrase_word(word: str, covered_tokens: set[str]) -> bool:
    normalized = word.lower()
    if normalized in _RAW_LEXICAL_STOP_WORDS:
        return False
    tokens = set(tokenize(word))
    return bool(tokens) and not tokens <= covered_tokens


def _lexical_anchor_facets(claim: str, *, covered_tokens: set[str]) -> tuple[str, ...]:
    anchors: list[str] = []
    for token in tokenize(claim):
        if (
            len(token) <= 3
            or token in _LEXICAL_FACET_STOP_TOKENS
            or token in _NEGATION_TOKENS
            or token in covered_tokens
        ):
            continue
        anchors.append(token)
    return _dedupe(anchors)[:8]


def _excluded_facets(claim: str) -> tuple[EvidenceFacet, ...]:
    phrases: list[str] = []
    if "itself" in claim.lower():
        phrases.extend(_INDIRECTION_MARKERS)
    return tuple(
        EvidenceFacet(name=f"excluded_{index}", phrases=(phrase,))
        for index, phrase in enumerate(_dedupe(phrases), start=1)
    )


def _dedupe(values: list[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    deduped: list[str] = []
    for value in values:
        normalized = value.lower()
        if normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(value)
    return tuple(deduped)


def _facet_name(index: int, phrase: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9]+", "_", phrase.lower()).strip("_")
    return cleaned or f"facet_{index}"
