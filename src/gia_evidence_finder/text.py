from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Iterable, Mapping

TOKEN_RE = re.compile(r"[a-zA-Z0-9_./:-]+")

STOP_WORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "can",
        "do",
        "does",
        "for",
        "from",
        "has",
        "have",
        "how",
        "in",
        "into",
        "is",
        "it",
        "its",
        "of",
        "on",
        "or",
        "that",
        "the",
        "these",
        "this",
        "those",
        "through",
        "to",
        "use",
        "uses",
        "using",
        "what",
        "when",
        "where",
        "which",
        "who",
        "why",
        "with",
    }
)

SYNONYM_GROUPS = (
    ("taxonomy", "concept", "ontology", "vocabulary", "universe", "catalog"),
    ("proof", "evidence", "support", "reason", "explanation", "trace", "audit"),
    ("typed", "schema", "structured", "contract", "validated"),
    ("remote", "external", "api", "openrouter", "service"),
    ("fake", "deterministic", "mock", "test-only", "local"),
    ("secret", "token", "credential", "header", "admin"),
    ("container", "docker", "testcontainers", "service"),
    ("quality", "lint", "typecheck", "mypy", "pytest", "ruff", "test"),
    ("reset", "replay", "clean", "rerun"),
    ("vector", "embedding", "semantic", "pgvector"),
    ("database", "postgres", "storage", "relational"),
    ("claim", "intent", "requirement", "statement"),
    ("only", "just", "merely", "solely"),
)

SYNONYM_INDEX: dict[str, str] = {
    token: group[0] for group in SYNONYM_GROUPS for token in group
}


def normalize_text(text: str) -> str:
    return " ".join(text.lower().strip().split())


def tokenize(text: str) -> tuple[str, ...]:
    tokens: list[str] = []
    for raw_token in TOKEN_RE.findall(text.lower()):
        for part in re.split(r"[/.:_-]+", raw_token):
            token = _stem_token(part)
            if len(token) > 1 and token not in STOP_WORDS:
                tokens.append(SYNONYM_INDEX.get(token, token))
    return tuple(tokens)


def token_counts(text: str) -> Mapping[str, float]:
    return dict(Counter(tokenize(text)))


def cosine(left: Mapping[str, float], right: Mapping[str, float]) -> float:
    if not left or not right:
        return 0.0
    numerator = sum(value * right.get(key, 0.0) for key, value in left.items())
    left_norm = math.sqrt(sum(value * value for value in left.values()))
    right_norm = math.sqrt(sum(value * value for value in right.values()))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return max(0.0, min(1.0, numerator / (left_norm * right_norm)))


def jaccard(left: Iterable[str], right: Iterable[str]) -> float:
    left_set = set(left)
    right_set = set(right)
    if not left_set or not right_set:
        return 0.0
    return len(left_set & right_set) / len(left_set | right_set)


def char_ngram_counts(text: str, n: int = 4) -> Mapping[str, float]:
    normalized = re.sub(r"\s+", " ", normalize_text(text))
    if len(normalized) <= n:
        return {normalized: 1.0} if normalized else {}
    return dict(Counter(normalized[index : index + n] for index in range(len(normalized) - n + 1)))


def soft_similarity(left: str, right: str) -> float:
    token_score = cosine(token_counts(left), token_counts(right))
    overlap_score = jaccard(tokenize(left), tokenize(right))
    char_score = cosine(char_ngram_counts(left), char_ngram_counts(right))
    exact_score = exact_phrase_score(left, right)
    return _clamp(
        (0.48 * token_score)
        + (0.18 * overlap_score)
        + (0.20 * char_score)
        + (0.14 * exact_score)
    )


def exact_phrase_score(left: str, right: str) -> float:
    left_terms = tuple(token for token in tokenize(left) if len(token) > 2)
    if not left_terms:
        return 0.0
    normalized_right = f" {' '.join(tokenize(right))} "
    hits = sum(1 for term in left_terms if f" {term} " in normalized_right)
    return hits / len(left_terms)


def max_similarity(queries: Iterable[str], text: str) -> float:
    return max((soft_similarity(query, text) for query in queries), default=0.0)


def _stem_token(token: str) -> str:
    if len(token) > 5 and token.endswith("ies"):
        return f"{token[:-3]}y"
    for suffix in ("ing", "ers", "er", "ed", "es", "s"):
        if len(token) > len(suffix) + 3 and token.endswith(suffix):
            return token[: -len(suffix)]
    return token


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))
