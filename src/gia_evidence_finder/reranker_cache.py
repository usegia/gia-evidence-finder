from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol


class PairRerankerProtocol(Protocol):
    name: str

    def score_pairs(self, pairs: Sequence[tuple[str, str]]) -> tuple[float, ...]: ...


@dataclass(frozen=True)
class PairScoreCacheKey:
    model_name: str
    query_sha256: str
    document_sha256: str
    version: str = "pair_reranker_score_v1"

    def __post_init__(self) -> None:
        if not self.model_name.strip():
            raise ValueError("cache key model_name must not be empty")
        _validate_sha256("query_sha256", self.query_sha256)
        _validate_sha256("document_sha256", self.document_sha256)

    @classmethod
    def from_pair(cls, *, model_name: str, pair: tuple[str, str]) -> PairScoreCacheKey:
        query, document = pair
        return cls(
            model_name=model_name,
            query_sha256=_text_sha256(query),
            document_sha256=_text_sha256(document),
        )

    @property
    def id(self) -> str:
        payload = json.dumps(
            self.to_json_dict(),
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def to_json_dict(self) -> dict[str, str]:
        return {
            "version": self.version,
            "model_name": self.model_name,
            "query_sha256": self.query_sha256,
            "document_sha256": self.document_sha256,
        }

    @classmethod
    def from_json_dict(cls, payload: object, *, line_number: int) -> PairScoreCacheKey:
        if not isinstance(payload, dict):
            raise ValueError(f"line {line_number}: key must be an object")
        return cls(
            version=_required_string(payload, "version", line_number),
            model_name=_required_string(payload, "model_name", line_number),
            query_sha256=_required_string(payload, "query_sha256", line_number),
            document_sha256=_required_string(payload, "document_sha256", line_number),
        )


@dataclass(frozen=True)
class PairScoreCacheEntry:
    key: PairScoreCacheKey
    score: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.score):
            raise ValueError("cache entry score must be finite")

    def to_json_dict(self) -> dict[str, object]:
        return {
            "id": self.key.id,
            "key": self.key.to_json_dict(),
            "score": self.score,
        }

    @classmethod
    def from_json_dict(cls, payload: object, *, line_number: int) -> PairScoreCacheEntry:
        if not isinstance(payload, dict):
            raise ValueError(f"line {line_number}: cache entry must be an object")
        key = PairScoreCacheKey.from_json_dict(
            payload.get("key"),
            line_number=line_number,
        )
        score = payload.get("score")
        if not isinstance(score, int | float):
            raise ValueError(f"line {line_number}: score must be numeric")
        entry_id = payload.get("id")
        if entry_id is not None and entry_id != key.id:
            raise ValueError(f"line {line_number}: cache entry id does not match key")
        return cls(key=key, score=float(score))


class PairScoreCache(Protocol):
    def get(self, key: PairScoreCacheKey) -> float | None: ...

    def set(self, key: PairScoreCacheKey, score: float) -> bool: ...


@dataclass
class InMemoryPairScoreCache:
    _scores: dict[str, float] = field(default_factory=dict)

    @property
    def size(self) -> int:
        return len(self._scores)

    def get(self, key: PairScoreCacheKey) -> float | None:
        return self._scores.get(key.id)

    def set(self, key: PairScoreCacheKey, score: float) -> bool:
        _validate_score(score)
        current = self._scores.get(key.id)
        if current == score:
            return False
        self._scores[key.id] = float(score)
        return True


class JsonlPairScoreCache:
    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self._scores: dict[str, float] = {}
        if self.path.exists():
            self._load()

    @property
    def size(self) -> int:
        return len(self._scores)

    def get(self, key: PairScoreCacheKey) -> float | None:
        return self._scores.get(key.id)

    def set(self, key: PairScoreCacheKey, score: float) -> bool:
        _validate_score(score)
        current = self._scores.get(key.id)
        if current == score:
            return False
        self._scores[key.id] = float(score)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        entry = PairScoreCacheEntry(key=key, score=float(score))
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry.to_json_dict(), sort_keys=True) + "\n")
        return True

    def _load(self) -> None:
        for line_number, line in enumerate(self.path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"line {line_number}: invalid JSON cache entry") from exc
            entry = PairScoreCacheEntry.from_json_dict(payload, line_number=line_number)
            self._scores[entry.key.id] = entry.score


@dataclass
class PairScoreCacheStats:
    requests: int = 0
    pairs: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    model_calls: int = 0
    scores_written: int = 0

    def to_json_dict(self) -> dict[str, int]:
        return {
            "requests": self.requests,
            "pairs": self.pairs,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "model_calls": self.model_calls,
            "scores_written": self.scores_written,
        }


class CachedPairRerankerModel:
    def __init__(
        self,
        *,
        model: PairRerankerProtocol,
        cache: PairScoreCache,
        batch_size: int | None = None,
        name: str | None = None,
    ) -> None:
        if batch_size is not None and batch_size < 1:
            raise ValueError("batch_size must be positive when provided")
        self.model = model
        self.cache = cache
        self.batch_size = batch_size
        self.name = name or model.name
        self.stats = PairScoreCacheStats()

    def score_pairs(self, pairs: Sequence[tuple[str, str]]) -> tuple[float, ...]:
        self.stats.requests += 1
        self.stats.pairs += len(pairs)
        if not pairs:
            return ()

        scores: list[float | None] = [None] * len(pairs)
        misses: list[_PairScoreMiss] = []
        miss_index_by_key: dict[str, int] = {}
        for index, pair in enumerate(pairs):
            key = PairScoreCacheKey.from_pair(model_name=self.model.name, pair=pair)
            cached_score = self.cache.get(key)
            if cached_score is not None:
                scores[index] = cached_score
                self.stats.cache_hits += 1
                continue
            miss_index = miss_index_by_key.get(key.id)
            if miss_index is None:
                miss_index_by_key[key.id] = len(misses)
                misses.append(_PairScoreMiss(key=key, pair=pair, indexes=[index]))
            else:
                misses[miss_index].indexes.append(index)
            self.stats.cache_misses += 1

        for batch in _batches(tuple(misses), self.batch_size):
            batch_scores = self.model.score_pairs(tuple(miss.pair for miss in batch))
            self.stats.model_calls += 1
            if len(batch_scores) != len(batch):
                raise RuntimeError(
                    f"reranker returned {len(batch_scores)} scores for {len(batch)} pairs"
                )
            for miss, score in zip(batch, batch_scores, strict=True):
                if self.cache.set(miss.key, float(score)):
                    self.stats.scores_written += 1
                for index in miss.indexes:
                    scores[index] = float(score)

        return tuple(_required_score(score) for score in scores)


@dataclass
class _PairScoreMiss:
    key: PairScoreCacheKey
    pair: tuple[str, str]
    indexes: list[int]


def _batches(
    items: tuple[_PairScoreMiss, ...],
    batch_size: int | None,
) -> tuple[tuple[_PairScoreMiss, ...], ...]:
    if not items:
        return ()
    if batch_size is None:
        return (items,)
    return tuple(items[index : index + batch_size] for index in range(0, len(items), batch_size))


def _text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _validate_sha256(name: str, value: str) -> None:
    if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise ValueError(f"{name} must be a lowercase sha256 hex digest")


def _validate_score(score: float) -> None:
    if not math.isfinite(score):
        raise ValueError("score must be finite")


def _required_score(score: float | None) -> float:
    if score is None:
        raise RuntimeError("missing cached reranker score")
    return score


def _required_string(payload: dict[object, object], key: str, line_number: int) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"line {line_number}: {key} must be a non-empty string")
    return value
