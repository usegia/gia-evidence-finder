from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pytest

from gia_evidence_finder.reranker_cache import (
    CachedPairRerankerModel,
    InMemoryPairScoreCache,
    JsonlPairScoreCache,
    PairScoreCacheKey,
)


class CountingPairReranker:
    name = "counting_pair_reranker"

    def __init__(self) -> None:
        self.calls = 0
        self.batch_sizes: list[int] = []

    def score_pairs(self, pairs: Sequence[tuple[str, str]]) -> tuple[float, ...]:
        self.calls += 1
        self.batch_sizes.append(len(pairs))
        return tuple(float(len(document)) for _query, document in pairs)


class BadPairReranker:
    name = "bad_pair_reranker"

    def score_pairs(self, pairs: Sequence[tuple[str, str]]) -> tuple[float, ...]:
        del pairs
        return (1.0,)


def test_cached_pair_reranker_deduplicates_and_persists_scores(tmp_path: Path) -> None:
    cache_path = tmp_path / "pair-scores.jsonl"
    model = CountingPairReranker()
    cached = CachedPairRerankerModel(
        model=model,
        cache=JsonlPairScoreCache(cache_path),
        batch_size=2,
    )

    scores = cached.score_pairs((("intent", "alpha"), ("intent", "alpha"), ("intent", "beta")))

    assert scores == (5.0, 5.0, 4.0)
    assert model.calls == 1
    assert model.batch_sizes == [2]
    assert cached.stats.cache_hits == 0
    assert cached.stats.cache_misses == 3
    assert cached.stats.scores_written == 2

    second_model = CountingPairReranker()
    second_cached = CachedPairRerankerModel(
        model=second_model,
        cache=JsonlPairScoreCache(cache_path),
    )

    second_scores = second_cached.score_pairs((("intent", "alpha"), ("intent", "beta")))

    assert second_scores == (5.0, 4.0)
    assert second_model.calls == 0
    assert second_cached.stats.cache_hits == 2
    assert second_cached.stats.cache_misses == 0
    assert len(cache_path.read_text(encoding="utf-8").splitlines()) == 2


def test_cached_pair_reranker_batches_unique_misses() -> None:
    model = CountingPairReranker()
    cached = CachedPairRerankerModel(
        model=model,
        cache=InMemoryPairScoreCache(),
        batch_size=2,
    )

    scores = cached.score_pairs(
        tuple(("intent", f"document {index}") for index in range(5))
    )

    assert len(scores) == 5
    assert model.calls == 3
    assert model.batch_sizes == [2, 2, 1]
    assert cached.stats.model_calls == 3
    assert cached.stats.scores_written == 5


def test_cached_pair_reranker_rejects_bad_model_score_count() -> None:
    cached = CachedPairRerankerModel(
        model=BadPairReranker(),
        cache=InMemoryPairScoreCache(),
        batch_size=3,
    )

    with pytest.raises(RuntimeError, match="returned 1 scores for 2 pairs"):
        cached.score_pairs((("intent", "alpha"), ("intent", "beta")))


def test_pair_score_cache_key_is_stable_and_model_scoped() -> None:
    left = PairScoreCacheKey.from_pair(model_name="model-a", pair=("intent", "doc"))
    same = PairScoreCacheKey.from_pair(model_name="model-a", pair=("intent", "doc"))
    other_model = PairScoreCacheKey.from_pair(model_name="model-b", pair=("intent", "doc"))

    assert left.id == same.id
    assert left.id != other_model.id
