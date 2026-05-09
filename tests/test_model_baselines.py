from __future__ import annotations

from collections.abc import Sequence

import pytest

from gia_evidence_finder import (
    EvidenceExtractor,
    ExternalRerankerExtractor,
    ExternalRerankScore,
    MarkdownSpanParser,
    SupportLabel,
    TypedDecisionRerankExtractor,
)
from gia_evidence_finder.contracts import DocumentSpan, EvidenceDocument, IntentSpec
from gia_evidence_finder.model_baselines import (
    BM25Extractor,
    CohereRerankerClient,
    CrossEncoderRerankExtractor,
    EmbeddingRetrieverExtractor,
)
from gia_evidence_finder.ranking import ScoreResult


class FakeEmbeddingModel:
    name = "fake_embedding"
    _vocabulary = (
        "schema_name",
        "project_management_benchmark",
        "query",
        "payload",
        "seed",
        "postgres",
    )

    def encode(self, texts: Sequence[str]) -> tuple[tuple[float, ...], ...]:
        vectors: list[tuple[float, ...]] = []
        for text in texts:
            lowered = text.lower()
            vectors.append(
                tuple(1.0 if token in lowered else 0.0 for token in self._vocabulary)
            )
        return tuple(vectors)


class FakePairReranker:
    name = "fake_pair_reranker"

    def score_pairs(self, pairs: Sequence[tuple[str, str]]) -> tuple[float, ...]:
        scores: list[float] = []
        for _query, document in pairs:
            if "schema_name" in document and "project_management_benchmark" in document:
                scores.append(5.0)
            elif "seed" in document.lower():
                scores.append(-2.0)
            else:
                scores.append(0.0)
        return tuple(scores)


class FakeExternalRerankerClient:
    name = "fake_external_reranker"

    def rerank(
        self,
        *,
        query: str,
        documents: Sequence[str],
        top_n: int,
    ) -> tuple[ExternalRerankScore, ...]:
        scored = sorted(
            (
                ExternalRerankScore(
                    index=index,
                    score=0.99 if "schema_name" in document else 0.05,
                )
                for index, document in enumerate(documents)
            ),
            key=lambda item: item.score,
            reverse=True,
        )
        return tuple(scored[:top_n])


class ZeroRanker:
    def score(self, intent: IntentSpec, span: DocumentSpan) -> ScoreResult:
        del intent, span
        return ScoreResult(score=0.0, features={"zero": 0.0}, reasons=())


class LowSchemaRanker:
    def score(self, intent: IntentSpec, span: DocumentSpan) -> ScoreResult:
        del intent
        score = 0.4 if "schema_name" in span.text else 0.0
        return ScoreResult(
            score=score,
            features={"low_schema": score},
            reasons=("low schema support",) if score else (),
        )


def test_embedding_retriever_extractor_uses_model_scores() -> None:
    document = _fixture_document()
    intent = _fixture_intent()

    result = EmbeddingRetrieverExtractor(embedding_model=FakeEmbeddingModel()).extract(
        intent,
        document,
    )

    assert not result.abstained
    assert result.matches[0].label == SupportLabel.SUPPORTS
    assert "schema_name" in result.matches[0].span.text
    assert "embedding_similarity:fake_embedding" in result.matches[0].features


def test_bm25_extractor_scores_document_corpus_terms() -> None:
    document = _fixture_document()
    intent = _fixture_intent()

    result = BM25Extractor(support_threshold=0.4).extract(intent, document)

    assert not result.abstained
    assert "schema_name" in result.matches[0].span.text
    assert "bm25_normalized_score" in result.matches[0].features


def test_cross_encoder_reranker_can_use_explicit_support_threshold() -> None:
    document = _fixture_document()
    intent = _fixture_intent()

    result = CrossEncoderRerankExtractor(
        reranker_model=FakePairReranker(),
        support_threshold=1.0,
    ).extract(
        intent,
        document,
    )

    assert result.abstained
    assert result.candidates[0].label == SupportLabel.NEAR_MISS


def test_cross_encoder_reranker_extracts_after_first_stage() -> None:
    document = _fixture_document()
    intent = _fixture_intent()

    result = CrossEncoderRerankExtractor(reranker_model=FakePairReranker()).extract(
        intent,
        document,
    )

    assert not result.abstained
    assert "schema_name" in result.matches[0].span.text
    assert "cross_encoder_score:fake_pair_reranker" in result.matches[0].features


def test_cross_encoder_reranker_can_require_first_stage_support() -> None:
    document = _fixture_document()
    intent = _fixture_intent()

    result = CrossEncoderRerankExtractor(
        reranker_model=FakePairReranker(),
        first_stage=EvidenceExtractor(ranker=ZeroRanker()),
        require_first_stage_support=True,
    ).extract(
        intent,
        document,
    )

    assert result.abstained
    assert result.candidates[0].label == SupportLabel.NEAR_MISS


def test_cross_encoder_reranker_can_preserve_first_stage_labels() -> None:
    document = _fixture_document()
    intent = _fixture_intent()

    result = CrossEncoderRerankExtractor(
        reranker_model=FakePairReranker(),
        first_stage=EvidenceExtractor(ranker=ZeroRanker()),
        preserve_first_stage_labels=True,
    ).extract(
        intent,
        document,
    )

    assert result.abstained
    assert result.candidates[0].label == SupportLabel.REJECT
    assert result.candidates[0].features["first_stage_score"] == 0.0
    assert result.candidates[0].features["first_stage:zero"] == 0.0


def test_typed_decision_rerank_extractor_uses_model_for_ordering_only() -> None:
    document = _fixture_document()
    intent = _fixture_intent()

    result = TypedDecisionRerankExtractor(
        reranker_model=FakePairReranker(),
        first_stage=EvidenceExtractor(ranker=ZeroRanker()),
    ).extract(
        intent,
        document,
    )

    assert result.abstained
    assert result.candidates[0].label == SupportLabel.REJECT
    assert "schema_name" in result.candidates[0].span.text


def test_typed_decision_rerank_extractor_can_calibrate_first_stage_labels() -> None:
    document = _fixture_document()
    intent = _fixture_intent()

    result = TypedDecisionRerankExtractor(
        reranker_model=FakePairReranker(),
        first_stage=EvidenceExtractor(ranker=LowSchemaRanker()),
        first_stage_support_threshold=0.3,
    ).extract(
        intent,
        document,
    )

    assert not result.abstained
    assert result.matches[0].label == SupportLabel.SUPPORTS
    assert "schema_name" in result.matches[0].span.text


def test_external_reranker_boundary_honors_client_ranking() -> None:
    document = _fixture_document()
    intent = _fixture_intent()

    result = ExternalRerankerExtractor(client=FakeExternalRerankerClient()).extract(
        intent,
        document,
    )

    assert not result.abstained
    assert "schema_name" in result.matches[0].span.text
    assert "external_rerank_score:fake_external_reranker" in result.matches[0].features


def test_cohere_client_requires_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("COHERE_API_KEY", raising=False)

    with pytest.raises(RuntimeError, match="COHERE_API_KEY"):
        CohereRerankerClient()


def _fixture_document() -> EvidenceDocument:
    return MarkdownSpanParser().parse(
        """# Fixture

Seed the project-management benchmark example into Postgres.

Frontend requests can select the fixture by including schema_name
project_management_benchmark in query payloads.
""",
        document_id="fixture",
    )


def _fixture_intent() -> IntentSpec:
    return IntentSpec(
        id="fixture_payload",
        label="request payload fixture selection",
        description="Find evidence that schema_name selects the project-management fixture.",
        positive_examples=("schema_name project_management_benchmark query payloads",),
        negative_examples=("seed project-management benchmark example",),
        min_support_score=0.5,
    )
