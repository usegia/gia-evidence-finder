from __future__ import annotations

import json
import math
import os
import urllib.error
import urllib.request
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from importlib import import_module
from typing import Any, Protocol, cast

from gia_evidence_finder.calibration import SupportThresholdOverrideExtractor
from gia_evidence_finder.contracts import (
    DocumentSpan,
    EvidenceDocument,
    ExtractionResult,
    IntentSpec,
    SpanMatch,
    SupportLabel,
)
from gia_evidence_finder.evaluation import ExtractorProtocol
from gia_evidence_finder.extractor import EvidenceExtractor
from gia_evidence_finder.ranking import KeywordOverlapBaseline
from gia_evidence_finder.text import tokenize


class TextEmbeddingModel(Protocol):
    name: str

    def encode(self, texts: Sequence[str]) -> tuple[tuple[float, ...], ...]: ...


class PairRerankerModel(Protocol):
    name: str

    def score_pairs(self, pairs: Sequence[tuple[str, str]]) -> tuple[float, ...]: ...


@dataclass(frozen=True)
class ExternalRerankScore:
    index: int
    score: float

    def __post_init__(self) -> None:
        if self.index < 0:
            raise ValueError("rerank score index must be non-negative")


class ExternalRerankerClient(Protocol):
    name: str

    def rerank(
        self,
        *,
        query: str,
        documents: Sequence[str],
        top_n: int,
    ) -> tuple[ExternalRerankScore, ...]: ...


@dataclass(frozen=True)
class BM25Extractor:
    candidate_limit: int = 30
    max_matches: int = 3
    support_threshold: float | None = None
    k1: float = 1.5
    b: float = 0.75

    def __post_init__(self) -> None:
        _validate_support_threshold(self.support_threshold)
        if self.k1 <= 0:
            raise ValueError("k1 must be positive")
        if not 0.0 <= self.b <= 1.0:
            raise ValueError("b must be between 0 and 1")

    def extract(
        self,
        intent: IntentSpec,
        document: EvidenceDocument,
        *,
        candidate_limit: int | None = None,
        max_matches: int | None = None,
    ) -> ExtractionResult:
        effective_candidate_limit = candidate_limit or self.candidate_limit
        effective_max_matches = max_matches or self.max_matches
        query_terms = tuple(tokenize(_query_text(intent)))
        span_tokens = tuple(tuple(tokenize(span.context_text)) for span in document.spans)
        raw_scores = _bm25_scores(
            query_terms=query_terms,
            documents=span_tokens,
            k1=self.k1,
            b=self.b,
        )
        max_score = max(raw_scores, default=0.0)
        scored = tuple(
            _ScoredSpan(
                span=span,
                score=0.0 if max_score <= 0.0 else score / max_score,
            )
            for span, score in zip(document.spans, raw_scores, strict=True)
        )
        return _result_from_scored_spans(
            intent,
            document,
            scored,
            candidate_limit=effective_candidate_limit,
            max_matches=effective_max_matches,
            feature_name="bm25_normalized_score",
            reason="BM25 lexical retrieval",
            support_threshold=self.support_threshold,
        )


@dataclass(frozen=True)
class EmbeddingRetrieverExtractor:
    embedding_model: TextEmbeddingModel
    candidate_limit: int = 30
    max_matches: int = 3
    support_threshold: float | None = None

    def __post_init__(self) -> None:
        _validate_support_threshold(self.support_threshold)

    def extract(
        self,
        intent: IntentSpec,
        document: EvidenceDocument,
        *,
        candidate_limit: int | None = None,
        max_matches: int | None = None,
    ) -> ExtractionResult:
        effective_candidate_limit = candidate_limit or self.candidate_limit
        effective_max_matches = max_matches or self.max_matches
        query = _query_text(intent)
        vectors = self.embedding_model.encode(
            (query, *(span.context_text for span in document.spans))
        )
        query_vector = vectors[0]
        scored = tuple(
            _ScoredSpan(span=span, score=_cosine(query_vector, vector))
            for span, vector in zip(document.spans, vectors[1:], strict=True)
        )
        return _result_from_scored_spans(
            intent,
            document,
            scored,
            candidate_limit=effective_candidate_limit,
            max_matches=effective_max_matches,
            feature_name=f"embedding_similarity:{self.embedding_model.name}",
            reason=f"embedding retrieval with {self.embedding_model.name}",
            support_threshold=self.support_threshold,
        )


@dataclass(frozen=True)
class CrossEncoderRerankExtractor:
    reranker_model: PairRerankerModel
    first_stage: ExtractorProtocol = field(
        default_factory=lambda: EvidenceExtractor(ranker=KeywordOverlapBaseline())
    )
    first_stage_limit: int = 30
    candidate_limit: int = 30
    max_matches: int = 3
    support_threshold: float | None = None
    require_first_stage_support: bool = False
    preserve_first_stage_labels: bool = False

    def __post_init__(self) -> None:
        _validate_support_threshold(self.support_threshold)

    def extract(
        self,
        intent: IntentSpec,
        document: EvidenceDocument,
        *,
        candidate_limit: int | None = None,
        max_matches: int | None = None,
    ) -> ExtractionResult:
        effective_candidate_limit = candidate_limit or self.candidate_limit
        effective_max_matches = max_matches or self.max_matches
        first_stage_result = self.first_stage.extract(
            intent,
            document,
            candidate_limit=self.first_stage_limit,
            max_matches=self.first_stage_limit,
        )
        first_stage_matches = first_stage_result.candidates
        first_stage_spans = tuple(match.span for match in first_stage_matches)
        query = _query_text(intent)
        raw_scores = self.reranker_model.score_pairs(
            tuple((query, span.context_text) for span in first_stage_spans)
        )
        scored = tuple(
            _ScoredSpan(
                span=match.span,
                score=_sigmoid(score),
                first_stage_label=match.label,
                first_stage_score=match.score,
                first_stage_features=match.features,
                first_stage_reasons=match.reasons,
            )
            for match, score in zip(first_stage_matches, raw_scores, strict=True)
        )
        return _result_from_scored_spans(
            intent,
            document,
            scored,
            candidate_limit=effective_candidate_limit,
            max_matches=effective_max_matches,
            feature_name=f"cross_encoder_score:{self.reranker_model.name}",
            reason=f"cross-encoder reranking with {self.reranker_model.name}",
            support_threshold=self.support_threshold,
            require_first_stage_support=self.require_first_stage_support,
            preserve_first_stage_labels=self.preserve_first_stage_labels,
        )


@dataclass(frozen=True)
class TypedDecisionRerankExtractor:
    reranker_model: PairRerankerModel
    first_stage: ExtractorProtocol = field(default_factory=EvidenceExtractor.default)
    first_stage_limit: int = 30
    candidate_limit: int = 30
    max_matches: int = 3
    first_stage_support_threshold: float | None = None

    def __post_init__(self) -> None:
        _validate_support_threshold(self.first_stage_support_threshold)

    def extract(
        self,
        intent: IntentSpec,
        document: EvidenceDocument,
        *,
        candidate_limit: int | None = None,
        max_matches: int | None = None,
    ) -> ExtractionResult:
        first_stage = (
            self.first_stage
            if self.first_stage_support_threshold is None
            else SupportThresholdOverrideExtractor(
                self.first_stage,
                self.first_stage_support_threshold,
            )
        )
        return CrossEncoderRerankExtractor(
            reranker_model=self.reranker_model,
            first_stage=first_stage,
            first_stage_limit=self.first_stage_limit,
            candidate_limit=self.candidate_limit,
            max_matches=self.max_matches,
            preserve_first_stage_labels=True,
        ).extract(
            intent,
            document,
            candidate_limit=candidate_limit,
            max_matches=max_matches,
        )


@dataclass(frozen=True)
class ExternalRerankerExtractor:
    client: ExternalRerankerClient
    first_stage: EvidenceExtractor = field(
        default_factory=lambda: EvidenceExtractor(ranker=KeywordOverlapBaseline())
    )
    first_stage_limit: int = 50
    candidate_limit: int = 30
    max_matches: int = 3
    support_threshold: float | None = None

    def __post_init__(self) -> None:
        _validate_support_threshold(self.support_threshold)

    def extract(
        self,
        intent: IntentSpec,
        document: EvidenceDocument,
        *,
        candidate_limit: int | None = None,
        max_matches: int | None = None,
    ) -> ExtractionResult:
        effective_candidate_limit = candidate_limit or self.candidate_limit
        effective_max_matches = max_matches or self.max_matches
        first_stage_result = self.first_stage.extract(
            intent,
            document,
            candidate_limit=self.first_stage_limit,
            max_matches=self.first_stage_limit,
        )
        first_stage_spans = tuple(match.span for match in first_stage_result.candidates)
        scores = self.client.rerank(
            query=_query_text(intent),
            documents=tuple(span.context_text for span in first_stage_spans),
            top_n=min(effective_candidate_limit, len(first_stage_spans)),
        )
        scored = tuple(
            _ScoredSpan(span=first_stage_spans[score.index], score=_clamp(score.score))
            for score in scores
            if score.index < len(first_stage_spans)
        )
        return _result_from_scored_spans(
            intent,
            document,
            scored,
            candidate_limit=effective_candidate_limit,
            max_matches=effective_max_matches,
            feature_name=f"external_rerank_score:{self.client.name}",
            reason=f"external reranking with {self.client.name}",
            support_threshold=self.support_threshold,
        )


class SentenceTransformerEmbeddingModel:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        self.name = model_name
        try:
            module = import_module("sentence_transformers")
        except ImportError as exc:
            raise RuntimeError(
                "sentence-transformers is required for SentenceTransformerEmbeddingModel. "
                "Install the model extra or run with `uv run --extra models ...`."
            ) from exc
        sentence_transformer = cast(Any, module.SentenceTransformer)
        try:
            self._model = sentence_transformer(model_name)
        except Exception as exc:
            raise RuntimeError(f"failed to load embedding model {model_name!r}: {exc}") from exc

    def encode(self, texts: Sequence[str]) -> tuple[tuple[float, ...], ...]:
        vectors = self._model.encode(
            list(texts),
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return tuple(tuple(float(value) for value in row.tolist()) for row in vectors)


class SentenceTransformerCrossEncoderModel:
    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        *,
        trust_remote_code: bool = False,
    ) -> None:
        self.name = model_name
        try:
            module = import_module("sentence_transformers")
        except ImportError as exc:
            raise RuntimeError(
                "sentence-transformers is required for SentenceTransformerCrossEncoderModel. "
                "Install the model extra or run with `uv run --extra models ...`."
            ) from exc
        cross_encoder = cast(Any, module.CrossEncoder)
        try:
            self._model = cross_encoder(model_name, trust_remote_code=trust_remote_code)
        except Exception as exc:
            raise RuntimeError(
                f"failed to load cross-encoder model {model_name!r}: {exc}"
            ) from exc

    def score_pairs(self, pairs: Sequence[tuple[str, str]]) -> tuple[float, ...]:
        scores = self._model.predict(list(pairs), show_progress_bar=False)
        return tuple(float(score) for score in scores)


class TransformersSequenceClassificationRerankerModel:
    def __init__(
        self,
        model_name: str,
        *,
        max_length: int = 512,
        device: str = "cpu",
        trust_remote_code: bool = False,
    ) -> None:
        self.name = model_name
        self._max_length = max_length
        self._device = device
        try:
            transformers = import_module("transformers")
            torch = import_module("torch")
        except ImportError as exc:
            raise RuntimeError(
                "transformers and torch are required for "
                "TransformersSequenceClassificationRerankerModel. "
                "Install the model extra or run with `uv run --extra models ...`."
            ) from exc
        tokenizer_cls = cast(Any, transformers.AutoTokenizer)
        model_cls = cast(Any, transformers.AutoModelForSequenceClassification)
        self._torch = torch
        try:
            self._tokenizer = tokenizer_cls.from_pretrained(
                model_name,
                trust_remote_code=trust_remote_code,
            )
            self._model = model_cls.from_pretrained(
                model_name,
                trust_remote_code=trust_remote_code,
            )
        except Exception as exc:
            raise RuntimeError(
                f"failed to load transformers reranker model {model_name!r}: {exc}"
            ) from exc
        self._model.to(device)
        self._model.eval()

    def score_pairs(self, pairs: Sequence[tuple[str, str]]) -> tuple[float, ...]:
        if not pairs:
            return ()
        encoded = self._tokenizer(
            [list(pair) for pair in pairs],
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=self._max_length,
        )
        encoded = {key: value.to(self._device) for key, value in encoded.items()}
        with self._torch.no_grad():
            outputs = self._model(**encoded, return_dict=True)
            scores = outputs.logits.view(-1).float().cpu().tolist()
        return tuple(float(score) for score in scores)


class CohereRerankerClient:
    def __init__(
        self,
        *,
        model_name: str = "rerank-v4.0-pro",
        api_key: str | None = None,
        endpoint: str = "https://api.cohere.com/v2/rerank",
        timeout_seconds: float = 30.0,
    ) -> None:
        self.name = f"cohere:{model_name}"
        self._model_name = model_name
        self._api_key = api_key or os.environ.get("COHERE_API_KEY")
        self._endpoint = endpoint
        self._timeout_seconds = timeout_seconds
        if not self._api_key:
            raise RuntimeError("COHERE_API_KEY is required for CohereRerankerClient")

    def rerank(
        self,
        *,
        query: str,
        documents: Sequence[str],
        top_n: int,
    ) -> tuple[ExternalRerankScore, ...]:
        body = json.dumps(
            {
                "model": self._model_name,
                "query": query,
                "documents": list(documents),
                "top_n": top_n,
            }
        ).encode("utf-8")
        request = urllib.request.Request(
            self._endpoint,
            data=body,
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self._timeout_seconds) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Cohere rerank request failed: {exc}") from exc
        return _cohere_scores(payload)


@dataclass(frozen=True)
class _ScoredSpan:
    span: DocumentSpan
    score: float
    first_stage_label: SupportLabel | None = None
    first_stage_score: float | None = None
    first_stage_features: Mapping[str, float] = field(default_factory=dict)
    first_stage_reasons: tuple[str, ...] = ()


def _result_from_scored_spans(
    intent: IntentSpec,
    document: EvidenceDocument,
    scored: tuple[_ScoredSpan, ...],
    *,
    candidate_limit: int,
    max_matches: int,
    feature_name: str,
    reason: str,
    support_threshold: float | None = None,
    require_first_stage_support: bool = False,
    preserve_first_stage_labels: bool = False,
) -> ExtractionResult:
    ranked = sorted(scored, key=lambda item: item.score, reverse=True)
    candidates = tuple(
        SpanMatch(
            span=item.span,
            label=_label_for_scored_span(
                intent,
                item,
                support_threshold=support_threshold,
                require_first_stage_support=require_first_stage_support,
                preserve_first_stage_labels=preserve_first_stage_labels,
            ),
            score=round(_clamp(item.score), 4),
            features=_candidate_features(item, feature_name=feature_name),
            reasons=_candidate_reasons(item, reason=reason),
        )
        for item in ranked[:candidate_limit]
    )
    selected: list[SpanMatch] = []
    selected_ids: set[str] = set()
    for candidate in candidates:
        if candidate.label != SupportLabel.SUPPORTS:
            continue
        if candidate.span.id in selected_ids:
            continue
        selected.append(candidate)
        selected_ids.add(candidate.span.id)
        if candidate.span.parent_id is not None:
            selected_ids.add(candidate.span.parent_id)
        if len(selected) >= max_matches:
            break
    return ExtractionResult(
        intent=intent,
        document=document,
        matches=tuple(selected),
        candidates=candidates,
        abstained=not selected,
        trace=(f"ranked {len(scored)} spans with {reason}",),
    )


def _query_text(intent: IntentSpec) -> str:
    return "\n".join((intent.label, intent.description, *intent.positive_examples))


def _candidate_features(item: _ScoredSpan, *, feature_name: str) -> dict[str, float]:
    features = {feature_name: round(_clamp(item.score), 4)}
    if item.first_stage_score is not None:
        features["first_stage_score"] = round(_clamp(item.first_stage_score), 4)
    for name, value in item.first_stage_features.items():
        features[f"first_stage:{name}"] = round(float(value), 4)
    return features


def _candidate_reasons(item: _ScoredSpan, *, reason: str) -> tuple[str, ...]:
    return (
        reason,
        *(f"first-stage: {first_stage_reason}" for first_stage_reason in item.first_stage_reasons),
    )


def _label_for_scored_span(
    intent: IntentSpec,
    item: _ScoredSpan,
    *,
    support_threshold: float | None = None,
    require_first_stage_support: bool = False,
    preserve_first_stage_labels: bool = False,
) -> SupportLabel:
    if preserve_first_stage_labels and item.first_stage_label is not None:
        return item.first_stage_label
    label = _label_for_score(intent, item.score, support_threshold=support_threshold)
    if (
        require_first_stage_support
        and label == SupportLabel.SUPPORTS
        and item.first_stage_label != SupportLabel.SUPPORTS
    ):
        return SupportLabel.NEAR_MISS
    return label


def _bm25_scores(
    *,
    query_terms: tuple[str, ...],
    documents: tuple[tuple[str, ...], ...],
    k1: float,
    b: float,
) -> tuple[float, ...]:
    if not query_terms or not documents:
        return tuple(0.0 for _ in documents)
    doc_freqs: Counter[str] = Counter()
    for document in documents:
        doc_freqs.update(set(document))
    doc_count = len(documents)
    avg_doc_len = sum(len(document) for document in documents) / doc_count
    unique_query_terms = tuple(dict.fromkeys(query_terms))
    scores: list[float] = []
    for document in documents:
        term_counts = Counter(document)
        doc_len = len(document)
        score = 0.0
        for term in unique_query_terms:
            term_frequency = term_counts[term]
            if term_frequency == 0:
                continue
            inverse_doc_frequency = math.log(
                1.0 + ((doc_count - doc_freqs[term] + 0.5) / (doc_freqs[term] + 0.5))
            )
            length_norm = 1.0 - b + b * (doc_len / avg_doc_len if avg_doc_len else 0.0)
            score += inverse_doc_frequency * (
                (term_frequency * (k1 + 1.0)) / (term_frequency + k1 * length_norm)
            )
        scores.append(score)
    return tuple(scores)


def _label_for_score(
    intent: IntentSpec,
    score: float,
    *,
    support_threshold: float | None = None,
) -> SupportLabel:
    min_support_score = (
        support_threshold if support_threshold is not None else intent.min_support_score
    )
    if score >= min_support_score:
        return SupportLabel.SUPPORTS
    if score >= min_support_score * intent.near_miss_ratio:
        return SupportLabel.NEAR_MISS
    return SupportLabel.REJECT


def _cosine(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    if len(left) != len(right):
        return 0.0
    numerator = sum(
        left_value * right_value for left_value, right_value in zip(left, right, strict=True)
    )
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return _clamp(numerator / (left_norm * right_norm))


def _sigmoid(value: float) -> float:
    if value >= 0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))


def _cohere_scores(payload: object) -> tuple[ExternalRerankScore, ...]:
    if not isinstance(payload, dict):
        raise RuntimeError("Cohere rerank response must be a JSON object")
    results = payload.get("results")
    if not isinstance(results, list):
        raise RuntimeError("Cohere rerank response missing results list")
    scores: list[ExternalRerankScore] = []
    for result in results:
        if not isinstance(result, dict):
            continue
        index = result.get("index")
        raw_score = result.get("relevance_score", result.get("score"))
        if not isinstance(index, int) or not isinstance(raw_score, int | float):
            continue
        scores.append(ExternalRerankScore(index=index, score=float(raw_score)))
    return tuple(scores)


def _validate_support_threshold(value: float | None) -> None:
    if value is not None and not 0.0 < value <= 1.0:
        raise ValueError("support threshold must be in (0, 1]")
