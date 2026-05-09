from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import Any

from gia_evidence_finder.contracts import BenchmarkCase, BenchmarkSuite, SupportLabel, TrainingPair
from gia_evidence_finder.evaluation import ExtractorProtocol

DEFAULT_RERANKER_LABEL_SCORES: Mapping[SupportLabel, float] = MappingProxyType(
    {
        SupportLabel.SUPPORTS: 1.0,
        SupportLabel.NEAR_MISS: 0.0,
        SupportLabel.CONTRADICTS: 0.0,
        SupportLabel.INSUFFICIENT_CONTEXT: 0.0,
        SupportLabel.REJECT: 0.0,
    }
)


@dataclass(frozen=True)
class RerankerTrainingExample:
    query: str
    text: str
    label_score: float
    source_label: SupportLabel
    intent_id: str
    span_id: str
    metadata: Mapping[str, str]

    def __post_init__(self) -> None:
        if not self.query.strip():
            raise ValueError("training example query must not be empty")
        if not self.text.strip():
            raise ValueError("training example text must not be empty")
        if not 0.0 <= self.label_score <= 1.0:
            raise ValueError("training example label_score must be between 0 and 1")
        if not self.intent_id.strip():
            raise ValueError("training example intent_id must not be empty")
        if not self.span_id.strip():
            raise ValueError("training example span_id must not be empty")
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))


def hard_negative_pairs(
    extractor: ExtractorProtocol,
    cases: tuple[BenchmarkCase, ...],
    *,
    negatives_per_case: int = 3,
) -> tuple[TrainingPair, ...]:
    pairs: list[TrainingPair] = []
    for case in cases:
        result = extractor.extract(case.intent, case.document, candidate_limit=30)
        support_ids = set(case.support_span_ids)
        near_miss_ids = set(case.near_miss_span_ids)
        contradiction_ids = set(case.contradiction_span_ids)
        insufficient_context_ids = set(case.insufficient_context_span_ids)
        forbidden_ids = set(case.forbidden_span_ids)
        for span_id in support_ids:
            span = case.document.span_by_id(span_id)
            pairs.append(
                TrainingPair(
                    intent_id=case.intent.id,
                    span_id=span.id,
                    text=span.context_text,
                    label=SupportLabel.SUPPORTS,
                    query=_query_text(case),
                    metadata=_base_metadata(case, source="support"),
                )
            )
        negatives = 0
        for candidate in result.candidates:
            if candidate.span.id in support_ids:
                continue
            if candidate.span.id in near_miss_ids:
                label = SupportLabel.NEAR_MISS
            elif candidate.span.id in contradiction_ids:
                label = SupportLabel.CONTRADICTS
            elif candidate.span.id in insufficient_context_ids:
                label = SupportLabel.INSUFFICIENT_CONTEXT
            elif candidate.span.id in forbidden_ids:
                label = SupportLabel.REJECT
            else:
                label = SupportLabel.REJECT
            pairs.append(
                TrainingPair(
                    intent_id=case.intent.id,
                    span_id=candidate.span.id,
                    text=candidate.span.context_text,
                    label=label,
                    query=_query_text(case),
                    score=candidate.score,
                    metadata=_base_metadata(case, source="hard_negative"),
                )
            )
            negatives += 1
            if negatives >= negatives_per_case:
                break
    return tuple(pairs)


def training_pairs_from_suite(
    extractor: ExtractorProtocol,
    suite: BenchmarkSuite,
    *,
    negatives_per_case: int = 3,
) -> tuple[TrainingPair, ...]:
    return tuple(
        replace(
            pair,
            metadata={
                **pair.metadata,
                "suite_id": suite.id,
                "suite_name": suite.name,
            },
        )
        for pair in hard_negative_pairs(
            extractor,
            suite.cases,
            negatives_per_case=negatives_per_case,
        )
    )


def training_pairs_jsonl(pairs: tuple[TrainingPair, ...]) -> str:
    return "\n".join(json.dumps(_pair_payload(pair), sort_keys=True) for pair in pairs)


def reranker_training_examples_from_jsonl(
    jsonl_text: str,
    *,
    label_scores: Mapping[SupportLabel, float] = DEFAULT_RERANKER_LABEL_SCORES,
) -> tuple[RerankerTrainingExample, ...]:
    examples: list[RerankerTrainingExample] = []
    for line_number, line in enumerate(jsonl_text.splitlines(), start=1):
        if not line.strip():
            continue
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise ValueError(f"line {line_number}: expected a JSON object")
        label = SupportLabel(_required_string(payload, "label", line_number))
        if label not in label_scores:
            raise ValueError(f"line {line_number}: no score configured for label {label.value!r}")
        examples.append(
            RerankerTrainingExample(
                query=_required_string(payload, "query", line_number),
                text=_required_string(payload, "text", line_number),
                label_score=label_scores[label],
                source_label=label,
                intent_id=_required_string(payload, "intent_id", line_number),
                span_id=_required_string(payload, "span_id", line_number),
                metadata=_metadata(payload.get("metadata", {}), line_number),
            )
        )
    if not examples:
        raise ValueError("training JSONL did not contain any examples")
    return tuple(examples)


def _query_text(case: BenchmarkCase) -> str:
    return "\n".join(case.intent.query_texts)


def _base_metadata(case: BenchmarkCase, *, source: str) -> dict[str, str]:
    metadata = {
        "case_id": case.id,
        "document_id": case.document.id,
        "source": source,
        "expect_abstain": str(case.expect_abstain).lower(),
    }
    if case.document.source:
        metadata["document_source"] = case.document.source
    return metadata


def _pair_payload(pair: TrainingPair) -> dict[str, object]:
    return {
        "intent_id": pair.intent_id,
        "query": pair.query,
        "span_id": pair.span_id,
        "text": pair.text,
        "label": pair.label.value,
        "score": pair.score,
        "metadata": dict(pair.metadata),
    }


def _required_string(payload: dict[Any, Any], key: str, line_number: int) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"line {line_number}: {key} must be a non-empty string")
    return value


def _metadata(value: object, line_number: int) -> dict[str, str]:
    if not isinstance(value, dict):
        raise ValueError(f"line {line_number}: metadata must be an object")
    return {str(key): str(item) for key, item in value.items()}
