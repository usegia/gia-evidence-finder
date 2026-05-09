from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from gia_evidence_finder.contracts import BenchmarkCase, BenchmarkSuite
from gia_evidence_finder.evaluation import CaseEvaluation


@dataclass(frozen=True)
class FailureCandidate:
    span_id: str
    score: float
    text: str

    def to_json_dict(self) -> dict[str, object]:
        return {
            "span_id": self.span_id,
            "score": self.score,
            "text": self.text,
        }


@dataclass(frozen=True)
class FailureCase:
    case_id: str
    document_id: str
    buckets: tuple[str, ...]
    intent_label: str
    expect_abstain: bool
    support_span_ids: tuple[str, ...]
    matched_span_ids: tuple[str, ...]
    top_span_id: str | None
    top_label: str | None
    top_score: float | None
    first_support_rank: int | None
    support_texts: tuple[str, ...]
    matched_texts: tuple[str, ...]
    top_text: str | None
    top_candidates: tuple[FailureCandidate, ...]
    curation_source: str
    difficulty: str
    phenomena: tuple[str, ...]

    def to_json_dict(self) -> dict[str, object]:
        return {
            "case_id": self.case_id,
            "document_id": self.document_id,
            "buckets": list(self.buckets),
            "intent_label": self.intent_label,
            "expect_abstain": self.expect_abstain,
            "support_span_ids": list(self.support_span_ids),
            "matched_span_ids": list(self.matched_span_ids),
            "top_span_id": self.top_span_id,
            "top_label": self.top_label,
            "top_score": self.top_score,
            "first_support_rank": self.first_support_rank,
            "support_texts": list(self.support_texts),
            "matched_texts": list(self.matched_texts),
            "top_text": self.top_text,
            "top_candidates": [candidate.to_json_dict() for candidate in self.top_candidates],
            "curation_source": self.curation_source,
            "difficulty": self.difficulty,
            "phenomena": list(self.phenomena),
        }


@dataclass(frozen=True)
class FailureReport:
    case_count: int
    failure_count: int
    bucket_counts: dict[str, int]
    failures: tuple[FailureCase, ...]

    def to_json_dict(self) -> dict[str, object]:
        return {
            "case_count": self.case_count,
            "failure_count": self.failure_count,
            "bucket_counts": self.bucket_counts,
            "failures": [failure.to_json_dict() for failure in self.failures],
        }


def build_failure_report(
    suite: BenchmarkSuite,
    evaluations: tuple[CaseEvaluation, ...],
) -> FailureReport:
    cases_by_id = {case.id: case for case in suite.cases}
    failures: list[FailureCase] = []
    for evaluation in evaluations:
        case = cases_by_id[evaluation.case_id]
        buckets = _failure_buckets(evaluation)
        if not buckets:
            continue
        failures.append(_failure_case(case, evaluation, buckets))
    bucket_counts: Counter[str] = Counter()
    for failure in failures:
        bucket_counts.update(failure.buckets)
    return FailureReport(
        case_count=len(evaluations),
        failure_count=len(failures),
        bucket_counts=dict(sorted(bucket_counts.items())),
        failures=tuple(failures),
    )


def _failure_buckets(evaluation: CaseEvaluation) -> tuple[str, ...]:
    buckets: list[str] = []
    if not evaluation.decision_correct:
        buckets.append("decision_error")
    if evaluation.support_span_ids and not evaluation.top1_is_support:
        buckets.append("top_rank_missed_support")
    if evaluation.forbidden_top1:
        buckets.append("forbidden_top_rank")
    if evaluation.forbidden_supported_top1:
        buckets.append("forbidden_supported_top_rank")
    for negative in evaluation.negative_label_evaluations:
        if negative.top1:
            buckets.append(f"{negative.label.value}_top_rank")
        if negative.supported_top1:
            buckets.append(f"{negative.label.value}_supported_top_rank")
    return tuple(dict.fromkeys(buckets))


def _failure_case(
    case: BenchmarkCase,
    evaluation: CaseEvaluation,
    buckets: tuple[str, ...],
) -> FailureCase:
    return FailureCase(
        case_id=case.id,
        document_id=case.document.id,
        buckets=buckets,
        intent_label=case.intent.label,
        expect_abstain=case.expect_abstain,
        support_span_ids=case.support_span_ids,
        matched_span_ids=evaluation.matched_span_ids,
        top_span_id=evaluation.top_span_id,
        top_label=evaluation.top_label,
        top_score=evaluation.top_score,
        first_support_rank=evaluation.first_support_rank,
        support_texts=_span_texts(case, case.support_span_ids),
        matched_texts=_span_texts(case, evaluation.matched_span_ids),
        top_text=_span_text(case, evaluation.top_span_id),
        top_candidates=tuple(
            FailureCandidate(
                span_id=candidate.span_id,
                score=candidate.score,
                text=_span_text(case, candidate.span_id) or "",
            )
            for candidate in evaluation.candidate_scores[:5]
        ),
        curation_source=case.curation.source,
        difficulty=case.curation.difficulty,
        phenomena=case.curation.phenomena,
    )


def _span_texts(case: BenchmarkCase, span_ids: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(text for span_id in span_ids if (text := _span_text(case, span_id)) is not None)


def _span_text(case: BenchmarkCase, span_id: str | None) -> str | None:
    if span_id is None:
        return None
    try:
        return case.document.span_by_id(span_id).text
    except KeyError:
        return None
