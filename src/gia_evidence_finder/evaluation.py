from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from gia_evidence_finder.contracts import (
    BenchmarkCase,
    EvidenceDocument,
    ExtractionResult,
    IntentSpec,
    SpanMatch,
    SupportLabel,
)

NEGATIVE_DIAGNOSTIC_LABELS: tuple[SupportLabel, ...] = (
    SupportLabel.NEAR_MISS,
    SupportLabel.CONTRADICTS,
    SupportLabel.INSUFFICIENT_CONTEXT,
)


@dataclass(frozen=True)
class CandidateScore:
    span_id: str
    score: float
    label: SupportLabel


@dataclass(frozen=True)
class EvaluationReport:
    case_count: int
    support_case_count: int
    abstain_case_count: int
    mean_reciprocal_rank: float
    recall_at_1: float
    recall_at_3: float
    recall_at_5: float
    top1_support_accuracy: float
    abstain_accuracy: float
    decision_accuracy: float
    forbidden_top1_rate: float
    forbidden_supported_top1_rate: float
    negative_label_reports: tuple[NegativeLabelReport, ...] = ()


@dataclass(frozen=True)
class NegativeLabelReport:
    label: SupportLabel
    case_count: int
    span_label_count: int
    top1_rate: float
    supported_top1_rate: float


@dataclass(frozen=True)
class NegativeLabelEvaluation:
    label: SupportLabel
    span_ids: tuple[str, ...]
    top1: bool
    supported_top1: bool


@dataclass(frozen=True)
class CaseEvaluation:
    case_id: str
    document_id: str
    expect_abstain: bool
    support_span_ids: tuple[str, ...]
    forbidden_span_ids: tuple[str, ...]
    ranked_span_ids: tuple[str, ...]
    candidate_scores: tuple[CandidateScore, ...]
    matched_span_ids: tuple[str, ...]
    top_span_id: str | None
    top_label: str | None
    top_score: float | None
    first_support_rank: int | None
    top1_is_support: bool
    abstained: bool
    decision_correct: bool
    forbidden_top1: bool
    forbidden_supported_top1: bool
    negative_label_evaluations: tuple[NegativeLabelEvaluation, ...] = ()


@dataclass(frozen=True)
class DetailedEvaluationReport:
    summary: EvaluationReport
    cases: tuple[CaseEvaluation, ...]

    @property
    def failures(self) -> tuple[CaseEvaluation, ...]:
        return tuple(
            case
            for case in self.cases
            if not case.decision_correct
            or (case.support_span_ids and not case.top1_is_support)
            or case.forbidden_supported_top1
            or any(
                evaluation.supported_top1
                for evaluation in case.negative_label_evaluations
            )
        )


class ExtractorProtocol(Protocol):
    def extract(
        self,
        intent: IntentSpec,
        document: EvidenceDocument,
        *,
        candidate_limit: int | None = None,
        max_matches: int | None = None,
    ) -> ExtractionResult: ...


def evaluate_suite(
    extractor: ExtractorProtocol,
    cases: tuple[BenchmarkCase, ...],
) -> EvaluationReport:
    return evaluate_suite_detailed(extractor, cases).summary


def evaluate_suite_detailed(
    extractor: ExtractorProtocol,
    cases: tuple[BenchmarkCase, ...],
) -> DetailedEvaluationReport:
    if not cases:
        raise ValueError("evaluation suite must include at least one case")
    reciprocal_ranks: list[float] = []
    recall_1 = 0
    recall_3 = 0
    recall_5 = 0
    top1_support = 0
    abstain_correct = 0
    decision_correct = 0
    forbidden_top1 = 0
    forbidden_supported_top1 = 0
    support_case_count = 0
    abstain_case_count = 0
    case_evaluations: list[CaseEvaluation] = []
    for case in cases:
        result = extractor.extract(case.intent, case.document, candidate_limit=30)
        ranked_ids = tuple(match.span.id for match in result.candidates)
        first_rank = _first_support_rank(ranked_ids, case.support_span_ids)
        case_evaluation = _case_evaluation(case, result, ranked_ids, first_rank)
        case_evaluations.append(case_evaluation)
        if case.support_span_ids:
            support_case_count += 1
            reciprocal_ranks.append(0.0 if first_rank is None else 1.0 / first_rank)
            recall_1 += int(_has_support(ranked_ids[:1], case.support_span_ids))
            recall_3 += int(_has_support(ranked_ids[:3], case.support_span_ids))
            recall_5 += int(_has_support(ranked_ids[:5], case.support_span_ids))
            top1_support += int(_top1_is_support(result.matches, case.support_span_ids))
        if case.expect_abstain:
            abstain_case_count += 1
            abstain_correct += int(result.abstained)
        decision_correct += int(case_evaluation.decision_correct)
        if case_evaluation.forbidden_top1:
            forbidden_top1 += 1
        if case_evaluation.forbidden_supported_top1:
            forbidden_supported_top1 += 1
    count = len(cases)
    return DetailedEvaluationReport(
        summary=EvaluationReport(
            case_count=count,
            support_case_count=support_case_count,
            abstain_case_count=abstain_case_count,
            mean_reciprocal_rank=_round_rate(sum(reciprocal_ranks), support_case_count),
            recall_at_1=_round_rate(recall_1, support_case_count),
            recall_at_3=_round_rate(recall_3, support_case_count),
            recall_at_5=_round_rate(recall_5, support_case_count),
            top1_support_accuracy=_round_rate(top1_support, support_case_count),
            abstain_accuracy=_round_rate(abstain_correct, abstain_case_count, empty_value=1.0),
            decision_accuracy=_round_rate(decision_correct, count),
            forbidden_top1_rate=_round_rate(forbidden_top1, count),
            forbidden_supported_top1_rate=_round_rate(forbidden_supported_top1, count),
            negative_label_reports=_negative_label_reports(case_evaluations),
        ),
        cases=tuple(case_evaluations),
    )


def _first_support_rank(ranked_ids: tuple[str, ...], support_ids: tuple[str, ...]) -> int | None:
    if not support_ids:
        return None
    support = set(support_ids)
    for index, span_id in enumerate(ranked_ids, start=1):
        if span_id in support:
            return index
    return None


def _has_support(ranked_ids: tuple[str, ...], support_ids: tuple[str, ...]) -> bool:
    if not support_ids:
        return False
    support = set(support_ids)
    return any(span_id in support for span_id in ranked_ids)


def _top1_is_support(matches: tuple[SpanMatch, ...], support_ids: tuple[str, ...]) -> bool:
    return bool(matches and matches[0].span.id in set(support_ids))


def _case_evaluation(
    case: BenchmarkCase,
    result: ExtractionResult,
    ranked_ids: tuple[str, ...],
    first_support_rank: int | None,
) -> CaseEvaluation:
    top_candidate = result.candidates[0] if result.candidates else None
    negative_label_evaluations = _negative_label_evaluations(case, top_candidate, ranked_ids)
    return CaseEvaluation(
        case_id=case.id,
        document_id=case.document.id,
        expect_abstain=case.expect_abstain,
        support_span_ids=case.support_span_ids,
        forbidden_span_ids=case.forbidden_span_ids,
        ranked_span_ids=ranked_ids,
        candidate_scores=tuple(
            CandidateScore(span_id=match.span.id, score=match.score, label=match.label)
            for match in result.candidates
        ),
        matched_span_ids=tuple(match.span.id for match in result.matches),
        top_span_id=top_candidate.span.id if top_candidate is not None else None,
        top_label=top_candidate.label.value if top_candidate is not None else None,
        top_score=top_candidate.score if top_candidate is not None else None,
        first_support_rank=first_support_rank,
        top1_is_support=_has_support(ranked_ids[:1], case.support_span_ids),
        abstained=result.abstained,
        decision_correct=result.abstained == case.expect_abstain,
        forbidden_top1=bool(ranked_ids and ranked_ids[0] in case.forbidden_span_ids),
        forbidden_supported_top1=bool(
            top_candidate is not None
            and top_candidate.span.id in case.forbidden_span_ids
            and top_candidate.label.value == "supports"
        ),
        negative_label_evaluations=negative_label_evaluations,
    )


def _negative_label_evaluations(
    case: BenchmarkCase,
    top_candidate: SpanMatch | None,
    ranked_ids: tuple[str, ...],
) -> tuple[NegativeLabelEvaluation, ...]:
    top_span_id = ranked_ids[0] if ranked_ids else None
    return tuple(
        NegativeLabelEvaluation(
            label=label,
            span_ids=span_ids,
            top1=top_span_id in span_ids,
            supported_top1=bool(
                top_candidate is not None
                and top_candidate.span.id in span_ids
                and top_candidate.label == SupportLabel.SUPPORTS
            ),
        )
        for label, span_ids in _negative_label_span_ids(case)
    )


def _negative_label_span_ids(
    case: BenchmarkCase,
) -> tuple[tuple[SupportLabel, tuple[str, ...]], ...]:
    span_ids_by_label = {
        SupportLabel.NEAR_MISS: case.near_miss_span_ids,
        SupportLabel.CONTRADICTS: case.contradiction_span_ids,
        SupportLabel.INSUFFICIENT_CONTEXT: case.insufficient_context_span_ids,
    }
    return tuple((label, span_ids_by_label[label]) for label in NEGATIVE_DIAGNOSTIC_LABELS)


def _negative_label_reports(
    cases: list[CaseEvaluation] | tuple[CaseEvaluation, ...],
) -> tuple[NegativeLabelReport, ...]:
    reports: list[NegativeLabelReport] = []
    for label in NEGATIVE_DIAGNOSTIC_LABELS:
        evaluations = tuple(
            evaluation
            for case in cases
            for evaluation in case.negative_label_evaluations
            if evaluation.label == label and evaluation.span_ids
        )
        case_count = len(evaluations)
        reports.append(
            NegativeLabelReport(
                label=label,
                case_count=case_count,
                span_label_count=sum(len(evaluation.span_ids) for evaluation in evaluations),
                top1_rate=_round_rate(
                    sum(1 for evaluation in evaluations if evaluation.top1),
                    case_count,
                ),
                supported_top1_rate=_round_rate(
                    sum(1 for evaluation in evaluations if evaluation.supported_top1),
                    case_count,
                ),
            )
        )
    return tuple(reports)


def _round_rate(numerator: float, denominator: int, *, empty_value: float = 0.0) -> float:
    if denominator == 0:
        return empty_value
    return round(numerator / denominator, 4)
