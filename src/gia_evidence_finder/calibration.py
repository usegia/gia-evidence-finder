from __future__ import annotations

from dataclasses import dataclass, replace

from gia_evidence_finder.contracts import (
    DocumentSpan,
    EvidenceDocument,
    ExtractionResult,
    IntentSpec,
    SpanMatch,
    SupportLabel,
)
from gia_evidence_finder.evaluation import (
    NEGATIVE_DIAGNOSTIC_LABELS,
    CandidateScore,
    CaseEvaluation,
    DetailedEvaluationReport,
    EvaluationReport,
    ExtractorProtocol,
    NegativeLabelReport,
)

DEFAULT_THRESHOLDS: tuple[float, ...] = (
    0.05,
    0.10,
    0.15,
    0.20,
    0.25,
    0.30,
    0.35,
    0.40,
    0.45,
    0.50,
    0.55,
    0.60,
    0.65,
    0.70,
    0.75,
    0.80,
    0.85,
    0.90,
    0.95,
)


@dataclass(frozen=True)
class CalibrationPoint:
    threshold: float
    report: EvaluationReport


@dataclass(frozen=True)
class CalibrationReport:
    points: tuple[CalibrationPoint, ...]
    best: CalibrationPoint


@dataclass(frozen=True)
class SupportThresholdOverrideExtractor:
    extractor: ExtractorProtocol
    support_threshold: float

    def __post_init__(self) -> None:
        if not 0.0 < self.support_threshold <= 1.0:
            raise ValueError("support_threshold must be in (0, 1]")

    def extract(
        self,
        intent: IntentSpec,
        document: EvidenceDocument,
        *,
        candidate_limit: int | None = None,
        max_matches: int | None = None,
    ) -> ExtractionResult:
        adjusted_intent = replace(intent, min_support_score=self.support_threshold)
        return self.extractor.extract(
            adjusted_intent,
            document,
            candidate_limit=candidate_limit,
            max_matches=max_matches,
        )


@dataclass(frozen=True)
class CandidateScoreThresholdExtractor:
    extractor: ExtractorProtocol
    support_threshold: float
    max_matches: int = 3

    def __post_init__(self) -> None:
        if not 0.0 < self.support_threshold <= 1.0:
            raise ValueError("support_threshold must be in (0, 1]")
        if self.max_matches <= 0:
            raise ValueError("max_matches must be positive")

    def extract(
        self,
        intent: IntentSpec,
        document: EvidenceDocument,
        *,
        candidate_limit: int | None = None,
        max_matches: int | None = None,
    ) -> ExtractionResult:
        effective_max_matches = max_matches or self.max_matches
        result = self.extractor.extract(
            intent,
            document,
            candidate_limit=candidate_limit,
            max_matches=max_matches,
        )
        candidates = tuple(
            SpanMatch(
                span=candidate.span,
                label=_threshold_label_for_candidate(
                    intent,
                    candidate,
                    support_threshold=self.support_threshold,
                ),
                score=candidate.score,
                features=candidate.features,
                reasons=candidate.reasons,
            )
            for candidate in result.candidates
        )
        matches = _select_thresholded_matches(candidates, effective_max_matches)
        return ExtractionResult(
            intent=result.intent,
            document=result.document,
            matches=matches,
            candidates=candidates,
            abstained=not matches,
            trace=(
                *result.trace,
                f"applied calibrated support threshold {self.support_threshold:.2f}",
            ),
        )


def calibrate_thresholds(
    detailed_report: DetailedEvaluationReport,
    *,
    thresholds: tuple[float, ...] = DEFAULT_THRESHOLDS,
) -> CalibrationReport:
    if not thresholds:
        raise ValueError("thresholds must not be empty")
    points = tuple(
        CalibrationPoint(
            threshold=threshold,
            report=_report_at_threshold(detailed_report.cases, threshold),
        )
        for threshold in thresholds
    )
    return CalibrationReport(points=points, best=max(points, key=_objective_key))


def _report_at_threshold(
    cases: tuple[CaseEvaluation, ...],
    threshold: float,
) -> EvaluationReport:
    if not 0.0 < threshold <= 1.0:
        raise ValueError("threshold must be in (0, 1]")
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
    for case in cases:
        support_ids = set(case.support_span_ids)
        ranked_ids = case.ranked_span_ids
        selected_ids = tuple(
            candidate.span_id
            for candidate in case.candidate_scores
            if _candidate_is_threshold_support(candidate, threshold)
        )
        if case.support_span_ids:
            support_case_count += 1
            reciprocal_ranks.append(
                0.0 if case.first_support_rank is None else 1.0 / case.first_support_rank
            )
            recall_1 += int(_has_support(ranked_ids[:1], support_ids))
            recall_3 += int(_has_support(ranked_ids[:3], support_ids))
            recall_5 += int(_has_support(ranked_ids[:5], support_ids))
            top1_support += int(bool(selected_ids and selected_ids[0] in support_ids))
        abstained = not selected_ids
        if case.expect_abstain:
            abstain_case_count += 1
            abstain_correct += int(abstained)
        decision_correct += int(abstained == case.expect_abstain)
        if case.forbidden_top1:
            forbidden_top1 += 1
        if _top_candidate_is_forbidden_support(case, threshold):
            forbidden_supported_top1 += 1
    count = len(cases)
    return EvaluationReport(
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
        negative_label_reports=_negative_label_reports_at_threshold(cases, threshold),
    )


def _objective_key(point: CalibrationPoint) -> tuple[float, float, float, float, float, float]:
    report = point.report
    return (
        report.decision_accuracy,
        report.top1_support_accuracy,
        report.abstain_accuracy,
        -report.forbidden_supported_top1_rate,
        report.mean_reciprocal_rank,
        -point.threshold,
    )


def _top_candidate_is_forbidden_support(case: CaseEvaluation, threshold: float) -> bool:
    if not case.candidate_scores:
        return False
    top_candidate = case.candidate_scores[0]
    return top_candidate.span_id in case.forbidden_span_ids and _candidate_is_threshold_support(
        top_candidate,
        threshold,
    )


def _negative_label_reports_at_threshold(
    cases: tuple[CaseEvaluation, ...],
    threshold: float,
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
                    sum(
                        1
                        for case in cases
                        for evaluation in case.negative_label_evaluations
                        if evaluation.label == label
                        and evaluation.span_ids
                        and _top_candidate_is_label_support(case, evaluation.span_ids, threshold)
                    ),
                    case_count,
                ),
            )
        )
    return tuple(reports)


def _top_candidate_is_label_support(
    case: CaseEvaluation,
    span_ids: tuple[str, ...],
    threshold: float,
) -> bool:
    if not case.candidate_scores:
        return False
    top_candidate = case.candidate_scores[0]
    return top_candidate.span_id in span_ids and _candidate_is_threshold_support(
        top_candidate,
        threshold,
    )


def _candidate_is_threshold_support(candidate: CandidateScore, threshold: float) -> bool:
    return candidate.label != SupportLabel.CONTRADICTS and candidate.score >= threshold


def _label_for_score(
    intent: IntentSpec,
    score: float,
    *,
    support_threshold: float,
) -> SupportLabel:
    if score >= support_threshold:
        return SupportLabel.SUPPORTS
    if score >= support_threshold * intent.near_miss_ratio:
        return SupportLabel.NEAR_MISS
    return SupportLabel.REJECT


def _threshold_label_for_candidate(
    intent: IntentSpec,
    candidate: SpanMatch,
    *,
    support_threshold: float,
) -> SupportLabel:
    if candidate.label == SupportLabel.CONTRADICTS:
        return SupportLabel.CONTRADICTS
    return _label_for_score(intent, candidate.score, support_threshold=support_threshold)


def _select_thresholded_matches(
    candidates: tuple[SpanMatch, ...],
    max_matches: int,
) -> tuple[SpanMatch, ...]:
    selected: list[SpanMatch] = []
    selected_ids: set[str] = set()
    for candidate in candidates:
        if candidate.label != SupportLabel.SUPPORTS:
            continue
        if _conflicts_with_selected(candidate.span, selected_ids):
            continue
        selected.append(candidate)
        selected_ids.add(candidate.span.id)
        if candidate.span.parent_id is not None:
            selected_ids.add(candidate.span.parent_id)
        if len(selected) >= max_matches:
            break
    return tuple(selected)


def _conflicts_with_selected(span: DocumentSpan, selected_ids: set[str]) -> bool:
    if span.id in selected_ids:
        return True
    return span.parent_id is not None and span.parent_id in selected_ids


def _has_support(ranked_ids: tuple[str, ...], support_ids: set[str]) -> bool:
    return bool(support_ids and any(span_id in support_ids for span_id in ranked_ids))


def _round_rate(numerator: float, denominator: int, *, empty_value: float = 0.0) -> float:
    if denominator == 0:
        return empty_value
    return round(numerator / denominator, 4)
