from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from gia_evidence_finder.contracts import (
    DocumentSpan,
    EvidenceDocument,
    ExtractionResult,
    IntentSpec,
    SpanMatch,
    SupportLabel,
)
from gia_evidence_finder.ranking import IntentAwareRanker, SpanRanker


@dataclass(frozen=True)
class EvidenceExtractor:
    ranker: SpanRanker
    candidate_limit: int = 20
    max_matches: int = 3

    @classmethod
    def default(cls) -> EvidenceExtractor:
        return cls(ranker=IntentAwareRanker())

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
        candidates = self._rank_candidates(intent, document, effective_candidate_limit)
        matches = self._select_matches(intent, candidates, effective_max_matches)
        abstained = not matches
        trace = (
            f"ranked {len(document.spans)} spans",
            f"kept {len(candidates)} candidates",
            "abstained" if abstained else f"selected {len(matches)} support spans",
        )
        return ExtractionResult(
            intent=intent,
            document=document,
            matches=matches,
            candidates=candidates,
            abstained=abstained,
            trace=trace,
        )

    def _rank_candidates(
        self,
        intent: IntentSpec,
        document: EvidenceDocument,
        candidate_limit: int,
    ) -> tuple[SpanMatch, ...]:
        scored: list[SpanMatch] = []
        for span in document.spans:
            score = self.ranker.score(intent, span)
            label = _label_for_score(intent, score.score, score.features)
            final_score = _candidate_score(intent, score.score, label, score.features)
            scored.append(
                SpanMatch(
                    span=span,
                    label=label,
                    score=final_score,
                    features=score.features,
                    reasons=score.reasons,
                )
            )
        has_support = any(match.label == SupportLabel.SUPPORTS for match in scored)
        scored.sort(
            key=lambda match: (
                _label_sort_priority(match.label, has_support=has_support),
                match.score,
                _granularity_sort(match.span),
            ),
            reverse=True,
        )
        return tuple(scored[:candidate_limit])

    def _select_matches(
        self,
        intent: IntentSpec,
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


def _label_for_score(
    intent: IntentSpec,
    score: float,
    features: Mapping[str, float],
) -> SupportLabel:
    if features.get("preferred_span_kind", 1.0) <= 0.0:
        return SupportLabel.REJECT
    if _is_quantifier_flip(features):
        return SupportLabel.CONTRADICTS
    if _is_only_not_trap(features):
        return SupportLabel.NEAR_MISS
    if _is_insufficient_context(features):
        if (
            score >= intent.min_support_score * intent.near_miss_ratio
            or features.get("direct_similarity", 0.0) >= 0.55
            or features.get("relation_coverage", 0.0) >= 1.0
        ):
            return SupportLabel.INSUFFICIENT_CONTEXT
        return SupportLabel.REJECT
    if _is_contradiction(features):
        return SupportLabel.CONTRADICTS
    if _is_unaligned_negative_intent(features):
        if (
            features.get("required_facet_coverage", 0.0) > 0.0
            and features.get("direct_similarity", 0.0) >= 0.45
        ):
            return SupportLabel.CONTRADICTS
        return SupportLabel.REJECT
    if (
        _has_required_quantifier(features)
        and features.get("required_facet_score", 1.0) > 0.0
        and features.get("required_facet_coverage", 1.0) < 0.5
    ):
        return SupportLabel.REJECT
    if _has_required_quantifier(features) and features.get("quantifier_coverage", 1.0) < 1.0:
        if score >= intent.min_support_score * intent.near_miss_ratio:
            return SupportLabel.INSUFFICIENT_CONTEXT
        return SupportLabel.REJECT
    if score >= intent.min_support_score:
        return SupportLabel.SUPPORTS
    if score >= intent.min_support_score * intent.near_miss_ratio:
        return SupportLabel.NEAR_MISS
    return SupportLabel.REJECT


def _is_contradiction(features: Mapping[str, float]) -> bool:
    explicit_counterclaim = (
        features.get("explicit_contradiction_cue", 0.0) > 0.0
        and (
            features.get("required_facet_coverage", 0.0) > 0.0
            or features.get("direct_similarity", 0.0) >= 0.45
            or features.get("context_similarity", 0.0) >= 0.45
        )
    )
    quantifier_flip = _is_quantifier_flip(features)
    strong_polarity_flip = (
        features.get("shared_polarity_anchor_count", 0.0) >= 2.0
        and features.get("polarity_mismatch", 0.0) >= 0.35
        and features.get("contradiction_score", 0.0) >= 0.30
    )
    exclusive_or_modal_flip = (
        features.get("shared_polarity_anchor_count", 0.0) >= 4.0
        and features.get("polarity_mismatch", 0.0) >= 0.25
        and features.get("direct_similarity", 0.0) >= 0.65
        and features.get("span_negated_anchor_coverage", 0.0)
        > features.get("intent_negated_anchor_coverage", 0.0)
    )
    return (
        explicit_counterclaim
        or quantifier_flip
        or strong_polarity_flip
        or exclusive_or_modal_flip
    )


def _is_insufficient_context(features: Mapping[str, float]) -> bool:
    return (
        features.get("insufficient_context_cue", 0.0) > 0.0
        or features.get("missing_relation_keyword_cue", 0.0) > 0.0
    ) and (
        features.get("required_facet_coverage", 0.0) > 0.0
        or features.get("relation_score", 0.0) > 0.0
        or features.get("direct_similarity", 0.0) >= 0.40
    )


def _is_quantifier_flip(features: Mapping[str, float]) -> bool:
    return (
        features.get("quantifier_requirement_count", 0.0) > 0.0
        and features.get("quantifier_contradiction", 0.0) > 0.0
        and features.get("required_facet_coverage", 0.0) > 0.0
    )


def _is_only_not_trap(features: Mapping[str, float]) -> bool:
    return (
        features.get("only_not_trap_cue", 0.0) > 0.0
        and features.get("required_facet_coverage", 0.0) > 0.0
        and features.get("direct_similarity", 0.0) >= 0.55
    )


def _is_unaligned_negative_intent(features: Mapping[str, float]) -> bool:
    return (
        features.get("intent_denies_presence_cue", 0.0) > 0.0
        and features.get("span_denies_presence_cue", 0.0) <= 0.0
        and features.get("intent_negated_anchor_coverage", 0.0) > 0.0
    )


def _has_required_quantifier(features: Mapping[str, float]) -> bool:
    return features.get("quantifier_requirement_count", 0.0) > 0.0


def _candidate_score(
    intent: IntentSpec,
    score: float,
    label: SupportLabel,
    features: Mapping[str, float],
) -> float:
    if label == SupportLabel.SUPPORTS:
        return score
    if label == SupportLabel.REJECT:
        if features.get("preferred_span_kind", 1.0) <= 0.0:
            return round(min(score, 0.05), 4)
        return score
    if label == SupportLabel.INSUFFICIENT_CONTEXT:
        diagnostic_floor = min(
            0.74,
            0.28
            + (0.14 * features.get("insufficient_context_cue", 0.0))
            + (0.18 * features.get("missing_relation_keyword_cue", 0.0))
            + (0.06 * features.get("relation_coverage", 0.0))
            + (0.10 * features.get("required_facet_coverage", 0.0))
            + (0.05 * features.get("direct_similarity", 0.0)),
        )
        return round(max(score, diagnostic_floor), 4)
    if label == SupportLabel.NEAR_MISS:
        diagnostic_floor = min(
            0.68,
            0.34
            + (0.20 * features.get("only_not_trap_cue", 0.0))
            + (0.06 * features.get("negative_similarity", 0.0))
            + (0.15 * features.get("direct_similarity", 0.0)),
        )
        return round(max(score, diagnostic_floor), 4)
    diagnostic_floor = min(
        0.68,
        0.30
        + (0.22 * features.get("explicit_contradiction_cue", 0.0))
        + (0.16 * features.get("quantifier_contradiction", 0.0))
        + (0.12 * features.get("contradiction_score", 0.0))
        + (0.10 * features.get("required_facet_coverage", 0.0))
        + (0.06 * features.get("direct_similarity", 0.0)),
    )
    return round(max(score, diagnostic_floor), 4)


def _conflicts_with_selected(span: DocumentSpan, selected_ids: set[str]) -> bool:
    if span.id in selected_ids:
        return True
    return span.parent_id is not None and span.parent_id in selected_ids


def _granularity_sort(span: DocumentSpan) -> int:
    if span.parent_id is not None:
        return 2
    return 1


def _label_sort_priority(label: SupportLabel, *, has_support: bool) -> int:
    if has_support:
        return 1 if label == SupportLabel.SUPPORTS else 0
    if label in {
        SupportLabel.CONTRADICTS,
        SupportLabel.NEAR_MISS,
        SupportLabel.INSUFFICIENT_CONTEXT,
    }:
        return 1
    return 0
