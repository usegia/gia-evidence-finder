from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Protocol

from gia_evidence_finder.contracts import (
    DocumentSpan,
    EvidenceFacet,
    EvidenceRelation,
    IntentSpec,
    SpanKind,
)
from gia_evidence_finder.polarity import PolarityScore, polarity_score
from gia_evidence_finder.quantifiers import quantifier_score
from gia_evidence_finder.text import exact_phrase_score, max_similarity, soft_similarity, tokenize


@dataclass(frozen=True)
class ScoreResult:
    score: float
    features: dict[str, float]
    reasons: tuple[str, ...]


class SpanRanker(Protocol):
    def score(self, intent: IntentSpec, span: DocumentSpan) -> ScoreResult: ...


@dataclass(frozen=True)
class IntentAwareRanker:
    """Deterministic high-quality starter ranker.

    This ranker is deliberately not BM25. It combines typed intent examples,
    required facets, excluded facets, heading context, span-kind priors, and
    soft lexical similarity. A future model can replace this class through the
    SpanRanker protocol.
    """

    facet_match_threshold: float = 0.18

    def score(self, intent: IntentSpec, span: DocumentSpan) -> ScoreResult:
        direct_text = span.text
        context_text = span.context_text
        direct_similarity = max_similarity(intent.query_texts, direct_text)
        context_similarity = max_similarity(intent.query_texts, context_text)
        positive_similarity = max_similarity(intent.positive_examples, context_text)
        negative_similarity = max_similarity(intent.negative_examples, context_text)
        required_score, required_coverage = _facet_score(
            intent.required_facets,
            context_text,
            threshold=self.facet_match_threshold,
        )
        relation_score, relation_coverage, relation_bridge_score = _relation_score(
            intent.relations,
            context_text,
            threshold=self.facet_match_threshold,
        )
        quantifiers = quantifier_score(intent.quantifier_requirements, context_text)
        if intent.excluded_facets:
            excluded_score, _ = _facet_score(
                intent.excluded_facets,
                context_text,
                threshold=self.facet_match_threshold,
            )
        else:
            excluded_score = 0.0
        heading_similarity = soft_similarity(intent.description, " ".join(span.heading_path))
        exact_score = max(exact_phrase_score(query, context_text) for query in intent.query_texts)
        polarity = polarity_score(_polarity_intent_texts(intent), context_text)
        safety_cues = _safety_cue_features(intent, context_text)
        kind_prior = _kind_prior(intent, span.kind)
        preferred_span_kind = 1.0 if span.kind in intent.preferred_span_kinds else 0.0
        granularity_bonus = 0.04 if span.kind == SpanKind.SENTENCE else 0.0
        missing_required_penalty = 0.22 * (1.0 - required_coverage)
        relation_bonus = 0.18 * relation_score if intent.relations else 0.0
        missing_relation_penalty = (
            0.24 * (1.0 - relation_coverage) if intent.relations else 0.0
        )
        negative_penalty = 0.22 * negative_similarity
        excluded_penalty = 0.28 * excluded_score
        relation_bridge_penalty = 0.65 * relation_bridge_score
        missing_quantifier_penalty = 0.32 * (1.0 - quantifiers.coverage)
        quantifier_mismatch_penalty = 0.55 * quantifiers.mismatch
        polarity_bonus = (
            0.04 * polarity.alignment if polarity.intent_negated_anchor_coverage else 0.0
        )
        polarity_penalty = _polarity_penalty(polarity)
        safety_penalty = (
            (0.46 * safety_cues["insufficient_context_cue"])
            + (0.52 * safety_cues["explicit_contradiction_cue"])
            + (0.34 * safety_cues["only_not_trap_cue"])
            + (0.20 * safety_cues["missing_relation_keyword_cue"])
        )
        score = (
            (0.24 * context_similarity)
            + (0.18 * direct_similarity)
            + (0.18 * positive_similarity)
            + (0.20 * required_score)
            + relation_bonus
            + (0.12 * quantifiers.score if intent.quantifier_requirements else 0.0)
            + (0.08 * exact_score)
            + (0.05 * heading_similarity)
            + kind_prior
            + granularity_bonus
            + polarity_bonus
            - missing_required_penalty
            - missing_relation_penalty
            - negative_penalty
            - excluded_penalty
            - relation_bridge_penalty
            - missing_quantifier_penalty
            - quantifier_mismatch_penalty
            - polarity_penalty
            - safety_penalty
        )
        score = max(0.0, min(1.0, score))
        features = {
            "context_similarity": round(context_similarity, 4),
            "direct_similarity": round(direct_similarity, 4),
            "positive_similarity": round(positive_similarity, 4),
            "negative_similarity": round(negative_similarity, 4),
            "required_facet_score": round(required_score, 4),
            "required_facet_coverage": round(required_coverage, 4),
            "relation_score": round(relation_score, 4),
            "relation_coverage": round(relation_coverage, 4),
            "relation_bridge_score": round(relation_bridge_score, 4),
            "quantifier_score": round(quantifiers.score, 4),
            "quantifier_coverage": round(quantifiers.coverage, 4),
            "quantifier_mismatch": round(quantifiers.mismatch, 4),
            "quantifier_contradiction": round(quantifiers.contradiction, 4),
            "quantifier_requirement_count": float(quantifiers.requirement_count),
            "quantifier_matched_count": float(quantifiers.matched_count),
            "quantifier_missing_count": float(quantifiers.missing_count),
            "quantifier_role_mismatch": round(quantifiers.role_mismatch, 4),
            "quantifier_role_missing_count": float(quantifiers.role_missing_count),
            "excluded_facet_score": round(excluded_score, 4),
            "exact_score": round(exact_score, 4),
            "heading_similarity": round(heading_similarity, 4),
            "polarity_alignment": round(polarity.alignment, 4),
            "polarity_mismatch": round(polarity.mismatch, 4),
            "contradiction_score": round(polarity.contradiction, 4),
            "intent_negated_anchor_coverage": round(polarity.intent_negated_anchor_coverage, 4),
            "span_negated_anchor_coverage": round(polarity.span_negated_anchor_coverage, 4),
            "shared_polarity_anchor_count": float(polarity.shared_anchor_count),
            "kind_prior": round(kind_prior, 4),
            "preferred_span_kind": preferred_span_kind,
            "granularity_bonus": round(granularity_bonus, 4),
            "missing_required_penalty": round(missing_required_penalty, 4),
            "missing_relation_penalty": round(missing_relation_penalty, 4),
            "negative_penalty": round(negative_penalty, 4),
            "excluded_penalty": round(excluded_penalty, 4),
            "relation_bridge_penalty": round(relation_bridge_penalty, 4),
            "missing_quantifier_penalty": round(missing_quantifier_penalty, 4),
            "quantifier_mismatch_penalty": round(quantifier_mismatch_penalty, 4),
            "polarity_bonus": round(polarity_bonus, 4),
            "polarity_penalty": round(polarity_penalty, 4),
            "insufficient_context_cue": round(safety_cues["insufficient_context_cue"], 4),
            "explicit_contradiction_cue": round(safety_cues["explicit_contradiction_cue"], 4),
            "only_not_trap_cue": round(safety_cues["only_not_trap_cue"], 4),
            "intent_denies_presence_cue": round(safety_cues["intent_denies_presence_cue"], 4),
            "span_denies_presence_cue": round(safety_cues["span_denies_presence_cue"], 4),
            "missing_relation_keyword_cue": round(
                safety_cues["missing_relation_keyword_cue"],
                4,
            ),
            "safety_penalty": round(safety_penalty, 4),
        }
        reasons = _reasons(features)
        return ScoreResult(score=round(score, 4), features=features, reasons=reasons)


@dataclass(frozen=True)
class KeywordOverlapBaseline:
    """Weak baseline for measurement only."""

    def score(self, intent: IntentSpec, span: DocumentSpan) -> ScoreResult:
        query_tokens = set(tokenize(" ".join(intent.query_texts)))
        span_tokens = set(tokenize(span.context_text))
        if not query_tokens or not span_tokens:
            score = 0.0
        else:
            score = len(query_tokens & span_tokens) / len(query_tokens)
        return ScoreResult(
            score=round(score, 4),
            features={"keyword_overlap": round(score, 4)},
            reasons=("keyword overlap baseline",),
        )


def _facet_score(
    facets: tuple[EvidenceFacet, ...],
    text: str,
    *,
    threshold: float,
) -> tuple[float, float]:
    if not facets:
        return 1.0, 1.0
    weighted_score = 0.0
    total_weight = 0.0
    covered = 0
    required_count = 0
    for facet in facets:
        score = max_similarity(facet.phrases, text)
        weighted_score += score * facet.weight
        total_weight += facet.weight
        if facet.required:
            required_count += 1
            if score >= threshold:
                covered += 1
    coverage = 1.0 if required_count == 0 else covered / required_count
    return weighted_score / total_weight, coverage


def _relation_score(
    relations: tuple[EvidenceRelation, ...],
    text: str,
    *,
    threshold: float,
) -> tuple[float, float, float]:
    if not relations:
        return 0.0, 1.0, 0.0
    weighted_score = 0.0
    total_weight = 0.0
    covered = 0
    required_count = 0
    bridge_score = 0.0
    for relation in relations:
        component_scores = _relation_component_scores(relation, text)
        relation_score = sum(component_scores) / len(component_scores)
        weighted_score += relation_score * relation.weight
        total_weight += relation.weight
        if relation.required:
            required_count += 1
            if all(score >= threshold for score in component_scores):
                covered += 1
        if relation.forbidden_bridge_phrases:
            bridge_score = max(
                bridge_score,
                max_similarity(relation.forbidden_bridge_phrases, text),
            )
    coverage = 1.0 if required_count == 0 else covered / required_count
    return weighted_score / total_weight, coverage, bridge_score


def _relation_component_scores(relation: EvidenceRelation, text: str) -> tuple[float, ...]:
    scores = [
        max_similarity(relation.subject_phrases, text),
        max_similarity(relation.predicate_phrases, text),
    ]
    if relation.object_phrases:
        scores.append(max_similarity(relation.object_phrases, text))
    if relation.modifier_phrases:
        scores.append(max_similarity(relation.modifier_phrases, text))
    return tuple(scores)


def _kind_prior(intent: IntentSpec, kind: SpanKind) -> float:
    if kind not in intent.preferred_span_kinds:
        return -0.08
    index = intent.preferred_span_kinds.index(kind)
    return max(0.0, 0.05 - (0.01 * index))


def _polarity_intent_texts(intent: IntentSpec) -> tuple[str, ...]:
    return (
        intent.label,
        intent.description,
        *intent.positive_examples,
        *(relation.query_text for relation in intent.relations),
    )


def _polarity_penalty(polarity: PolarityScore) -> float:
    mismatch = polarity.mismatch
    contradiction = polarity.contradiction
    penalty = (0.36 * mismatch) + (0.34 * contradiction)
    if (
        polarity.intent_negated_anchor_coverage > 0.0
        and polarity.span_negated_anchor_coverage > 0.0
    ):
        penalty *= 0.35
    return penalty


def _safety_cue_features(intent: IntentSpec, text: str) -> dict[str, float]:
    intent_text = _normalize_for_cues(
        " ".join(
            (
                intent.description,
                *intent.positive_examples,
                *(relation.query_text for relation in intent.relations),
            )
        )
    )
    span_text = _normalize_for_cues(text)
    return {
        "insufficient_context_cue": 1.0
        if _has_insufficient_context_cue(span_text)
        else 0.0,
        "explicit_contradiction_cue": 1.0
        if _has_explicit_contradiction_cue(intent_text, span_text)
        else 0.0,
        "only_not_trap_cue": 1.0 if _has_only_not_trap(intent_text, span_text) else 0.0,
        "intent_denies_presence_cue": 1.0 if _intent_denies_presence(intent_text) else 0.0,
        "span_denies_presence_cue": 1.0 if _span_denies_presence(span_text) else 0.0,
        "missing_relation_keyword_cue": 1.0
        if _has_missing_relation_keyword(intent_text, span_text)
        else 0.0,
    }


def _normalize_for_cues(text: str) -> str:
    normalized = text.lower()
    replacements = (
        ("doesn't", "does not"),
        ("don't", "do not"),
        ("didn't", "did not"),
        ("can't", "can not"),
        ("cannot", "can not"),
        ("won't", "will not"),
        ("isn't", "is not"),
        ("aren't", "are not"),
    )
    for source, replacement in replacements:
        normalized = normalized.replace(source, replacement)
    return re.sub(r"\s+", " ", normalized).strip()


def _has_insufficient_context_cue(text: str) -> bool:
    proof_refusals = (
        "does not prove",
        "do not prove",
        "did not prove",
        "does not establish",
        "do not establish",
        "does not show",
        "not enough context",
        "insufficient context",
        "not sufficient context",
        "cannot infer",
        "can not infer",
    )
    if any(phrase in text for phrase in proof_refusals):
        return True
    return bool(
        re.search(
            r"\bmentions?\b.{0,90}\bbut\b.{0,90}\b(?:not|never)\b.{0,40}\b(?:prove|establish|show)\b",
            text,
        )
    )


def _has_explicit_contradiction_cue(intent_text: str, span_text: str) -> bool:
    if _contains_any(span_text, ("does not forbid", "do not forbid")):
        return _contains_any(intent_text, ("forbid", "forbids", "forbidden")) or _contains_any(
            span_text,
            ("requires", "required", "allows", "permits"),
        )
    if _contains_any(span_text, ("does not deny", "do not deny")):
        return _intent_denies_presence(intent_text) or _contains_any(
            span_text,
            ("lists", "listed", "confirms", "describes"),
        )
    if _contains_any(span_text, ("does not remove", "do not remove")):
        return _intent_denies_presence(intent_text) or _contains_any(
            span_text,
            ("keeps", "kept", "confirms", "remains", "still"),
        )
    if _contains_any(span_text, ("does not lack", "do not lack")):
        return _intent_denies_presence(intent_text)
    return False


def _contains_any(text: str, phrases: tuple[str, ...]) -> bool:
    return any(phrase in text for phrase in phrases)


def _intent_denies_presence(intent_text: str) -> bool:
    return bool(
        re.search(
            (
                r"\b(?:has no|have no|had no|no|not|without|does not have|"
                r"do not have|lacks?|lack of|missing)\b"
            ),
            intent_text,
        )
    )


def _span_denies_presence(span_text: str) -> bool:
    return bool(
        re.search(
            (
                r"\b(?:has no|have no|had no|no|without|does not|do not|"
                r"did not|not about|not described|not advertised|not supported|"
                r"not permitted|not allowed|lacks?|lack of|missing|removed|"
                r"unavailable|unsupported)\b"
            ),
            span_text,
        )
    )


def _has_missing_relation_keyword(intent_text: str, span_text: str) -> bool:
    relation_keywords = ("employer", "source", "owns", "guarantees", "inside", "itself provides")
    required = tuple(keyword for keyword in relation_keywords if keyword in intent_text)
    if not required:
        return False
    missing = tuple(keyword for keyword in required if keyword not in span_text)
    return len(missing) == len(required)


def _has_only_not_trap(intent_text: str, span_text: str) -> bool:
    if "not only" in intent_text or "not only" in span_text:
        return False
    return bool(
        re.search(r"\bonly\b.{0,60}\bnot\b", intent_text)
        or re.search(r"\bonly\b.{0,60}\bnot\b", span_text)
    )


def _reasons(features: dict[str, float]) -> tuple[str, ...]:
    reasons: list[str] = []
    if features["required_facet_coverage"] >= 1.0:
        reasons.append("all required facets covered")
    elif features["required_facet_coverage"] > 0.0:
        reasons.append("some required facets covered")
    if features["relation_coverage"] >= 1.0 and features["relation_score"] > 0.0:
        reasons.append("all required relation anchors covered")
    elif features["relation_coverage"] > 0.0 and features["relation_score"] > 0.0:
        reasons.append("some required relation anchors covered")
    if features["quantifier_requirement_count"] > 0.0:
        if features["quantifier_coverage"] >= 1.0:
            reasons.append("all required quantifiers covered")
        elif features["quantifier_contradiction"] > 0.0:
            reasons.append("required quantifier contradicted")
        else:
            reasons.append("required quantifier missing")
    if features["positive_similarity"] >= 0.22:
        reasons.append("similar to positive examples")
    if features["exact_score"] >= 0.4:
        reasons.append("contains exact intent anchors")
    if (
        features["negative_penalty"] > 0.05
        or features["excluded_penalty"] > 0.05
        or features["relation_bridge_penalty"] > 0.05
        or features["safety_penalty"] > 0.05
    ):
        reasons.append("penalized by negative or excluded evidence")
    if features["insufficient_context_cue"] > 0.0:
        reasons.append("span explicitly withholds proof")
    if features["explicit_contradiction_cue"] > 0.0:
        reasons.append("span explicitly counters the intent")
    if features["only_not_trap_cue"] > 0.0:
        reasons.append("span contains an only-not trap")
    if features["contradiction_score"] >= 0.35:
        reasons.append("polarity contradicts the intent")
    elif features["polarity_bonus"] > 0.0:
        reasons.append("negated anchors align with the intent")
    return tuple(reasons)
