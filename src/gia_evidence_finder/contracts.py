from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from types import MappingProxyType


class SpanKind(StrEnum):
    HEADING = "heading"
    PARAGRAPH = "paragraph"
    SENTENCE = "sentence"
    BULLET = "bullet"
    CODE = "code"
    TABLE_ROW = "table_row"


class SupportLabel(StrEnum):
    SUPPORTS = "supports"
    NEAR_MISS = "near_miss"
    CONTRADICTS = "contradicts"
    INSUFFICIENT_CONTEXT = "insufficient_context"
    REJECT = "reject"
    ABSTAIN = "abstain"


class QuantifierKind(StrEnum):
    DATE = "date"
    YEAR = "year"
    NUMBER = "number"
    MONEY = "money"
    DURATION = "duration"
    MULTIPLIER = "multiplier"
    PERCENT = "percent"


class QuantifierOperator(StrEnum):
    EQ = "eq"
    LT = "lt"
    LTE = "lte"
    GT = "gt"
    GTE = "gte"


@dataclass(frozen=True)
class DocumentSpan:
    id: str
    document_id: str
    kind: SpanKind
    text: str
    ordinal: int
    heading_path: tuple[str, ...] = ()
    parent_id: str | None = None
    previous_id: str | None = None
    next_id: str | None = None
    char_start: int = 0
    char_end: int = 0
    metadata: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_identifier(self.id, "span id")
        _require_identifier(self.document_id, "span document_id")
        _require_identifier(self.text, "span text")
        if self.ordinal < 0:
            raise ValueError("span ordinal must be non-negative")
        if self.char_start < 0 or self.char_end < self.char_start:
            raise ValueError("span offsets must be ordered and non-negative")
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

    @property
    def context_text(self) -> str:
        if not self.heading_path:
            return self.text
        return "\n".join((*self.heading_path, self.text))


@dataclass(frozen=True)
class EvidenceDocument:
    id: str
    spans: tuple[DocumentSpan, ...]
    source: str | None = None
    metadata: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_identifier(self.id, "document id")
        if not self.spans:
            raise ValueError("evidence document must include at least one span")
        seen: set[str] = set()
        for span in self.spans:
            if span.document_id != self.id:
                raise ValueError("span document_id must match document id")
            if span.id in seen:
                raise ValueError(f"duplicate span id {span.id!r}")
            seen.add(span.id)
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

    def span_by_id(self, span_id: str) -> DocumentSpan:
        for span in self.spans:
            if span.id == span_id:
                return span
        raise KeyError(span_id)


@dataclass(frozen=True)
class EvidenceFacet:
    name: str
    phrases: tuple[str, ...]
    required: bool = True
    weight: float = 1.0

    def __post_init__(self) -> None:
        _require_identifier(self.name, "facet name")
        _require_non_empty_strings(self.phrases, "facet phrases")
        if self.weight <= 0:
            raise ValueError("facet weight must be positive")


@dataclass(frozen=True)
class EvidenceRelation:
    name: str
    subject_phrases: tuple[str, ...]
    predicate_phrases: tuple[str, ...]
    object_phrases: tuple[str, ...] = ()
    modifier_phrases: tuple[str, ...] = ()
    forbidden_bridge_phrases: tuple[str, ...] = ()
    required: bool = True
    weight: float = 1.0

    def __post_init__(self) -> None:
        _require_identifier(self.name, "relation name")
        _require_non_empty_strings(self.subject_phrases, "relation subject phrases")
        _require_non_empty_strings(self.predicate_phrases, "relation predicate phrases")
        _require_strings(self.object_phrases, "relation object phrases")
        _require_strings(self.modifier_phrases, "relation modifier phrases")
        _require_strings(self.forbidden_bridge_phrases, "relation forbidden bridge phrases")
        if self.weight <= 0:
            raise ValueError("relation weight must be positive")

    @property
    def query_text(self) -> str:
        parts = (
            *self.subject_phrases,
            *self.predicate_phrases,
            *self.object_phrases,
            *self.modifier_phrases,
        )
        return " ".join(parts)


@dataclass(frozen=True)
class EvidenceQuantifier:
    kind: QuantifierKind
    value: float
    surface: str
    unit: str = ""
    operator: QuantifierOperator = QuantifierOperator.EQ
    normalized: str = ""

    def __post_init__(self) -> None:
        _require_identifier(self.surface, "quantifier surface")
        if self.unit:
            _require_identifier(self.unit, "quantifier unit")
        if self.normalized:
            _require_identifier(self.normalized, "quantifier normalized")


@dataclass(frozen=True)
class QuantifierBinding:
    quantifier: EvidenceQuantifier
    role: str = ""
    subject_terms: tuple[str, ...] = ()
    predicate_terms: tuple[str, ...] = ()
    local_text: str = ""
    binding_confidence: float = 0.0

    def __post_init__(self) -> None:
        if self.role:
            _require_identifier(self.role, "quantifier binding role")
        _require_strings(self.subject_terms, "quantifier binding subject terms")
        _require_strings(self.predicate_terms, "quantifier binding predicate terms")
        if self.local_text:
            _require_identifier(self.local_text, "quantifier binding local text")
        if not 0.0 <= self.binding_confidence <= 1.0:
            raise ValueError("quantifier binding confidence must be between 0 and 1")


@dataclass(frozen=True)
class QuantifierRequirement:
    name: str
    quantifier: EvidenceQuantifier
    role: str = ""
    binding_required: bool = False
    subject_terms: tuple[str, ...] = ()
    predicate_terms: tuple[str, ...] = ()
    required: bool = True
    weight: float = 1.0

    def __post_init__(self) -> None:
        _require_identifier(self.name, "quantifier requirement name")
        if self.role:
            _require_identifier(self.role, "quantifier requirement role")
        _require_strings(self.subject_terms, "quantifier requirement subject terms")
        _require_strings(self.predicate_terms, "quantifier requirement predicate terms")
        if self.weight <= 0:
            raise ValueError("quantifier requirement weight must be positive")


@dataclass(frozen=True)
class IntentSpec:
    id: str
    label: str
    description: str
    positive_examples: tuple[str, ...] = ()
    negative_examples: tuple[str, ...] = ()
    required_facets: tuple[EvidenceFacet, ...] = ()
    excluded_facets: tuple[EvidenceFacet, ...] = ()
    relations: tuple[EvidenceRelation, ...] = ()
    quantifier_requirements: tuple[QuantifierRequirement, ...] = ()
    preferred_span_kinds: tuple[SpanKind, ...] = (
        SpanKind.SENTENCE,
        SpanKind.BULLET,
        SpanKind.PARAGRAPH,
        SpanKind.CODE,
        SpanKind.TABLE_ROW,
    )
    min_support_score: float = 0.55
    near_miss_ratio: float = 0.82

    def __post_init__(self) -> None:
        _require_identifier(self.id, "intent id")
        _require_identifier(self.label, "intent label")
        _require_identifier(self.description, "intent description")
        _require_non_empty_strings(self.positive_examples or (self.description,), "intent text")
        _require_strings(self.positive_examples, "positive examples")
        _require_strings(self.negative_examples, "negative examples")
        if not 0.0 < self.min_support_score <= 1.0:
            raise ValueError("min_support_score must be in (0, 1]")
        if not 0.0 < self.near_miss_ratio < 1.0:
            raise ValueError("near_miss_ratio must be in (0, 1)")
        if not self.preferred_span_kinds:
            raise ValueError("preferred_span_kinds must not be empty")

    @property
    def query_texts(self) -> tuple[str, ...]:
        relation_texts = tuple(relation.query_text for relation in self.relations)
        return (self.label, self.description, *self.positive_examples, *relation_texts)


@dataclass(frozen=True)
class SpanMatch:
    span: DocumentSpan
    label: SupportLabel
    score: float
    features: Mapping[str, float]
    reasons: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not 0.0 <= self.score <= 1.0:
            raise ValueError("match score must be between 0 and 1")
        object.__setattr__(self, "features", MappingProxyType(dict(self.features)))


@dataclass(frozen=True)
class ExtractionResult:
    intent: IntentSpec
    document: EvidenceDocument
    matches: tuple[SpanMatch, ...]
    candidates: tuple[SpanMatch, ...]
    abstained: bool
    trace: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.abstained and self.matches:
            raise ValueError("abstained result cannot include support matches")


@dataclass(frozen=True)
class BenchmarkCuration:
    reviewed: bool = False
    source: str = "manual"
    difficulty: str = "unspecified"
    phenomena: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _require_identifier(self.source, "benchmark curation source")
        _require_identifier(self.difficulty, "benchmark curation difficulty")
        _require_strings(self.phenomena, "benchmark curation phenomena")
        _require_strings(self.notes, "benchmark curation notes")


@dataclass(frozen=True)
class BenchmarkCase:
    id: str
    intent: IntentSpec
    document: EvidenceDocument
    support_span_ids: tuple[str, ...] = ()
    near_miss_span_ids: tuple[str, ...] = ()
    contradiction_span_ids: tuple[str, ...] = ()
    insufficient_context_span_ids: tuple[str, ...] = ()
    forbidden_span_ids: tuple[str, ...] = ()
    expect_abstain: bool = False
    curation: BenchmarkCuration = field(default_factory=BenchmarkCuration)

    def __post_init__(self) -> None:
        _require_identifier(self.id, "benchmark case id")
        if self.expect_abstain and self.support_span_ids:
            raise ValueError("abstain cases cannot define support spans")
        existing = {span.id for span in self.document.spans}
        for label, span_ids in (
            ("support", self.support_span_ids),
            ("near miss", self.near_miss_span_ids),
            ("contradiction", self.contradiction_span_ids),
            ("insufficient context", self.insufficient_context_span_ids),
            ("forbidden", self.forbidden_span_ids),
        ):
            missing = tuple(span_id for span_id in span_ids if span_id not in existing)
            if missing:
                raise ValueError(f"{label} span ids not in document: {missing!r}")


@dataclass(frozen=True)
class BenchmarkSuite:
    id: str
    name: str
    cases: tuple[BenchmarkCase, ...]
    description: str = ""
    metadata: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_identifier(self.id, "benchmark suite id")
        _require_identifier(self.name, "benchmark suite name")
        if not self.cases:
            raise ValueError("benchmark suite must include at least one case")
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))


@dataclass(frozen=True)
class TrainingPair:
    intent_id: str
    span_id: str
    text: str
    label: SupportLabel
    query: str = ""
    score: float | None = None
    metadata: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_identifier(self.intent_id, "training pair intent_id")
        _require_identifier(self.span_id, "training pair span_id")
        _require_identifier(self.text, "training pair text")
        if self.query:
            _require_identifier(self.query, "training pair query")
        if self.score is not None and not 0.0 <= self.score <= 1.0:
            raise ValueError("training pair score must be between 0 and 1")
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))


def _require_identifier(value: str, label: str) -> None:
    if not value or not value.strip():
        raise ValueError(f"{label} must not be empty")


def _require_non_empty_strings(values: Sequence[str], label: str) -> None:
    if not values:
        raise ValueError(f"{label} must not be empty")
    _require_strings(values, label)


def _require_strings(values: Sequence[str], label: str) -> None:
    for value in values:
        _require_identifier(value, label)
