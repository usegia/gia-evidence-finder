from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType

from gia_evidence_finder.benchmark_series import BenchmarkSplit, benchmark_series_suites
from gia_evidence_finder.contracts import BenchmarkCase, BenchmarkSuite, SupportLabel

JUDGE_LABELS: tuple[SupportLabel, ...] = (
    SupportLabel.SUPPORTS,
    SupportLabel.NEAR_MISS,
    SupportLabel.CONTRADICTS,
    SupportLabel.INSUFFICIENT_CONTEXT,
    SupportLabel.REJECT,
)


@dataclass(frozen=True)
class JudgeCriterion:
    id: str
    description: str

    def __post_init__(self) -> None:
        if not self.id.strip():
            raise ValueError("criterion id must not be empty")
        if not self.description.strip():
            raise ValueError("criterion description must not be empty")

    def to_json_dict(self) -> dict[str, str]:
        return {"id": self.id, "description": self.description}


@dataclass(frozen=True)
class ModelJudgeRubric:
    id: str
    version: str
    criteria: tuple[JudgeCriterion, ...]
    label_definitions: Mapping[SupportLabel, str]

    def __post_init__(self) -> None:
        if not self.id.strip():
            raise ValueError("rubric id must not be empty")
        if not self.version.strip():
            raise ValueError("rubric version must not be empty")
        if not self.criteria:
            raise ValueError("rubric must include at least one criterion")
        missing = tuple(label for label in JUDGE_LABELS if label not in self.label_definitions)
        if missing:
            raise ValueError(f"rubric missing label definitions: {missing!r}")
        object.__setattr__(
            self,
            "label_definitions",
            MappingProxyType(dict(self.label_definitions)),
        )

    def to_json_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "version": self.version,
            "criteria": [criterion.to_json_dict() for criterion in self.criteria],
            "label_definitions": {
                label.value: self.label_definitions[label] for label in JUDGE_LABELS
            },
        }


DEFAULT_MODEL_JUDGE_RUBRIC = ModelJudgeRubric(
    id="gia_evidence_finder_span_judge",
    version="2026-05-08",
    criteria=(
        JudgeCriterion(
            id="direct_support",
            description=(
                "A support label requires the candidate span itself to directly prove the "
                "intent, without relying on unstated assumptions or another section."
            ),
        ),
        JudgeCriterion(
            id="minimality",
            description=(
                "Prefer the smallest sentence, bullet, table row, or paragraph that proves "
                "the claim. Larger context may explain but should not replace minimal evidence."
            ),
        ),
        JudgeCriterion(
            id="near_miss",
            description=(
                "A near miss is related or shares anchors but lacks at least one required "
                "facet, relation, modifier, polarity, or scope."
            ),
        ),
        JudgeCriterion(
            id="contradiction",
            description=(
                "A contradiction explicitly says the claim is false or gives the opposite "
                "policy, polarity, relation, or scope."
            ),
        ),
        JudgeCriterion(
            id="insufficient_context",
            description=(
                "Insufficient context contains some relevant terms but cannot establish the "
                "claim by itself."
            ),
        ),
    ),
    label_definitions={
        SupportLabel.SUPPORTS: "The candidate span directly supports the intent.",
        SupportLabel.NEAR_MISS: "The span is related but misses a required detail.",
        SupportLabel.CONTRADICTS: "The span contradicts the intent.",
        SupportLabel.INSUFFICIENT_CONTEXT: "The span is relevant but incomplete.",
        SupportLabel.REJECT: "The span is not useful evidence for the intent.",
    },
)


@dataclass(frozen=True)
class ModelJudgeRequest:
    id: str
    case_id: str
    document_id: str
    span_id: str
    intent_label: str
    intent_description: str
    positive_examples: tuple[str, ...]
    candidate_text: str
    heading_path: tuple[str, ...]
    expected_label: SupportLabel | None
    source_hash: str
    metadata: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for value, label in (
            (self.id, "request id"),
            (self.case_id, "case id"),
            (self.document_id, "document id"),
            (self.span_id, "span id"),
            (self.intent_label, "intent label"),
            (self.intent_description, "intent description"),
            (self.candidate_text, "candidate text"),
            (self.source_hash, "source hash"),
        ):
            if not value.strip():
                raise ValueError(f"{label} must not be empty")
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

    def to_json_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "case_id": self.case_id,
            "document_id": self.document_id,
            "span_id": self.span_id,
            "intent_label": self.intent_label,
            "intent_description": self.intent_description,
            "positive_examples": list(self.positive_examples),
            "candidate_text": self.candidate_text,
            "heading_path": list(self.heading_path),
            "expected_label": self.expected_label.value if self.expected_label else None,
            "source_hash": self.source_hash,
            "metadata": dict(self.metadata),
        }

    def prompt(self, rubric: ModelJudgeRubric = DEFAULT_MODEL_JUDGE_RUBRIC) -> str:
        return "\n".join(
            (
                "You are judging whether one source span supports a typed evidence intent.",
                "Return only JSON matching the provided schema.",
                "",
                "Rubric:",
                json.dumps(rubric.to_json_dict(), indent=2, sort_keys=True),
                "",
                "Response schema:",
                json.dumps(model_judge_response_schema(), indent=2, sort_keys=True),
                "",
                "Request:",
                json.dumps(self.to_json_dict(), indent=2, sort_keys=True),
            )
        )


@dataclass(frozen=True)
class ModelJudgeDecision:
    request_id: str
    label: SupportLabel
    confidence: float
    rationale: str
    judge_model: str
    rubric_id: str
    rubric_version: str
    source_hash: str
    pass_id: str

    def __post_init__(self) -> None:
        if not self.request_id.strip():
            raise ValueError("decision request_id must not be empty")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("decision confidence must be between 0 and 1")
        for value, label in (
            (self.rationale, "rationale"),
            (self.judge_model, "judge_model"),
            (self.rubric_id, "rubric_id"),
            (self.rubric_version, "rubric_version"),
            (self.source_hash, "source_hash"),
            (self.pass_id, "pass_id"),
        ):
            if not value.strip():
                raise ValueError(f"decision {label} must not be empty")

    def to_json_dict(self) -> dict[str, object]:
        return {
            "request_id": self.request_id,
            "label": self.label.value,
            "confidence": self.confidence,
            "rationale": self.rationale,
            "judge_model": self.judge_model,
            "rubric_id": self.rubric_id,
            "rubric_version": self.rubric_version,
            "source_hash": self.source_hash,
            "pass_id": self.pass_id,
        }


@dataclass(frozen=True)
class ModelJudgeAgreement:
    request_id: str
    decision_count: int
    pass_count: int
    labels: tuple[SupportLabel, ...]
    majority_label: SupportLabel
    unanimous: bool
    mean_confidence: float

    def to_json_dict(self) -> dict[str, object]:
        return {
            "request_id": self.request_id,
            "decision_count": self.decision_count,
            "pass_count": self.pass_count,
            "labels": [label.value for label in self.labels],
            "majority_label": self.majority_label.value,
            "unanimous": self.unanimous,
            "mean_confidence": self.mean_confidence,
        }


@dataclass(frozen=True)
class ModelJudgeAgreementReport:
    request_count: int
    decision_count: int
    unanimous_count: int
    disagreement_count: int
    insufficient_pass_count: int
    agreements: tuple[ModelJudgeAgreement, ...]
    warnings: tuple[str, ...]

    def to_json_dict(self) -> dict[str, object]:
        return {
            "request_count": self.request_count,
            "decision_count": self.decision_count,
            "unanimous_count": self.unanimous_count,
            "disagreement_count": self.disagreement_count,
            "insufficient_pass_count": self.insufficient_pass_count,
            "agreements": [agreement.to_json_dict() for agreement in self.agreements],
            "warnings": list(self.warnings),
        }


def model_judge_response_schema() -> dict[str, object]:
    return {
        "type": "object",
        "required": ["label", "confidence", "rationale"],
        "additionalProperties": False,
        "properties": {
            "label": {"type": "string", "enum": [label.value for label in JUDGE_LABELS]},
            "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
            "rationale": {"type": "string", "minLength": 1},
        },
    }


def model_judge_requests_from_suite(
    suite: BenchmarkSuite,
    *,
    include_unlabeled_forbidden: bool = True,
) -> tuple[ModelJudgeRequest, ...]:
    requests: list[ModelJudgeRequest] = []
    for case in suite.cases:
        labeled_spans = _labeled_span_ids(
            case,
            include_unlabeled_forbidden=include_unlabeled_forbidden,
        )
        for span_id, expected_label in labeled_spans:
            span = case.document.span_by_id(span_id)
            requests.append(
                ModelJudgeRequest(
                    id=f"{case.id}:{span_id}",
                    case_id=case.id,
                    document_id=case.document.id,
                    span_id=span.id,
                    intent_label=case.intent.label,
                    intent_description=case.intent.description,
                    positive_examples=case.intent.positive_examples,
                    candidate_text=span.text,
                    heading_path=span.heading_path,
                    expected_label=expected_label,
                    source_hash=_source_hash(case, span.id),
                    metadata={
                        "suite_id": suite.id,
                        "curation_source": case.curation.source,
                        "expect_abstain": str(case.expect_abstain).lower(),
                    },
                )
            )
    return tuple(requests)


def write_model_judge_requests_jsonl(
    requests: tuple[ModelJudgeRequest, ...],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        "\n".join(json.dumps(request.to_json_dict(), sort_keys=True) for request in requests)
        + ("\n" if requests else ""),
        encoding="utf-8",
    )


def decisions_from_model_response_jsonl(
    *,
    response_jsonl: Path,
    request_jsonl: Path,
    judge_model: str,
    pass_id: str,
    rubric: ModelJudgeRubric = DEFAULT_MODEL_JUDGE_RUBRIC,
) -> tuple[ModelJudgeDecision, ...]:
    requests = {request.id: request for request in read_model_judge_requests_jsonl(request_jsonl)}
    decisions: list[ModelJudgeDecision] = []
    for line_number, line in enumerate(
        response_jsonl.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not line.strip():
            continue
        payload = _mapping(json.loads(line), f"line {line_number}")
        request_id = _string(payload, "request_id", line_number)
        try:
            request = requests[request_id]
        except KeyError as exc:
            raise ValueError(f"line {line_number}: unknown request_id {request_id!r}") from exc
        decisions.append(
            ModelJudgeDecision(
                request_id=request_id,
                label=SupportLabel(_string(payload, "label", line_number)),
                confidence=_float(payload, "confidence", line_number),
                rationale=_string(payload, "rationale", line_number),
                judge_model=judge_model,
                rubric_id=rubric.id,
                rubric_version=rubric.version,
                source_hash=request.source_hash,
                pass_id=pass_id,
            )
        )
    if not decisions:
        raise ValueError("model judge response JSONL did not contain any decisions")
    return tuple(decisions)


def read_model_judge_requests_jsonl(path: Path) -> tuple[ModelJudgeRequest, ...]:
    requests: list[ModelJudgeRequest] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        payload = _mapping(json.loads(line), f"line {line_number}")
        expected = payload.get("expected_label")
        requests.append(
            ModelJudgeRequest(
                id=_string(payload, "id", line_number),
                case_id=_string(payload, "case_id", line_number),
                document_id=_string(payload, "document_id", line_number),
                span_id=_string(payload, "span_id", line_number),
                intent_label=_string(payload, "intent_label", line_number),
                intent_description=_string(payload, "intent_description", line_number),
                positive_examples=_strings(payload, "positive_examples", line_number),
                candidate_text=_string(payload, "candidate_text", line_number),
                heading_path=_strings(payload, "heading_path", line_number),
                expected_label=SupportLabel(expected) if isinstance(expected, str) else None,
                source_hash=_string(payload, "source_hash", line_number),
                metadata=_metadata(payload.get("metadata", {}), line_number),
            )
        )
    if not requests:
        raise ValueError("model judge request JSONL did not contain any requests")
    return tuple(requests)


def read_model_judge_decisions_jsonl(path: Path) -> tuple[ModelJudgeDecision, ...]:
    decisions: list[ModelJudgeDecision] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        payload = _mapping(json.loads(line), f"line {line_number}")
        decisions.append(
            ModelJudgeDecision(
                request_id=_string(payload, "request_id", line_number),
                label=SupportLabel(_string(payload, "label", line_number)),
                confidence=_float(payload, "confidence", line_number),
                rationale=_string(payload, "rationale", line_number),
                judge_model=_string(payload, "judge_model", line_number),
                rubric_id=_string(payload, "rubric_id", line_number),
                rubric_version=_string(payload, "rubric_version", line_number),
                source_hash=_string(payload, "source_hash", line_number),
                pass_id=_string(payload, "pass_id", line_number),
            )
        )
    if not decisions:
        raise ValueError("model judge decision JSONL did not contain any decisions")
    return tuple(decisions)


def model_judge_agreement_report(
    decisions: tuple[ModelJudgeDecision, ...],
    *,
    min_passes: int = 2,
) -> ModelJudgeAgreementReport:
    if min_passes < 1:
        raise ValueError("min_passes must be positive")
    grouped: dict[str, list[ModelJudgeDecision]] = defaultdict(list)
    for decision in decisions:
        grouped[decision.request_id].append(decision)
    agreements: list[ModelJudgeAgreement] = []
    insufficient = 0
    warnings: list[str] = []
    for request_id, group in sorted(grouped.items()):
        pass_count = len({decision.pass_id for decision in group})
        if pass_count < min_passes:
            insufficient += 1
            warnings.append(
                f"{request_id} has {pass_count} judge passes; target is at least {min_passes}"
            )
        labels = tuple(decision.label for decision in group)
        majority_label = Counter(labels).most_common(1)[0][0]
        agreements.append(
            ModelJudgeAgreement(
                request_id=request_id,
                decision_count=len(group),
                pass_count=pass_count,
                labels=labels,
                majority_label=majority_label,
                unanimous=len(set(labels)) == 1,
                mean_confidence=round(
                    sum(decision.confidence for decision in group) / len(group),
                    4,
                ),
            )
        )
    unanimous_count = sum(1 for agreement in agreements if agreement.unanimous)
    return ModelJudgeAgreementReport(
        request_count=len(agreements),
        decision_count=len(decisions),
        unanimous_count=unanimous_count,
        disagreement_count=len(agreements) - unanimous_count,
        insufficient_pass_count=insufficient,
        agreements=tuple(agreements),
        warnings=tuple(warnings),
    )


def suites_for_judge_export(
    *,
    series_id: str,
    split: BenchmarkSplit | None,
) -> tuple[BenchmarkSuite, ...]:
    from gia_evidence_finder.benchmark_series import benchmark_series_by_id

    return benchmark_series_suites(benchmark_series_by_id(series_id), split=split)


def _labeled_span_ids(
    case: BenchmarkCase,
    *,
    include_unlabeled_forbidden: bool,
) -> tuple[tuple[str, SupportLabel], ...]:
    labels: list[tuple[str, SupportLabel]] = []
    labels.extend((span_id, SupportLabel.SUPPORTS) for span_id in case.support_span_ids)
    labels.extend((span_id, SupportLabel.NEAR_MISS) for span_id in case.near_miss_span_ids)
    labels.extend((span_id, SupportLabel.CONTRADICTS) for span_id in case.contradiction_span_ids)
    labels.extend(
        (span_id, SupportLabel.INSUFFICIENT_CONTEXT)
        for span_id in case.insufficient_context_span_ids
    )
    already_labeled = {span_id for span_id, _label in labels}
    if include_unlabeled_forbidden:
        labels.extend(
            (span_id, SupportLabel.REJECT)
            for span_id in case.forbidden_span_ids
            if span_id not in already_labeled
        )
    return tuple(dict.fromkeys(labels))


def _source_hash(case: BenchmarkCase, span_id: str) -> str:
    span = case.document.span_by_id(span_id)
    payload = {
        "document_id": case.document.id,
        "document_source": case.document.source,
        "span_id": span.id,
        "span_text": span.text,
        "intent_label": case.intent.label,
        "intent_description": case.intent.description,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def _mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, dict):
        raise ValueError(f"{label}: expected object")
    return value


def _string(payload: Mapping[str, object], key: str, line_number: int) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"line {line_number}: {key} must be a non-empty string")
    return value


def _strings(payload: Mapping[str, object], key: str, line_number: int) -> tuple[str, ...]:
    value = payload.get(key)
    if not isinstance(value, list):
        raise ValueError(f"line {line_number}: {key} must be a list")
    result: list[str] = []
    for item in value:
        if not isinstance(item, str):
            raise ValueError(f"line {line_number}: {key} must contain only strings")
        result.append(item)
    return tuple(result)


def _float(payload: Mapping[str, object], key: str, line_number: int) -> float:
    value = payload.get(key)
    if not isinstance(value, int | float):
        raise ValueError(f"line {line_number}: {key} must be numeric")
    return float(value)


def _metadata(value: object, line_number: int) -> dict[str, str]:
    if not isinstance(value, dict):
        raise ValueError(f"line {line_number}: metadata must be an object")
    return {str(key): str(item) for key, item in value.items()}
