from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from gia_evidence_finder.benchmark_series import BenchmarkSeries, BenchmarkSplit
from gia_evidence_finder.contracts import (
    BenchmarkCase,
    BenchmarkCuration,
    BenchmarkSuite,
    EvidenceDocument,
    EvidenceFacet,
    EvidenceRelation,
    IntentSpec,
    SpanKind,
)
from gia_evidence_finder.parsing import MarkdownSpanParser


@dataclass(frozen=True)
class ReviewedCaseLoadResult:
    suite: BenchmarkSuite
    split_counts: dict[str, int]
    document_count: int


@dataclass(frozen=True)
class ReviewedSeriesLoadResult:
    series: BenchmarkSeries
    split_counts: dict[str, int]
    document_count: int


@dataclass(frozen=True)
class _ReviewedCaseWithSplit:
    case: BenchmarkCase
    split: BenchmarkSplit


def load_reviewed_cases_jsonl(
    *,
    reviewed_jsonl: Path,
    source_dir: Path,
    suite_id: str,
    suite_name: str | None = None,
) -> ReviewedCaseLoadResult:
    entries = _load_reviewed_case_entries(
        reviewed_jsonl=reviewed_jsonl,
        source_dir=source_dir,
    )
    cases = tuple(entry.case for entry in entries)
    suite = BenchmarkSuite(
        id=suite_id,
        name=suite_name or suite_id,
        cases=cases,
        description="Reviewed promoted evidence benchmark cases.",
        metadata={
            "source_dir": str(source_dir),
            "reviewed_jsonl": str(reviewed_jsonl),
            "case_count": str(len(cases)),
        },
    )
    return ReviewedCaseLoadResult(
        suite=suite,
        split_counts=_split_counts(entries),
        document_count=len({case.document.id for case in cases}),
    )


def load_reviewed_series_jsonl(
    *,
    reviewed_jsonl: Path,
    source_dir: Path,
    series_id: str,
    series_name: str | None = None,
) -> ReviewedSeriesLoadResult:
    entries = _load_reviewed_case_entries(
        reviewed_jsonl=reviewed_jsonl,
        source_dir=source_dir,
    )
    missing = tuple(
        split
        for split in BenchmarkSplit
        if not any(entry.split == split for entry in entries)
    )
    if missing:
        labels = ", ".join(split.value for split in missing)
        raise ValueError(f"reviewed benchmark series missing splits: {labels}")
    splits = {
        split: _split_suite(
            series_id=series_id,
            split=split,
            cases=tuple(entry.case for entry in entries if entry.split == split),
            source_dir=source_dir,
            reviewed_jsonl=reviewed_jsonl,
        )
        for split in BenchmarkSplit
    }
    series = BenchmarkSeries(
        id=series_id,
        name=series_name or series_id,
        description="Reviewed promoted evidence benchmark series.",
        splits=splits,
        metadata={
            "source_dir": str(source_dir),
            "reviewed_jsonl": str(reviewed_jsonl),
            "case_count": str(len(entries)),
            "train_case_count": str(len(splits[BenchmarkSplit.TRAIN].cases)),
            "dev_case_count": str(len(splits[BenchmarkSplit.DEV].cases)),
            "test_case_count": str(len(splits[BenchmarkSplit.TEST].cases)),
            "status": "reviewed_external",
        },
    )
    return ReviewedSeriesLoadResult(
        series=series,
        split_counts=_split_counts(entries),
        document_count=len({entry.case.document.id for entry in entries}),
    )


def reviewed_case_template_from_queue_item(
    queue_item: Mapping[str, object],
    *,
    split: BenchmarkSplit = BenchmarkSplit.TRAIN,
) -> dict[str, object]:
    document_id = _string(queue_item, "document_id")
    span_id = _string(queue_item, "span_id")
    suggested_label = _string(queue_item, "suggested_label")
    return {
        "id": f"{document_id}.{span_id}.reviewed",
        "split": split.value,
        "document_id": document_id,
        "label": suggested_label,
        "description": f"Find evidence for: {suggested_label}",
        "positive_examples": [suggested_label],
        "required_facets": [
            {
                "name": "review_required_anchor",
                "phrases": list(_strings(queue_item, "anchor_terms")),
            }
        ],
        "support_span_ids": [span_id],
        "near_miss_span_ids": [],
        "contradiction_span_ids": [],
        "insufficient_context_span_ids": [],
        "forbidden_span_ids": [],
        "expect_abstain": False,
        "curation": {
            "reviewed": False,
            "source": "queue_template",
            "difficulty": "unreviewed",
            "phenomena": ["direct_support"],
            "notes": ["Template generated from curation queue; review before use."],
        },
    }


def write_reviewed_case_template_jsonl(
    *,
    queue_jsonl: Path,
    output_jsonl: Path,
    split: BenchmarkSplit = BenchmarkSplit.TRAIN,
) -> int:
    rows: list[str] = []
    for queue_item in _queue_items_from_jsonl(queue_jsonl):
        template = reviewed_case_template_from_queue_item(queue_item, split=split)
        rows.append(json.dumps(template, sort_keys=True))
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    output_jsonl.write_text("\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")
    return len(rows)


def write_reviewed_series_template_jsonl(
    *,
    queue_jsonl: Path,
    output_jsonl: Path,
    train_ratio: float = 0.7,
    dev_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: str = "reviewed-series-v1",
) -> dict[str, int]:
    queue_items = _queue_items_from_jsonl(queue_jsonl)
    document_ids = tuple(sorted({_string(item, "document_id") for item in queue_items}))
    assignments = _document_split_assignments(
        document_ids,
        train_ratio=train_ratio,
        dev_ratio=dev_ratio,
        test_ratio=test_ratio,
        seed=seed,
    )
    counts: Counter[str] = Counter()
    rows: list[str] = []
    for queue_item in queue_items:
        split = assignments[_string(queue_item, "document_id")]
        template = reviewed_case_template_from_queue_item(queue_item, split=split)
        rows.append(json.dumps(template, sort_keys=True))
        counts[split.value] += 1
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    output_jsonl.write_text("\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")
    return dict(sorted(counts.items()))


def _load_reviewed_case_entries(
    *,
    reviewed_jsonl: Path,
    source_dir: Path,
) -> tuple[_ReviewedCaseWithSplit, ...]:
    documents = _load_documents_from_source_dir(source_dir)
    entries: list[_ReviewedCaseWithSplit] = []
    for line_number, line in enumerate(
        reviewed_jsonl.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not line.strip():
            continue
        payload = _mapping(json.loads(line), f"line {line_number}")
        case = _case_from_payload(payload, documents, line_number=line_number)
        split = _split_from_payload(payload, line_number=line_number)
        entries.append(_ReviewedCaseWithSplit(case=case, split=split))
    if not entries:
        raise ValueError("reviewed case JSONL must include at least one case")
    return tuple(entries)


def _queue_items_from_jsonl(queue_jsonl: Path) -> tuple[Mapping[str, object], ...]:
    queue_items: list[Mapping[str, object]] = []
    for line_number, line in enumerate(
        queue_jsonl.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not line.strip():
            continue
        queue_items.append(_mapping(json.loads(line), f"queue item line {line_number}"))
    if not queue_items:
        raise ValueError("curation queue JSONL must include at least one item")
    return tuple(queue_items)


def _document_split_assignments(
    document_ids: tuple[str, ...],
    *,
    train_ratio: float,
    dev_ratio: float,
    test_ratio: float,
    seed: str,
) -> dict[str, BenchmarkSplit]:
    if len(document_ids) < len(BenchmarkSplit):
        raise ValueError("reviewed series templates require at least three documents")
    ratios = {
        BenchmarkSplit.TRAIN: train_ratio,
        BenchmarkSplit.DEV: dev_ratio,
        BenchmarkSplit.TEST: test_ratio,
    }
    _validate_split_ratios(ratios)
    counts = _split_document_counts(len(document_ids), ratios)
    ordered = sorted(
        document_ids,
        key=lambda document_id: hashlib.sha256(
            f"{seed}:{document_id}".encode()
        ).hexdigest(),
    )
    assignments: dict[str, BenchmarkSplit] = {}
    offset = 0
    for split in BenchmarkSplit:
        for document_id in ordered[offset : offset + counts[split]]:
            assignments[document_id] = split
        offset += counts[split]
    return assignments


def _validate_split_ratios(ratios: Mapping[BenchmarkSplit, float]) -> None:
    for split, ratio in ratios.items():
        if ratio <= 0:
            raise ValueError(f"{split.value}_ratio must be positive")
    total = sum(ratios.values())
    if abs(total - 1.0) > 0.000001:
        raise ValueError("train/dev/test ratios must sum to 1.0")


def _split_document_counts(
    document_count: int,
    ratios: Mapping[BenchmarkSplit, float],
) -> dict[BenchmarkSplit, int]:
    raw_counts = {split: ratios[split] * document_count for split in BenchmarkSplit}
    counts = {split: int(raw_counts[split]) for split in BenchmarkSplit}
    remainder = document_count - sum(counts.values())
    split_order = sorted(
        BenchmarkSplit,
        key=lambda split: raw_counts[split] - counts[split],
        reverse=True,
    )
    for split in split_order[:remainder]:
        counts[split] += 1
    for split in BenchmarkSplit:
        if counts[split] > 0:
            continue
        donor = max(BenchmarkSplit, key=lambda candidate: counts[candidate])
        if counts[donor] <= 1:
            raise ValueError("unable to assign every split at least one document")
        counts[donor] -= 1
        counts[split] = 1
    return counts


def _split_counts(entries: tuple[_ReviewedCaseWithSplit, ...]) -> dict[str, int]:
    counts = Counter(entry.split.value for entry in entries)
    return dict(sorted(counts.items()))


def _split_suite(
    *,
    series_id: str,
    split: BenchmarkSplit,
    cases: tuple[BenchmarkCase, ...],
    source_dir: Path,
    reviewed_jsonl: Path,
) -> BenchmarkSuite:
    return BenchmarkSuite(
        id=f"{series_id}_{split.value}",
        name=f"{series_id} {split.value} split",
        description=f"{series_id} reviewed {split.value} split.",
        cases=cases,
        metadata={
            "series_id": series_id,
            "split": split.value,
            "case_count": str(len(cases)),
            "source_dir": str(source_dir),
            "reviewed_jsonl": str(reviewed_jsonl),
        },
    )


def _load_documents_from_source_dir(source_dir: Path) -> dict[str, EvidenceDocument]:
    manifest_path = source_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"manifest not found: {manifest_path}")
    manifest = _mapping(json.loads(manifest_path.read_text(encoding="utf-8")), "manifest")
    entries = _mappings(manifest, "sources")
    parser = MarkdownSpanParser()
    documents: dict[str, EvidenceDocument] = {}
    for entry in entries:
        document_id = _string(entry, "id")
        filename = _string(entry, "filename")
        raw_url = _string(entry, "raw_url")
        text = (source_dir / filename).read_text(encoding="utf-8")
        documents[document_id] = parser.parse(text, document_id=document_id, source=raw_url)
    return documents


def _case_from_payload(
    payload: Mapping[str, object],
    documents: Mapping[str, EvidenceDocument],
    *,
    line_number: int,
) -> BenchmarkCase:
    document_id = _string(payload, "document_id")
    try:
        document = documents[document_id]
    except KeyError as exc:
        raise ValueError(
            f"line {line_number}: document_id {document_id!r} not found in source manifest"
        ) from exc
    curation = _curation_from_payload(_mapping(payload.get("curation"), "curation"))
    if not curation.reviewed:
        raise ValueError(f"line {line_number}: reviewed benchmark cases must be reviewed")
    case = BenchmarkCase(
        id=_string(payload, "id"),
        intent=IntentSpec(
            id=_string(payload, "id"),
            label=_string(payload, "label"),
            description=_string(payload, "description"),
            positive_examples=_strings(payload, "positive_examples"),
            negative_examples=_strings(payload, "negative_examples", default=()),
            required_facets=_facets(payload, "required_facets"),
            excluded_facets=_facets(payload, "excluded_facets"),
            relations=_relations(payload, "relations"),
            preferred_span_kinds=_span_kinds(payload, "preferred_span_kinds"),
            min_support_score=_float(payload, "min_support_score", default=0.55),
            near_miss_ratio=_float(payload, "near_miss_ratio", default=0.82),
        ),
        document=document,
        support_span_ids=_strings(payload, "support_span_ids", default=()),
        near_miss_span_ids=_strings(payload, "near_miss_span_ids", default=()),
        contradiction_span_ids=_strings(payload, "contradiction_span_ids", default=()),
        insufficient_context_span_ids=_strings(
            payload,
            "insufficient_context_span_ids",
            default=(),
        ),
        forbidden_span_ids=_strings(payload, "forbidden_span_ids", default=()),
        expect_abstain=_bool(payload, "expect_abstain", default=False),
        curation=curation,
    )
    _validate_reviewed_case(case, line_number=line_number)
    return case


def _validate_reviewed_case(case: BenchmarkCase, *, line_number: int) -> None:
    if not case.expect_abstain and not case.support_span_ids:
        raise ValueError(f"line {line_number}: supported cases must include support spans")
    if case.expect_abstain and case.support_span_ids:
        raise ValueError(f"line {line_number}: abstain cases cannot include support spans")
    if not case.curation.phenomena:
        raise ValueError(f"line {line_number}: reviewed cases must include phenomena")
    if (
        not case.support_span_ids
        and not case.near_miss_span_ids
        and not case.contradiction_span_ids
        and not case.insufficient_context_span_ids
        and not case.forbidden_span_ids
    ):
        raise ValueError(f"line {line_number}: reviewed case must label at least one span")


def _split_from_payload(payload: Mapping[str, object], *, line_number: int) -> BenchmarkSplit:
    value = _string(payload, "split")
    try:
        return BenchmarkSplit(value)
    except ValueError as exc:
        raise ValueError(f"line {line_number}: unknown split {value!r}") from exc


def _curation_from_payload(payload: Mapping[str, object]) -> BenchmarkCuration:
    return BenchmarkCuration(
        reviewed=_bool(payload, "reviewed"),
        source=_string(payload, "source"),
        difficulty=_string(payload, "difficulty"),
        phenomena=_strings(payload, "phenomena"),
        notes=_strings(payload, "notes", default=()),
    )


def _facets(payload: Mapping[str, object], key: str) -> tuple[EvidenceFacet, ...]:
    return tuple(
        EvidenceFacet(
            name=_string(item, "name"),
            phrases=_strings(item, "phrases"),
            required=_bool(item, "required", default=True),
            weight=_float(item, "weight", default=1.0),
        )
        for item in _mappings(payload, key, default=())
    )


def _relations(payload: Mapping[str, object], key: str) -> tuple[EvidenceRelation, ...]:
    return tuple(
        EvidenceRelation(
            name=_string(item, "name"),
            subject_phrases=_strings(item, "subject_phrases"),
            predicate_phrases=_strings(item, "predicate_phrases"),
            object_phrases=_strings(item, "object_phrases", default=()),
            modifier_phrases=_strings(item, "modifier_phrases", default=()),
            forbidden_bridge_phrases=_strings(
                item,
                "forbidden_bridge_phrases",
                default=(),
            ),
            required=_bool(item, "required", default=True),
            weight=_float(item, "weight", default=1.0),
        )
        for item in _mappings(payload, key, default=())
    )


def _span_kinds(payload: Mapping[str, object], key: str) -> tuple[SpanKind, ...]:
    values = _strings(payload, key, default=())
    if not values:
        return (
            SpanKind.SENTENCE,
            SpanKind.BULLET,
            SpanKind.PARAGRAPH,
            SpanKind.CODE,
            SpanKind.TABLE_ROW,
        )
    try:
        return tuple(SpanKind(value) for value in values)
    except ValueError as exc:
        raise ValueError(f"unknown preferred span kind in {key!r}") from exc


def _mapping(value: object, field: str) -> Mapping[str, object]:
    if not isinstance(value, dict):
        raise ValueError(f"{field} must be an object")
    return cast(Mapping[str, object], value)


def _mappings(
    payload: Mapping[str, object],
    key: str,
    *,
    default: tuple[Mapping[str, object], ...] | None = None,
) -> tuple[Mapping[str, object], ...]:
    value = payload.get(key)
    if value is None and default is not None:
        return default
    if not isinstance(value, list):
        raise ValueError(f"{key} must be a list")
    return tuple(_mapping(item, key) for item in value)


def _strings(
    payload: Mapping[str, object],
    key: str,
    *,
    default: tuple[str, ...] | None = None,
) -> tuple[str, ...]:
    value = payload.get(key)
    if value is None and default is not None:
        return default
    if not isinstance(value, list):
        raise ValueError(f"{key} must be a list")
    result: list[str] = []
    for item in value:
        if not isinstance(item, str):
            raise ValueError(f"{key} must contain only strings")
        result.append(item)
    return tuple(result)


def _string(payload: Mapping[str, object], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str):
        raise ValueError(f"{key} must be a string")
    return value


def _bool(
    payload: Mapping[str, object],
    key: str,
    *,
    default: bool | None = None,
) -> bool:
    value = payload.get(key)
    if value is None and default is not None:
        return default
    if not isinstance(value, bool):
        raise ValueError(f"{key} must be a boolean")
    return value


def _float(
    payload: Mapping[str, object],
    key: str,
    *,
    default: float,
) -> float:
    value = payload.get(key)
    if value is None:
        return default
    if not isinstance(value, int | float):
        raise ValueError(f"{key} must be numeric")
    return float(value)
