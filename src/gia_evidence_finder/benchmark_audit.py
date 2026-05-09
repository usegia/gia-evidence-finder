from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, replace

from gia_evidence_finder.benchmark_series import BenchmarkSeries, BenchmarkSplit
from gia_evidence_finder.contracts import BenchmarkCase, BenchmarkSuite


@dataclass(frozen=True)
class BenchmarkSuiteAudit:
    suite_id: str
    case_count: int
    reviewed_case_count: int
    unreviewed_case_count: int
    support_case_count: int
    abstain_case_count: int
    relation_case_count: int
    support_label_count: int
    near_miss_label_count: int
    contradiction_label_count: int
    insufficient_context_label_count: int
    forbidden_label_count: int
    document_count: int
    curation_source_counts: dict[str, int]
    difficulty_counts: dict[str, int]
    phenomenon_counts: dict[str, int]
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class BenchmarkSeriesAudit:
    series_id: str
    case_count: int
    reviewed_case_count: int
    train_case_count: int
    dev_case_count: int
    test_case_count: int
    label_counts: dict[str, int]
    curation_source_counts: dict[str, int]
    difficulty_counts: dict[str, int]
    phenomenon_counts: dict[str, int]
    domain_counts: dict[str, int]
    genre_counts: dict[str, int]
    split_audits: dict[str, BenchmarkSuiteAudit]
    warnings: tuple[str, ...] = ()


def audit_benchmark_suite(suite: BenchmarkSuite) -> BenchmarkSuiteAudit:
    cases = suite.cases
    support_case_count = sum(1 for case in cases if case.support_span_ids)
    abstain_case_count = sum(1 for case in cases if case.expect_abstain)
    relation_case_count = sum(1 for case in cases if case.intent.relations)
    reviewed_case_count = sum(1 for case in cases if case.curation.reviewed)
    difficulty_counts = Counter(case.curation.difficulty for case in cases)
    curation_source_counts = Counter(case.curation.source for case in cases)
    phenomenon_counts = Counter[str]()
    for case in cases:
        phenomenon_counts.update(case.curation.phenomena)
    audit = BenchmarkSuiteAudit(
        suite_id=suite.id,
        case_count=len(cases),
        reviewed_case_count=reviewed_case_count,
        unreviewed_case_count=len(cases) - reviewed_case_count,
        support_case_count=support_case_count,
        abstain_case_count=abstain_case_count,
        relation_case_count=relation_case_count,
        support_label_count=sum(len(case.support_span_ids) for case in cases),
        near_miss_label_count=sum(len(case.near_miss_span_ids) for case in cases),
        contradiction_label_count=sum(len(case.contradiction_span_ids) for case in cases),
        insufficient_context_label_count=sum(
            len(case.insufficient_context_span_ids) for case in cases
        ),
        forbidden_label_count=sum(len(case.forbidden_span_ids) for case in cases),
        document_count=len({case.document.id for case in cases}),
        curation_source_counts=dict(sorted(curation_source_counts.items())),
        difficulty_counts=dict(sorted(difficulty_counts.items())),
        phenomenon_counts=dict(sorted(phenomenon_counts.items())),
    )
    return _with_suite_warnings(audit)


def audit_benchmark_series(series: BenchmarkSeries) -> BenchmarkSeriesAudit:
    split_audits = {
        split.value: audit_benchmark_suite(series.splits[split]) for split in BenchmarkSplit
    }
    label_counts = Counter[str]()
    curation_source_counts = Counter[str]()
    difficulty_counts = Counter[str]()
    phenomenon_counts = Counter[str]()
    domain_counts = Counter[str]()
    genre_counts = Counter[str]()
    for case in series.cases:
        label_counts.update(_case_label_counts(case))
        curation_source_counts.update((case.curation.source,))
        difficulty_counts.update((case.curation.difficulty,))
        phenomenon_counts.update(case.curation.phenomena)
        domain = case.document.metadata.get("domain")
        genre = case.document.metadata.get("genre")
        if domain:
            domain_counts.update((domain,))
        if genre:
            genre_counts.update((genre,))
    audit = BenchmarkSeriesAudit(
        series_id=series.id,
        case_count=len(series.cases),
        reviewed_case_count=sum(
            split_audit.reviewed_case_count for split_audit in split_audits.values()
        ),
        train_case_count=split_audits[BenchmarkSplit.TRAIN.value].case_count,
        dev_case_count=split_audits[BenchmarkSplit.DEV.value].case_count,
        test_case_count=split_audits[BenchmarkSplit.TEST.value].case_count,
        label_counts=dict(sorted(label_counts.items())),
        curation_source_counts=dict(sorted(curation_source_counts.items())),
        difficulty_counts=dict(sorted(difficulty_counts.items())),
        phenomenon_counts=dict(sorted(phenomenon_counts.items())),
        domain_counts=dict(sorted(domain_counts.items())),
        genre_counts=dict(sorted(genre_counts.items())),
        split_audits=split_audits,
    )
    return _with_series_warnings(audit)


def _case_label_counts(case: BenchmarkCase) -> Counter[str]:
    counts = Counter[str]()
    counts["support"] += len(case.support_span_ids)
    counts["near_miss"] += len(case.near_miss_span_ids)
    counts["contradiction"] += len(case.contradiction_span_ids)
    counts["insufficient_context"] += len(case.insufficient_context_span_ids)
    counts["forbidden"] += len(case.forbidden_span_ids)
    counts["abstain_case"] += int(case.expect_abstain)
    counts["relation_case"] += int(bool(case.intent.relations))
    return counts


def _with_suite_warnings(audit: BenchmarkSuiteAudit) -> BenchmarkSuiteAudit:
    warnings: list[str] = []
    if audit.reviewed_case_count == 0:
        warnings.append("suite has no reviewed cases")
    if audit.support_case_count == 0:
        warnings.append("suite has no supported cases")
    if audit.abstain_case_count == 0:
        warnings.append("suite has no abstention cases")
    if audit.forbidden_label_count == 0:
        warnings.append("suite has no forbidden labels")
    return replace(audit, warnings=tuple(warnings))


def _with_series_warnings(audit: BenchmarkSeriesAudit) -> BenchmarkSeriesAudit:
    warnings: list[str] = []
    if audit.reviewed_case_count < 200:
        warnings.append("series has fewer than 200 reviewed cases")
    if audit.split_audits[BenchmarkSplit.DEV.value].reviewed_case_count < 50:
        warnings.append("dev split has fewer than 50 reviewed cases")
    if audit.split_audits[BenchmarkSplit.TEST.value].reviewed_case_count < 50:
        warnings.append("test split has fewer than 50 reviewed cases")
    for label in ("near_miss", "contradiction", "insufficient_context"):
        if audit.label_counts.get(label, 0) == 0:
            warnings.append(f"series has no {label} labels")
    for phenomenon in ("direct_support", "abstention", "hard_negative", "relation_binding"):
        if audit.phenomenon_counts.get(phenomenon, 0) == 0:
            warnings.append(f"series has no {phenomenon} phenomenon coverage")
    split_warnings = tuple(
        f"{split}: {warning}"
        for split, split_audit in audit.split_audits.items()
        for warning in split_audit.warnings
    )
    return BenchmarkSeriesAudit(
        series_id=audit.series_id,
        case_count=audit.case_count,
        reviewed_case_count=audit.reviewed_case_count,
        train_case_count=audit.train_case_count,
        dev_case_count=audit.dev_case_count,
        test_case_count=audit.test_case_count,
        label_counts=audit.label_counts,
        curation_source_counts=audit.curation_source_counts,
        difficulty_counts=audit.difficulty_counts,
        phenomenon_counts=audit.phenomenon_counts,
        domain_counts=audit.domain_counts,
        genre_counts=audit.genre_counts,
        split_audits=audit.split_audits,
        warnings=(*warnings, *split_warnings),
    )


def audit_expanded_domain_series(series: BenchmarkSeries) -> BenchmarkSeriesAudit:
    audit = audit_benchmark_series(series)
    warnings = list(audit.warnings)
    if audit.case_count < 300:
        warnings.append("expanded domain series has fewer than 300 reviewed cases")
    document_ids = {case.document.id for case in series.cases}
    if len(document_ids) < 40:
        warnings.append("expanded domain series has fewer than 40 source documents")
    non_technical_cases = sum(
        count for domain, count in audit.domain_counts.items() if domain != "technical_product"
    )
    if audit.case_count and non_technical_cases / audit.case_count < 0.25:
        warnings.append("expanded domain series has less than 25% non-technical coverage")
    negative_label_count = sum(
        audit.label_counts.get(label, 0)
        for label in ("near_miss", "contradiction", "insufficient_context")
    )
    if negative_label_count < 50:
        warnings.append("expanded domain series has fewer than 50 negative diagnostic labels")
    if len(audit.domain_counts) < 4:
        warnings.append("expanded domain series has fewer than four domains")
    if len(audit.genre_counts) < 12:
        warnings.append("expanded domain series has fewer than twelve genres")
    split_documents = {
        split.value: {case.document.id for case in series.splits[split].cases}
        for split in BenchmarkSplit
    }
    for left in BenchmarkSplit:
        for right in BenchmarkSplit:
            if left.value >= right.value:
                continue
            overlap = split_documents[left.value] & split_documents[right.value]
            if overlap:
                warnings.append(
                    f"{left.value}/{right.value} split document overlap: {sorted(overlap)!r}"
                )
    missing_metadata = [
        case.id
        for case in series.cases
        if not case.document.metadata.get("source_hash")
        or not case.document.metadata.get("review_source")
        or not case.curation.reviewed
    ]
    if missing_metadata:
        warnings.append(
            f"expanded domain series has cases missing review/source metadata: "
            f"{missing_metadata[:5]!r}"
        )
    return BenchmarkSeriesAudit(
        series_id=audit.series_id,
        case_count=audit.case_count,
        reviewed_case_count=audit.reviewed_case_count,
        train_case_count=audit.train_case_count,
        dev_case_count=audit.dev_case_count,
        test_case_count=audit.test_case_count,
        label_counts=audit.label_counts,
        curation_source_counts=audit.curation_source_counts,
        difficulty_counts=audit.difficulty_counts,
        phenomenon_counts=audit.phenomenon_counts,
        domain_counts=audit.domain_counts,
        genre_counts=audit.genre_counts,
        split_audits=audit.split_audits,
        warnings=tuple(warnings),
    )
