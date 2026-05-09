from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from types import MappingProxyType

from gia_evidence_finder.contracts import BenchmarkCase, BenchmarkSuite
from gia_evidence_finder.document_benchmarks import non_readme_benchmark_suite
from gia_evidence_finder.domain_benchmarks import domain_benchmark_suite
from gia_evidence_finder.polarity_benchmarks import polarity_benchmark_suite
from gia_evidence_finder.quantifier_benchmarks import (
    quantifier_benchmark_suite,
    quantifier_binding_benchmark_suite,
)
from gia_evidence_finder.readme_benchmarks import (
    adversarial_readme_benchmark_suite,
    hard_readme_benchmark_suite,
    popular_readme_benchmark_suite,
    relation_readme_benchmark_suite,
)


class BenchmarkSplit(StrEnum):
    TRAIN = "train"
    DEV = "dev"
    TEST = "test"


@dataclass(frozen=True)
class BenchmarkSeries:
    id: str
    name: str
    splits: Mapping[BenchmarkSplit, BenchmarkSuite]
    description: str = ""
    metadata: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.id.strip():
            raise ValueError("benchmark series id must not be empty")
        if not self.name.strip():
            raise ValueError("benchmark series name must not be empty")
        if set(self.splits) != {BenchmarkSplit.TRAIN, BenchmarkSplit.DEV, BenchmarkSplit.TEST}:
            raise ValueError("benchmark series must define train, dev, and test splits")
        object.__setattr__(self, "splits", MappingProxyType(dict(self.splits)))
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

    @property
    def cases(self) -> tuple[BenchmarkCase, ...]:
        return tuple(case for suite in self.splits.values() for case in suite.cases)


def evidence_benchmark_v2() -> BenchmarkSeries:
    popular = popular_readme_benchmark_suite()
    hard = hard_readme_benchmark_suite()
    adversarial = adversarial_readme_benchmark_suite()
    relation = relation_readme_benchmark_suite()
    train_cases = (*popular.cases, *hard.cases)
    dev_cases = adversarial.cases
    test_cases = relation.cases
    return BenchmarkSeries(
        id="evidence_benchmark_v2",
        name="Evidence extraction benchmark v2",
        description=(
            "Frozen starter train/dev/test split for typed claim-to-evidence extraction. "
            "The split uses popular README cases for broad recall, hard/adversarial cases "
            "for abstention, and relation cases for exact subject/predicate binding."
        ),
        splits={
            BenchmarkSplit.TRAIN: _split_suite(
                series_id="evidence_benchmark_v2",
                split=BenchmarkSplit.TRAIN,
                cases=train_cases,
                source_suites=(popular.id, hard.id),
            ),
            BenchmarkSplit.DEV: _split_suite(
                series_id="evidence_benchmark_v2",
                split=BenchmarkSplit.DEV,
                cases=dev_cases,
                source_suites=(adversarial.id,),
            ),
            BenchmarkSplit.TEST: _split_suite(
                series_id="evidence_benchmark_v2",
                split=BenchmarkSplit.TEST,
                cases=test_cases,
                source_suites=(relation.id,),
            ),
        },
        metadata={
            "case_count": str(len(train_cases) + len(dev_cases) + len(test_cases)),
            "train_case_count": str(len(train_cases)),
            "dev_case_count": str(len(dev_cases)),
            "test_case_count": str(len(test_cases)),
            "status": "starter_frozen",
        },
    )


def mixed_evidence_benchmark_v3() -> BenchmarkSeries:
    v2 = evidence_benchmark_v2()
    non_readme = non_readme_benchmark_suite()
    non_readme_train = tuple(
        case
        for case in non_readme.cases
        if case.document.id in {"storage_spec", "api_runbook", "planning_spec"}
    )
    non_readme_dev = tuple(
        case for case in non_readme.cases if case.document.id == "evidence_release_notes"
    )
    non_readme_test = tuple(
        case for case in non_readme.cases if case.document.id == "issue_discussion"
    )
    train_cases = (*v2.splits[BenchmarkSplit.TRAIN].cases, *non_readme_train)
    dev_cases = (*v2.splits[BenchmarkSplit.DEV].cases, *non_readme_dev)
    test_cases = (*v2.splits[BenchmarkSplit.TEST].cases, *non_readme_test)
    return BenchmarkSeries(
        id="mixed_evidence_benchmark_v3",
        name="Mixed evidence extraction benchmark v3",
        description=(
            "Frozen mixed train/dev/test split that keeps the README v2 cases and adds "
            "curated non-README spec, runbook, release-note, and issue-style documents."
        ),
        splits={
            BenchmarkSplit.TRAIN: _split_suite(
                series_id="mixed_evidence_benchmark_v3",
                split=BenchmarkSplit.TRAIN,
                cases=train_cases,
                source_suites=("evidence_benchmark_v2_train", "non_readme_docs_v1"),
            ),
            BenchmarkSplit.DEV: _split_suite(
                series_id="mixed_evidence_benchmark_v3",
                split=BenchmarkSplit.DEV,
                cases=dev_cases,
                source_suites=("evidence_benchmark_v2_dev", "non_readme_docs_v1"),
            ),
            BenchmarkSplit.TEST: _split_suite(
                series_id="mixed_evidence_benchmark_v3",
                split=BenchmarkSplit.TEST,
                cases=test_cases,
                source_suites=("evidence_benchmark_v2_test", "non_readme_docs_v1"),
            ),
        },
        metadata={
            "case_count": str(len(train_cases) + len(dev_cases) + len(test_cases)),
            "train_case_count": str(len(train_cases)),
            "dev_case_count": str(len(dev_cases)),
            "test_case_count": str(len(test_cases)),
            "status": "mixed_frozen",
            "non_readme_case_count": str(len(non_readme.cases)),
        },
    )


def polarity_evidence_benchmark_v1() -> BenchmarkSeries:
    polarity = polarity_benchmark_suite()
    train_cases = tuple(
        case
        for case in polarity.cases
        if case.document.id in {"ultrachess_readme", "api_access_runbook"}
    )
    dev_cases = tuple(
        case
        for case in polarity.cases
        if case.document.id in {"planner_contract", "evidence_runtime"}
    )
    test_cases = tuple(case for case in polarity.cases if case.document.id == "language_policy")
    return BenchmarkSeries(
        id="polarity_evidence_benchmark_v1",
        name="Polarity evidence extraction benchmark v1",
        description=(
            "Focused train/dev/test split for negation, contradiction, negative "
            "support, without/reject wording, and not-just exclusivity behavior."
        ),
        splits={
            BenchmarkSplit.TRAIN: _split_suite(
                series_id="polarity_evidence_benchmark_v1",
                split=BenchmarkSplit.TRAIN,
                cases=train_cases,
                source_suites=(polarity.id,),
            ),
            BenchmarkSplit.DEV: _split_suite(
                series_id="polarity_evidence_benchmark_v1",
                split=BenchmarkSplit.DEV,
                cases=dev_cases,
                source_suites=(polarity.id,),
            ),
            BenchmarkSplit.TEST: _split_suite(
                series_id="polarity_evidence_benchmark_v1",
                split=BenchmarkSplit.TEST,
                cases=test_cases,
                source_suites=(polarity.id,),
            ),
        },
        metadata={
            "case_count": str(len(train_cases) + len(dev_cases) + len(test_cases)),
            "train_case_count": str(len(train_cases)),
            "dev_case_count": str(len(dev_cases)),
            "test_case_count": str(len(test_cases)),
            "status": "focused_polarity",
            "source_suite": polarity.id,
        },
    )


def domain_evidence_benchmark_v4() -> BenchmarkSeries:
    domain = domain_benchmark_suite()
    train_cases = tuple(
        case
        for case in domain.cases
        if case.document.metadata.get("genre")
        in {
            "ticket description",
            "comment thread",
            "prd snippet",
            "resume",
            "profile bio",
            "listing description",
            "lease clause",
            "api doc",
            "spec",
        }
    )
    dev_cases = tuple(
        case
        for case in domain.cases
        if case.document.metadata.get("genre")
        in {
            "incident update",
            "company page",
            "building rules",
            "release notes",
            "runbook",
        }
    )
    test_cases = tuple(
        case
        for case in domain.cases
        if case.document.metadata.get("genre")
        in {
            "changelog",
            "press snippet",
            "investor blurb",
            "advisor blurb",
            "tenant review",
            "inspection note",
            "issue discussion",
            "PR discussion",
        }
    )
    return BenchmarkSeries(
        id="domain_evidence_benchmark_v4",
        name="Domain evidence extraction benchmark v4",
        description=(
            "Expanded reviewed domain benchmark with project-management, people-search, "
            "apartment-search, and technical/product source excerpts. The split is "
            "document-isolated and stresses retrieval, support decisions, and safety "
            "refusal metrics."
        ),
        splits={
            BenchmarkSplit.TRAIN: _split_suite(
                series_id="domain_evidence_benchmark_v4",
                split=BenchmarkSplit.TRAIN,
                cases=train_cases,
                source_suites=(domain.id,),
            ),
            BenchmarkSplit.DEV: _split_suite(
                series_id="domain_evidence_benchmark_v4",
                split=BenchmarkSplit.DEV,
                cases=dev_cases,
                source_suites=(domain.id,),
            ),
            BenchmarkSplit.TEST: _split_suite(
                series_id="domain_evidence_benchmark_v4",
                split=BenchmarkSplit.TEST,
                cases=test_cases,
                source_suites=(domain.id,),
            ),
        },
        metadata={
            "case_count": str(len(train_cases) + len(dev_cases) + len(test_cases)),
            "train_case_count": str(len(train_cases)),
            "dev_case_count": str(len(dev_cases)),
            "test_case_count": str(len(test_cases)),
            "document_count": domain.metadata["document_count"],
            "status": "expanded_reviewed",
            "source_suite": domain.id,
        },
    )


def quantifier_evidence_benchmark_v1() -> BenchmarkSeries:
    quantifier = quantifier_benchmark_suite()
    train_cases = tuple(
        case
        for case in quantifier.cases
        if case.document.id in {"project_dates", "ticket_metrics"}
    )
    dev_cases = tuple(
        case
        for case in quantifier.cases
        if case.document.id in {"people_dates", "technical_metrics"}
    )
    test_cases = tuple(
        case for case in quantifier.cases if case.document.id == "apartment_listing"
    )
    return BenchmarkSeries(
        id="quantifier_evidence_benchmark_v1",
        name="Quantifier evidence extraction benchmark v1",
        description=(
            "Focused train/dev/test split for deterministic numeric, date, currency, "
            "duration, percent, multiplier, and threshold evidence behavior."
        ),
        splits={
            BenchmarkSplit.TRAIN: _split_suite(
                series_id="quantifier_evidence_benchmark_v1",
                split=BenchmarkSplit.TRAIN,
                cases=train_cases,
                source_suites=(quantifier.id,),
            ),
            BenchmarkSplit.DEV: _split_suite(
                series_id="quantifier_evidence_benchmark_v1",
                split=BenchmarkSplit.DEV,
                cases=dev_cases,
                source_suites=(quantifier.id,),
            ),
            BenchmarkSplit.TEST: _split_suite(
                series_id="quantifier_evidence_benchmark_v1",
                split=BenchmarkSplit.TEST,
                cases=test_cases,
                source_suites=(quantifier.id,),
            ),
        },
        metadata={
            "case_count": str(len(train_cases) + len(dev_cases) + len(test_cases)),
            "train_case_count": str(len(train_cases)),
            "dev_case_count": str(len(dev_cases)),
            "test_case_count": str(len(test_cases)),
            "status": "focused_quantifier",
            "source_suite": quantifier.id,
        },
    )


def quantifier_binding_evidence_benchmark_v2() -> BenchmarkSeries:
    binding = quantifier_binding_benchmark_suite()
    train_cases = binding.cases[:8]
    dev_cases = binding.cases[8:16]
    test_cases = binding.cases[16:]
    return BenchmarkSeries(
        id="quantifier_binding_evidence_benchmark_v2",
        name="Quantifier binding evidence benchmark v2",
        description=(
            "Focused train/dev/test split for role-bound date, number, money, "
            "metric, count, and version evidence behavior."
        ),
        splits={
            BenchmarkSplit.TRAIN: _split_suite(
                series_id="quantifier_binding_evidence_benchmark_v2",
                split=BenchmarkSplit.TRAIN,
                cases=train_cases,
                source_suites=(binding.id,),
            ),
            BenchmarkSplit.DEV: _split_suite(
                series_id="quantifier_binding_evidence_benchmark_v2",
                split=BenchmarkSplit.DEV,
                cases=dev_cases,
                source_suites=(binding.id,),
            ),
            BenchmarkSplit.TEST: _split_suite(
                series_id="quantifier_binding_evidence_benchmark_v2",
                split=BenchmarkSplit.TEST,
                cases=test_cases,
                source_suites=(binding.id,),
            ),
        },
        metadata={
            "case_count": str(len(binding.cases)),
            "train_case_count": str(len(train_cases)),
            "dev_case_count": str(len(dev_cases)),
            "test_case_count": str(len(test_cases)),
            "status": "focused_quantifier_binding",
            "source_suite": binding.id,
        },
    )


def benchmark_series() -> tuple[BenchmarkSeries, ...]:
    return (
        evidence_benchmark_v2(),
        mixed_evidence_benchmark_v3(),
        polarity_evidence_benchmark_v1(),
        domain_evidence_benchmark_v4(),
        quantifier_evidence_benchmark_v1(),
        quantifier_binding_evidence_benchmark_v2(),
    )


def benchmark_series_by_id(series_id: str) -> BenchmarkSeries:
    series_by_id = {series.id: series for series in benchmark_series()}
    try:
        return series_by_id[series_id]
    except KeyError as exc:
        raise ValueError(f"unknown benchmark series {series_id!r}") from exc


def benchmark_series_suites(
    series: BenchmarkSeries,
    *,
    split: BenchmarkSplit | None = None,
) -> tuple[BenchmarkSuite, ...]:
    if split is not None:
        return (series.splits[split],)
    return tuple(series.splits[split_id] for split_id in BenchmarkSplit)


def _split_suite(
    *,
    series_id: str,
    split: BenchmarkSplit,
    cases: tuple[BenchmarkCase, ...],
    source_suites: tuple[str, ...],
) -> BenchmarkSuite:
    return BenchmarkSuite(
        id=f"{series_id}_{split.value}",
        name=f"{series_id} {split.value} split",
        description=f"{series_id} frozen {split.value} split.",
        cases=cases,
        metadata={
            "series_id": series_id,
            "split": split.value,
            "case_count": str(len(cases)),
            "source_suites": ",".join(source_suites),
        },
    )
