from __future__ import annotations

from gia_evidence_finder import (
    BenchmarkSplit,
    benchmark_series_by_id,
    benchmark_series_suites,
    domain_evidence_benchmark_v4,
    evidence_benchmark_v2,
)


def test_evidence_benchmark_v2_defines_frozen_splits() -> None:
    series = evidence_benchmark_v2()

    assert series.id == "evidence_benchmark_v2"
    assert set(series.splits) == {BenchmarkSplit.TRAIN, BenchmarkSplit.DEV, BenchmarkSplit.TEST}
    assert series.metadata["status"] == "starter_frozen"
    assert int(series.metadata["train_case_count"]) > int(series.metadata["dev_case_count"])
    assert int(series.metadata["test_case_count"]) == 16


def test_benchmark_series_lookup_and_split_selection() -> None:
    series = benchmark_series_by_id("evidence_benchmark_v2")

    all_suites = benchmark_series_suites(series)
    test_suite = benchmark_series_suites(series, split=BenchmarkSplit.TEST)

    assert [suite.metadata["split"] for suite in all_suites] == ["train", "dev", "test"]
    assert len(test_suite) == 1
    assert test_suite[0].id == "evidence_benchmark_v2_test"


def test_domain_evidence_benchmark_v4_defines_document_isolated_splits() -> None:
    series = domain_evidence_benchmark_v4()

    split_documents = {
        split: {case.document.id for case in suite.cases} for split, suite in series.splits.items()
    }

    assert series.id == "domain_evidence_benchmark_v4"
    assert int(series.metadata["case_count"]) == 320
    assert int(series.metadata["document_count"]) == 40
    assert int(series.metadata["dev_case_count"]) >= 50
    assert int(series.metadata["test_case_count"]) >= 50
    assert not (split_documents[BenchmarkSplit.TRAIN] & split_documents[BenchmarkSplit.DEV])
    assert not (split_documents[BenchmarkSplit.TRAIN] & split_documents[BenchmarkSplit.TEST])
    assert not (split_documents[BenchmarkSplit.DEV] & split_documents[BenchmarkSplit.TEST])
    assert benchmark_series_by_id(series.id).id == series.id
