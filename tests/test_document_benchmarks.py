from __future__ import annotations

from gia_evidence_finder import (
    BenchmarkSplit,
    EvidenceExtractor,
    KeywordOverlapBaseline,
    benchmark_series_by_id,
    benchmark_series_suites,
    document_benchmark_suite_by_id,
    document_benchmark_suites,
    evaluate_suite,
    mixed_evidence_benchmark_v3,
    non_readme_benchmark_suite,
)


def test_non_readme_benchmark_suite_has_genre_diversity() -> None:
    suite = non_readme_benchmark_suite()

    assert suite.id == "non_readme_docs_v1"
    assert int(suite.metadata["document_count"]) == 5
    assert len(suite.cases) == 15
    assert {"spec", "runbook", "release_notes", "issue"} <= set(
        suite.metadata["genres"].split(",")
    )
    assert any(case.expect_abstain for case in suite.cases)
    assert all(case.curation.reviewed for case in suite.cases)
    assert {
        case.curation.source for case in suite.cases
    } == {"curated_non_readme_excerpt"}
    assert document_benchmark_suite_by_id(suite.id).id == suite.id
    assert [suite.id for suite in document_benchmark_suites()] == ["non_readme_docs_v1"]


def test_intent_aware_ranker_beats_keyword_on_non_readme_suite() -> None:
    suite = non_readme_benchmark_suite()

    default_report = evaluate_suite(EvidenceExtractor.default(), suite.cases)
    keyword_report = evaluate_suite(EvidenceExtractor(ranker=KeywordOverlapBaseline()), suite.cases)

    assert default_report.mean_reciprocal_rank >= keyword_report.mean_reciprocal_rank
    assert default_report.recall_at_3 >= 0.85
    assert default_report.forbidden_supported_top1_rate == 0.0


def test_mixed_evidence_benchmark_v3_registers_non_readme_cases() -> None:
    series = mixed_evidence_benchmark_v3()

    assert series.id == "mixed_evidence_benchmark_v3"
    assert series.metadata["status"] == "mixed_frozen"
    assert int(series.metadata["non_readme_case_count"]) == 15
    assert benchmark_series_by_id(series.id).id == series.id
    assert {
        case.document.id
        for suite in benchmark_series_suites(series)
        for case in suite.cases
    } >= {
        "storage_spec",
        "api_runbook",
        "planning_spec",
        "evidence_release_notes",
        "issue_discussion",
    }
    assert benchmark_series_suites(series, split=BenchmarkSplit.TEST)[0].id == (
        "mixed_evidence_benchmark_v3_test"
    )
