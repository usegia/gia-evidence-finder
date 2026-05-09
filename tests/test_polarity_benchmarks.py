from __future__ import annotations

from gia_evidence_finder import (
    BenchmarkSplit,
    EvidenceExtractor,
    KeywordOverlapBaseline,
    SupportLabel,
    benchmark_series_by_id,
    benchmark_series_suites,
    evaluate_suite,
    polarity_benchmark_suite,
    polarity_benchmark_suite_by_id,
    polarity_benchmark_suites,
    polarity_evidence_benchmark_v1,
    polarity_score,
)


def test_polarity_score_detects_negated_anchor_mismatch() -> None:
    score = polarity_score(
        ("ultrachess includes evaluation search and an opening book",),
        "No evaluation, no search, no opening book.",
    )

    assert score.shared_anchor_count >= 3
    assert score.mismatch >= 0.8
    assert score.contradiction >= 0.8


def test_polarity_score_allows_negative_claim_support() -> None:
    score = polarity_score(
        ("ultrachess has no evaluation no search and no opening book",),
        "No evaluation, no search, no opening book.",
    )

    assert score.shared_anchor_count >= 3
    assert score.mismatch == 0.0
    assert score.intent_negated_anchor_coverage > 0.0
    assert score.span_negated_anchor_coverage > 0.0


def test_polarity_score_treats_not_just_as_exclusivity_not_entity_negation() -> None:
    positive = polarity_score(
        ("modules may be developed in Python",),
        "Modules may be developed in any dynamic language, not just Python.",
    )
    exclusive = polarity_score(
        ("modules may be developed only in Python",),
        "Modules may be developed in any dynamic language, not just Python.",
    )

    assert positive.mismatch == 0.0
    assert exclusive.mismatch > positive.mismatch


def test_polarity_benchmark_suite_has_targeted_label_coverage() -> None:
    suite = polarity_benchmark_suite()

    assert suite.id == "polarity_negation_v1"
    assert int(suite.metadata["document_count"]) == 5
    assert len(suite.cases) >= 20
    assert int(suite.metadata["contradiction_case_count"]) >= 8
    assert int(suite.metadata["negative_support_case_count"]) >= 6
    assert all(case.curation.reviewed for case in suite.cases)
    assert polarity_benchmark_suite_by_id(suite.id).id == suite.id
    assert [suite.id for suite in polarity_benchmark_suites()] == ["polarity_negation_v1"]


def test_default_extractor_clears_polarity_benchmark() -> None:
    suite = polarity_benchmark_suite()
    report = evaluate_suite(EvidenceExtractor.default(), suite.cases)

    assert report.recall_at_3 == 1.0
    assert report.top1_support_accuracy >= 0.95
    assert report.abstain_accuracy == 1.0
    assert report.decision_accuracy == 1.0
    assert report.forbidden_supported_top1_rate == 0.0
    assert {
        negative.label: negative.supported_top1_rate
        for negative in report.negative_label_reports
    }[SupportLabel.CONTRADICTS] == 0.0


def test_default_extractor_beats_keyword_on_polarity_decisions() -> None:
    suite = polarity_benchmark_suite()
    default_report = evaluate_suite(EvidenceExtractor.default(), suite.cases)
    keyword_report = evaluate_suite(EvidenceExtractor(ranker=KeywordOverlapBaseline()), suite.cases)

    assert default_report.decision_accuracy > keyword_report.decision_accuracy
    assert default_report.forbidden_supported_top1_rate < (
        keyword_report.forbidden_supported_top1_rate
    )


def test_polarity_series_is_registered_with_train_dev_test_splits() -> None:
    series = polarity_evidence_benchmark_v1()

    assert series.id == "polarity_evidence_benchmark_v1"
    assert series.metadata["status"] == "focused_polarity"
    assert benchmark_series_by_id(series.id).id == series.id
    assert {
        suite.metadata["split"] for suite in benchmark_series_suites(series)
    } == {"train", "dev", "test"}
    assert benchmark_series_suites(series, split=BenchmarkSplit.TEST)[0].cases
