from __future__ import annotations

from gia_evidence_finder import (
    BenchmarkSplit,
    EvidenceExtractor,
    KeywordOverlapBaseline,
    adversarial_readme_benchmark_suite,
    benchmark_series_suites,
    evaluate_suite,
    evidence_benchmark_v2,
    hard_readme_benchmark_suite,
    popular_readme_benchmark_suite,
    readme_baseline_specs,
    readme_benchmark_suite_by_id,
    readme_benchmark_suites,
    relation_readme_benchmark_suite,
    run_calibrated_holdout,
    run_popular_readme_baselines,
    run_readme_baselines,
)


def test_popular_readme_benchmark_suite_has_curated_source_metadata() -> None:
    suite = popular_readme_benchmark_suite()

    assert suite.id == "popular_readme_v1"
    assert int(suite.metadata["document_count"]) >= 8
    assert len(suite.cases) >= 20
    assert all(case.document.source for case in suite.cases)
    assert {
        case.document.id
        for case in suite.cases
    } >= {"react", "kubernetes", "django", "fastapi", "pytorch", "rust", "vscode", "ansible"}


def test_hard_readme_benchmark_suite_adds_adversarial_cases() -> None:
    suite = hard_readme_benchmark_suite()

    assert suite.id == "hard_readme_v1"
    assert suite.metadata["source_suite"] == "popular_readme_v1"
    assert int(suite.metadata["document_count"]) >= 8
    assert len(suite.cases) >= 16
    assert any(case.expect_abstain for case in suite.cases)
    assert any(case.forbidden_span_ids for case in suite.cases)
    assert readme_benchmark_suite_by_id(suite.id).id == suite.id
    assert {suite.id for suite in readme_benchmark_suites()} >= {
        "popular_readme_v1",
        "hard_readme_v1",
        "adversarial_readme_v1",
        "relation_readme_v1",
    }


def test_adversarial_readme_benchmark_suite_is_diagnostic() -> None:
    suite = adversarial_readme_benchmark_suite()

    assert suite.id == "adversarial_readme_v1"
    assert suite.metadata["diagnostic"] == "true"
    assert len(suite.cases) >= 16
    assert any(case.expect_abstain for case in suite.cases)
    assert any(case.intent.excluded_facets for case in suite.cases)
    assert any(case.support_span_ids for case in suite.cases)
    assert readme_benchmark_suite_by_id(suite.id).id == suite.id


def test_relation_readme_benchmark_suite_focuses_claim_binding() -> None:
    suite = relation_readme_benchmark_suite()

    assert suite.id == "relation_readme_v1"
    assert suite.metadata["focus"] == "relation_binding"
    assert len(suite.cases) == 16
    assert sum(1 for case in suite.cases if case.expect_abstain) == 8
    assert all(case.forbidden_span_ids for case in suite.cases)
    assert readme_benchmark_suite_by_id(suite.id).id == suite.id


def test_intent_aware_ranker_beats_keyword_baseline_on_popular_readmes() -> None:
    suite = popular_readme_benchmark_suite()

    default_report = evaluate_suite(EvidenceExtractor.default(), suite.cases)
    keyword_report = evaluate_suite(EvidenceExtractor(ranker=KeywordOverlapBaseline()), suite.cases)

    assert default_report.mean_reciprocal_rank > keyword_report.mean_reciprocal_rank
    assert default_report.top1_support_accuracy > keyword_report.top1_support_accuracy
    assert default_report.recall_at_3 >= 0.95
    assert default_report.forbidden_top1_rate == 0.0


def test_intent_aware_ranker_clears_hard_readme_suite() -> None:
    suite = hard_readme_benchmark_suite()
    support_cases = tuple(case for case in suite.cases if not case.expect_abstain)
    abstain_cases = tuple(case for case in suite.cases if case.expect_abstain)

    default_report = evaluate_suite(EvidenceExtractor.default(), support_cases)
    keyword_report = evaluate_suite(
        EvidenceExtractor(ranker=KeywordOverlapBaseline()),
        support_cases,
    )
    abstain_report = evaluate_suite(EvidenceExtractor.default(), abstain_cases)

    assert default_report.mean_reciprocal_rank >= keyword_report.mean_reciprocal_rank
    assert default_report.recall_at_3 >= 0.95
    assert default_report.top1_support_accuracy >= 0.95
    assert default_report.forbidden_top1_rate == 0.0
    assert abstain_report.abstain_accuracy == 1.0


def test_popular_readme_baseline_experiment_runner_reports_named_results() -> None:
    results = run_popular_readme_baselines(popular_readme_benchmark_suite())

    reports = {result.name: result.report for result in results}

    assert set(reports) == {
        "intent_aware_default",
        "keyword_overlap_baseline",
        "bm25_baseline",
    }
    assert reports["intent_aware_default"].case_count >= 20
    assert all(result.cases for result in results)
    assert (
        reports["intent_aware_default"].mean_reciprocal_rank
        > reports["keyword_overlap_baseline"].mean_reciprocal_rank
    )


def test_generic_readme_baseline_runner_accepts_any_readme_suite() -> None:
    results = run_readme_baselines(hard_readme_benchmark_suite())

    reports = {result.name: result.report for result in results}

    assert set(reports) == {
        "intent_aware_default",
        "keyword_overlap_baseline",
        "bm25_baseline",
    }
    assert reports["intent_aware_default"].case_count >= 16


def test_readme_baseline_runner_can_attach_threshold_calibration() -> None:
    results = run_readme_baselines(relation_readme_benchmark_suite(), calibrate=True)

    calibrations = {result.name: result.calibration for result in results}

    assert set(calibrations) == {
        "intent_aware_default",
        "keyword_overlap_baseline",
        "bm25_baseline",
    }
    assert all(calibration is not None for calibration in calibrations.values())
    assert calibrations["intent_aware_default"] is not None
    assert calibrations["intent_aware_default"].best.report.case_count == 16


def test_calibrated_holdout_uses_dev_threshold_on_test_split() -> None:
    series = evidence_benchmark_v2()
    dev_suite = benchmark_series_suites(series, split=BenchmarkSplit.DEV)[0]
    test_suite = benchmark_series_suites(series, split=BenchmarkSplit.TEST)[0]

    results = run_calibrated_holdout(
        readme_baseline_specs(),
        dev_suite=dev_suite,
        test_suite=test_suite,
    )

    reports = {result.name: result for result in results}
    assert set(reports) == {
        "intent_aware_default",
        "keyword_overlap_baseline",
        "bm25_baseline",
    }
    assert reports["intent_aware_default"].dev.calibration is not None
    assert reports["intent_aware_default"].selected_threshold is None
    assert reports["keyword_overlap_baseline"].dev.calibration is not None
    assert reports["keyword_overlap_baseline"].selected_threshold == (
        reports["keyword_overlap_baseline"].dev.calibration.best.threshold
    )
    assert reports["intent_aware_default"].test.report.case_count == 16
