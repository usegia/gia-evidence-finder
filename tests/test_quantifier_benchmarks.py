from __future__ import annotations

from gia_evidence_finder import (
    BenchmarkSplit,
    EvidenceExtractor,
    KeywordOverlapBaseline,
    QuantifierKind,
    SupportLabel,
    benchmark_series_by_id,
    benchmark_series_suites,
    bind_quantifiers,
    compile_intent,
    evaluate_suite,
    extract_quantifiers,
    quantifier_benchmark_suite,
    quantifier_benchmark_suite_by_id,
    quantifier_benchmark_suites,
    quantifier_binding_benchmark_suite,
    quantifier_binding_evidence_benchmark_v2,
    quantifier_evidence_benchmark_v1,
    quantifier_score,
    requirements_from_text,
)


def test_extract_quantifiers_parses_dates_numbers_and_units() -> None:
    quantifiers = extract_quantifiers(
        "Created on 2025-05-20 with latency under 100 ms, $3,200 rent, and 94% cache."
    )

    assert {quantifier.kind for quantifier in quantifiers} >= {
        QuantifierKind.DATE,
        QuantifierKind.DURATION,
        QuantifierKind.MONEY,
        QuantifierKind.PERCENT,
    }
    assert any(quantifier.unit == "ms" for quantifier in quantifiers)


def test_quantifier_score_detects_wrong_year_contradiction() -> None:
    requirements = requirements_from_text("OAuth hardening project was created in 2025")
    score = quantifier_score(
        requirements,
        "OAuth hardening project was created in 2024.",
    )

    assert score.requirement_count == 1
    assert score.coverage == 0.0
    assert score.contradiction == 1.0


def test_quantifier_score_supports_threshold_comparisons() -> None:
    requirements = requirements_from_text("API latency is under 100 ms")
    supported = quantifier_score(requirements, "API latency is 80 ms at p95.")
    contradicted = quantifier_score(requirements, "API latency is 150 ms at p95.")

    assert supported.coverage == 1.0
    assert contradicted.contradiction == 1.0


def test_bind_quantifiers_assigns_local_roles() -> None:
    bindings = bind_quantifiers(
        "Project started in 2024 and ended in 2025 with rent $3,200 and deposit $1,000."
    )

    role_by_surface = {binding.quantifier.surface: binding.role for binding in bindings}

    assert role_by_surface["2024"] == "started"
    assert role_by_surface["2025"] == "ended"
    assert role_by_surface["$3,200"] == "rent"
    assert role_by_surface["$1,000"] == "deposit"


def test_bind_quantifiers_leaves_unknown_role_unbound() -> None:
    bindings = bind_quantifiers("The appendix mentions 42 points.")

    assert bindings
    assert bindings[0].role == ""


def test_compiled_intent_rejects_semantically_similar_wrong_year() -> None:
    suite = quantifier_benchmark_suite()
    document = next(case.document for case in suite.cases if case.document.id == "project_dates")
    intent = compile_intent(
        intent_id="claim",
        claim="OAuth hardening project was created in 2026",
        min_support_score=0.50,
    )

    result = EvidenceExtractor.default().extract(intent, document)

    assert result.abstained
    assert result.candidates[0].label == SupportLabel.CONTRADICTS
    assert "2025" in result.candidates[0].span.text


def test_compiled_intent_rejects_swapped_role_dates() -> None:
    suite = quantifier_binding_benchmark_suite()
    case = next(case for case in suite.cases if case.id == "binding.started_ended")
    result = EvidenceExtractor.default().extract(case.intent, case.document)

    swapped = next(
        candidate
        for candidate in result.candidates
        if "started in 2025" in candidate.span.text
    )

    assert result.matches[0].span.id in case.support_span_ids
    assert swapped.label == SupportLabel.CONTRADICTS
    assert swapped.features["quantifier_contradiction"] > 0.0


def test_quantifier_benchmark_suite_has_targeted_label_coverage() -> None:
    suite = quantifier_benchmark_suite()

    assert suite.id == "quantifier_numeric_date_v1"
    assert int(suite.metadata["document_count"]) == 5
    assert len(suite.cases) >= 20
    assert int(suite.metadata["contradiction_case_count"]) >= 15
    assert all(case.curation.reviewed for case in suite.cases)
    assert quantifier_benchmark_suite_by_id(suite.id).id == suite.id
    assert [suite.id for suite in quantifier_benchmark_suites()] == [
        "quantifier_numeric_date_v1",
        "quantifier_binding_v2",
    ]


def test_quantifier_binding_benchmark_suite_has_targeted_label_coverage() -> None:
    suite = quantifier_binding_benchmark_suite()

    assert suite.id == "quantifier_binding_v2"
    assert int(suite.metadata["case_count"]) >= 24
    assert all(case.curation.reviewed for case in suite.cases)
    assert quantifier_benchmark_suite_by_id(suite.id).id == suite.id


def test_default_extractor_clears_quantifier_benchmark() -> None:
    suite = quantifier_benchmark_suite()
    report = evaluate_suite(EvidenceExtractor.default(), suite.cases)

    assert report.recall_at_3 >= 0.95
    assert report.top1_support_accuracy >= 0.90
    assert report.abstain_accuracy == 1.0
    assert report.decision_accuracy >= 0.95
    assert report.forbidden_supported_top1_rate == 0.0
    assert {
        negative.label: negative.supported_top1_rate
        for negative in report.negative_label_reports
    }[SupportLabel.CONTRADICTS] == 0.0


def test_default_extractor_clears_quantifier_binding_benchmark() -> None:
    suite = quantifier_binding_benchmark_suite()
    report = evaluate_suite(EvidenceExtractor.default(), suite.cases)

    assert report.recall_at_3 >= 0.95
    assert report.top1_support_accuracy >= 0.90
    assert report.decision_accuracy >= 0.95
    assert report.forbidden_supported_top1_rate == 0.0
    assert {
        negative.label: negative.supported_top1_rate
        for negative in report.negative_label_reports
    }[SupportLabel.CONTRADICTS] == 0.0


def test_default_extractor_beats_keyword_on_quantifier_decisions() -> None:
    suite = quantifier_benchmark_suite()
    default_report = evaluate_suite(EvidenceExtractor.default(), suite.cases)
    keyword_report = evaluate_suite(EvidenceExtractor(ranker=KeywordOverlapBaseline()), suite.cases)

    assert default_report.decision_accuracy > keyword_report.decision_accuracy


def test_default_extractor_beats_keyword_on_quantifier_binding_decisions() -> None:
    suite = quantifier_binding_benchmark_suite()
    default_report = evaluate_suite(EvidenceExtractor.default(), suite.cases)
    keyword_report = evaluate_suite(EvidenceExtractor(ranker=KeywordOverlapBaseline()), suite.cases)

    assert default_report.decision_accuracy > keyword_report.decision_accuracy


def test_quantifier_series_is_registered_with_train_dev_test_splits() -> None:
    series = quantifier_evidence_benchmark_v1()

    assert series.id == "quantifier_evidence_benchmark_v1"
    assert series.metadata["status"] == "focused_quantifier"
    assert benchmark_series_by_id(series.id).id == series.id
    assert {
        suite.metadata["split"] for suite in benchmark_series_suites(series)
    } == {"train", "dev", "test"}
    assert benchmark_series_suites(series, split=BenchmarkSplit.TEST)[0].cases


def test_quantifier_binding_series_is_registered_with_train_dev_test_splits() -> None:
    series = quantifier_binding_evidence_benchmark_v2()

    assert series.id == "quantifier_binding_evidence_benchmark_v2"
    assert series.metadata["status"] == "focused_quantifier_binding"
    assert benchmark_series_by_id(series.id).id == series.id
    assert {
        suite.metadata["split"] for suite in benchmark_series_suites(series)
    } == {"train", "dev", "test"}
