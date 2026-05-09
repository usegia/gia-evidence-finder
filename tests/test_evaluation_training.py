from __future__ import annotations

from gia_evidence_finder import (
    BenchmarkCase,
    BenchmarkSuite,
    CandidateScoreThresholdExtractor,
    DocumentSpan,
    EvidenceDocument,
    EvidenceExtractor,
    EvidenceFacet,
    ExtractionResult,
    ExtractorExperimentSpec,
    HoldoutDecisionPolicy,
    IntentSpec,
    SpanMatch,
    SupportLabel,
    SupportThresholdOverrideExtractor,
    calibrate_thresholds,
    evaluate_suite,
    evaluate_suite_detailed,
    hard_negative_pairs,
    reranker_training_examples_from_jsonl,
    run_calibrated_holdout,
    training_pairs_from_suite,
    training_pairs_jsonl,
)
from gia_evidence_finder.ranking import ScoreResult


class _PinnedSpanRanker:
    def __init__(self, span_id: str, *, score: float = 0.9) -> None:
        self._span_id = span_id
        self._score = score

    def score(self, intent: IntentSpec, span: DocumentSpan) -> ScoreResult:
        del intent
        score = self._score if span.id == self._span_id else 0.0
        return ScoreResult(
            score=score,
            features={"pinned_score": score},
            reasons=("pinned test score",) if score else (),
        )


class _HighScoreRejectExtractor:
    def extract(
        self,
        intent: IntentSpec,
        document: EvidenceDocument,
        *,
        candidate_limit: int | None = None,
        max_matches: int | None = None,
    ) -> ExtractionResult:
        del candidate_limit, max_matches
        top_span = document.spans[0]
        candidate = SpanMatch(
            span=top_span,
            label=SupportLabel.REJECT,
            score=1.0,
            features={"test_score": 1.0},
            reasons=("intentionally high-scored reject",),
        )
        return ExtractionResult(
            intent=intent,
            document=document,
            matches=(),
            candidates=(candidate,),
            abstained=True,
        )


def _span_id_containing(document: EvidenceDocument, text: str) -> str:
    return next(span.id for span in document.spans if text in span.text)


def test_evaluation_report_scores_support_abstention_and_forbidden_rate(
    evidence_document: EvidenceDocument,
) -> None:
    semantic_intent = IntentSpec(
        id="semantic_support",
        label="taxonomy-free semantic search",
        description="Find evidence that uncurated text can be searched before concept packs exist.",
        positive_examples=("open semantic gates search uncurated customer text",),
        required_facets=(
            EvidenceFacet("uncurated", ("uncurated customer text",)),
            EvidenceFacet("concept packs", ("before curated concept packs exist",)),
        ),
        min_support_score=0.43,
    )
    absent_intent = IntentSpec(
        id="absent_mobile",
        label="mobile notification support",
        description="Find evidence for mobile push notifications.",
        positive_examples=("mobile push notifications",),
        required_facets=(EvidenceFacet("mobile push", ("mobile push notifications",)),),
        min_support_score=0.55,
    )
    support_id = _span_id_containing(evidence_document, "Open semantic gates search")
    seed_id = _span_id_containing(evidence_document, "Seed the project-management")

    report = evaluate_suite(
        EvidenceExtractor.default(),
        (
            BenchmarkCase(
                id="semantic",
                intent=semantic_intent,
                document=evidence_document,
                support_span_ids=(support_id,),
                forbidden_span_ids=(seed_id,),
            ),
            BenchmarkCase(
                id="absent",
                intent=absent_intent,
                document=evidence_document,
                expect_abstain=True,
            ),
        ),
    )

    assert report.case_count == 2
    assert report.support_case_count == 1
    assert report.abstain_case_count == 1
    assert report.recall_at_3 == 1.0
    assert report.abstain_accuracy == 1.0
    assert report.decision_accuracy == 1.0
    assert report.forbidden_top1_rate == 0.0


def test_detailed_evaluation_report_exposes_case_diagnostics(
    evidence_document: EvidenceDocument,
) -> None:
    support_id = _span_id_containing(evidence_document, "Open semantic gates search")
    report = evaluate_suite_detailed(
        EvidenceExtractor.default(),
        (
            BenchmarkCase(
                id="semantic",
                intent=IntentSpec(
                    id="semantic_support",
                    label="taxonomy-free semantic search",
                    description=(
                        "Find evidence that uncurated text can be searched before concept packs."
                    ),
                    positive_examples=("open semantic gates search uncurated customer text",),
                    min_support_score=0.43,
                ),
                document=evidence_document,
                support_span_ids=(support_id,),
            ),
        ),
    )

    case = report.cases[0]

    assert report.summary.case_count == 1
    assert case.case_id == "semantic"
    assert case.first_support_rank is not None
    assert case.ranked_span_ids
    assert case.decision_correct
    assert report.summary.diagnostic_top1_accuracy == 1.0
    assert report.summary.abstain_diagnostic_top1_accuracy == 1.0
    assert report.failures == ((case,) if not case.top1_is_support else ())


def test_calibrate_thresholds_replays_candidate_scores(
    evidence_document: EvidenceDocument,
) -> None:
    support_id = _span_id_containing(evidence_document, "Open semantic gates search")
    detailed = evaluate_suite_detailed(
        EvidenceExtractor.default(),
        (
            BenchmarkCase(
                id="semantic",
                intent=IntentSpec(
                    id="semantic_support",
                    label="taxonomy-free semantic search",
                    description=(
                        "Find evidence that uncurated text can be searched before concept packs."
                    ),
                    positive_examples=("open semantic gates search uncurated customer text",),
                    min_support_score=0.43,
                ),
                document=evidence_document,
                support_span_ids=(support_id,),
            ),
        ),
    )

    calibration = calibrate_thresholds(detailed, thresholds=(0.1, 0.5, 0.9))

    assert len(calibration.points) == 3
    assert calibration.best.threshold in {0.1, 0.5, 0.9}
    assert calibration.points[0].report.case_count == 1
    assert detailed.cases[0].candidate_scores


def test_evaluation_report_tracks_special_negative_false_support(
    evidence_document: EvidenceDocument,
) -> None:
    support_id = _span_id_containing(evidence_document, "Frontend requests can select")
    contradiction_id = _span_id_containing(evidence_document, "Seed the project-management")
    extractor = EvidenceExtractor(ranker=_PinnedSpanRanker(contradiction_id, score=0.8))

    detailed = evaluate_suite_detailed(
        extractor,
        (
            BenchmarkCase(
                id="fixture_wrongly_seeded",
                intent=IntentSpec(
                    id="fixture_request_body",
                    label="select fixture by request body schema name",
                    description="Find the request payload evidence for choosing a fixture.",
                    positive_examples=("schema_name project_management_benchmark in payloads",),
                    min_support_score=0.43,
                ),
                document=evidence_document,
                support_span_ids=(support_id,),
                contradiction_span_ids=(contradiction_id,),
            ),
        ),
    )
    reports_by_label = {
        report.label: report
        for report in detailed.summary.negative_label_reports
    }
    contradiction_report = reports_by_label[SupportLabel.CONTRADICTS]
    case = detailed.cases[0]
    contradiction_case = next(
        evaluation
        for evaluation in case.negative_label_evaluations
        if evaluation.label == SupportLabel.CONTRADICTS
    )
    calibration = calibrate_thresholds(detailed, thresholds=(0.5, 0.95))
    low_threshold_report = {
        report.label: report
        for report in calibration.points[0].report.negative_label_reports
    }[SupportLabel.CONTRADICTS]
    high_threshold_report = {
        report.label: report
        for report in calibration.points[1].report.negative_label_reports
    }[SupportLabel.CONTRADICTS]

    assert contradiction_report.case_count == 1
    assert contradiction_report.span_label_count == 1
    assert contradiction_report.top1_rate == 1.0
    assert contradiction_report.labeled_top1_rate == 0.0
    assert contradiction_report.supported_top1_rate == 1.0
    assert contradiction_case.top1
    assert not contradiction_case.labeled_top1
    assert contradiction_case.supported_top1
    assert not case.diagnostic_top1
    assert low_threshold_report.supported_top1_rate == 1.0
    assert high_threshold_report.supported_top1_rate == 0.0


def test_support_threshold_override_applies_dev_selected_threshold(
    evidence_document: EvidenceDocument,
) -> None:
    extractor = SupportThresholdOverrideExtractor(EvidenceExtractor.default(), 0.99)
    intent = IntentSpec(
        id="semantic_support",
        label="taxonomy-free semantic search",
        description="Find evidence that uncurated text can be searched before concept packs.",
        positive_examples=("open semantic gates search uncurated customer text",),
        min_support_score=0.43,
    )

    result = extractor.extract(intent, evidence_document)

    assert result.abstained
    assert result.candidates[0].score < 0.99


def test_candidate_score_threshold_relabels_any_extractor_result(
    evidence_document: EvidenceDocument,
) -> None:
    extractor = CandidateScoreThresholdExtractor(EvidenceExtractor.default(), 0.99)
    intent = IntentSpec(
        id="semantic_support",
        label="taxonomy-free semantic search",
        description="Find evidence that uncurated text can be searched before concept packs.",
        positive_examples=("open semantic gates search uncurated customer text",),
        min_support_score=0.43,
    )

    result = extractor.extract(intent, evidence_document)

    assert result.abstained
    assert result.candidates[0].score < 0.99
    assert result.trace[-1] == "applied calibrated support threshold 0.99"


def test_calibrated_holdout_can_preserve_extractor_labels(
    evidence_document: EvidenceDocument,
) -> None:
    suite = BenchmarkSuite(
        id="label_preserve",
        name="Label preserve",
        cases=(
            BenchmarkCase(
                id="absent",
                intent=IntentSpec(
                    id="absent_mobile",
                    label="mobile push notifications",
                    description="Find evidence for mobile push notifications.",
                    positive_examples=("mobile push notifications",),
                ),
                document=evidence_document,
                forbidden_span_ids=(evidence_document.spans[0].id,),
                expect_abstain=True,
            ),
        ),
    )

    holdout = run_calibrated_holdout(
        (
            ExtractorExperimentSpec(
                "label_preserving",
                _HighScoreRejectExtractor(),
                decision_policy=HoldoutDecisionPolicy.EXTRACTOR_LABELS,
            ),
        ),
        dev_suite=suite,
        test_suite=suite,
    )[0]

    assert holdout.decision_policy == HoldoutDecisionPolicy.EXTRACTOR_LABELS
    assert holdout.selected_threshold is None
    assert holdout.dev.calibration is not None
    assert holdout.test.report.decision_accuracy == 1.0
    assert holdout.test.report.forbidden_supported_top1_rate == 0.0
    assert holdout.test.cases[0].top_label == SupportLabel.REJECT.value


def test_hard_negative_pairs_export_support_near_miss_and_reject_labels(
    evidence_document: EvidenceDocument,
) -> None:
    intent = IntentSpec(
        id="fixture_request_body",
        label="select fixture by request body schema name",
        description="Find the request payload evidence for choosing a fixture.",
        positive_examples=("schema_name project_management_benchmark in query payloads",),
        negative_examples=("seed project-management example into postgres",),
        required_facets=(
            EvidenceFacet("schema field", ("schema_name",)),
            EvidenceFacet("fixture", ("project_management_benchmark",)),
        ),
        min_support_score=0.43,
    )
    support_id = _span_id_containing(evidence_document, "Frontend requests can select")
    seed_id = _span_id_containing(evidence_document, "Seed the project-management")

    pairs = hard_negative_pairs(
        EvidenceExtractor.default(),
        (
            BenchmarkCase(
                id="fixture",
                intent=intent,
                document=evidence_document,
                support_span_ids=(support_id,),
                near_miss_span_ids=(seed_id,),
            ),
        ),
        negatives_per_case=5,
    )

    labels = {pair.label for pair in pairs}
    assert SupportLabel.SUPPORTS in labels
    assert SupportLabel.NEAR_MISS in labels or SupportLabel.REJECT in labels
    assert any(pair.span_id == support_id and pair.label == SupportLabel.SUPPORTS for pair in pairs)
    assert all(pair.query for pair in pairs)
    assert all(pair.metadata["case_id"] == "fixture" for pair in pairs)


def test_hard_negative_pairs_preserve_special_negative_labels(
    evidence_document: EvidenceDocument,
) -> None:
    intent = IntentSpec(
        id="fixture_request_body",
        label="select fixture by request body schema name",
        description="Find the request payload evidence for choosing a fixture.",
        positive_examples=("schema_name project_management_benchmark in query payloads",),
        min_support_score=0.43,
    )
    support_id = _span_id_containing(evidence_document, "Frontend requests can select")
    seed_id = _span_id_containing(evidence_document, "Seed the project-management")

    pairs = hard_negative_pairs(
        EvidenceExtractor.default(),
        (
            BenchmarkCase(
                id="fixture",
                intent=intent,
                document=evidence_document,
                support_span_ids=(support_id,),
                contradiction_span_ids=(seed_id,),
            ),
        ),
        negatives_per_case=5,
    )

    assert any(
        pair.span_id == seed_id and pair.label == SupportLabel.CONTRADICTS
        for pair in pairs
    )


def test_training_pairs_from_suite_export_jsonl(
    evidence_document: EvidenceDocument,
) -> None:
    support_id = _span_id_containing(evidence_document, "Frontend requests can select")
    suite_case = BenchmarkCase(
        id="fixture",
        intent=IntentSpec(
            id="fixture_request_body",
            label="select fixture by request body schema name",
            description="Find the request payload evidence for choosing a fixture.",
            positive_examples=("schema_name project_management_benchmark in query payloads",),
            required_facets=(
                EvidenceFacet("schema field", ("schema_name",)),
                EvidenceFacet("fixture", ("project_management_benchmark",)),
            ),
            min_support_score=0.43,
        ),
        document=evidence_document,
        support_span_ids=(support_id,),
    )
    pairs = training_pairs_from_suite(
        EvidenceExtractor.default(),
        BenchmarkSuite(id="fixture_suite", name="Fixture suite", cases=(suite_case,)),
        negatives_per_case=2,
    )
    payload = training_pairs_jsonl(pairs)

    assert len(payload.splitlines()) == 3
    assert '"suite_id": "fixture_suite"' in payload
    assert '"query": "select fixture by request body schema name' in payload


def test_reranker_training_examples_parse_exported_pair_jsonl(
    evidence_document: EvidenceDocument,
) -> None:
    support_id = _span_id_containing(evidence_document, "Frontend requests can select")
    seed_id = _span_id_containing(evidence_document, "Seed the project-management")
    suite_case = BenchmarkCase(
        id="fixture",
        intent=IntentSpec(
            id="fixture_request_body",
            label="select fixture by request body schema name",
            description="Find the request payload evidence for choosing a fixture.",
            positive_examples=("schema_name project_management_benchmark in query payloads",),
            min_support_score=0.43,
        ),
        document=evidence_document,
        support_span_ids=(support_id,),
        near_miss_span_ids=(seed_id,),
    )
    pairs = training_pairs_from_suite(
        EvidenceExtractor.default(),
        BenchmarkSuite(id="fixture_suite", name="Fixture suite", cases=(suite_case,)),
        negatives_per_case=3,
    )

    examples = reranker_training_examples_from_jsonl(training_pairs_jsonl(pairs))

    assert {example.label_score for example in examples} == {0.0, 1.0}
    assert any(
        example.source_label == SupportLabel.SUPPORTS and example.label_score == 1.0
        for example in examples
    )
    assert any(
        example.source_label != SupportLabel.SUPPORTS and example.label_score == 0.0
        for example in examples
    )
    assert all(example.metadata["suite_id"] == "fixture_suite" for example in examples)
