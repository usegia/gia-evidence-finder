from __future__ import annotations

from gia_evidence_finder import (
    BenchmarkCase,
    BenchmarkSuite,
    DocumentSpan,
    EvidenceDocument,
    EvidenceExtractor,
    EvidenceFacet,
    IntentSpec,
    build_failure_report,
    evaluate_suite_detailed,
)
from gia_evidence_finder.ranking import ScoreResult


class _PinnedSpanRanker:
    def __init__(self, span_id: str, *, score: float = 0.8) -> None:
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


def test_failure_report_explains_rank_and_false_support(
    evidence_document: EvidenceDocument,
) -> None:
    support_id = _span_id_containing(evidence_document, "Frontend requests can select")
    contradiction_id = _span_id_containing(evidence_document, "Seed the project-management")
    suite = BenchmarkSuite(
        id="failure",
        name="Failure",
        cases=(
            BenchmarkCase(
                id="fixture_wrongly_seeded",
                intent=IntentSpec(
                    id="fixture_request_body",
                    label="select fixture by request body schema name",
                    description="Find the request payload evidence for choosing a fixture.",
                    positive_examples=("schema_name project_management_benchmark in payloads",),
                    required_facets=(
                        EvidenceFacet("schema field", ("schema_name",)),
                        EvidenceFacet("fixture", ("project_management_benchmark",)),
                    ),
                    min_support_score=0.43,
                ),
                document=evidence_document,
                support_span_ids=(support_id,),
                contradiction_span_ids=(contradiction_id,),
                forbidden_span_ids=(contradiction_id,),
            ),
        ),
    )
    detailed = evaluate_suite_detailed(
        EvidenceExtractor(ranker=_PinnedSpanRanker(contradiction_id)),
        suite.cases,
    )

    report = build_failure_report(suite, detailed.cases)
    failure = report.failures[0]

    assert report.failure_count == 1
    assert "top_rank_missed_support" in failure.buckets
    assert "forbidden_supported_top_rank" in failure.buckets
    assert "contradicts_supported_top_rank" in failure.buckets
    assert failure.support_texts == (
        evidence_document.span_by_id(support_id).text,
    )
    assert failure.top_text == evidence_document.span_by_id(contradiction_id).text
    assert report.bucket_counts["forbidden_supported_top_rank"] == 1


def _span_id_containing(document: EvidenceDocument, text: str) -> str:
    return next(span.id for span in document.spans if text in span.text)
