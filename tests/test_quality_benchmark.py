from __future__ import annotations

from gia_evidence_finder import (
    BenchmarkCase,
    EvidenceDocument,
    EvidenceExtractor,
    EvidenceFacet,
    IntentSpec,
    KeywordOverlapBaseline,
    SpanKind,
    evaluate_suite,
)


def test_default_ranker_clears_local_quality_benchmark(
    evidence_document: EvidenceDocument,
) -> None:
    cases = _quality_cases(evidence_document)
    support_cases = tuple(case for case in cases if not case.expect_abstain)
    abstain_cases = tuple(case for case in cases if case.expect_abstain)

    support_report = evaluate_suite(EvidenceExtractor.default(), support_cases)
    abstain_report = evaluate_suite(EvidenceExtractor.default(), abstain_cases)

    assert support_report.recall_at_3 >= 0.85
    assert support_report.top1_support_accuracy >= 0.75
    assert support_report.forbidden_top1_rate == 0.0
    assert abstain_report.abstain_accuracy == 1.0


def test_default_ranker_beats_keyword_overlap_on_quality_benchmark(
    evidence_document: EvidenceDocument,
) -> None:
    cases = tuple(case for case in _quality_cases(evidence_document) if not case.expect_abstain)
    default_report = evaluate_suite(EvidenceExtractor.default(), cases)
    baseline_report = evaluate_suite(EvidenceExtractor(ranker=KeywordOverlapBaseline()), cases)

    assert default_report.mean_reciprocal_rank >= baseline_report.mean_reciprocal_rank
    assert default_report.top1_support_accuracy >= baseline_report.top1_support_accuracy


def _quality_cases(document: EvidenceDocument) -> tuple[BenchmarkCase, ...]:
    semantic_id = _span_id_containing(document, "Open semantic gates search")
    quality_id = _span_id_containing(document, "uv run ruff check")
    admin_id = _span_id_containing(document, "Production mutating routes require")
    fixture_id = _span_id_containing(document, "Frontend requests can select")
    migration_id = _span_id_containing(document, "Postgres migrations use a ledger")
    provider_id = _span_id_containing(
        document,
        "Production benchmark gates must use Qwen3",
        kind=SpanKind.BULLET,
    )
    evidence_id = _span_id_containing(document, "Search results carry evidence")
    seed_id = _span_id_containing(document, "Seed the project-management")
    showcase_id = _span_id_containing(document, "bun run typecheck")

    return (
        BenchmarkCase(
            id="semantic_without_taxonomy",
            intent=IntentSpec(
                id="semantic_without_taxonomy",
                label="search uncurated prose before taxonomy",
                description="Find evidence for semantic search before curated taxonomy setup.",
                positive_examples=("open semantic gates search uncurated customer text",),
                required_facets=(
                    EvidenceFacet("uncurated", ("uncurated customer text",)),
                    EvidenceFacet("concept packs", ("before curated concept packs exist",)),
                ),
                min_support_score=0.43,
            ),
            document=document,
            support_span_ids=(semantic_id,),
        ),
        BenchmarkCase(
            id="quality_command",
            intent=IntentSpec(
                id="quality_command",
                label="full quality command",
                description="Find the command that runs lint, typing, and tests.",
                positive_examples=("uv run ruff check mypy pytest",),
                negative_examples=("bun run typecheck lint",),
                required_facets=(
                    EvidenceFacet("ruff", ("ruff check",)),
                    EvidenceFacet("mypy", ("mypy",)),
                    EvidenceFacet("pytest", ("pytest",)),
                ),
                preferred_span_kinds=(SpanKind.CODE, SpanKind.SENTENCE, SpanKind.PARAGRAPH),
                min_support_score=0.42,
            ),
            document=document,
            support_span_ids=(quality_id,),
            forbidden_span_ids=(showcase_id,),
        ),
        BenchmarkCase(
            id="admin_token",
            intent=IntentSpec(
                id="admin_token",
                label="admin token required for mutating routes",
                description="Find evidence that write-like admin API operations require a token.",
                positive_examples=("mutating routes require x-traverce-admin-token header",),
                required_facets=(
                    EvidenceFacet("mutating", ("mutating routes",)),
                    EvidenceFacet("token", ("x-traverce-admin-token",)),
                ),
                min_support_score=0.43,
            ),
            document=document,
            support_span_ids=(admin_id,),
        ),
        BenchmarkCase(
            id="fixture_payload",
            intent=IntentSpec(
                id="fixture_payload",
                label="fixture selected in request payload",
                description=(
                    "Find evidence that schema_name selects the project-management fixture."
                ),
                positive_examples=("schema_name project_management_benchmark query payloads",),
                negative_examples=("seed project-management benchmark example",),
                required_facets=(
                    EvidenceFacet("schema", ("schema_name",)),
                    EvidenceFacet("fixture", ("project_management_benchmark",)),
                ),
                min_support_score=0.43,
            ),
            document=document,
            support_span_ids=(fixture_id,),
            forbidden_span_ids=(seed_id,),
        ),
        BenchmarkCase(
            id="migration_safety",
            intent=IntentSpec(
                id="migration_safety",
                label="safe serialized database upgrades",
                description="Find evidence that migrations are checksummed and protected by locks.",
                positive_examples=(
                    "ledger checksum verification advisory-lock protected upgrades",
                ),
                required_facets=(
                    EvidenceFacet("ledger", ("ledger",)),
                    EvidenceFacet("checksum", ("checksum verification",)),
                    EvidenceFacet("lock", ("advisory-lock protected",)),
                ),
                min_support_score=0.43,
            ),
            document=document,
            support_span_ids=(migration_id,),
        ),
        BenchmarkCase(
            id="production_embedding_provider",
            intent=IntentSpec(
                id="production_embedding_provider",
                label="real benchmark embeddings provider",
                description="Find evidence that benchmarks must use real Qwen embeddings.",
                positive_examples=(
                    "Qwen3 embeddings through OpenRouter deterministic provider test-only",
                ),
                required_facets=(
                    EvidenceFacet("qwen", ("Qwen3 embeddings",)),
                    EvidenceFacet("openrouter", ("OpenRouter",)),
                    EvidenceFacet("test only", ("deterministic provider is test-only",)),
                ),
                min_support_score=0.43,
            ),
            document=document,
            support_span_ids=(provider_id,),
        ),
        BenchmarkCase(
            id="auditable_results",
            intent=IntentSpec(
                id="auditable_results",
                label="auditable search result evidence",
                description="Find evidence that results explain why records matched.",
                positive_examples=("evidence explains why each record matched",),
                required_facets=(
                    EvidenceFacet("evidence", ("evidence",)),
                    EvidenceFacet("why", ("why each record matched",)),
                ),
                min_support_score=0.43,
            ),
            document=document,
            support_span_ids=(evidence_id,),
        ),
        BenchmarkCase(
            id="unsupported_mobile_notifications",
            intent=IntentSpec(
                id="unsupported_mobile_notifications",
                label="mobile push notifications",
                description="Find evidence for mobile push notifications.",
                positive_examples=("mobile push notifications",),
                required_facets=(
                    EvidenceFacet("mobile", ("mobile",)),
                    EvidenceFacet("push", ("push notifications",)),
                ),
                min_support_score=0.55,
            ),
            document=document,
            expect_abstain=True,
        ),
    )


def _span_id_containing(
    document: EvidenceDocument,
    text: str,
    *,
    kind: SpanKind | None = None,
) -> str:
    matches = [
        span
        for span in document.spans
        if text in span.text and (kind is None or span.kind == kind)
    ]
    if not matches:
        raise AssertionError(f"no span contains {text!r}")
    return min(matches, key=lambda span: len(span.text)).id
