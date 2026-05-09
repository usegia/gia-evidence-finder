from __future__ import annotations

from gia_evidence_finder import (
    EvidenceDocument,
    EvidenceExtractor,
    EvidenceFacet,
    EvidenceRelation,
    IntentSpec,
    KeywordOverlapBaseline,
    MarkdownSpanParser,
    SpanKind,
    SupportLabel,
)
from gia_evidence_finder.extractor import EvidenceExtractor as Extractor


def test_extracts_paraphrased_open_semantic_gate(evidence_document: EvidenceDocument) -> None:
    intent = IntentSpec(
        id="taxonomy_free_semantic",
        label="taxonomy-free semantic document search",
        description=(
            "Find evidence that the product searches free-form customer prose even before a "
            "curated taxonomy has been installed."
        ),
        positive_examples=(
            "open semantic gates search customer text before curated concept packs exist",
            "embedding-backed matching before a curated concept universe exists",
        ),
        required_facets=(
            EvidenceFacet("open text", ("uncurated customer text", "free-form customer prose")),
            EvidenceFacet(
                "before taxonomy",
                ("before curated concept packs exist", "before taxonomy"),
            ),
        ),
        min_support_score=0.43,
    )

    result = EvidenceExtractor.default().extract(intent, evidence_document)

    assert not result.abstained
    assert result.matches[0].label == SupportLabel.SUPPORTS
    assert "Open semantic gates search uncurated customer text" in result.matches[0].span.text
    assert result.matches[0].span.kind == SpanKind.SENTENCE
    assert result.matches[0].features["required_facet_coverage"] == 1.0


def test_selects_command_for_quality_gate_not_showcase_check(
    evidence_document: EvidenceDocument,
) -> None:
    intent = IntentSpec(
        id="full_quality_gate",
        label="full repository quality command",
        description="Find the command that combines linting, static typing, and the test suite.",
        positive_examples=("uv run ruff check followed by mypy and pytest",),
        negative_examples=("bun run typecheck and lint for showcase frontend only",),
        required_facets=(
            EvidenceFacet("lint", ("ruff check",)),
            EvidenceFacet("typing", ("mypy", "static typing")),
            EvidenceFacet("tests", ("pytest", "test suite")),
        ),
        preferred_span_kinds=(SpanKind.CODE, SpanKind.SENTENCE, SpanKind.PARAGRAPH),
        min_support_score=0.42,
    )

    result = EvidenceExtractor.default().extract(intent, evidence_document)

    assert not result.abstained
    assert result.matches[0].span.kind == SpanKind.CODE
    assert result.matches[0].span.text == "uv run ruff check . && uv run mypy && uv run pytest"
    assert "bun run typecheck" not in result.matches[0].span.text


def test_negative_examples_reject_related_seed_endpoint(
    evidence_document: EvidenceDocument,
) -> None:
    intent = IntentSpec(
        id="fixture_request_body",
        label="select fixture by request body schema name",
        description=(
            "Find evidence that callers choose the project-management example by putting a "
            "schema name in query request payloads."
        ),
        positive_examples=("include schema_name project_management_benchmark in /query payloads",),
        negative_examples=("seed the project-management benchmark example into postgres",),
        required_facets=(
            EvidenceFacet("schema field", ("schema_name",)),
            EvidenceFacet("fixture name", ("project_management_benchmark",)),
            EvidenceFacet("query payload", ("/query payloads", "/query/structured payloads")),
        ),
        min_support_score=0.43,
    )

    result = EvidenceExtractor.default().extract(intent, evidence_document)

    assert not result.abstained
    assert result.matches[0].span.text.startswith("Frontend requests can select the fixture")
    assert "Seed the project-management" not in result.matches[0].span.text


def test_abstains_when_required_facets_are_absent(evidence_document: EvidenceDocument) -> None:
    intent = IntentSpec(
        id="mobile_push_notifications",
        label="mobile push notification support",
        description="Find evidence that the product sends mobile push notifications.",
        positive_examples=("sends mobile push notifications",),
        required_facets=(
            EvidenceFacet("mobile", ("mobile",)),
            EvidenceFacet("push", ("push notification",)),
        ),
        min_support_score=0.55,
    )

    result = EvidenceExtractor.default().extract(intent, evidence_document)

    assert result.abstained
    assert not result.matches
    assert result.candidates[0].label != SupportLabel.SUPPORTS


def test_relation_bridge_rejects_attribute_transfer() -> None:
    document = MarkdownSpanParser().parse(
        """# FastAPI

FastAPI is very high performance, on par with NodeJS and Go, thanks to
Starlette and Pydantic.

Pydantic validates Python type annotations.
""",
        document_id="fastapi",
    )
    intent = IntentSpec(
        id="pydantic_performance",
        label="Pydantic itself matches NodeJS and Go performance",
        description="Find direct evidence that Pydantic itself is on par with NodeJS and Go.",
        positive_examples=("Pydantic itself is on par with NodeJS and Go",),
        relations=(
            EvidenceRelation(
                name="subject_performance",
                subject_phrases=("Pydantic",),
                predicate_phrases=("performance", "on par"),
                object_phrases=("NodeJS", "Go"),
                forbidden_bridge_phrases=("thanks to", "based on", "derived from"),
            ),
        ),
        min_support_score=0.55,
    )

    result = EvidenceExtractor.default().extract(intent, document)

    assert result.abstained
    assert result.candidates[0].span.text.startswith("FastAPI is very high performance")
    assert result.candidates[0].features["relation_bridge_penalty"] > 0.0


def test_explicit_counterclaim_is_contradiction_not_support() -> None:
    document = MarkdownSpanParser().parse(
        (
            "# PR discussion\n\n"
            "PR discussion requires open semantic gate diagnostics.\n\n"
            "Contradiction: PR discussion does not forbid open semantic gate diagnostics; "
            "it requires open semantic gate diagnostics.\n"
        ),
        document_id="counterclaim",
    )
    intent = IntentSpec(
        id="forbidden_diagnostics",
        label="PR discussion forbids open semantic gate diagnostics",
        description="Find evidence that PR discussion forbids open semantic gate diagnostics.",
        positive_examples=("PR discussion forbids open semantic gate diagnostics",),
        required_facets=(
            EvidenceFacet("subject", ("PR discussion",)),
            EvidenceFacet("detail", ("open semantic gate diagnostics",)),
        ),
        min_support_score=0.42,
    )

    result = EvidenceExtractor.default().extract(intent, document)

    assert result.abstained
    assert result.candidates[0].label == SupportLabel.CONTRADICTS
    assert any(
        candidate.features["explicit_contradiction_cue"] == 1.0
        and candidate.label == SupportLabel.CONTRADICTS
        for candidate in result.candidates
    )


def test_explicit_proof_refusal_is_insufficient_context_not_support() -> None:
    document = MarkdownSpanParser().parse(
        (
            "# Dependency note\n\n"
            "A dependent library is mentioned near open semantic gate diagnostics, but this "
            "does not prove the dependency provides it.\n"
        ),
        document_id="proof_refusal",
    )
    intent = IntentSpec(
        id="dependency_provides_diagnostics",
        label="dependent library provides diagnostics",
        description=(
            "Find evidence that a dependent library itself provides open semantic gate "
            "diagnostics."
        ),
        positive_examples=("dependent library itself provides open semantic gate diagnostics",),
        relations=(
            EvidenceRelation(
                name="dependency_provides",
                subject_phrases=("dependent library",),
                predicate_phrases=("provides",),
                object_phrases=("open semantic gate diagnostics",),
            ),
        ),
        min_support_score=0.42,
    )

    result = EvidenceExtractor.default().extract(intent, document)

    assert result.abstained
    assert result.candidates[0].label == SupportLabel.INSUFFICIENT_CONTEXT
    assert result.candidates[0].features["insufficient_context_cue"] == 1.0


def test_only_not_trap_is_near_miss_not_support() -> None:
    document = MarkdownSpanParser().parse(
        """# Listing note

Near miss: Cedar House only offers not elevator outages.
""",
        document_id="only_not_trap",
    )
    intent = IntentSpec(
        id="only_not_elevator",
        label="Cedar House only offers not elevator outages",
        description="Find evidence that Cedar House only offers not elevator outages.",
        positive_examples=("Cedar House only offers not elevator outages",),
        required_facets=(
            EvidenceFacet("subject", ("Cedar House",)),
            EvidenceFacet("detail", ("elevator outages",)),
        ),
        min_support_score=0.42,
    )

    result = EvidenceExtractor.default().extract(intent, document)

    assert result.abstained
    assert result.candidates[0].label == SupportLabel.NEAR_MISS
    assert result.candidates[0].features["only_not_trap_cue"] == 1.0


def test_keyword_baseline_is_available_only_as_measurement_baseline(
    evidence_document: EvidenceDocument,
) -> None:
    intent = IntentSpec(
        id="auditable_results",
        label="auditable result explanations",
        description="Find evidence that search outputs explain why records matched.",
        positive_examples=("evidence explains why each record matched",),
        min_support_score=0.2,
    )

    result = Extractor(ranker=KeywordOverlapBaseline()).extract(intent, evidence_document)

    assert result.candidates
    assert "keyword_overlap" in result.candidates[0].features
