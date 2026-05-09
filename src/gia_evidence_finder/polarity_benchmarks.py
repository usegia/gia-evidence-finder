from __future__ import annotations

from dataclasses import dataclass

from gia_evidence_finder.contracts import (
    BenchmarkCase,
    BenchmarkCuration,
    BenchmarkSuite,
    DocumentSpan,
    EvidenceDocument,
    EvidenceFacet,
    IntentSpec,
    SpanKind,
)
from gia_evidence_finder.parsing import MarkdownSpanParser


@dataclass(frozen=True)
class PolarityExcerpt:
    id: str
    source_url: str
    genre: str
    text: str


POLARITY_EXCERPTS: tuple[PolarityExcerpt, ...] = (
    PolarityExcerpt(
        id="ultrachess_readme",
        source_url="https://github.com/yahorbarkouski/ultrachess/blob/main/README.md",
        genre="readme",
        text="""# ultrachess

A Rust chess engine compiled to WebAssembly behind a typed TypeScript API with
zero runtime dependencies.

Legal move generation, FEN / SAN / PGN, perft, and Zobrist hashing at
native-Rust speed.

No evaluation, no search, no opening book.

Chess960, crazyhouse, atomic, antichess, and three-check are not supported.
""",
    ),
    PolarityExcerpt(
        id="api_access_runbook",
        source_url="repo://README.md#postgres-execution-api",
        genre="runbook",
        text="""# API Access

Mutating admin routes for schema creation, graph writes, concept packs,
benchmark seeding, and migration upgrades require the x-traverce-admin-token
header.

Query, status, and health routes remain readable without the admin token.
""",
    ),
    PolarityExcerpt(
        id="planner_contract",
        source_url="repo://docs/specs/natural-language-planning.md",
        genre="spec",
        text="""# Planner Contract

Planner repair is a deterministic guardrail that may normalize schema-proven
shapes but must not invent missing user intent.

Wrong semantic scope should fail validation with a useful diagnostic instead
of being silently moved to another node or edge space.
""",
    ),
    PolarityExcerpt(
        id="evidence_runtime",
        source_url="repo://packages/gia-evidence-finder/README.md",
        genre="design_note",
        text="""# Evidence Runtime

The library should return source evidence, not prose answers.

The team rejected adding an LLM answer generator to the hot path.

Batch reranking caches pair scores so repeated benchmarks do not rescore
unchanged pairs.
""",
    ),
    PolarityExcerpt(
        id="language_policy",
        source_url="repo://README.md#design-principles",
        genre="policy",
        text="""# Language Policy

Modules may be developed in any dynamic language, not just Python.

Python remains the reference implementation for the core package.
""",
    ),
)


def polarity_benchmark_suite() -> BenchmarkSuite:
    documents = _documents()
    cases: list[BenchmarkCase] = []
    cases.extend(_ultrachess_cases(documents["ultrachess_readme"]))
    cases.extend(_api_access_cases(documents["api_access_runbook"]))
    cases.extend(_planner_cases(documents["planner_contract"]))
    cases.extend(_runtime_cases(documents["evidence_runtime"]))
    cases.extend(_language_policy_cases(documents["language_policy"]))
    contradiction_cases = sum(1 for case in cases if case.contradiction_span_ids)
    negative_support_cases = sum(
        1
        for case in cases
        if case.support_span_ids and "negative_support" in case.curation.phenomena
    )
    return BenchmarkSuite(
        id="polarity_negation_v1",
        name="Polarity and negation evidence benchmark",
        description=(
            "Curated cases that separately test positive support, positive claims "
            "contradicted by negated spans, negative claims supported by negated spans, "
            "without/avoid/reject wording, and not-just exclusivity."
        ),
        cases=tuple(cases),
        metadata={
            "document_count": str(len(documents)),
            "case_count": str(len(cases)),
            "contradiction_case_count": str(contradiction_cases),
            "negative_support_case_count": str(negative_support_cases),
            "genres": ",".join(sorted({excerpt.genre for excerpt in POLARITY_EXCERPTS})),
        },
    )


def polarity_benchmark_suites() -> tuple[BenchmarkSuite, ...]:
    return (polarity_benchmark_suite(),)


def polarity_benchmark_suite_by_id(suite_id: str) -> BenchmarkSuite:
    suites = {suite.id: suite for suite in polarity_benchmark_suites()}
    try:
        return suites[suite_id]
    except KeyError as exc:
        raise ValueError(f"unknown polarity benchmark suite {suite_id!r}") from exc


def _documents() -> dict[str, EvidenceDocument]:
    parser = MarkdownSpanParser()
    return {
        excerpt.id: parser.parse(excerpt.text, document_id=excerpt.id, source=excerpt.source_url)
        for excerpt in POLARITY_EXCERPTS
    }


def _ultrachess_cases(document: EvidenceDocument) -> tuple[BenchmarkCase, ...]:
    wasm = _span(document, "Rust chess engine compiled")
    features = _span(document, "Legal move generation")
    no_eval = _span(document, "No evaluation")
    variants = _span(document, "are not supported")
    return (
        _case(
            "polarity.ultrachess_wasm_api_support",
            document,
            "ultrachess is Rust WebAssembly chess engine with typed TypeScript API",
            "Find evidence that ultrachess is a Rust chess engine compiled to WebAssembly.",
            ("Rust chess engine compiled WebAssembly typed TypeScript API",),
            support=(wasm,),
            forbidden=(no_eval,),
            facets=(
                EvidenceFacet("rust", ("Rust",)),
                EvidenceFacet("wasm", ("WebAssembly",)),
                EvidenceFacet("typescript", ("TypeScript API",)),
            ),
            phenomena=("positive_support", "readme_claim"),
        ),
        _case(
            "polarity.ultrachess_features_support",
            document,
            "ultrachess supports legal move generation FEN SAN PGN perft and Zobrist",
            "Find evidence that ultrachess supports legal move generation and notation helpers.",
            ("legal move generation FEN SAN PGN perft Zobrist hashing",),
            support=(features,),
            forbidden=(no_eval,),
            facets=(
                EvidenceFacet("move generation", ("Legal move generation",)),
                EvidenceFacet("notation", ("FEN", "SAN", "PGN")),
                EvidenceFacet("zobrist", ("Zobrist",)),
            ),
            phenomena=("positive_support", "readme_claim"),
        ),
        _case(
            "polarity.ultrachess_eval_search_positive_contradicted",
            document,
            "ultrachess includes evaluation search and an opening book",
            "Find evidence that ultrachess includes evaluation, search, and an opening book.",
            ("evaluation search opening book",),
            support=(),
            contradiction=(no_eval,),
            forbidden=(no_eval,),
            facets=(
                EvidenceFacet("evaluation", ("evaluation",)),
                EvidenceFacet("search", ("search",)),
                EvidenceFacet("opening book", ("opening book",)),
            ),
            expect_abstain=True,
            phenomena=("contradiction", "negated_capability", "readme_claim"),
        ),
        _case(
            "polarity.ultrachess_eval_search_negative_supported",
            document,
            "ultrachess has no evaluation search or opening book",
            "Find evidence that ultrachess has no evaluation, no search, and no opening book.",
            ("no evaluation no search no opening book",),
            support=(no_eval,),
            forbidden=(features,),
            facets=(
                EvidenceFacet("evaluation", ("evaluation",)),
                EvidenceFacet("search", ("search",)),
                EvidenceFacet("opening book", ("opening book",)),
            ),
            phenomena=("negative_support", "negated_capability", "readme_claim"),
        ),
        _case(
            "polarity.ultrachess_variants_positive_contradicted",
            document,
            "ultrachess supports Chess960 crazyhouse atomic antichess and three-check",
            "Find evidence that ultrachess supports Chess960 and chess variants.",
            ("supports Chess960 crazyhouse atomic antichess three-check variants",),
            support=(),
            contradiction=(variants,),
            forbidden=(variants,),
            facets=(
                EvidenceFacet("chess960", ("Chess960",)),
                EvidenceFacet("variants", ("crazyhouse", "atomic", "three-check")),
            ),
            expect_abstain=True,
            phenomena=("contradiction", "not_supported", "readme_claim"),
        ),
        _case(
            "polarity.ultrachess_variants_negative_supported",
            document,
            "ultrachess does not support Chess960 crazyhouse atomic antichess or three-check",
            "Find evidence that ultrachess does not support Chess960 and chess variants.",
            ("Chess960 crazyhouse atomic antichess three-check are not supported",),
            support=(variants,),
            forbidden=(features,),
            facets=(
                EvidenceFacet("chess960", ("Chess960",)),
                EvidenceFacet("variants", ("crazyhouse", "atomic", "three-check")),
            ),
            phenomena=("negative_support", "not_supported", "readme_claim"),
        ),
    )


def _api_access_cases(document: EvidenceDocument) -> tuple[BenchmarkCase, ...]:
    admin = _span(document, "Mutating admin routes")
    public = _span(document, "remain readable without")
    return (
        _case(
            "polarity.api_mutating_routes_require_token",
            document,
            "mutating admin routes require the admin token",
            "Find evidence that mutating admin routes require the admin token header.",
            ("mutating admin routes require x-traverce-admin-token header",),
            support=(admin,),
            forbidden=(public,),
            facets=(
                EvidenceFacet("mutating", ("Mutating admin routes",)),
                EvidenceFacet("token", ("x-traverce-admin-token", "admin token")),
            ),
            phenomena=("positive_support", "without_scope"),
        ),
        _case(
            "polarity.api_query_routes_require_token_contradicted",
            document,
            "query status and health routes require the admin token",
            "Find evidence that query, status, and health routes require the admin token.",
            ("query status health routes require admin token",),
            support=(),
            contradiction=(public,),
            forbidden=(public, admin),
            facets=(
                EvidenceFacet("routes", ("Query", "status", "health")),
                EvidenceFacet("token", ("admin token",)),
            ),
            expect_abstain=True,
            phenomena=("contradiction", "without_scope"),
        ),
        _case(
            "polarity.api_query_routes_without_token_supported",
            document,
            "query status and health routes are readable without the admin token",
            "Find evidence that query, status, and health routes are readable without token.",
            ("query status health routes readable without admin token",),
            support=(public,),
            forbidden=(admin,),
            facets=(
                EvidenceFacet("routes", ("Query", "status", "health")),
                EvidenceFacet("without token", ("without the admin token",)),
            ),
            phenomena=("negative_support", "without_scope"),
        ),
    )


def _planner_cases(document: EvidenceDocument) -> tuple[BenchmarkCase, ...]:
    no_invent = _span(document, "must not invent")
    fail_scope = _span(document, "should fail validation")
    return (
        _case(
            "polarity.repair_invents_intent_contradicted",
            document,
            "planner repair invents missing user intent",
            "Find evidence that planner repair invents missing user intent.",
            ("planner repair invents missing user intent",),
            support=(),
            contradiction=(no_invent,),
            forbidden=(no_invent,),
            facets=(
                EvidenceFacet("repair", ("Planner repair",)),
                EvidenceFacet("intent", ("missing user intent",)),
            ),
            expect_abstain=True,
            phenomena=("contradiction", "must_not"),
        ),
        _case(
            "polarity.repair_must_not_invent_supported",
            document,
            "planner repair must not invent missing user intent",
            "Find evidence that planner repair must not invent missing user intent.",
            ("planner repair must not invent missing user intent",),
            support=(no_invent,),
            forbidden=(fail_scope,),
            facets=(
                EvidenceFacet("repair", ("Planner repair",)),
                EvidenceFacet("must not", ("must not invent",)),
            ),
            phenomena=("negative_support", "must_not"),
        ),
        _case(
            "polarity.wrong_scope_silent_move_contradicted",
            document,
            "wrong semantic scope is silently moved to another space",
            "Find evidence that wrong semantic scope is silently moved to another graph space.",
            ("wrong semantic scope silently moved another node edge space",),
            support=(),
            contradiction=(fail_scope,),
            forbidden=(fail_scope,),
            facets=(
                EvidenceFacet("wrong scope", ("Wrong semantic scope",)),
                EvidenceFacet("silent move", ("silently moved",)),
            ),
            expect_abstain=True,
            phenomena=("contradiction", "instead_of"),
        ),
        _case(
            "polarity.wrong_scope_fail_validation_supported",
            document,
            "wrong semantic scope should fail validation instead of being silently moved",
            "Find evidence that wrong semantic scope should fail validation instead of moving.",
            ("wrong semantic scope fail validation instead of silently moved",),
            support=(fail_scope,),
            forbidden=(no_invent,),
            facets=(
                EvidenceFacet("wrong scope", ("Wrong semantic scope",)),
                EvidenceFacet("fail validation", ("fail validation",)),
            ),
            phenomena=("negative_support", "instead_of"),
        ),
    )


def _runtime_cases(document: EvidenceDocument) -> tuple[BenchmarkCase, ...]:
    source = _span(document, "source evidence")
    rejected = _span(document, "rejected adding")
    cache = _span(document, "do not rescore")
    return (
        _case(
            "polarity.runtime_returns_source_evidence_supported",
            document,
            "the library returns source evidence",
            "Find evidence that the library returns source evidence.",
            ("library return source evidence",),
            support=(source,),
            forbidden=(rejected,),
            facets=(EvidenceFacet("source evidence", ("source evidence",)),),
            phenomena=("positive_support", "not_prose"),
        ),
        _case(
            "polarity.runtime_prose_answers_contradicted",
            document,
            "the library returns prose answers",
            "Find evidence that the library returns prose answers.",
            ("library return prose answers",),
            support=(),
            contradiction=(source,),
            forbidden=(source,),
            facets=(EvidenceFacet("prose answers", ("prose answers",)),),
            expect_abstain=True,
            phenomena=("contradiction", "not_prose"),
        ),
        _case(
            "polarity.runtime_llm_hot_path_contradicted",
            document,
            "the hot path adds an LLM answer generator",
            "Find evidence that the hot path adds an LLM answer generator.",
            ("hot path adds LLM answer generator",),
            support=(),
            contradiction=(rejected,),
            forbidden=(rejected,),
            facets=(
                EvidenceFacet("llm", ("LLM answer generator",)),
                EvidenceFacet("hot path", ("hot path",)),
            ),
            expect_abstain=True,
            phenomena=("contradiction", "rejected_action"),
        ),
        _case(
            "polarity.runtime_no_rescore_supported",
            document,
            "repeated benchmarks do not rescore unchanged pairs",
            "Find evidence that repeated benchmarks do not rescore unchanged pairs.",
            ("repeated benchmarks do not rescore unchanged pairs",),
            support=(cache,),
            forbidden=(source,),
            facets=(
                EvidenceFacet("benchmarks", ("repeated benchmarks",)),
                EvidenceFacet("rescore", ("do not rescore",)),
            ),
            phenomena=("negative_support", "do_not"),
        ),
    )


def _language_policy_cases(document: EvidenceDocument) -> tuple[BenchmarkCase, ...]:
    dynamic = _span(document, "not just Python")
    reference = _span(document, "reference implementation")
    return (
        _case(
            "polarity.language_dynamic_supported",
            document,
            "modules may be developed in any dynamic language",
            "Find evidence that modules may be developed in any dynamic language.",
            ("modules developed any dynamic language",),
            support=(dynamic,),
            forbidden=(reference,),
            facets=(
                EvidenceFacet("modules", ("Modules",)),
                EvidenceFacet("dynamic language", ("any dynamic language",)),
            ),
            phenomena=("positive_support", "not_just"),
        ),
        _case(
            "polarity.language_only_python_contradicted",
            document,
            "modules may be developed only in Python",
            "Find evidence that modules may be developed only in Python.",
            ("modules developed only Python",),
            support=(),
            contradiction=(dynamic,),
            forbidden=(dynamic,),
            facets=(
                EvidenceFacet("modules", ("Modules",)),
                EvidenceFacet("python", ("Python",)),
            ),
            expect_abstain=True,
            phenomena=("contradiction", "not_just"),
        ),
        _case(
            "polarity.language_python_reference_supported",
            document,
            "Python is the reference implementation for the core package",
            "Find evidence that Python is the reference implementation for the core package.",
            ("Python reference implementation core package",),
            support=(reference,),
            forbidden=(dynamic,),
            facets=(
                EvidenceFacet("python", ("Python",)),
                EvidenceFacet("reference", ("reference implementation",)),
            ),
            phenomena=("positive_support", "not_just"),
        ),
    )


def _case(
    case_id: str,
    document: EvidenceDocument,
    label: str,
    description: str,
    positive_examples: tuple[str, ...],
    *,
    support: tuple[DocumentSpan, ...],
    near_miss: tuple[DocumentSpan, ...] = (),
    contradiction: tuple[DocumentSpan, ...] = (),
    insufficient_context: tuple[DocumentSpan, ...] = (),
    forbidden: tuple[DocumentSpan, ...] = (),
    facets: tuple[EvidenceFacet, ...] = (),
    expect_abstain: bool = False,
    phenomena: tuple[str, ...],
    min_support_score: float = 0.55,
) -> BenchmarkCase:
    return BenchmarkCase(
        id=case_id,
        intent=IntentSpec(
            id=case_id,
            label=label,
            description=description,
            positive_examples=positive_examples,
            required_facets=facets,
            min_support_score=min_support_score,
        ),
        document=document,
        support_span_ids=tuple(span.id for span in support),
        near_miss_span_ids=tuple(span.id for span in near_miss),
        contradiction_span_ids=tuple(span.id for span in contradiction),
        insufficient_context_span_ids=tuple(span.id for span in insufficient_context),
        forbidden_span_ids=tuple(
            dict.fromkeys(span.id for span in (*near_miss, *contradiction, *forbidden))
        ),
        expect_abstain=expect_abstain,
        curation=BenchmarkCuration(
            reviewed=True,
            source="curated_polarity_excerpt",
            difficulty="hard" if contradiction else "medium",
            phenomena=phenomena,
            notes=(
                "Curated polarity benchmark case for negation, contradiction, or "
                "negative support behavior.",
            ),
        ),
    )


def _span(
    document: EvidenceDocument,
    contains: str,
    *,
    kind: SpanKind | None = None,
) -> DocumentSpan:
    matches = [
        span
        for span in document.spans
        if contains in span.text and (kind is None or span.kind == kind)
    ]
    if not matches:
        raise ValueError(f"could not find span containing {contains!r} in {document.id}")
    if len(matches) > 1:
        matches.sort(key=lambda span: (span.kind != SpanKind.SENTENCE, len(span.text)))
    return matches[0]
