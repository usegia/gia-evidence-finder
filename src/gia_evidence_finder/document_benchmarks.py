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
class DocumentExcerpt:
    id: str
    source_url: str
    genre: str
    text: str


DOCUMENT_EXCERPTS: tuple[DocumentExcerpt, ...] = (
    DocumentExcerpt(
        id="storage_spec",
        source_url="repo://docs/specs/storage-interfaces.md",
        genre="spec",
        text="""# Storage Interfaces

## Write Epochs

Write epochs are append-only records that group schema, entity, edge, claim,
profile, and index-version mutations under one committed transaction.

Each write epoch stores enough replay metadata for the integrity doctor to
verify that persisted graph records can be replayed.

The memory store is deterministic for unit tests but does not claim Postgres
advisory-lock migration behavior.

## Integrity

Integrity checks validate replayability, migration checksums, dangling
references, and semantic index-version coverage.
""",
    ),
    DocumentExcerpt(
        id="api_runbook",
        source_url="repo://README.md#postgres-execution-api",
        genre="runbook",
        text="""# Production API Runbook

Mutating admin routes for schema creation, graph writes, concept packs,
benchmark seeding, and migration upgrades require the x-traverce-admin-token
header.

Query, status, and health routes remain readable without the admin token.

The local showcase can select the project-management fixture by sending
schema_name project_management_benchmark in query payloads.
""",
    ),
    DocumentExcerpt(
        id="planning_spec",
        source_url="repo://docs/specs/natural-language-planning.md",
        genre="spec",
        text="""# Natural-Language Planning

The natural-language compiler must emit binding-first typed payloads rather
than executable string queries.

Planner repair is a deterministic guardrail that may normalize schema-proven
shapes but must not invent missing user intent.

Wrong semantic scope should fail validation with a useful diagnostic instead
of being silently moved to another node or edge space.
""",
    ),
    DocumentExcerpt(
        id="evidence_release_notes",
        source_url="repo://packages/gia-evidence-finder/docs/sota-progress-2026-05-08.md",
        genre="release_notes",
        text="""# Evidence Extraction Release Notes

The evidence extractor emits stable span offsets, heading paths, and span kinds
for Markdown documents.

Model adjudication records the judge model, rubric version, source hash, and
disagreement group before a case can count as reviewed.

Batch reranking caches pair scores by query hash, span hash, and model identity
so repeated extraction benchmarks do not rescore unchanged pairs.
""",
    ),
    DocumentExcerpt(
        id="issue_discussion",
        source_url="repo://issues/42",
        genre="issue",
        text="""# Issue 42: Over-Broad Evidence

A user reported that a chunk retriever returned the whole installation section
when the claim only asked about Docker Compose.

The proposed fix is to return minimal sentence or bullet spans and keep the
larger paragraph only as surrounding context.

The team rejected adding an LLM answer generator to the hot path because the
library should return source evidence, not prose.
""",
    ),
)


def non_readme_benchmark_suite() -> BenchmarkSuite:
    documents = _documents()
    cases: list[BenchmarkCase] = []
    cases.extend(_storage_cases(documents["storage_spec"]))
    cases.extend(_api_cases(documents["api_runbook"]))
    cases.extend(_planning_cases(documents["planning_spec"]))
    cases.extend(_release_note_cases(documents["evidence_release_notes"]))
    cases.extend(_issue_cases(documents["issue_discussion"]))
    return BenchmarkSuite(
        id="non_readme_docs_v1",
        name="Non-README document evidence benchmark",
        description=(
            "Curated evidence extraction cases from spec, runbook, release-note, "
            "and issue-style documents. The suite stresses long-lived product docs, "
            "operational claims, curation metadata, and over-broad chunk traps."
        ),
        cases=tuple(cases),
        metadata={
            "document_count": str(len(documents)),
            "case_count": str(len(cases)),
            "genres": ",".join(sorted({excerpt.genre for excerpt in DOCUMENT_EXCERPTS})),
        },
    )


def document_benchmark_suites() -> tuple[BenchmarkSuite, ...]:
    return (non_readme_benchmark_suite(),)


def document_benchmark_suite_by_id(suite_id: str) -> BenchmarkSuite:
    suites = {suite.id: suite for suite in document_benchmark_suites()}
    try:
        return suites[suite_id]
    except KeyError as exc:
        raise ValueError(f"unknown document benchmark suite {suite_id!r}") from exc


def _documents() -> dict[str, EvidenceDocument]:
    parser = MarkdownSpanParser()
    return {
        excerpt.id: parser.parse(
            excerpt.text,
            document_id=excerpt.id,
            source=excerpt.source_url,
        )
        for excerpt in DOCUMENT_EXCERPTS
    }


def _storage_cases(document: EvidenceDocument) -> tuple[BenchmarkCase, ...]:
    epoch = _span(document, "group schema")
    replay = _span(document, "stores enough replay metadata")
    memory = _span(document, "memory store is deterministic")
    return (
        _case(
            "docs.storage.write_epoch_transaction",
            document,
            "write epochs group mutations in one committed transaction",
            (
                "Find evidence that write epochs group graph mutations under one "
                "committed transaction."
            ),
            ("write epochs group mutations committed transaction",),
            (epoch,),
            forbidden=(memory,),
            facets=(
                EvidenceFacet("write epochs", ("Write epochs",)),
                EvidenceFacet("transaction", ("committed transaction",)),
            ),
            phenomena=("direct_support", "spec_claim", "hard_negative"),
        ),
        _case(
            "docs.storage.replay_metadata",
            document,
            "integrity doctor can verify replay metadata",
            "Find evidence that write epochs store replay metadata for integrity verification.",
            ("replay metadata integrity doctor verify graph records",),
            (replay,),
            forbidden=(epoch,),
            facets=(
                EvidenceFacet("replay", ("replay metadata",)),
                EvidenceFacet("integrity", ("integrity doctor", "verify")),
            ),
            phenomena=("direct_support", "spec_claim", "near_miss"),
        ),
        _case(
            "docs.storage.memory_no_advisory_locks",
            document,
            "memory store uses Postgres advisory locks",
            "Find evidence that the memory store uses Postgres advisory locks for migrations.",
            ("memory store Postgres advisory locks migrations",),
            (),
            contradiction=(memory,),
            forbidden=(memory,),
            facets=(
                EvidenceFacet("memory store", ("memory store",)),
                EvidenceFacet("advisory locks", ("advisory-lock", "advisory locks")),
            ),
            excluded_facets=(EvidenceFacet("does not claim", ("does not claim",)),),
            expect_abstain=True,
            phenomena=("abstention", "contradiction", "spec_claim"),
        ),
    )


def _api_cases(document: EvidenceDocument) -> tuple[BenchmarkCase, ...]:
    admin = _span(document, "Mutating admin routes")
    readable = _span(document, "remain readable")
    fixture = _span(document, "schema_name project_management_benchmark")
    return (
        _case(
            "docs.api.admin_token",
            document,
            "mutating admin routes require admin token",
            "Find evidence that mutating admin routes require the x-traverce-admin-token header.",
            ("mutating admin routes require x-traverce-admin-token",),
            (admin,),
            forbidden=(readable,),
            facets=(
                EvidenceFacet("mutating", ("Mutating admin routes",)),
                EvidenceFacet("token", ("x-traverce-admin-token",)),
            ),
            phenomena=("direct_support", "runbook_claim", "hard_negative"),
        ),
        _case(
            "docs.api.query_public",
            document,
            "query status and health routes are readable without token",
            (
                "Find evidence that query, status, and health routes remain readable "
                "without the admin token."
            ),
            ("query status health readable without admin token",),
            (readable,),
            forbidden=(admin,),
            facets=(
                EvidenceFacet("routes", ("Query", "status", "health")),
                EvidenceFacet("readable", ("readable without",)),
            ),
            phenomena=("direct_support", "runbook_claim", "relation_binding"),
        ),
        _case(
            "docs.api.fixture_payload",
            document,
            "project-management fixture selected through schema_name payload",
            "Find evidence that the project-management fixture can be selected in query payloads.",
            ("schema_name project_management_benchmark query payloads",),
            (fixture,),
            forbidden=(admin,),
            facets=(
                EvidenceFacet("schema field", ("schema_name",)),
                EvidenceFacet("fixture", ("project_management_benchmark",)),
            ),
            phenomena=("direct_support", "runbook_claim", "exact_anchor"),
        ),
    )


def _planning_cases(document: EvidenceDocument) -> tuple[BenchmarkCase, ...]:
    binding = _span(document, "binding-first typed payloads")
    repair = _span(document, "must not invent missing user intent")
    wrong_scope = _span(document, "Wrong semantic scope")
    return (
        _case(
            "docs.planning.binding_first_payloads",
            document,
            "natural-language compiler emits binding-first typed payloads",
            "Find evidence that the natural-language compiler emits binding-first typed payloads.",
            ("natural-language compiler binding-first typed payloads",),
            (binding,),
            forbidden=(repair,),
            facets=(
                EvidenceFacet("compiler", ("natural-language compiler",)),
                EvidenceFacet("typed payloads", ("binding-first typed payloads",)),
            ),
            phenomena=("direct_support", "spec_claim", "typed_contract"),
        ),
        _case(
            "docs.planning.repair_no_intent_invention",
            document,
            "planner repair does not invent missing user intent",
            "Find evidence that planner repair must not invent missing user intent.",
            ("planner repair must not invent missing user intent",),
            (repair,),
            forbidden=(binding,),
            facets=(
                EvidenceFacet("repair", ("Planner repair",)),
                EvidenceFacet("no invention", ("must not invent",)),
            ),
            phenomena=("direct_support", "spec_claim", "hard_negative"),
        ),
        _case(
            "docs.planning.wrong_scope_not_silent",
            document,
            "wrong semantic scope is silently moved",
            (
                "Find evidence that wrong semantic scope should be silently moved to "
                "another graph space."
            ),
            ("wrong semantic scope silently moved node edge space",),
            (),
            contradiction=(wrong_scope,),
            forbidden=(wrong_scope,),
            facets=(
                EvidenceFacet("wrong scope", ("Wrong semantic scope",)),
                EvidenceFacet("silent move", ("silently moved",)),
            ),
            negative_examples=("should fail validation instead of being silently moved",),
            excluded_facets=(EvidenceFacet("fail validation", ("fail validation", "instead of")),),
            expect_abstain=True,
            phenomena=("abstention", "contradiction", "spec_claim"),
        ),
    )


def _release_note_cases(document: EvidenceDocument) -> tuple[BenchmarkCase, ...]:
    offsets = _span(document, "stable span offsets")
    judge = _span(document, "judge model")
    cache = _span(document, "Batch reranking caches")
    return (
        _case(
            "docs.release.stable_spans",
            document,
            "extractor emits stable offsets heading paths and span kinds",
            "Find evidence that the evidence extractor emits stable span metadata for Markdown.",
            ("stable span offsets heading paths span kinds Markdown",),
            (offsets,),
            forbidden=(judge,),
            facets=(
                EvidenceFacet("offsets", ("stable span offsets",)),
                EvidenceFacet("heading", ("heading paths",)),
            ),
            phenomena=("direct_support", "release_note", "metadata_claim"),
        ),
        _case(
            "docs.release.model_adjudication_metadata",
            document,
            "model adjudication records judge metadata before review",
            (
                "Find evidence that model adjudication records judge model, rubric, "
                "source hash, and disagreement group."
            ),
            ("model adjudication judge model rubric version source hash disagreement group",),
            (judge,),
            forbidden=(cache,),
            facets=(
                EvidenceFacet("judge", ("judge model",)),
                EvidenceFacet("rubric", ("rubric version",)),
                EvidenceFacet("source hash", ("source hash",)),
            ),
            phenomena=("direct_support", "release_note", "model_judgment"),
        ),
        _case(
            "docs.release.batch_cache_keys",
            document,
            "batch reranking caches scores by query span and model identity",
            (
                "Find evidence that batch reranking caches scores by query hash, span "
                "hash, and model identity."
            ),
            ("batch reranking caches query hash span hash model identity",),
            (cache,),
            forbidden=(judge,),
            facets=(
                EvidenceFacet("cache", ("caches pair scores",)),
                EvidenceFacet("keys", ("query hash", "span hash", "model identity")),
            ),
            phenomena=("direct_support", "release_note", "performance"),
        ),
    )


def _issue_cases(document: EvidenceDocument) -> tuple[BenchmarkCase, ...]:
    problem = _span(document, "whole installation section")
    fix = _span(document, "minimal sentence or bullet spans")
    no_llm = _span(document, "rejected adding an LLM answer generator")
    return (
        _case(
            "docs.issue.overbroad_chunk_problem",
            document,
            "chunk retriever returned whole installation section",
            "Find evidence that the reported problem was an over-broad installation section chunk.",
            ("chunk retriever returned whole installation section Docker Compose",),
            (problem,),
            forbidden=(fix,),
            facets=(
                EvidenceFacet("chunk retriever", ("chunk retriever",)),
                EvidenceFacet("whole section", ("whole installation section",)),
            ),
            phenomena=("direct_support", "issue_discussion", "overbroad_chunk"),
        ),
        _case(
            "docs.issue.minimal_span_fix",
            document,
            "fix returns minimal sentence or bullet spans",
            "Find evidence that the proposed fix returns minimal sentence or bullet spans.",
            ("proposed fix return minimal sentence bullet spans",),
            (fix,),
            forbidden=(problem,),
            facets=(
                EvidenceFacet("minimal", ("minimal sentence", "bullet spans")),
                EvidenceFacet("context", ("surrounding context",)),
            ),
            phenomena=("direct_support", "issue_discussion", "minimal_span"),
        ),
        _case(
            "docs.issue.no_llm_hot_path",
            document,
            "LLM answer generator rejected for hot path",
            "Find evidence that adding an LLM answer generator to the hot path was rejected.",
            ("rejected LLM answer generator hot path source evidence not prose",),
            (no_llm,),
            forbidden=(problem,),
            facets=(
                EvidenceFacet("rejected", ("rejected",)),
                EvidenceFacet("hot path", ("hot path",)),
                EvidenceFacet("source evidence", ("source evidence", "not prose")),
            ),
            phenomena=("direct_support", "issue_discussion", "no_llm_hot_path"),
        ),
    )


def _case(
    case_id: str,
    document: EvidenceDocument,
    label: str,
    description: str,
    positive_examples: tuple[str, ...],
    support: tuple[DocumentSpan, ...],
    *,
    near_miss: tuple[DocumentSpan, ...] = (),
    contradiction: tuple[DocumentSpan, ...] = (),
    insufficient_context: tuple[DocumentSpan, ...] = (),
    forbidden: tuple[DocumentSpan, ...] = (),
    facets: tuple[EvidenceFacet, ...] = (),
    excluded_facets: tuple[EvidenceFacet, ...] = (),
    negative_examples: tuple[str, ...] = (),
    expect_abstain: bool = False,
    phenomena: tuple[str, ...] = ("direct_support",),
    difficulty: str = "medium",
    min_support_score: float = 0.55,
) -> BenchmarkCase:
    return BenchmarkCase(
        id=case_id,
        intent=IntentSpec(
            id=case_id,
            label=label,
            description=description,
            positive_examples=positive_examples,
            negative_examples=negative_examples,
            required_facets=facets,
            excluded_facets=excluded_facets,
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
            source="curated_non_readme_excerpt",
            difficulty=difficulty,
            phenomena=phenomena,
            notes=(
                "Curated non-README benchmark case for specs, runbooks, release notes, or issues.",
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
