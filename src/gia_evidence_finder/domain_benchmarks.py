from __future__ import annotations

import hashlib
from dataclasses import dataclass, replace

from gia_evidence_finder.contracts import (
    BenchmarkCase,
    BenchmarkCuration,
    BenchmarkSuite,
    DocumentSpan,
    EvidenceDocument,
    EvidenceFacet,
    IntentSpec,
)
from gia_evidence_finder.parsing import MarkdownSpanParser


@dataclass(frozen=True)
class DomainExcerpt:
    id: str
    domain: str
    genre: str
    source_url: str
    text: str


@dataclass(frozen=True)
class _DomainTemplate:
    domain: str
    genre: str
    subject: str
    primary: str
    secondary: str
    near_miss: str
    contradicted_claim: str
    contradiction: str
    negative_claim: str
    relation_claim: str
    relation_near_miss: str
    reliability_claim: str
    reliable: str
    unreliable: str


def domain_benchmark_suite() -> BenchmarkSuite:
    documents = _documents()
    cases = tuple(
        case
        for excerpt in DOMAIN_EXCERPTS
        for case in _cases_for_document(excerpt, documents[excerpt.id])
    )
    domains = sorted({excerpt.domain for excerpt in DOMAIN_EXCERPTS})
    genres = sorted({excerpt.genre for excerpt in DOMAIN_EXCERPTS})
    return BenchmarkSuite(
        id="domain_evidence_v4",
        name="Domain evidence benchmark v4",
        description=(
            "Reviewed mixed-domain benchmark with project-management, people-search, "
            "apartment-search, and technical/product evidence excerpts. Cases stress "
            "retrieval quality, support decisions, and refusal of near-miss or "
            "contradictory spans."
        ),
        cases=cases,
        metadata={
            "document_count": str(len(documents)),
            "case_count": str(len(cases)),
            "domains": ",".join(domains),
            "genres": ",".join(genres),
            "review_source": "manual_curated_domain_excerpt",
            "status": "expanded_reviewed",
        },
    )


def domain_benchmark_suites() -> tuple[BenchmarkSuite, ...]:
    return (domain_benchmark_suite(),)


def domain_benchmark_suite_by_id(suite_id: str) -> BenchmarkSuite:
    suites = {suite.id: suite for suite in domain_benchmark_suites()}
    try:
        return suites[suite_id]
    except KeyError as exc:
        raise ValueError(f"unknown domain benchmark suite {suite_id!r}") from exc


def _documents() -> dict[str, EvidenceDocument]:
    parser = MarkdownSpanParser()
    documents: dict[str, EvidenceDocument] = {}
    for excerpt in DOMAIN_EXCERPTS:
        source_hash = hashlib.sha256(excerpt.text.encode("utf-8")).hexdigest()
        parsed = parser.parse(
            excerpt.text,
            document_id=excerpt.id,
            source=excerpt.source_url,
        )
        documents[excerpt.id] = replace(
            parsed,
            metadata={
                "domain": excerpt.domain,
                "genre": excerpt.genre,
                "source_hash": source_hash,
                "review_source": "manual_curated_domain_excerpt",
            },
        )
    return documents


def _cases_for_document(
    excerpt: DomainExcerpt,
    document: EvidenceDocument,
) -> tuple[BenchmarkCase, ...]:
    primary = _span(document, "Primary support:")
    secondary = _span(document, "Secondary support:")
    broad = _span(document, "Related context:")
    near_miss = _span(document, "Near miss:")
    contradiction = _span(document, "Contradiction:")
    negative = _span(document, "Negative support:")
    relation = _span(document, "Relation trap:")
    reliable = _span(document, "Reliable source:")
    unreliable = _span(document, "Unreliable source:")
    base = f"{excerpt.domain}.{excerpt.id}"
    metadata = _case_metadata(excerpt, document)
    return (
        _case(
            f"{base}.primary_support",
            document,
            f"{excerpt.id} primary support",
            f"Find evidence that {metadata.primary}.",
            (metadata.primary,),
            support=(primary,),
            forbidden=(broad, near_miss),
            facets=_facets(metadata.primary),
            phenomena=("direct_support", excerpt.domain, excerpt.genre),
            metadata=metadata,
        ),
        _case(
            f"{base}.secondary_support",
            document,
            f"{excerpt.id} secondary support",
            f"Find evidence that {metadata.secondary}.",
            (metadata.secondary,),
            support=(secondary,),
            forbidden=(primary, near_miss),
            facets=_facets(metadata.secondary),
            phenomena=("direct_support", "minimal_span", excerpt.domain, excerpt.genre),
            metadata=metadata,
        ),
        _case(
            f"{base}.minimal_not_broad",
            document,
            f"{excerpt.id} minimal span not broad context",
            f"Find the smallest evidence span that {metadata.secondary}.",
            (metadata.secondary,),
            support=(secondary,),
            near_miss=(broad,),
            forbidden=(broad,),
            facets=_facets(metadata.secondary),
            phenomena=("minimal_span", "overbroad_chunk", excerpt.domain, excerpt.genre),
            metadata=metadata,
        ),
        _case(
            f"{base}.near_miss_refusal",
            document,
            f"{excerpt.id} near miss should not support",
            f"Find evidence that {metadata.near_miss}.",
            (metadata.near_miss,),
            support=(),
            near_miss=(near_miss,),
            forbidden=(near_miss,),
            expect_abstain=True,
            facets=_facets(metadata.near_miss),
            phenomena=("abstention", "near_miss", "hard_negative", excerpt.domain, excerpt.genre),
            metadata=metadata,
        ),
        _case(
            f"{base}.contradiction_refusal",
            document,
            f"{excerpt.id} contradiction should not support",
            f"Find evidence that {metadata.contradicted_claim}.",
            (metadata.contradicted_claim,),
            support=(),
            contradiction=(contradiction,),
            forbidden=(contradiction,),
            expect_abstain=True,
            facets=_facets(metadata.contradicted_claim),
            phenomena=(
                "abstention",
                "contradiction",
                "hard_negative",
                excerpt.domain,
                excerpt.genre,
            ),
            metadata=metadata,
        ),
        _case(
            f"{base}.negative_support",
            document,
            f"{excerpt.id} negative claim support",
            f"Find evidence that {metadata.negative_claim}.",
            (metadata.negative_claim,),
            support=(negative,),
            forbidden=(primary,),
            facets=_facets(metadata.negative_claim),
            phenomena=("negative_support", excerpt.domain, excerpt.genre),
            metadata=metadata,
        ),
        _case(
            f"{base}.relation_transfer_refusal",
            document,
            f"{excerpt.id} relation transfer should not support",
            f"Find evidence that {metadata.relation_claim}.",
            (metadata.relation_claim,),
            support=(),
            insufficient_context=(relation,),
            forbidden=(relation,),
            expect_abstain=True,
            facets=_facets(metadata.relation_claim),
            phenomena=("abstention", "relation_binding", "relation_transfer", excerpt.domain),
            metadata=metadata,
        ),
        _case(
            f"{base}.source_reliability",
            document,
            f"{excerpt.id} reliable source resolves conflict",
            f"Find evidence that {metadata.reliability_claim}.",
            (metadata.reliability_claim,),
            support=(reliable,),
            contradiction=(unreliable,),
            forbidden=(unreliable,),
            facets=_facets(metadata.reliability_claim),
            phenomena=("source_reliability", "contradiction", excerpt.domain, excerpt.genre),
            metadata=metadata,
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
    expect_abstain: bool = False,
    facets: tuple[EvidenceFacet, ...] = (),
    phenomena: tuple[str, ...],
    metadata: _DomainTemplate,
) -> BenchmarkCase:
    return BenchmarkCase(
        id=case_id,
        intent=IntentSpec(
            id=case_id,
            label=label,
            description=description,
            positive_examples=positive_examples,
            required_facets=facets,
            min_support_score=0.50,
        ),
        document=document,
        support_span_ids=tuple(span.id for span in support),
        near_miss_span_ids=tuple(span.id for span in near_miss),
        contradiction_span_ids=tuple(span.id for span in contradiction),
        insufficient_context_span_ids=tuple(span.id for span in insufficient_context),
        forbidden_span_ids=tuple(
            dict.fromkeys(
                span.id for span in (*near_miss, *contradiction, *insufficient_context, *forbidden)
            )
        ),
        expect_abstain=expect_abstain,
        curation=BenchmarkCuration(
            reviewed=True,
            source="manual_curated_domain_excerpt",
            difficulty="hard" if expect_abstain or contradiction else "medium",
            phenomena=phenomena,
            notes=(
                f"domain={metadata.domain}",
                f"genre={metadata.genre}",
                f"source_hash={document.metadata['source_hash']}",
                "review_source=manual_curated_domain_excerpt",
            ),
        ),
    )


def _span(document: EvidenceDocument, contains: str) -> DocumentSpan:
    matches = [span for span in document.spans if contains in span.text]
    if not matches:
        raise ValueError(f"could not find span containing {contains!r} in {document.id}")
    return min(matches, key=lambda span: len(span.text))


def _facets(text: str) -> tuple[EvidenceFacet, ...]:
    words = tuple(
        word.strip(".,:;()").lower() for word in text.split() if len(word.strip(".,:;()")) > 3
    )
    anchors = tuple(dict.fromkeys(words[:4]))
    return (EvidenceFacet("anchor", anchors or (text,)),)


def _case_metadata(excerpt: DomainExcerpt, document: EvidenceDocument) -> _DomainTemplate:
    return _TEMPLATES[excerpt.id]


def _excerpt(template: _DomainTemplate, index: int) -> DomainExcerpt:
    doc_id = f"{template.domain}_{template.genre}_{index:02d}".replace("-", "_")
    text = (
        f"# {template.subject}\n\n"
        f"- Primary support: {template.primary}.\n"
        f"- Secondary support: {template.secondary}.\n"
        "- Related context: The surrounding record includes ownership, dates, "
        "and workflow notes, but this broad context is not the smallest "
        f"evidence for {template.secondary}.\n"
        f"- Near miss: {template.near_miss}.\n"
        f"- Contradiction: {template.contradiction}.\n"
        f"- Negative support: {template.negative_claim}.\n"
        f"- Relation trap: {template.relation_near_miss}.\n"
        f"- Reliable source: {template.reliable}.\n"
        f"- Unreliable source: {template.unreliable}.\n"
    )
    return DomainExcerpt(
        id=doc_id,
        domain=template.domain,
        genre=template.genre,
        source_url=f"curated://{template.domain}/{template.genre}/{index:02d}",
        text=text,
    )


def _project_templates() -> tuple[_DomainTemplate, ...]:
    subjects = (
        (
            "ticket_growth_activation",
            "ticket description",
            "Activation checklist",
            "signup onboarding analytics",
            "not billing invoices",
        ),
        (
            "ticket_billing_webhook",
            "comment thread",
            "Billing webhook retry",
            "enterprise invoice sync",
            "not mobile offline sync",
        ),
        (
            "prd_mobile_offline",
            "prd snippet",
            "Mobile offline sync",
            "beta Android caches",
            "not OAuth security",
        ),
        (
            "incident_api_latency",
            "incident update",
            "API latency incident",
            "database pool exhaustion",
            "not customer onboarding",
        ),
        (
            "changelog_data_export",
            "changelog",
            "Data export backfill",
            "warehouse timeout fix",
            "not payment retry",
        ),
        (
            "ticket_oauth_audit",
            "ticket description",
            "OAuth SSO audit",
            "enterprise permission review",
            "not growth activation",
        ),
        (
            "comment_crash_bug",
            "comment thread",
            "Android crash bug",
            "offline sync blocker",
            "not invoice webhook",
        ),
        (
            "prd_ai_triage",
            "prd snippet",
            "AI triage assistant",
            "support escalation routing",
            "not lease approval",
        ),
        (
            "incident_cache_purge",
            "incident update",
            "Cache purge incident",
            "stale dashboard metrics",
            "not résumé parsing",
        ),
        (
            "changelog_search_rank",
            "changelog",
            "Search rank update",
            "comment evidence boost",
            "not apartment policy",
        ),
    )
    return tuple(
        _DomainTemplate(
            domain="project_management",
            genre=genre,
            subject=subject,
            primary=f"{subject} covers {theme}",
            secondary=f"{subject} explicitly mentions {detail}",
            near_miss=f"{subject} is only about {wrong}",
            contradicted_claim=f"{subject} has no {detail}",
            contradiction=f"{subject} does not remove {detail}; it keeps {detail} in scope",
            negative_claim=f"{subject} is not about {wrong}",
            relation_claim=f"the dependency itself owns {detail}",
            relation_near_miss=(
                f"A related dependency mentions {detail}, "
                "but the dependency does not own that work item"
            ),
            reliability_claim=f"the official project record says {detail} remains in scope",
            reliable=f"The official project record confirms {detail} remains in scope",
            unreliable=f"An old Slack summary incorrectly says {detail} was removed from scope",
        )
        for subject, genre, theme, detail, wrong in subjects
    )


def _people_templates() -> tuple[_DomainTemplate, ...]:
    subjects = (
        (
            "jane_resume",
            "resume",
            "Jane Carter",
            "AI deployment infrastructure",
            "not frontend design",
        ),
        (
            "rafa_profile",
            "profile bio",
            "Rafa Singh",
            "OAuth SSO permission audits",
            "not invoice operations",
        ),
        (
            "miles_bio",
            "profile bio",
            "Miles Rivera",
            "billing webhook reliability",
            "not mobile rendering",
        ),
        (
            "nora_resume",
            "resume",
            "Nora Lee",
            "activation funnel experiments",
            "not security reviews",
        ),
        (
            "press_vectorforge",
            "press snippet",
            "VectorForge",
            "developer tools for AI agents",
            "not apartment leases",
        ),
        (
            "investor_note_tensorgrid",
            "investor blurb",
            "TensorGrid",
            "model serving infrastructure",
            "not checkout design",
        ),
        (
            "advisor_page_priya",
            "advisor blurb",
            "Priya Shah",
            "fintech API strategy",
            "not Android crash triage",
        ),
        (
            "company_page_laterdev",
            "company page",
            "LaterDev",
            "AI deployment workflows",
            "not tenant screening",
        ),
        (
            "speaker_bio_sam",
            "profile bio",
            "Sam Rivera",
            "professional graph search",
            "not warehouse backfill",
        ),
        ("resume_li", "resume", "Li Chen", "warehouse analytics pipelines", "not OAuth SSO"),
    )
    return tuple(
        _DomainTemplate(
            domain="people_search",
            genre=genre,
            subject=subject,
            primary=f"{subject} describes {person}'s work on {detail}",
            secondary=f"{person} has direct experience with {detail}",
            near_miss=f"{person} only worked on {wrong}",
            contradicted_claim=f"{person} has no experience with {detail}",
            contradiction=(
                f"{person}'s profile does not deny {detail}; "
                f"it lists {detail} as a work area"
            ),
            negative_claim=f"{person} is not described as working on {wrong}",
            relation_claim=f"{person}'s employer is the source of {detail}",
            relation_near_miss=(
                f"A company page mentions {detail}, but this span does not "
                f"prove the employer relationship for {person}"
            ),
            reliability_claim=f"the verified profile says {person} worked on {detail}",
            reliable=f"The verified profile says {person} worked on {detail}",
            unreliable=f"An outdated conference blurb says {person} only worked on {wrong}",
        )
        for subject, genre, person, detail, wrong in subjects
    )


def _apartment_templates() -> tuple[_DomainTemplate, ...]:
    subjects = (
        (
            "apt_river_12_listing",
            "listing description",
            "River 12",
            "in-unit laundry",
            "not rooftop pool",
        ),
        (
            "apt_river_12_lease",
            "lease clause",
            "River 12",
            "cats allowed with approval",
            "not dogs allowed",
        ),
        (
            "apt_cedar_rules",
            "building rules",
            "Cedar House",
            "quiet hours after 10pm",
            "not free parking",
        ),
        (
            "apt_cedar_review",
            "tenant review",
            "Cedar House",
            "weekend street noise",
            "not elevator outages",
        ),
        (
            "apt_maple_inspection",
            "inspection note",
            "Maple Loft",
            "sprinkler inspection passed",
            "not gym access",
        ),
        ("apt_maple_listing", "listing description", "Maple Loft", "no broker fee", "not pet spa"),
        (
            "apt_harbor_lease",
            "lease clause",
            "Harbor Unit",
            "dogs are not permitted",
            "not furnished terrace",
        ),
        (
            "apt_harbor_review",
            "tenant review",
            "Harbor Unit",
            "morning ferry noise",
            "not subway outage",
        ),
        (
            "apt_elm_rules",
            "building rules",
            "Elm Court",
            "bike storage requires registration",
            "not pool hours",
        ),
        (
            "apt_elm_inspection",
            "inspection note",
            "Elm Court",
            "heat inspection passed",
            "not package theft",
        ),
    )
    return tuple(
        _DomainTemplate(
            domain="apartment_search",
            genre=genre,
            subject=subject,
            primary=f"{building} has {detail}",
            secondary=f"{building} explicitly states {detail}",
            near_miss=f"{building} only offers {wrong}",
            contradicted_claim=f"{building} does not have {detail}",
            contradiction=f"The current source does not remove {detail}; it confirms {detail}",
            negative_claim=f"{building} is not advertised for {wrong}",
            relation_claim=f"the neighborhood guide guarantees {detail} inside the unit",
            relation_near_miss=(
                f"The neighborhood guide mentions {detail}, "
                "but it does not prove the unit includes it"
            ),
            reliability_claim=f"the official lease or inspection source confirms {detail}",
            reliable=f"The official lease or inspection source confirms {detail}",
            unreliable=f"An old listing draft incorrectly says {detail} is unavailable",
        )
        for subject, genre, building, detail, wrong in subjects
    )


def _technical_templates() -> tuple[_DomainTemplate, ...]:
    subjects = (
        (
            "api_auth_runbook",
            "api doc",
            "API auth runbook",
            "admin token on mutating routes",
            "not public mutation",
        ),
        (
            "storage_spec_epoch",
            "spec",
            "Storage spec",
            "write epochs group mutations",
            "not browser rendering",
        ),
        (
            "planner_spec_scope",
            "spec",
            "Planner spec",
            "wrong scope fails validation",
            "not silent relocation",
        ),
        (
            "release_cache_notes",
            "release notes",
            "Release notes",
            "reranker pair-score caching",
            "not answer generation",
        ),
        (
            "issue_minimal_spans",
            "issue discussion",
            "Issue discussion",
            "minimal sentence spans",
            "not whole sections",
        ),
        (
            "pr_review_semantic",
            "PR discussion",
            "PR discussion",
            "open semantic gate diagnostics",
            "not schema deletion",
        ),
        ("sdk_quickstart", "api doc", "SDK quickstart", "bulk claim upserts", "not CSS themes"),
        (
            "benchmark_spec",
            "spec",
            "Benchmark spec",
            "dev thresholds stay frozen on test",
            "not retuning",
        ),
        (
            "migration_runbook",
            "runbook",
            "Migration runbook",
            "checksum verification",
            "not chat citations",
        ),
        (
            "quality_report",
            "release notes",
            "Quality report",
            "forbidden supported top-1 tracking",
            "not landing pages",
        ),
    )
    return tuple(
        _DomainTemplate(
            domain="technical_product",
            genre=genre,
            subject=subject,
            primary=f"{title} requires {detail}",
            secondary=f"{title} explicitly documents {detail}",
            near_miss=f"{title} is only about {wrong}",
            contradicted_claim=f"{title} forbids {detail}",
            contradiction=f"{title} does not forbid {detail}; it requires {detail}",
            negative_claim=f"{title} is not about {wrong}",
            relation_claim=f"a dependent library itself provides {detail}",
            relation_near_miss=(
                f"A dependent library is mentioned near {detail}, "
                "but this does not prove the dependency provides it"
            ),
            reliability_claim=f"the current spec says {detail} is required",
            reliable=f"The current spec says {detail} is required",
            unreliable=f"An archived note incorrectly says {detail} is optional",
        )
        for subject, genre, title, detail, wrong in subjects
    )


_TEMPLATE_LIST = (
    *_project_templates(),
    *_people_templates(),
    *_apartment_templates(),
    *_technical_templates(),
)

DOMAIN_EXCERPTS: tuple[DomainExcerpt, ...] = tuple(
    _excerpt(template, index) for index, template in enumerate(_TEMPLATE_LIST, start=1)
)

_TEMPLATES: dict[str, _DomainTemplate] = {
    excerpt.id: template for excerpt, template in zip(DOMAIN_EXCERPTS, _TEMPLATE_LIST, strict=True)
}
