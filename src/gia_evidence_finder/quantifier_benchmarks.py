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
)
from gia_evidence_finder.parsing import MarkdownSpanParser
from gia_evidence_finder.quantifiers import requirements_from_text


@dataclass(frozen=True)
class QuantifierExcerpt:
    id: str
    domain: str
    text: str


def quantifier_benchmark_suite() -> BenchmarkSuite:
    documents = _documents()
    cases = (
        *_project_cases(documents["project_dates"]),
        *_ticket_metric_cases(documents["ticket_metrics"]),
        *_people_cases(documents["people_dates"]),
        *_apartment_cases(documents["apartment_listing"]),
        *_technical_cases(documents["technical_metrics"]),
    )
    contradiction_cases = sum(1 for case in cases if case.contradiction_span_ids)
    return BenchmarkSuite(
        id="quantifier_numeric_date_v1",
        name="Quantifier numeric and date evidence benchmark",
        description=(
            "Focused reviewed benchmark for exact years, dates, numeric thresholds, "
            "currency, percentages, durations, multipliers, and insufficient-context "
            "refusals."
        ),
        cases=cases,
        metadata={
            "document_count": str(len(documents)),
            "case_count": str(len(cases)),
            "contradiction_case_count": str(contradiction_cases),
            "status": "focused_quantifier",
            "review_source": "manual_curated_quantifier_excerpt",
        },
    )


def quantifier_benchmark_suites() -> tuple[BenchmarkSuite, ...]:
    return (quantifier_benchmark_suite(), quantifier_binding_benchmark_suite())


def quantifier_benchmark_suite_by_id(suite_id: str) -> BenchmarkSuite:
    suites = {suite.id: suite for suite in quantifier_benchmark_suites()}
    try:
        return suites[suite_id]
    except KeyError as exc:
        raise ValueError(f"unknown quantifier benchmark suite {suite_id!r}") from exc


def quantifier_binding_benchmark_suite() -> BenchmarkSuite:
    document = MarkdownSpanParser().parse(
        _BINDING_DOCUMENT,
        document_id="quantifier_binding_cases",
        source="curated://quantifier/binding/v2",
    )
    specs = (
        _binding_spec(
            "binding.started_ended",
            "Project started in 2024 and ended in 2025",
            "started in 2024 and ended in 2025",
            "started in 2025 and ended in 2024",
            "swapped_role",
        ),
        _binding_spec(
            "binding.created_completed",
            "Project created on 2024-01-10 and completed on 2025-03-01",
            "created on 2024-01-10 and completed on 2025-03-01",
            "created on 2025-03-01 and completed on 2024-01-10",
            "swapped_role",
        ),
        _binding_spec(
            "binding.founded_announced",
            "VectorForge was founded in 2023 and investment announced in 2024",
            "founded in 2023 and investment announced in 2024",
            "founded in 2024 and investment announced in 2023",
            "swapped_role",
        ),
        _binding_spec(
            "binding.worked_founded",
            "Jane worked at Stripe until 2023 and founded VectorForge in 2025",
            "worked at Stripe until 2023 and founded VectorForge in 2025",
            "worked at Stripe until 2025 and founded VectorForge in 2023",
            "swapped_role",
        ),
        _binding_spec(
            "binding.ticket_created_due",
            "Ticket created in 2024 and due in 2025",
            "ticket created in 2024 and due in 2025",
            "ticket created in 2025 and due in 2024",
            "swapped_role",
        ),
        _binding_spec(
            "binding.updated_completed",
            "Ticket updated in March 2025 and completed in April 2025",
            "updated in March 2025 and completed in April 2025",
            "updated in April 2025 and completed in March 2025",
            "swapped_role",
        ),
        _binding_spec(
            "binding.retry_latency",
            "Webhook retried 3 times and latency stayed under 100 ms",
            "retried 3 times and latency was 80 ms",
            "retried 100 times and latency was 3 ms",
            "swapped_role",
        ),
        _binding_spec(
            "binding.p95_latency",
            "p95 latency is under 100 ms",
            "p95 latency is 80 ms",
            "p50 latency is 80 ms",
            "wrong_role",
        ),
        _binding_spec(
            "binding.cache_hit_rate",
            "Cache hit rate reached 94%",
            "cache hit rate reached 94%",
            "cache hit rate reached 84%",
            "wrong_value",
        ),
        _binding_spec(
            "binding.speedup",
            "Benchmark was 55x faster after caching",
            "benchmark was 55x faster after caching",
            "unrelated import job was 55x faster",
            "wrong_subject",
        ),
        _binding_spec(
            "binding.rent_budget",
            "River 12 rent is under $3,500",
            "rent is $3,200",
            "deposit is $3,200 but rent is $3,800",
            "wrong_role",
        ),
        _binding_spec(
            "binding.monthly_rent",
            "River 12 monthly rent is $3,200",
            "monthly rent is $3,200",
            "broker fee is $3,200",
            "wrong_role",
        ),
        _binding_spec(
            "binding.bed_bath",
            "River 12 has 2 bedrooms and 1 bathroom",
            "2 bedrooms and 1 bathroom",
            "1 bedroom and 2 bathrooms",
            "swapped_role",
        ),
        _binding_spec(
            "binding.unit_number",
            "River 12 has 2 bedrooms",
            "2 bedrooms",
            "Unit 2 has 1 bedroom",
            "wrong_role",
        ),
        _binding_spec(
            "binding.available",
            "River 12 is available after June 1, 2026",
            "available June 15, 2026",
            "lease starts after June 1, 2026 but unit is available May 15, 2026",
            "wrong_role",
        ),
        _binding_spec(
            "binding.lease_term",
            "Minimum lease is 12 months",
            "minimum lease term is 12 months",
            "renewal option is 12 months; minimum lease term is 6 months",
            "wrong_role",
        ),
        _binding_spec(
            "binding.python_min",
            "Package supports Python 3.12 or newer",
            "supports Python 3.12",
            "supports Python 3.10 through Python 3.11",
            "wrong_value",
        ),
        _binding_spec(
            "binding.python_unsupported",
            "Python 3.11 is unsupported",
            "Python 3.11 is unsupported",
            "Python 3.11 is supported",
            "polarity_quantifier",
        ),
        _binding_spec(
            "binding.rent_deposit",
            "Rent is $3,200 and deposit is $1,000",
            "rent is $3,200 and deposit is $1,000",
            "rent is $1,000 and deposit is $3,200",
            "swapped_role",
        ),
        _binding_spec(
            "binding.fees",
            "Application fee is $50 and broker fee is $0",
            "application fee is $50 and broker fee is $0",
            "application fee is $0 and broker fee is $50",
            "swapped_role",
        ),
        _binding_spec(
            "binding.seats_projects",
            "Package includes 3 seats and 10 projects",
            "includes 3 seats and 10 projects",
            "includes 10 seats and 3 projects",
            "swapped_role",
        ),
        _binding_spec(
            "binding.engineers_designers",
            "Team has 8 engineers and 2 designers",
            "has 8 engineers and 2 designers",
            "has 2 engineers and 8 designers",
            "swapped_role",
        ),
        _binding_spec(
            "binding.incident_range",
            "Incident started on May 1, 2025 and ended on May 3, 2025",
            "started on May 1, 2025 and ended on May 3, 2025",
            "started on May 3, 2025 and ended on May 1, 2025",
            "swapped_role",
        ),
        _binding_spec(
            "binding.incomplete",
            "Project started in 2024 and ended in 2026",
            "started in 2024 and ended in 2025",
            "started in 2024 only",
            "insufficient_context",
        ),
    )
    cases = tuple(
        _binding_case(document, case_id, claim, support_text, negative_text, phenomenon)
        for case_id, claim, support_text, negative_text, phenomenon in specs
    )
    return BenchmarkSuite(
        id="quantifier_binding_v2",
        name="Quantifier binding benchmark v2",
        description=(
            "Reviewed role-bound quantifier benchmark for swapped dates, metrics, "
            "money fields, counts, versions, and insufficient-context spans."
        ),
        cases=cases,
        metadata={
            "document_count": "1",
            "case_count": str(len(cases)),
            "status": "focused_quantifier_binding",
            "review_source": "manual_curated_quantifier_binding_excerpt",
        },
    )


def _documents() -> dict[str, EvidenceDocument]:
    parser = MarkdownSpanParser()
    return {
        excerpt.id: parser.parse(
            excerpt.text,
            document_id=excerpt.id,
            source=f"curated://quantifier/{excerpt.domain}/{excerpt.id}",
        )
        for excerpt in QUANTIFIER_EXCERPTS
    }


def _project_cases(document: EvidenceDocument) -> tuple[BenchmarkCase, ...]:
    created_2025 = _span(document, "created in 2025")
    created_2024 = _span(document, "created in 2024")
    due_may = _span(document, "due on 2025-05-20")
    due_june = _span(document, "due on 2025-06-15")
    sparse = _span(document, "created after planning")
    return (
        _case(
            "quant.project.created_2025_supported",
            document,
            "OAuth hardening project was created in 2025",
            support=(created_2025,),
            contradiction=(created_2024,),
        ),
        _case(
            "quant.project.created_2026_contradicted",
            document,
            "OAuth hardening project was created in 2026",
            support=(),
            contradiction=(created_2025,),
            expect_abstain=True,
        ),
        _case(
            "quant.project.due_before_supported",
            document,
            "billing retry ticket is due before 2025-06-01",
            support=(due_may,),
            contradiction=(due_june,),
        ),
        _case(
            "quant.project.due_after_supported",
            document,
            "billing retry ticket is due after 2025-06-01",
            support=(due_june,),
            contradiction=(due_may,),
        ),
        _case(
            "quant.project.created_year_missing",
            document,
            "cache purge project was created in 2025",
            support=(),
            insufficient_context=(sparse,),
            expect_abstain=True,
        ),
    )


def _ticket_metric_cases(document: EvidenceDocument) -> tuple[BenchmarkCase, ...]:
    retries_three = _span(document, "retried 3 times")
    retries_two = _span(document, "retried 2 times")
    retries_five = _span(document, "retried 5 times")
    latency_80 = _span(document, "latency is 80 ms")
    latency_150 = _span(document, "latency is 150 ms")
    return (
        _case(
            "quant.ticket.at_least_three_retries",
            document,
            "webhook retried at least 3 times",
            support=(retries_three,),
            contradiction=(retries_two,),
        ),
        _case(
            "quant.ticket.at_most_three_retries",
            document,
            "webhook retried at most 3 times",
            support=(retries_three, retries_two),
            contradiction=(retries_five,),
        ),
        _case(
            "quant.ticket.latency_under_100",
            document,
            "API latency is under 100 ms",
            support=(latency_80,),
            contradiction=(latency_150,),
        ),
        _case(
            "quant.ticket.latency_over_100",
            document,
            "API latency is over 100 ms",
            support=(latency_150,),
            contradiction=(latency_80,),
        ),
    )


def _people_cases(document: EvidenceDocument) -> tuple[BenchmarkCase, ...]:
    jane_supported = _span(document, "worked at Stripe until 2023")
    jane_sparse = _span(document, "worked at Stripe and founded VectorForge")
    founded_2025 = _span(document, "founded VectorForge in 2025")
    announced_2024 = _span(document, "announced on 2024-03-15")
    announced_2023 = _span(document, "announced on 2023-11-01")
    return (
        _case(
            "quant.people.employment_before_founding",
            document,
            "Jane worked at Stripe before founding VectorForge in 2025",
            support=(jane_supported,),
            insufficient_context=(jane_sparse,),
        ),
        _case(
            "quant.people.founding_wrong_year",
            document,
            "Jane founded VectorForge in 2024",
            support=(),
            contradiction=(founded_2025,),
            expect_abstain=True,
        ),
        _case(
            "quant.people.investment_after_2024",
            document,
            "TensorGrid investment was announced after 2024-01-01",
            support=(announced_2024,),
            contradiction=(announced_2023,),
        ),
    )


def _apartment_cases(document: EvidenceDocument) -> tuple[BenchmarkCase, ...]:
    two_bed = _span(document, "2-bedroom apartment")
    one_bed = _span(document, "1-bedroom apartment")
    three_bed = _span(document, "3-bedroom unit")
    rent_3200 = _span(document, "$3,200 per month")
    rent_3800 = _span(document, "$3,800 per month")
    june_15 = _span(document, "available June 15, 2026")
    may_15 = _span(document, "available May 15, 2026")
    lease_12 = _span(document, "minimum lease term is 12 months")
    lease_6 = _span(document, "minimum lease term is 6 months")
    return (
        _case(
            "quant.apartment.exact_bedrooms",
            document,
            "River 12 apartment has 2 bedrooms",
            support=(two_bed,),
            contradiction=(one_bed,),
        ),
        _case(
            "quant.apartment.at_least_two_bedrooms",
            document,
            "Maple Loft apartment has at least 2 bedrooms",
            support=(three_bed,),
            contradiction=(one_bed,),
        ),
        _case(
            "quant.apartment.rent_under_budget",
            document,
            "River 12 rent is under $3,500",
            support=(rent_3200,),
            contradiction=(rent_3800,),
        ),
        _case(
            "quant.apartment.available_after_june_first",
            document,
            "River 12 is available after June 1, 2026",
            support=(june_15,),
            contradiction=(may_15,),
        ),
        _case(
            "quant.apartment.lease_at_least_12_months",
            document,
            "lease requires at least 12 months",
            support=(lease_12,),
            contradiction=(lease_6,),
        ),
    )


def _technical_cases(document: EvidenceDocument) -> tuple[BenchmarkCase, ...]:
    speed_55 = _span(document, "55x faster")
    speed_5 = _span(document, "was 5x faster")
    cache_94 = _span(document, "cache hit rate reached 94%")
    cache_84 = _span(document, "cache hit rate reached 84%")
    return (
        _case(
            "quant.technical.multiplier_55x",
            document,
            "benchmark was 55x faster",
            support=(speed_55,),
            contradiction=(speed_5,),
        ),
        _case(
            "quant.technical.cache_rate_over_90",
            document,
            "cache hit rate exceeds 90%",
            support=(cache_94,),
            contradiction=(cache_84,),
        ),
        _case(
            "quant.technical.cache_rate_under_90_contradicted",
            document,
            "cache hit rate exceeds 95%",
            support=(),
            contradiction=(cache_94,),
            expect_abstain=True,
        ),
    )


def _case(
    case_id: str,
    document: EvidenceDocument,
    claim: str,
    *,
    support: tuple[DocumentSpan, ...],
    contradiction: tuple[DocumentSpan, ...] = (),
    insufficient_context: tuple[DocumentSpan, ...] = (),
    expect_abstain: bool = False,
) -> BenchmarkCase:
    return BenchmarkCase(
        id=case_id,
        intent=IntentSpec(
            id=case_id,
            label=claim,
            description=f"Find direct evidence for this claim: {claim}",
            positive_examples=(claim,),
            required_facets=_facets(claim),
            quantifier_requirements=requirements_from_text(claim),
            min_support_score=0.45,
        ),
        document=document,
        support_span_ids=tuple(span.id for span in support),
        contradiction_span_ids=tuple(span.id for span in contradiction),
        insufficient_context_span_ids=tuple(span.id for span in insufficient_context),
        forbidden_span_ids=tuple(
            dict.fromkeys(span.id for span in (*contradiction, *insufficient_context))
        ),
        expect_abstain=expect_abstain,
        curation=BenchmarkCuration(
            reviewed=True,
            source="manual_curated_quantifier_excerpt",
            difficulty="hard" if contradiction or insufficient_context else "medium",
            phenomena=(
                "quantifier",
                "date_or_numeric",
                "contradiction" if contradiction else "direct_support",
            ),
            notes=("Curated deterministic quantifier benchmark case.",),
        ),
    )


def _binding_case(
    document: EvidenceDocument,
    case_id: str,
    claim: str,
    support_text: str,
    negative_text: str,
    phenomenon: str,
) -> BenchmarkCase:
    support = _span(document, support_text)
    negative = _span(document, negative_text)
    expect_abstain = phenomenon == "insufficient_context"
    return BenchmarkCase(
        id=case_id,
        intent=IntentSpec(
            id=case_id,
            label=claim,
            description=f"Find direct evidence for this claim: {claim}",
            positive_examples=(claim,),
            required_facets=_facets(claim),
            quantifier_requirements=requirements_from_text(claim),
            min_support_score=0.45,
        ),
        document=document,
        support_span_ids=() if expect_abstain else (support.id,),
        contradiction_span_ids=() if expect_abstain else (negative.id,),
        insufficient_context_span_ids=(negative.id,) if expect_abstain else (),
        forbidden_span_ids=(negative.id,),
        expect_abstain=expect_abstain,
        curation=BenchmarkCuration(
            reviewed=True,
            source="manual_curated_quantifier_binding_excerpt",
            difficulty="hard",
            phenomena=("quantifier_binding", phenomenon),
            notes=("Curated deterministic quantifier binding benchmark case.",),
        ),
    )


def _binding_spec(
    case_id: str,
    claim: str,
    support_text: str,
    negative_text: str,
    phenomenon: str,
) -> tuple[str, str, str, str, str]:
    return case_id, claim, support_text, negative_text, phenomenon


def _span(document: EvidenceDocument, contains: str) -> DocumentSpan:
    contains_normalized = contains.casefold()
    matches = [
        span for span in document.spans if contains_normalized in span.text.casefold()
    ]
    if not matches:
        raise ValueError(f"could not find span containing {contains!r} in {document.id}")
    return min(matches, key=lambda span: len(span.text))


def _facets(text: str) -> tuple[EvidenceFacet, ...]:
    stop_words = {
        "after",
        "before",
        "created",
        "direct",
        "evidence",
        "exceeds",
        "faster",
        "find",
        "least",
        "over",
        "requires",
        "support",
        "under",
    }
    anchors = tuple(
        dict.fromkeys(
            word.strip(".,:;()").lower()
            for word in text.split()
            if len(word.strip(".,:;()")) > 3
            and word.strip(".,:;()").lower() not in stop_words
            and not any(char.isdigit() for char in word)
        )
    )[:4]
    return tuple(
        EvidenceFacet(f"anchor_{index}", (anchor,))
        for index, anchor in enumerate(anchors or (text,), start=1)
    )


QUANTIFIER_EXCERPTS: tuple[QuantifierExcerpt, ...] = (
    QuantifierExcerpt(
        id="project_dates",
        domain="project_management",
        text=(
            "# Project dates\n\n"
            "- Support: OAuth hardening project was created in 2025.\n"
            "- Contradiction: OAuth hardening project was created in 2024.\n"
            "- Support: billing retry ticket is due on 2025-05-20.\n"
            "- Contradiction: billing retry ticket is due on 2025-06-15.\n"
            "- Insufficient context: cache purge project was created after planning.\n"
        ),
    ),
    QuantifierExcerpt(
        id="ticket_metrics",
        domain="project_management",
        text=(
            "# Ticket metrics\n\n"
            "- Support: webhook retried 3 times before succeeding.\n"
            "- Near miss: webhook retried 2 times before succeeding.\n"
            "- Contradiction: webhook retried 5 times before succeeding.\n"
            "- Support: API latency is 80 ms at p95.\n"
            "- Contradiction: API latency is 150 ms at p95.\n"
        ),
    ),
    QuantifierExcerpt(
        id="people_dates",
        domain="people_search",
        text=(
            "# People dates\n\n"
            "- Support: Jane worked at Stripe until 2023 and founded VectorForge in 2025.\n"
            "- Insufficient context: Jane worked at Stripe and founded VectorForge.\n"
            "- Support: Jane founded VectorForge in 2025.\n"
            "- Support: TensorGrid investment was announced on 2024-03-15.\n"
            "- Contradiction: TensorGrid investment was announced on 2023-11-01.\n"
        ),
    ),
    QuantifierExcerpt(
        id="apartment_listing",
        domain="apartment_search",
        text=(
            "# Apartment listing\n\n"
            "- Support: River 12 is a 2-bedroom apartment.\n"
            "- Contradiction: River 12 is a 1-bedroom apartment.\n"
            "- Support: Maple Loft is a 3-bedroom unit.\n"
            "- Support: River 12 rent is $3,200 per month.\n"
            "- Contradiction: River 12 rent is $3,800 per month.\n"
            "- Support: River 12 is available June 15, 2026.\n"
            "- Contradiction: River 12 is available May 15, 2026.\n"
            "- Support: minimum lease term is 12 months.\n"
            "- Contradiction: minimum lease term is 6 months.\n"
        ),
    ),
    QuantifierExcerpt(
        id="technical_metrics",
        domain="technical_product",
        text=(
            "# Technical metrics\n\n"
            "- Support: benchmark was 55x faster after pair-score caching.\n"
            "- Contradiction: benchmark was 5x faster after pair-score caching.\n"
            "- Support: cache hit rate reached 94% in the replay run.\n"
            "- Contradiction: cache hit rate reached 84% in the replay run.\n"
        ),
    ),
)


_BINDING_DOCUMENT = """# Quantifier binding cases

- Support: Project started in 2024 and ended in 2025.
- Contradiction: Project started in 2025 and ended in 2024.
- Support: Project created on 2024-01-10 and completed on 2025-03-01.
- Contradiction: Project created on 2025-03-01 and completed on 2024-01-10.
- Support: VectorForge was founded in 2023 and investment announced in 2024.
- Contradiction: VectorForge was founded in 2024 and investment announced in 2023.
- Support: Jane worked at Stripe until 2023 and founded VectorForge in 2025.
- Contradiction: Jane worked at Stripe until 2025 and founded VectorForge in 2023.
- Support: Ticket created in 2024 and due in 2025.
- Contradiction: Ticket created in 2025 and due in 2024.
- Support: Ticket updated in March 2025 and completed in April 2025.
- Contradiction: Ticket updated in April 2025 and completed in March 2025.
- Support: Webhook retried 3 times and latency was 80 ms.
- Contradiction: Webhook retried 100 times and latency was 3 ms.
- Support: API p95 latency is 80 ms.
- Near miss: API p50 latency is 80 ms.
- Support: Cache hit rate reached 94%.
- Contradiction: Cache hit rate reached 84%.
- Support: Benchmark was 55x faster after caching.
- Near miss: Unrelated import job was 55x faster.
- Support: River 12 rent is $3,200.
- Near miss: River 12 deposit is $3,200 but rent is $3,800.
- Support: River 12 monthly rent is $3,200.
- Near miss: River 12 broker fee is $3,200.
- Support: River 12 has 2 bedrooms and 1 bathroom.
- Contradiction: River 12 has 1 bedroom and 2 bathrooms.
- Support: River 12 has 2 bedrooms.
- Near miss: Unit 2 has 1 bedroom.
- Support: River 12 is available June 15, 2026.
- Near miss: Lease starts after June 1, 2026 but unit is available May 15, 2026.
- Support: Minimum lease term is 12 months.
- Near miss: Renewal option is 12 months; minimum lease term is 6 months.
- Support: Package supports Python 3.12.
- Contradiction: Package supports Python 3.10 through Python 3.11.
- Support: Python 3.11 is unsupported.
- Contradiction: Python 3.11 is supported.
- Support: Rent is $3,200 and deposit is $1,000.
- Contradiction: Rent is $1,000 and deposit is $3,200.
- Support: Application fee is $50 and broker fee is $0.
- Contradiction: Application fee is $0 and broker fee is $50.
- Support: Package includes 3 seats and 10 projects.
- Contradiction: Package includes 10 seats and 3 projects.
- Support: Team has 8 engineers and 2 designers.
- Contradiction: Team has 2 engineers and 8 designers.
- Support: Incident started on May 1, 2025 and ended on May 3, 2025.
- Contradiction: Incident started on May 3, 2025 and ended on May 1, 2025.
- Insufficient context: Project started in 2024 only.
"""
