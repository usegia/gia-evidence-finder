from __future__ import annotations

from gia_evidence_finder import (
    audit_expanded_domain_series,
    domain_benchmark_suite,
    domain_benchmark_suite_by_id,
    domain_benchmark_suites,
    domain_evidence_benchmark_v4,
)


def test_domain_benchmark_suite_meets_expansion_targets() -> None:
    suite = domain_benchmark_suite()
    domains = set(suite.metadata["domains"].split(","))
    genres = set(suite.metadata["genres"].split(","))

    assert suite.id == "domain_evidence_v4"
    assert int(suite.metadata["document_count"]) == 40
    assert len(suite.cases) == 320
    assert domains == {
        "apartment_search",
        "people_search",
        "project_management",
        "technical_product",
    }
    assert len(genres) >= 12
    assert all(case.curation.reviewed for case in suite.cases)
    assert all(case.document.metadata["source_hash"] for case in suite.cases)
    assert all(case.document.metadata["review_source"] for case in suite.cases)
    assert [suite.id for suite in domain_benchmark_suites()] == ["domain_evidence_v4"]
    assert domain_benchmark_suite_by_id(suite.id).id == suite.id


def test_expanded_domain_series_audit_has_no_warnings() -> None:
    audit = audit_expanded_domain_series(domain_evidence_benchmark_v4())

    assert audit.case_count == 320
    assert audit.reviewed_case_count == 320
    assert len(audit.domain_counts) == 4
    assert len(audit.genre_counts) >= 12
    assert audit.label_counts["near_miss"] >= 40
    assert audit.label_counts["contradiction"] >= 80
    assert audit.label_counts["insufficient_context"] >= 40
    assert audit.warnings == ()
