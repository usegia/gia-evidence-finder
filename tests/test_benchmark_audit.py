from __future__ import annotations

from dataclasses import replace

from gia_evidence_finder import (
    BenchmarkCuration,
    BenchmarkSuite,
    audit_benchmark_series,
    audit_benchmark_suite,
)
from gia_evidence_finder.benchmark_series import evidence_benchmark_v2
from gia_evidence_finder.readme_benchmarks import adversarial_readme_benchmark_suite


def test_benchmark_suite_audit_counts_special_labels() -> None:
    audit = audit_benchmark_suite(adversarial_readme_benchmark_suite())

    assert audit.case_count == 16
    assert audit.reviewed_case_count == 16
    assert audit.unreviewed_case_count == 0
    assert audit.curation_source_counts["curated_readme_excerpt"] == 16
    assert audit.difficulty_counts["hard"] >= 4
    assert audit.phenomenon_counts["contradiction"] >= 4
    assert audit.contradiction_label_count >= 4
    assert audit.forbidden_label_count >= audit.contradiction_label_count


def test_benchmark_suite_audit_keeps_unreviewed_cases_out_of_proof_count() -> None:
    source_suite = adversarial_readme_benchmark_suite()
    unreviewed_case = replace(
        source_suite.cases[0],
        curation=BenchmarkCuration(
            reviewed=False,
            source="generated_probe",
            difficulty="hard",
            phenomena=("contradiction",),
        ),
    )
    suite = BenchmarkSuite(
        id="generated_probe",
        name="Generated probe",
        cases=(unreviewed_case,),
    )

    audit = audit_benchmark_suite(suite)

    assert audit.case_count == 1
    assert audit.reviewed_case_count == 0
    assert audit.unreviewed_case_count == 1
    assert "suite has no reviewed cases" in audit.warnings


def test_benchmark_series_audit_reports_sota_readiness_gaps() -> None:
    audit = audit_benchmark_series(evidence_benchmark_v2())

    assert audit.series_id == "evidence_benchmark_v2"
    assert audit.case_count == 70
    assert audit.reviewed_case_count == 70
    assert audit.curation_source_counts["curated_readme_excerpt"] == 70
    assert audit.phenomenon_counts["relation_binding"] == 16
    assert audit.label_counts["contradiction"] >= 4
    assert audit.label_counts["insufficient_context"] >= 2
    assert "series has fewer than 200 reviewed cases" in audit.warnings
