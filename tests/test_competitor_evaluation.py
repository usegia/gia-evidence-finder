from __future__ import annotations

import pytest

from gia_evidence_finder import (
    competitor_evaluation_payload,
    competitor_markdown_report,
    competitor_registry,
    domain_evidence_benchmark_v4,
    required_local_competitor_ids,
)
from gia_evidence_finder.competitor_evaluation import _validate_registry


def test_competitor_registry_contains_required_local_baselines() -> None:
    registry = competitor_registry()
    by_id = {competitor.id: competitor for competitor in registry}

    assert set(required_local_competitor_ids()) <= set(by_id)
    assert by_id["bm25"].model_name == "bm25"
    assert by_id["cohere_rerank"].kind.value == "hosted"
    assert by_id["qwen3_reranker_06b"].supports_score_cache


def test_competitor_registry_rejects_duplicate_ids() -> None:
    competitor = competitor_registry()[0]

    with pytest.raises(ValueError, match="duplicate competitor id"):
        _validate_registry((competitor, competitor))


def test_competitor_evaluation_reports_three_axes_for_local_baselines() -> None:
    payload = competitor_evaluation_payload(
        series=domain_evidence_benchmark_v4(),
        split="test",
        competitor_ids=("typed_default", "keyword", "bm25"),
        include_hosted=False,
        registry=competitor_registry(),
        trust_remote_code=False,
        score_cache_path=None,
        score_batch_size=None,
    )

    experiments = payload["splits"]["test"]["experiments"]
    typed = experiments["intent_aware_default"]["axes"]

    assert payload["unavailable"] == {}
    assert "mrr" in typed
    assert "decision_accuracy" in typed
    assert "contradiction_supported_top1_rate" in typed
    assert "support_precision" in typed


def test_hosted_competitors_are_skipped_unless_requested() -> None:
    payload = competitor_evaluation_payload(
        series=domain_evidence_benchmark_v4(),
        split="test",
        competitor_ids=("cohere_rerank", "bm25"),
        include_hosted=False,
        registry=competitor_registry(),
        trust_remote_code=False,
        score_cache_path=None,
        score_batch_size=None,
    )

    competitor_ids = {competitor["id"] for competitor in payload["competitors"]}
    assert competitor_ids == {"bm25"}


def test_markdown_report_groups_axes() -> None:
    payload = competitor_evaluation_payload(
        series=domain_evidence_benchmark_v4(),
        split="test",
        competitor_ids=("typed_default",),
        include_hosted=False,
        registry=competitor_registry(),
        trust_remote_code=False,
        score_cache_path=None,
        score_batch_size=None,
    )

    markdown = competitor_markdown_report(payload)

    assert "### Retrieval" in markdown
    assert "### Decision" in markdown
    assert "### Safety" in markdown
