from __future__ import annotations

from gia_evidence_finder import baseline_profile_by_id, baseline_profiles


def test_baseline_registry_exposes_local_and_open_profiles() -> None:
    profiles = {profile.id: profile for profile in baseline_profiles()}

    assert {
        "local_smoke",
        "open_reranker_probe",
        "api_reranker_probe",
        "open_sota_probe",
    } <= set(profiles)
    assert profiles["local_smoke"].include_sentence_transformer_baselines
    assert "BAAI/bge-reranker-v2-m3" in profiles["open_reranker_probe"].transformers_reranker_models
    assert "rerank-v4.0-pro" in profiles["api_reranker_probe"].cohere_reranker_models
    assert "Qwen/Qwen3-Reranker-8B" in profiles["open_sota_probe"].cross_encoder_models


def test_baseline_profile_lookup_by_id() -> None:
    profile = baseline_profile_by_id("open_reranker_probe")

    assert profile.cross_encoder_models
    assert profile.transformers_reranker_models
