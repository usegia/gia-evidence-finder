from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BaselineProfile:
    id: str
    name: str
    description: str
    include_sentence_transformer_baselines: bool = False
    cross_encoder_models: tuple[str, ...] = ()
    transformers_reranker_models: tuple[str, ...] = ()
    cohere_reranker_models: tuple[str, ...] = ()


def baseline_profiles() -> tuple[BaselineProfile, ...]:
    return (
        BaselineProfile(
            id="local_smoke",
            name="Local smoke baselines",
            description=(
                "Runs bundled deterministic baselines plus MiniLM embedding and "
                "cross-encoder baselines."
            ),
            include_sentence_transformer_baselines=True,
        ),
        BaselineProfile(
            id="open_reranker_probe",
            name="Open reranker probe",
            description=(
                "Adds compact modern open reranker probes that are suitable for local "
                "CPU/GPU smoke runs when model dependencies are installed."
            ),
            cross_encoder_models=(
                "Qwen/Qwen3-Reranker-0.6B",
                "jinaai/jina-reranker-v2-base-multilingual",
            ),
            transformers_reranker_models=("BAAI/bge-reranker-v2-m3",),
        ),
        BaselineProfile(
            id="api_reranker_probe",
            name="Managed API reranker probe",
            description=(
                "Adds managed API reranker probes. Requires provider credentials such as "
                "COHERE_API_KEY."
            ),
            cohere_reranker_models=("rerank-v4.0-pro",),
        ),
        BaselineProfile(
            id="open_sota_probe",
            name="Open SOTA reranker probe",
            description=(
                "Adds larger open reranker probes. This profile is intentionally expensive "
                "and should usually run on GPU-backed hardware."
            ),
            cross_encoder_models=(
                "Qwen/Qwen3-Reranker-4B",
                "Qwen/Qwen3-Reranker-8B",
            ),
            transformers_reranker_models=("BAAI/bge-reranker-v2-m3",),
        ),
    )


def baseline_profile_by_id(profile_id: str) -> BaselineProfile:
    profiles = {profile.id: profile for profile in baseline_profiles()}
    try:
        return profiles[profile_id]
    except KeyError as exc:
        raise ValueError(f"unknown baseline profile {profile_id!r}") from exc
