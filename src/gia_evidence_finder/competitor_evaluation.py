from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any, cast

from gia_evidence_finder.benchmark_series import (
    BenchmarkSeries,
    BenchmarkSplit,
    benchmark_series_suites,
)
from gia_evidence_finder.experiments import (
    ExtractorExperimentSpec,
    RankerExperiment,
    cohere_reranker_baseline_spec,
    cross_encoder_baseline_spec,
    readme_baseline_specs,
    run_experiment_specs,
    sentence_transformer_baseline_specs,
    transformers_reranker_baseline_spec,
    typed_decision_cross_encoder_spec,
)


class CompetitorKind(StrEnum):
    LOCAL = "local"
    HOSTED = "hosted"


@dataclass(frozen=True)
class CompetitorCost:
    unit: str
    amount_usd: float | None = None
    notes: str = ""


@dataclass(frozen=True)
class CompetitorSpec:
    id: str
    name: str
    provider: str
    kind: CompetitorKind
    model_name: str
    description: str
    benchmark_modes: tuple[str, ...]
    supports_score_cache: bool = False
    supports_batching: bool = False
    cost: CompetitorCost | None = None
    build: Callable[[CompetitorBuildOptions], ExtractorExperimentSpec] | None = None

    def __post_init__(self) -> None:
        if not self.id.strip():
            raise ValueError("competitor id must not be empty")
        if not self.name.strip():
            raise ValueError("competitor name must not be empty")
        if not self.provider.strip():
            raise ValueError("competitor provider must not be empty")
        if not self.model_name.strip():
            raise ValueError("competitor model_name must not be empty")
        if not self.benchmark_modes:
            raise ValueError("competitor benchmark_modes must not be empty")


@dataclass(frozen=True)
class CompetitorBuildOptions:
    trust_remote_code: bool = False
    score_cache_path: Path | None = None
    score_batch_size: int | None = None
    first_stage_support_threshold: float | None = None


def competitor_registry() -> tuple[CompetitorSpec, ...]:
    specs = (
        CompetitorSpec(
            id="typed_default",
            name="gia typed deterministic default",
            provider="gia",
            kind=CompetitorKind.LOCAL,
            model_name="intent-aware-default",
            description="Deterministic typed support/abstain extractor.",
            benchmark_modes=("retrieval", "decision", "safety"),
            build=lambda options: readme_baseline_specs()[0],
        ),
        CompetitorSpec(
            id="keyword",
            name="Keyword overlap",
            provider="gia",
            kind=CompetitorKind.LOCAL,
            model_name="keyword-overlap",
            description="Lexical keyword overlap baseline.",
            benchmark_modes=("retrieval", "decision", "safety"),
            build=lambda options: readme_baseline_specs()[1],
        ),
        CompetitorSpec(
            id="bm25",
            name="BM25",
            provider="gia",
            kind=CompetitorKind.LOCAL,
            model_name="bm25",
            description="Local BM25 lexical retrieval baseline.",
            benchmark_modes=("retrieval", "decision", "safety"),
            build=lambda options: readme_baseline_specs()[2],
        ),
        CompetitorSpec(
            id="minilm_embedding",
            name="MiniLM embedding retrieval",
            provider="sentence-transformers",
            kind=CompetitorKind.LOCAL,
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            description="Embedding-only retrieval baseline.",
            benchmark_modes=("retrieval", "decision", "safety"),
            build=lambda options: sentence_transformer_baseline_specs(
                score_cache_path=options.score_cache_path,
                score_batch_size=options.score_batch_size,
            )[0],
        ),
        CompetitorSpec(
            id="minilm_cross_encoder",
            name="MiniLM cross-encoder",
            provider="sentence-transformers",
            kind=CompetitorKind.LOCAL,
            model_name="cross-encoder/ms-marco-MiniLM-L-6-v2",
            description="Generic MS MARCO MiniLM cross-encoder reranker.",
            benchmark_modes=("retrieval", "decision", "safety"),
            supports_score_cache=True,
            supports_batching=True,
            build=lambda options: sentence_transformer_baseline_specs(
                score_cache_path=options.score_cache_path,
                score_batch_size=options.score_batch_size,
            )[1],
        ),
        CompetitorSpec(
            id="bge_reranker_v2_m3",
            name="BGE reranker v2 m3",
            provider="BAAI",
            kind=CompetitorKind.LOCAL,
            model_name="BAAI/bge-reranker-v2-m3",
            description="Open-source sequence-classification reranker.",
            benchmark_modes=("retrieval", "decision", "safety"),
            supports_score_cache=True,
            supports_batching=True,
            build=lambda options: transformers_reranker_baseline_spec(
                model_name="BAAI/bge-reranker-v2-m3",
                trust_remote_code=options.trust_remote_code,
                score_cache_path=options.score_cache_path,
                score_batch_size=options.score_batch_size,
            ),
        ),
        CompetitorSpec(
            id="qwen3_reranker_06b",
            name="Qwen3 reranker 0.6B",
            provider="Qwen",
            kind=CompetitorKind.LOCAL,
            model_name="Qwen/Qwen3-Reranker-0.6B",
            description="Open Qwen reranker probe through sentence-transformers.",
            benchmark_modes=("retrieval", "decision", "safety"),
            supports_score_cache=True,
            supports_batching=True,
            build=lambda options: cross_encoder_baseline_spec(
                model_name="Qwen/Qwen3-Reranker-0.6B",
                trust_remote_code=options.trust_remote_code,
                score_cache_path=options.score_cache_path,
                score_batch_size=options.score_batch_size,
            ),
        ),
        CompetitorSpec(
            id="gia_minilm_l12_typed",
            name="gia fine-tuned MiniLM-L12 typed decision",
            provider="gia",
            kind=CompetitorKind.LOCAL,
            model_name="sentence-transformers/msmarco-MiniLM-L12-cos-v5",
            description=(
                "Typed-decision reranker slot for the current product candidate. "
                "Uses the configured model name unless a fine-tuned artifact is supplied."
            ),
            benchmark_modes=("retrieval", "decision", "safety"),
            supports_score_cache=True,
            supports_batching=True,
            build=lambda options: typed_decision_cross_encoder_spec(
                model_name="sentence-transformers/msmarco-MiniLM-L12-cos-v5",
                first_stage_support_threshold=options.first_stage_support_threshold,
                trust_remote_code=options.trust_remote_code,
                score_cache_path=options.score_cache_path,
                score_batch_size=options.score_batch_size,
            ),
        ),
        CompetitorSpec(
            id="cohere_rerank",
            name="Cohere rerank",
            provider="cohere",
            kind=CompetitorKind.HOSTED,
            model_name="rerank-v4.0-pro",
            description="Hosted Cohere reranker baseline. Requires COHERE_API_KEY.",
            benchmark_modes=("retrieval", "decision", "safety"),
            cost=CompetitorCost(unit="provider_pricing", notes="Resolved by provider account."),
            build=lambda options: cohere_reranker_baseline_spec(model_name="rerank-v4.0-pro"),
        ),
    )
    _validate_registry(specs)
    return specs


def required_local_competitor_ids() -> tuple[str, ...]:
    return (
        "typed_default",
        "keyword",
        "bm25",
        "minilm_embedding",
        "minilm_cross_encoder",
        "bge_reranker_v2_m3",
        "qwen3_reranker_06b",
        "gia_minilm_l12_typed",
    )


def competitor_evaluation_payload(
    *,
    series: BenchmarkSeries,
    split: str,
    competitor_ids: tuple[str, ...],
    include_hosted: bool,
    registry: tuple[CompetitorSpec, ...],
    trust_remote_code: bool,
    score_cache_path: Path | None,
    score_batch_size: int | None,
) -> dict[str, Any]:
    selected_specs = _selected_competitors(
        registry=registry,
        competitor_ids=competitor_ids or required_local_competitor_ids(),
        include_hosted=include_hosted,
    )
    unavailable: dict[str, str] = {}
    experiment_specs: list[ExtractorExperimentSpec] = []
    for spec in selected_specs:
        if spec.build is None:
            unavailable[spec.id] = "competitor has no executable local runner"
            continue
        try:
            experiment_specs.append(
                spec.build(
                    CompetitorBuildOptions(
                        trust_remote_code=trust_remote_code,
                        score_cache_path=score_cache_path if spec.supports_score_cache else None,
                        score_batch_size=score_batch_size if spec.supports_batching else None,
                        first_stage_support_threshold=None,
                    )
                )
            )
        except (ImportError, ModuleNotFoundError, RuntimeError, ValueError) as exc:
            unavailable[spec.id] = str(exc)
    suites = benchmark_series_suites(
        series,
        split=None if split == "all" else BenchmarkSplit(split),
    )
    split_payloads: dict[str, object] = {}
    for suite in suites:
        experiments = run_experiment_specs(tuple(experiment_specs), suite)
        split_payloads[suite.metadata["split"]] = {
            "suite_id": suite.id,
            "case_count": len(suite.cases),
            "experiments": {
                experiment.name: _experiment_axis_payload(experiment) for experiment in experiments
            },
        }
    return {
        "series_id": series.id,
        "split": split,
        "competitors": [_competitor_payload(spec) for spec in selected_specs],
        "splits": split_payloads,
        "unavailable": unavailable,
    }


def competitor_markdown_report(payload: Mapping[str, Any]) -> str:
    lines = [
        f"# Competitor Evaluation: {payload['series_id']}",
        "",
        f"Split: `{payload['split']}`",
        "",
    ]
    unavailable = payload.get("unavailable") or {}
    if unavailable:
        lines.extend(("## Unavailable", ""))
        for key, reason in unavailable.items():
            lines.append(f"- `{key}`: {reason}")
        lines.append("")
    for split, split_payload in payload["splits"].items():
        lines.extend((f"## {split.title()} Split", ""))
        experiments = split_payload["experiments"]
        lines.extend(
            _axis_table(
                "Retrieval",
                experiments,
                ("mrr", "recall_at_1", "recall_at_3", "recall_at_5", "top1_support_accuracy"),
            )
        )
        lines.extend(
            _axis_table(
                "Decision",
                experiments,
                ("decision_accuracy", "abstain_accuracy", "support_precision", "support_recall"),
            )
        )
        lines.extend(
            _axis_table(
                "Safety",
                experiments,
                (
                    "forbidden_supported_top1_rate",
                    "near_miss_supported_top1_rate",
                    "contradiction_supported_top1_rate",
                    "insufficient_context_supported_top1_rate",
                ),
            )
        )
    return "\n".join(lines)


def _axis_table(
    title: str,
    experiments: Mapping[str, Mapping[str, object]],
    columns: tuple[str, ...],
) -> list[str]:
    lines = [f"### {title}", ""]
    lines.append("| System | " + " | ".join(columns) + " |")
    lines.append("| --- | " + " | ".join("---:" for _ in columns) + " |")
    for name, payload in experiments.items():
        axes = cast(Mapping[str, object], payload["axes"])
        values = [str(axes.get(column, "")) for column in columns]
        lines.append(f"| `{name}` | " + " | ".join(values) + " |")
    lines.append("")
    return lines


def _selected_competitors(
    *,
    registry: tuple[CompetitorSpec, ...],
    competitor_ids: tuple[str, ...],
    include_hosted: bool,
) -> tuple[CompetitorSpec, ...]:
    by_id = {spec.id: spec for spec in registry}
    selected: list[CompetitorSpec] = []
    for competitor_id in competitor_ids:
        try:
            spec = by_id[competitor_id]
        except KeyError as exc:
            raise ValueError(f"unknown competitor {competitor_id!r}") from exc
        if spec.kind == CompetitorKind.HOSTED and not include_hosted:
            continue
        selected.append(spec)
    if include_hosted:
        selected.extend(
            spec
            for spec in registry
            if spec.kind == CompetitorKind.HOSTED and spec.id not in competitor_ids
        )
    return tuple(dict.fromkeys(selected))


def _experiment_axis_payload(experiment: RankerExperiment) -> dict[str, object]:
    report = experiment.report
    support_precision, support_recall = _support_precision_recall(experiment)
    negative = {
        item.label.value: item.supported_top1_rate for item in report.negative_label_reports
    }
    return {
        "elapsed_ms": experiment.elapsed_ms,
        "axes": {
            "mrr": report.mean_reciprocal_rank,
            "recall_at_1": report.recall_at_1,
            "recall_at_3": report.recall_at_3,
            "recall_at_5": report.recall_at_5,
            "top1_support_accuracy": report.top1_support_accuracy,
            "decision_accuracy": report.decision_accuracy,
            "abstain_accuracy": report.abstain_accuracy,
            "support_precision": support_precision,
            "support_recall": support_recall,
            "forbidden_supported_top1_rate": report.forbidden_supported_top1_rate,
            "near_miss_supported_top1_rate": negative.get("near_miss", 0.0),
            "contradiction_supported_top1_rate": negative.get("contradicts", 0.0),
            "insufficient_context_supported_top1_rate": negative.get("insufficient_context", 0.0),
        },
        "failure_examples": [
            {
                "case_id": case.case_id,
                "document_id": case.document_id,
                "top_span_id": case.top_span_id,
                "top_label": case.top_label,
                "first_support_rank": case.first_support_rank,
            }
            for case in experiment.cases
            if not case.decision_correct
            or case.forbidden_supported_top1
            or any(item.supported_top1 for item in case.negative_label_evaluations)
        ][:10],
    }


def _support_precision_recall(experiment: RankerExperiment) -> tuple[float, float]:
    predicted_support = 0
    correct_support = 0
    expected_support = 0
    for case in experiment.cases:
        expected_support += int(bool(case.support_span_ids))
        if case.top_label == "supports":
            predicted_support += 1
            correct_support += int(case.top1_is_support)
    precision = 1.0 if predicted_support == 0 else round(correct_support / predicted_support, 4)
    recall = 1.0 if expected_support == 0 else round(correct_support / expected_support, 4)
    return precision, recall


def _competitor_payload(spec: CompetitorSpec) -> dict[str, object]:
    payload: dict[str, object] = {
        "id": spec.id,
        "name": spec.name,
        "provider": spec.provider,
        "kind": spec.kind.value,
        "model_name": spec.model_name,
        "description": spec.description,
        "benchmark_modes": list(spec.benchmark_modes),
        "supports_score_cache": spec.supports_score_cache,
        "supports_batching": spec.supports_batching,
    }
    if spec.cost is not None:
        payload["cost"] = {
            "unit": spec.cost.unit,
            "amount_usd": spec.cost.amount_usd,
            "notes": spec.cost.notes,
        }
    return payload


def _validate_registry(specs: tuple[CompetitorSpec, ...]) -> None:
    seen: set[str] = set()
    for spec in specs:
        if spec.id in seen:
            raise ValueError(f"duplicate competitor id {spec.id!r}")
        seen.add(spec.id)
    missing = set(required_local_competitor_ids()) - {spec.id for spec in specs}
    if missing:
        raise ValueError(f"missing required local competitors: {sorted(missing)!r}")
