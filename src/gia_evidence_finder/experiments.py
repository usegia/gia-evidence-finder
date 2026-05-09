from __future__ import annotations

import time
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from gia_evidence_finder.calibration import (
    CalibrationReport,
    CandidateScoreThresholdExtractor,
    calibrate_thresholds,
)
from gia_evidence_finder.contracts import BenchmarkSuite
from gia_evidence_finder.evaluation import (
    CaseEvaluation,
    EvaluationReport,
    ExtractorProtocol,
    evaluate_suite_detailed,
)
from gia_evidence_finder.extractor import EvidenceExtractor
from gia_evidence_finder.model_baselines import (
    BM25Extractor,
    CohereRerankerClient,
    CrossEncoderRerankExtractor,
    EmbeddingRetrieverExtractor,
    ExternalRerankerExtractor,
    SentenceTransformerCrossEncoderModel,
    SentenceTransformerEmbeddingModel,
    TransformersSequenceClassificationRerankerModel,
    TypedDecisionRerankExtractor,
)
from gia_evidence_finder.ranking import KeywordOverlapBaseline
from gia_evidence_finder.readme_benchmarks import popular_readme_benchmark_suite
from gia_evidence_finder.reranker_cache import (
    CachedPairRerankerModel,
    InMemoryPairScoreCache,
    JsonlPairScoreCache,
    PairRerankerProtocol,
)


@dataclass(frozen=True)
class RankerExperiment:
    name: str
    report: EvaluationReport
    elapsed_ms: int = 0
    cases: tuple[CaseEvaluation, ...] = ()
    calibration: CalibrationReport | None = None


class HoldoutDecisionPolicy(StrEnum):
    SCORE_THRESHOLD = "score_threshold"
    EXTRACTOR_LABELS = "extractor_labels"


@dataclass(frozen=True)
class ExtractorExperimentSpec:
    name: str
    extractor: ExtractorProtocol
    decision_policy: HoldoutDecisionPolicy = HoldoutDecisionPolicy.SCORE_THRESHOLD


@dataclass(frozen=True)
class CalibratedHoldoutExperiment:
    name: str
    decision_policy: HoldoutDecisionPolicy
    selected_threshold: float | None
    dev: RankerExperiment
    test: RankerExperiment


def run_popular_readme_baselines(
    suite: BenchmarkSuite | None = None,
    *,
    calibrate: bool = False,
) -> tuple[RankerExperiment, ...]:
    active_suite = suite or popular_readme_benchmark_suite()
    return run_readme_baselines(active_suite, calibrate=calibrate)


def run_readme_baselines(
    suite: BenchmarkSuite,
    *,
    calibrate: bool = False,
) -> tuple[RankerExperiment, ...]:
    return run_experiment_specs(readme_baseline_specs(), suite, calibrate=calibrate)


def readme_baseline_specs() -> tuple[ExtractorExperimentSpec, ...]:
    return (
        ExtractorExperimentSpec(
            "intent_aware_default",
            EvidenceExtractor.default(),
            decision_policy=HoldoutDecisionPolicy.EXTRACTOR_LABELS,
        ),
        ExtractorExperimentSpec(
            "keyword_overlap_baseline",
            EvidenceExtractor(ranker=KeywordOverlapBaseline()),
        ),
        ExtractorExperimentSpec("bm25_baseline", BM25Extractor()),
    )


def run_popular_readme_sentence_transformer_baselines(
    suite: BenchmarkSuite | None = None,
    *,
    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    cross_encoder_model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    calibrate: bool = False,
    score_cache_path: Path | None = None,
    score_batch_size: int | None = None,
) -> tuple[RankerExperiment, ...]:
    active_suite = suite or popular_readme_benchmark_suite()
    return run_readme_sentence_transformer_baselines(
        active_suite,
        embedding_model_name=embedding_model_name,
        cross_encoder_model_name=cross_encoder_model_name,
        calibrate=calibrate,
        score_cache_path=score_cache_path,
        score_batch_size=score_batch_size,
    )


def run_readme_sentence_transformer_baselines(
    suite: BenchmarkSuite,
    *,
    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    cross_encoder_model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    calibrate: bool = False,
    score_cache_path: Path | None = None,
    score_batch_size: int | None = None,
) -> tuple[RankerExperiment, ...]:
    return run_experiment_specs(
        sentence_transformer_baseline_specs(
            embedding_model_name=embedding_model_name,
            cross_encoder_model_name=cross_encoder_model_name,
            score_cache_path=score_cache_path,
            score_batch_size=score_batch_size,
        ),
        suite,
        calibrate=calibrate,
    )


def sentence_transformer_baseline_specs(
    *,
    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    cross_encoder_model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    score_cache_path: Path | None = None,
    score_batch_size: int | None = None,
) -> tuple[ExtractorExperimentSpec, ...]:
    embedding_model = SentenceTransformerEmbeddingModel(embedding_model_name)
    cross_encoder_model = _cached_pair_reranker(
        SentenceTransformerCrossEncoderModel(cross_encoder_model_name),
        score_cache_path=score_cache_path,
        score_batch_size=score_batch_size,
    )
    embedding_extractor = EmbeddingRetrieverExtractor(embedding_model=embedding_model)
    cross_encoder_extractor = CrossEncoderRerankExtractor(reranker_model=cross_encoder_model)
    return (
        ExtractorExperimentSpec(
            f"sentence_transformer_embedding:{embedding_model_name}",
            embedding_extractor,
        ),
        ExtractorExperimentSpec(
            f"sentence_transformer_cross_encoder:{cross_encoder_model_name}",
            cross_encoder_extractor,
        ),
    )


def run_readme_cross_encoder_baseline(
    suite: BenchmarkSuite,
    *,
    model_name: str,
    support_threshold: float | None = None,
    trust_remote_code: bool = False,
    calibrate: bool = False,
    score_cache_path: Path | None = None,
    score_batch_size: int | None = None,
) -> RankerExperiment:
    return run_experiment_spec(
        cross_encoder_baseline_spec(
            model_name=model_name,
            support_threshold=support_threshold,
            trust_remote_code=trust_remote_code,
            score_cache_path=score_cache_path,
            score_batch_size=score_batch_size,
        ),
        suite,
        calibrate=calibrate,
    )


def cross_encoder_baseline_spec(
    *,
    model_name: str,
    support_threshold: float | None = None,
    trust_remote_code: bool = False,
    score_cache_path: Path | None = None,
    score_batch_size: int | None = None,
) -> ExtractorExperimentSpec:
    cross_encoder_model = _cached_pair_reranker(
        SentenceTransformerCrossEncoderModel(
            model_name,
            trust_remote_code=trust_remote_code,
        ),
        score_cache_path=score_cache_path,
        score_batch_size=score_batch_size,
    )
    extractor = CrossEncoderRerankExtractor(
        reranker_model=cross_encoder_model,
        support_threshold=support_threshold,
    )
    return ExtractorExperimentSpec(
        f"sentence_transformer_cross_encoder:{model_name}",
        extractor,
    )


def typed_decision_cross_encoder_spec(
    *,
    model_name: str,
    first_stage_support_threshold: float | None = None,
    trust_remote_code: bool = False,
    score_cache_path: Path | None = None,
    score_batch_size: int | None = None,
) -> ExtractorExperimentSpec:
    cross_encoder_model = _cached_pair_reranker(
        SentenceTransformerCrossEncoderModel(
            model_name,
            trust_remote_code=trust_remote_code,
        ),
        score_cache_path=score_cache_path,
        score_batch_size=score_batch_size,
    )
    extractor = TypedDecisionRerankExtractor(
        reranker_model=cross_encoder_model,
        first_stage_support_threshold=first_stage_support_threshold,
    )
    return ExtractorExperimentSpec(
        f"typed_decision_cross_encoder:{model_name}",
        extractor,
        decision_policy=HoldoutDecisionPolicy.EXTRACTOR_LABELS,
    )


def run_readme_transformers_reranker_baseline(
    suite: BenchmarkSuite,
    *,
    model_name: str,
    max_length: int = 512,
    device: str = "cpu",
    support_threshold: float | None = None,
    trust_remote_code: bool = False,
    calibrate: bool = False,
    score_cache_path: Path | None = None,
    score_batch_size: int | None = None,
) -> RankerExperiment:
    return run_experiment_spec(
        transformers_reranker_baseline_spec(
            model_name=model_name,
            max_length=max_length,
            device=device,
            support_threshold=support_threshold,
            trust_remote_code=trust_remote_code,
            score_cache_path=score_cache_path,
            score_batch_size=score_batch_size,
        ),
        suite,
        calibrate=calibrate,
    )


def transformers_reranker_baseline_spec(
    *,
    model_name: str,
    max_length: int = 512,
    device: str = "cpu",
    support_threshold: float | None = None,
    trust_remote_code: bool = False,
    score_cache_path: Path | None = None,
    score_batch_size: int | None = None,
) -> ExtractorExperimentSpec:
    reranker_model = _cached_pair_reranker(
        TransformersSequenceClassificationRerankerModel(
            model_name,
            max_length=max_length,
            device=device,
            trust_remote_code=trust_remote_code,
        ),
        score_cache_path=score_cache_path,
        score_batch_size=score_batch_size,
    )
    extractor = CrossEncoderRerankExtractor(
        reranker_model=reranker_model,
        support_threshold=support_threshold,
    )
    return ExtractorExperimentSpec(f"transformers_reranker:{model_name}", extractor)


def run_readme_cohere_reranker_baseline(
    suite: BenchmarkSuite,
    *,
    model_name: str = "rerank-v4.0-pro",
    support_threshold: float | None = None,
    calibrate: bool = False,
) -> RankerExperiment:
    return run_experiment_spec(
        cohere_reranker_baseline_spec(
            model_name=model_name,
            support_threshold=support_threshold,
        ),
        suite,
        calibrate=calibrate,
    )


def cohere_reranker_baseline_spec(
    *,
    model_name: str = "rerank-v4.0-pro",
    support_threshold: float | None = None,
) -> ExtractorExperimentSpec:
    client = CohereRerankerClient(model_name=model_name)
    extractor = ExternalRerankerExtractor(
        client=client,
        support_threshold=support_threshold,
    )
    return ExtractorExperimentSpec(
        f"external_reranker:cohere:{model_name}",
        extractor,
    )


def run_experiment_specs(
    specs: tuple[ExtractorExperimentSpec, ...],
    suite: BenchmarkSuite,
    *,
    calibrate: bool = False,
) -> tuple[RankerExperiment, ...]:
    return tuple(
        run_experiment_spec(spec, suite, calibrate=calibrate)
        for spec in specs
    )


def run_experiment_spec(
    spec: ExtractorExperimentSpec,
    suite: BenchmarkSuite,
    *,
    calibrate: bool = False,
) -> RankerExperiment:
    return _run_experiment(
        name=spec.name,
        extractor=spec.extractor,
        suite=suite,
        calibrate=calibrate,
    )


def run_calibrated_holdout(
    specs: tuple[ExtractorExperimentSpec, ...],
    *,
    dev_suite: BenchmarkSuite,
    test_suite: BenchmarkSuite,
) -> tuple[CalibratedHoldoutExperiment, ...]:
    experiments: list[CalibratedHoldoutExperiment] = []
    for spec in specs:
        dev_experiment = run_experiment_spec(spec, dev_suite, calibrate=True)
        test_extractor: ExtractorProtocol
        if spec.decision_policy == HoldoutDecisionPolicy.SCORE_THRESHOLD:
            if dev_experiment.calibration is None:
                raise RuntimeError("dev calibration was not computed")
            threshold = dev_experiment.calibration.best.threshold
            test_extractor = CandidateScoreThresholdExtractor(
                extractor=spec.extractor,
                support_threshold=threshold,
            )
        elif spec.decision_policy == HoldoutDecisionPolicy.EXTRACTOR_LABELS:
            threshold = None
            test_extractor = spec.extractor
        else:
            raise ValueError(f"unsupported decision policy {spec.decision_policy!r}")
        test_experiment = _run_experiment(
            name=spec.name,
            extractor=test_extractor,
            suite=test_suite,
            calibrate=False,
        )
        experiments.append(
            CalibratedHoldoutExperiment(
                name=spec.name,
                decision_policy=spec.decision_policy,
                selected_threshold=threshold,
                dev=dev_experiment,
                test=test_experiment,
            )
        )
    return tuple(experiments)


def calibrate_default_first_stage_support_threshold(
    suite: BenchmarkSuite,
    *,
    extractor: ExtractorProtocol | None = None,
) -> float:
    dev_experiment = run_experiment_spec(
        ExtractorExperimentSpec(
            "intent_aware_default",
            extractor or EvidenceExtractor.default(),
        ),
        suite,
        calibrate=True,
    )
    if dev_experiment.calibration is None:
        raise RuntimeError("default first-stage calibration was not computed")
    return dev_experiment.calibration.best.threshold


def _run_experiment(
    *,
    name: str,
    extractor: ExtractorProtocol,
    suite: BenchmarkSuite,
    calibrate: bool = False,
) -> RankerExperiment:
    started = time.monotonic()
    detailed_report = evaluate_suite_detailed(extractor, suite.cases)
    elapsed_ms = int((time.monotonic() - started) * 1000)
    return RankerExperiment(
        name=name,
        report=detailed_report.summary,
        elapsed_ms=elapsed_ms,
        cases=detailed_report.cases,
        calibration=calibrate_thresholds(detailed_report) if calibrate else None,
    )


def _cached_pair_reranker(
    model: PairRerankerProtocol,
    *,
    score_cache_path: Path | None,
    score_batch_size: int | None,
) -> PairRerankerProtocol:
    if score_cache_path is None and score_batch_size is None:
        return model
    cache = (
        JsonlPairScoreCache(score_cache_path)
        if score_cache_path is not None
        else InMemoryPairScoreCache()
    )
    return CachedPairRerankerModel(
        model=model,
        cache=cache,
        batch_size=score_batch_size,
    )
