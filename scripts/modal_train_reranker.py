from __future__ import annotations

import json
import sys
from pathlib import Path

import modal

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
REMOTE_SRC = "/opt/gia-evidence-finder/src"
RUNS_DIR = "/runs"
DEFAULT_RUN_NAME = "reviewed-msmarco-minilm"


def _ignore_source_path(path: Path) -> bool:
    return "__pycache__" in path.parts or path.suffix in {".pyc", ".pyo"}


image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "accelerate>=1.1.0",
        "datasets>=3.0.0",
        "sentence-transformers>=3.3.0",
        "transformers>=4.48.0",
        "torch>=2.4.0",
        "numpy>=1.26.0",
    )
    .add_local_dir(
        PACKAGE_ROOT / "src",
        remote_path=REMOTE_SRC,
        copy=True,
        ignore=_ignore_source_path,
    )
)
volume = modal.Volume.from_name("gia-evidence-finder-reranker-runs", create_if_missing=True)
app = modal.App("gia-evidence-finder-reranker-training")


@app.function(
    image=image,
    gpu="H200",
    timeout=60 * 60,
    volumes={RUNS_DIR: volume},
)
def train_cross_encoder_remote(
    *,
    train_jsonl: str,
    dev_jsonl: str,
    base_model: str,
    run_name: str,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    warmup_ratio: float,
    max_length: int,
    seed: int,
    reviewed_jsonl: str | None = None,
    source_files: dict[str, str] | None = None,
    series_id: str = "reviewed_external_series",
) -> dict[str, object]:
    sys.path.insert(0, REMOTE_SRC)

    import random

    import numpy as np
    import torch
    from sentence_transformers import CrossEncoder, InputExample
    from sentence_transformers.cross_encoder.evaluation import CEBinaryClassificationEvaluator
    from torch.utils.data import DataLoader

    from gia_evidence_finder.training import reranker_training_examples_from_jsonl

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    generator = torch.Generator()
    generator.manual_seed(seed)
    train_examples = reranker_training_examples_from_jsonl(train_jsonl)
    dev_examples = reranker_training_examples_from_jsonl(dev_jsonl)
    train_input_examples = [
        InputExample(texts=[example.query, example.text], label=example.label_score)
        for example in train_examples
    ]
    dev_input_examples = [
        InputExample(texts=[example.query, example.text], label=example.label_score)
        for example in dev_examples
    ]
    train_dataloader = DataLoader(
        train_input_examples,
        shuffle=True,
        batch_size=batch_size,
        generator=generator,
    )
    evaluator = CEBinaryClassificationEvaluator.from_input_examples(
        dev_input_examples,
        name="dev",
    )
    model = CrossEncoder(base_model, num_labels=1, max_length=max_length)
    total_steps = max(1, len(train_dataloader) * epochs)
    warmup_steps = max(1, int(total_steps * warmup_ratio))
    run_dir = Path(RUNS_DIR) / run_name
    model_dir = run_dir / "model"
    model.fit(
        train_dataloader=train_dataloader,
        evaluator=evaluator,
        epochs=epochs,
        warmup_steps=warmup_steps,
        optimizer_params={"lr": learning_rate},
        output_path=str(model_dir),
        save_best_model=True,
        use_amp=True,
        show_progress_bar=True,
    )
    model.save(str(model_dir))
    dev_scores = tuple(
        float(score)
        for score in model.predict(
            [(example.query, example.text) for example in dev_examples],
            show_progress_bar=False,
        )
    )
    dev_labels = tuple(example.label_score for example in dev_examples)
    metrics = {
        "base_model": base_model,
        "run_name": run_name,
        "model_dir": str(model_dir),
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "seed": seed,
        "warmup_steps": warmup_steps,
        "train_examples": len(train_examples),
        "dev_examples": len(dev_examples),
        "train_label_counts": _label_counts(
            tuple(example.label_score for example in train_examples)
        ),
        "dev_label_counts": _label_counts(dev_labels),
        "dev_binary_thresholds": _binary_threshold_metrics(dev_scores, dev_labels),
    }
    if reviewed_jsonl is not None and source_files is not None:
        metrics["extraction_holdout"] = _evaluate_trained_extractor(
            model=model,
            reviewed_jsonl=reviewed_jsonl,
            source_files=source_files,
            series_id=series_id,
            score_cache_jsonl=run_dir / "pair-score-cache.jsonl",
        )
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    volume.commit()
    return metrics


@app.function(
    image=image,
    gpu="H200",
    timeout=20 * 60,
    volumes={RUNS_DIR: volume},
)
def evaluate_saved_cross_encoder_remote(
    *,
    run_name: str,
    reviewed_jsonl: str,
    source_files: dict[str, str],
    series_id: str = "reviewed_external_series",
) -> dict[str, object]:
    sys.path.insert(0, REMOTE_SRC)

    from sentence_transformers import CrossEncoder

    model_dir = Path(RUNS_DIR) / run_name / "model"
    if not model_dir.exists():
        raise FileNotFoundError(f"saved model not found: {model_dir}")
    model = CrossEncoder(str(model_dir))
    return {
        "run_name": run_name,
        "model_dir": str(model_dir),
        "extraction_holdout": _evaluate_trained_extractor(
            model=model,
            reviewed_jsonl=reviewed_jsonl,
            source_files=source_files,
            series_id=series_id,
            score_cache_jsonl=Path(RUNS_DIR) / run_name / "pair-score-cache.jsonl",
        ),
    }


@app.local_entrypoint()
def main(
    train_jsonl_path: str = "",
    dev_jsonl_path: str = "",
    base_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    run_name: str = DEFAULT_RUN_NAME,
    evaluate_run_name: str = "",
    probe_id: str = "",
    list_probes: bool = False,
    epochs: int = 2,
    batch_size: int = 16,
    learning_rate: float = 2e-5,
    warmup_ratio: float = 0.1,
    max_length: int = 512,
    seed: int = 13,
    reviewed_jsonl_path: str = "",
    source_dir: str = "",
    series_id: str = "reviewed_external_series",
) -> None:
    from gia_evidence_finder.training_runs import (
        modal_reranker_probe_by_id,
        modal_reranker_probes,
    )

    if list_probes:
        print(
            json.dumps(
                {"probes": [probe.to_json_dict() for probe in modal_reranker_probes()]},
                indent=2,
            )
        )
        return
    if probe_id:
        probe = modal_reranker_probe_by_id(probe_id)
        base_model = probe.base_model
        if run_name == DEFAULT_RUN_NAME:
            run_name = probe.run_name
        epochs = probe.epochs
        batch_size = probe.batch_size
        learning_rate = probe.learning_rate
        warmup_ratio = probe.warmup_ratio
        max_length = probe.max_length
        seed = probe.seed
    if evaluate_run_name:
        if not reviewed_jsonl_path or not source_dir:
            raise ValueError("evaluation requires --reviewed-jsonl-path and --source-dir")
        reviewed_jsonl = Path(reviewed_jsonl_path).read_text(encoding="utf-8")
        source_files = _source_files(Path(source_dir))
        metrics = evaluate_saved_cross_encoder_remote.remote(
            run_name=evaluate_run_name,
            reviewed_jsonl=reviewed_jsonl,
            source_files=source_files,
            series_id=series_id,
        )
        print(json.dumps(metrics, indent=2))
        return
    if not train_jsonl_path or not dev_jsonl_path:
        raise ValueError("training requires train_jsonl_path and dev_jsonl_path")
    train_jsonl = Path(train_jsonl_path).read_text(encoding="utf-8")
    dev_jsonl = Path(dev_jsonl_path).read_text(encoding="utf-8")
    reviewed_jsonl = (
        Path(reviewed_jsonl_path).read_text(encoding="utf-8")
        if reviewed_jsonl_path
        else None
    )
    source_files = _source_files(Path(source_dir)) if source_dir else None
    metrics = train_cross_encoder_remote.remote(
        train_jsonl=train_jsonl,
        dev_jsonl=dev_jsonl,
        base_model=base_model,
        run_name=run_name,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        warmup_ratio=warmup_ratio,
        max_length=max_length,
        seed=seed,
        reviewed_jsonl=reviewed_jsonl,
        source_files=source_files,
        series_id=series_id,
    )
    print(json.dumps(metrics, indent=2))


def _source_files(source_dir: Path) -> dict[str, str]:
    return {
        path.name: path.read_text(encoding="utf-8")
        for path in sorted(source_dir.iterdir())
        if path.is_file()
    }


def _evaluate_trained_extractor(
    *,
    model: object,
    reviewed_jsonl: str,
    source_files: dict[str, str],
    series_id: str,
    score_cache_jsonl: Path,
) -> dict[str, object]:
    from gia_evidence_finder.benchmark_series import BenchmarkSplit
    from gia_evidence_finder.experiments import (
        ExtractorExperimentSpec,
        HoldoutDecisionPolicy,
        calibrate_default_first_stage_support_threshold,
        run_calibrated_holdout,
    )
    from gia_evidence_finder.extractor import EvidenceExtractor
    from gia_evidence_finder.model_baselines import (
        CrossEncoderRerankExtractor,
        TypedDecisionRerankExtractor,
    )
    from gia_evidence_finder.reranker_cache import (
        CachedPairRerankerModel,
        JsonlPairScoreCache,
    )
    from gia_evidence_finder.reviewed_cases import load_reviewed_series_jsonl

    source_dir = Path("/tmp/reviewed-sources")
    source_dir.mkdir(parents=True, exist_ok=True)
    for filename, text in source_files.items():
        (source_dir / filename).write_text(text, encoding="utf-8")
    reviewed_path = Path("/tmp/reviewed-cases.jsonl")
    reviewed_path.write_text(reviewed_jsonl, encoding="utf-8")
    series = load_reviewed_series_jsonl(
        reviewed_jsonl=reviewed_path,
        source_dir=source_dir,
        series_id=series_id,
    ).series
    raw_reranker = _InMemoryPairReranker(
        model=model,
        name=f"trained_cross_encoder:{series_id}",
    )
    reranker = CachedPairRerankerModel(
        model=raw_reranker,
        cache=JsonlPairScoreCache(score_cache_jsonl),
        batch_size=64,
    )
    typed_first_stage_threshold = calibrate_default_first_stage_support_threshold(
        series.splits[BenchmarkSplit.DEV],
    )
    holdouts = run_calibrated_holdout(
        (
            ExtractorExperimentSpec(
                f"{reranker.name}:keyword_first_stage",
                CrossEncoderRerankExtractor(reranker_model=reranker),
            ),
            ExtractorExperimentSpec(
                f"{reranker.name}:intent_first_stage",
                CrossEncoderRerankExtractor(
                    reranker_model=reranker,
                    first_stage=EvidenceExtractor.default(),
                ),
            ),
            ExtractorExperimentSpec(
                f"{reranker.name}:intent_first_stage_gated",
                CrossEncoderRerankExtractor(
                    reranker_model=reranker,
                    first_stage=EvidenceExtractor.default(),
                    require_first_stage_support=True,
                ),
            ),
            ExtractorExperimentSpec(
                f"{reranker.name}:intent_first_stage_preserve_labels",
                CrossEncoderRerankExtractor(
                    reranker_model=reranker,
                    first_stage=EvidenceExtractor.default(),
                    preserve_first_stage_labels=True,
                ),
                decision_policy=HoldoutDecisionPolicy.EXTRACTOR_LABELS,
            ),
            ExtractorExperimentSpec(
                f"{reranker.name}:calibrated_typed_decision",
                TypedDecisionRerankExtractor(
                    reranker_model=reranker,
                    first_stage=EvidenceExtractor.default(),
                    first_stage_support_threshold=typed_first_stage_threshold,
                ),
                decision_policy=HoldoutDecisionPolicy.EXTRACTOR_LABELS,
            ),
        ),
        dev_suite=series.splits[BenchmarkSplit.DEV],
        test_suite=series.splits[BenchmarkSplit.TEST],
    )
    return {
        "typed_first_stage_support_threshold": typed_first_stage_threshold,
        "pair_score_cache": {
            "path": str(score_cache_jsonl),
            "stats": reranker.stats.to_json_dict(),
        },
        "experiments": {
            holdout.name: {
                "decision_policy": holdout.decision_policy.value,
                "selected_threshold": holdout.selected_threshold,
                "first_stage_support_threshold": (
                    typed_first_stage_threshold
                    if holdout.name.endswith(":calibrated_typed_decision")
                    else None
                ),
                "dev_selected": _evaluation_report_payload(holdout.dev.calibration.best.report)
                if holdout.dev.calibration is not None
                else None,
                "test": _evaluation_report_payload(holdout.test.report),
            }
            for holdout in holdouts
        },
    }


class _InMemoryPairReranker:
    def __init__(self, *, model: object, name: str) -> None:
        self._model = model
        self.name = name

    def score_pairs(
        self,
        pairs: list[tuple[str, str]] | tuple[tuple[str, str], ...],
    ) -> tuple[float, ...]:
        scores = self._model.predict(list(pairs), show_progress_bar=False)
        return tuple(float(score) for score in scores)


def _evaluation_report_payload(report: object) -> dict[str, object]:
    return {
        "case_count": report.case_count,
        "support_case_count": report.support_case_count,
        "abstain_case_count": report.abstain_case_count,
        "mean_reciprocal_rank": report.mean_reciprocal_rank,
        "recall_at_1": report.recall_at_1,
        "recall_at_3": report.recall_at_3,
        "recall_at_5": report.recall_at_5,
        "top1_support_accuracy": report.top1_support_accuracy,
        "abstain_accuracy": report.abstain_accuracy,
        "decision_accuracy": report.decision_accuracy,
        "forbidden_top1_rate": report.forbidden_top1_rate,
        "forbidden_supported_top1_rate": report.forbidden_supported_top1_rate,
    }


def _label_counts(labels: tuple[float, ...]) -> dict[str, int]:
    positives = sum(1 for label in labels if label >= 0.5)
    negatives = len(labels) - positives
    return {"positive": positives, "negative": negatives}


def _binary_threshold_metrics(
    scores: tuple[float, ...],
    labels: tuple[float, ...],
) -> list[dict[str, float]]:
    if len(scores) != len(labels):
        raise ValueError("scores and labels must have the same length")
    thresholds = tuple(index / 100 for index in range(5, 100, 5))
    return [
        {
            "threshold": threshold,
            **_binary_metrics_at_threshold(scores, labels, threshold),
        }
        for threshold in thresholds
    ]


def _binary_metrics_at_threshold(
    scores: tuple[float, ...],
    labels: tuple[float, ...],
    threshold: float,
) -> dict[str, float]:
    true_positive = 0
    false_positive = 0
    true_negative = 0
    false_negative = 0
    for score, label in zip(scores, labels, strict=True):
        predicted_positive = score >= threshold
        actual_positive = label >= 0.5
        if predicted_positive and actual_positive:
            true_positive += 1
        elif predicted_positive and not actual_positive:
            false_positive += 1
        elif not predicted_positive and actual_positive:
            false_negative += 1
        else:
            true_negative += 1
    precision = _safe_divide(true_positive, true_positive + false_positive)
    recall = _safe_divide(true_positive, true_positive + false_negative)
    return {
        "accuracy": round(
            _safe_divide(true_positive + true_negative, len(scores)),
            4,
        ),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(_safe_divide(2 * precision * recall, precision + recall), 4),
    }


def _safe_divide(numerator: float, denominator: int | float) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator
