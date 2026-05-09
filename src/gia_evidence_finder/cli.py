from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

from gia_evidence_finder.baseline_registry import (
    baseline_profile_by_id,
    baseline_profiles,
)
from gia_evidence_finder.benchmark_audit import (
    BenchmarkSeriesAudit,
    BenchmarkSuiteAudit,
    audit_benchmark_series,
    audit_benchmark_suite,
    audit_expanded_domain_series,
)
from gia_evidence_finder.benchmark_series import (
    BenchmarkSeries,
    BenchmarkSplit,
    benchmark_series_by_id,
    benchmark_series_suites,
)
from gia_evidence_finder.calibration import CalibrationPoint
from gia_evidence_finder.competitor_evaluation import (
    competitor_evaluation_payload,
    competitor_markdown_report,
    competitor_registry,
)
from gia_evidence_finder.contracts import BenchmarkCase, BenchmarkSuite, IntentSpec
from gia_evidence_finder.curation_audit import (
    CurationQueueAudit,
    audit_curation_queue_jsonl,
)
from gia_evidence_finder.curation_queue import (
    curation_queue_from_fetched_readmes,
    curation_queue_from_local_documents,
    write_curation_queue_jsonl,
)
from gia_evidence_finder.evaluation import CaseEvaluation, NegativeLabelReport
from gia_evidence_finder.experiments import (
    CalibratedHoldoutExperiment,
    ExtractorExperimentSpec,
    RankerExperiment,
    calibrate_default_first_stage_support_threshold,
    cohere_reranker_baseline_spec,
    cross_encoder_baseline_spec,
    readme_baseline_specs,
    run_calibrated_holdout,
    run_experiment_spec,
    run_experiment_specs,
    run_readme_baselines,
    run_readme_cohere_reranker_baseline,
    run_readme_cross_encoder_baseline,
    run_readme_sentence_transformer_baselines,
    run_readme_transformers_reranker_baseline,
    sentence_transformer_baseline_specs,
    transformers_reranker_baseline_spec,
    typed_decision_cross_encoder_spec,
)
from gia_evidence_finder.extractor import EvidenceExtractor
from gia_evidence_finder.failure_analysis import build_failure_report
from gia_evidence_finder.intent_compiler import compile_intent
from gia_evidence_finder.model_judgment import (
    DEFAULT_MODEL_JUDGE_RUBRIC,
    decisions_from_model_response_jsonl,
    model_judge_agreement_report,
    model_judge_requests_from_suite,
    model_judge_response_schema,
    read_model_judge_decisions_jsonl,
    write_model_judge_requests_jsonl,
)
from gia_evidence_finder.parsing import MarkdownSpanParser
from gia_evidence_finder.readme_benchmarks import (
    readme_benchmark_suite_by_id,
    readme_benchmark_suites,
)
from gia_evidence_finder.reviewed_cases import (
    load_reviewed_cases_jsonl,
    load_reviewed_series_jsonl,
    write_reviewed_case_template_jsonl,
    write_reviewed_series_template_jsonl,
)
from gia_evidence_finder.source_catalog import (
    ReadmeSource,
    fetch_readme_sources,
    readme_source_by_id,
    readme_source_catalog,
    write_fetched_readmes,
)
from gia_evidence_finder.training import training_pairs_from_suite, training_pairs_jsonl
from gia_evidence_finder.training_runs import modal_reranker_probes


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract typed evidence spans from documents.")
    subcommands = parser.add_subparsers(dest="command", required=True)

    extract_parser = subcommands.add_parser("extract", help="Extract spans from a Markdown file.")
    extract_parser.add_argument("document", type=Path)
    extract_parser.add_argument("--intent-id", default="claim")
    extract_parser.add_argument("--claim")
    extract_parser.add_argument("--label")
    extract_parser.add_argument("--description")
    extract_parser.add_argument("--positive-example", action="append", default=[])
    extract_parser.add_argument("--max-matches", type=int, default=3)

    benchmark_parser = subcommands.add_parser(
        "benchmark-readmes",
        help="Run bundled README benchmark baselines.",
    )
    benchmark_parser.add_argument(
        "--include-models",
        action="store_true",
        help="Also run optional sentence-transformers embedding and cross-encoder baselines.",
    )
    benchmark_parser.add_argument(
        "--baseline-profile",
        action="append",
        choices=_baseline_profile_ids(),
        default=[],
        help="Expand a curated baseline profile.",
    )
    benchmark_parser.add_argument(
        "--cross-encoder-model",
        action="append",
        default=[],
        help="Also run a sentence-transformers CrossEncoder-compatible reranker model.",
    )
    benchmark_parser.add_argument(
        "--typed-decision-cross-encoder-model",
        action="append",
        default=[],
        help=(
            "Also run a CrossEncoder-compatible reranker for ordering while preserving "
            "typed first-stage support decisions."
        ),
    )
    benchmark_parser.add_argument(
        "--reranker-model",
        action="append",
        default=[],
        help=(
            "Also run a Hugging Face transformers sequence-classification reranker model. "
            "May be passed multiple times."
        ),
    )
    benchmark_parser.add_argument(
        "--cohere-reranker-model",
        action="append",
        default=[],
        help="Also run a Cohere reranker model. Requires COHERE_API_KEY.",
    )
    benchmark_parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Allow Hugging Face optional model baselines to execute custom model code.",
    )
    benchmark_parser.add_argument(
        "--reranker-support-threshold",
        type=float,
        default=None,
        help="Optional support threshold for --reranker-model scores after sigmoid.",
    )
    _add_reranker_score_cache_args(benchmark_parser)
    benchmark_parser.add_argument(
        "--details",
        action="store_true",
        help="Include per-case diagnostics for failed or risky cases.",
    )
    benchmark_parser.add_argument(
        "--calibrate-thresholds",
        action="store_true",
        help="Include support-threshold sweep results for each experiment.",
    )
    benchmark_parser.add_argument(
        "--suite",
        default="popular_readme_v1",
        choices=(
            "popular_readme_v1",
            "hard_readme_v1",
            "adversarial_readme_v1",
            "relation_readme_v1",
            "all",
        ),
        help="README benchmark suite to run.",
    )
    series_parser = subcommands.add_parser(
        "benchmark-series",
        help="Run a frozen train/dev/test benchmark series.",
    )
    series_parser.add_argument("--series", default="evidence_benchmark_v2")
    series_parser.add_argument(
        "--split",
        default="all",
        choices=("all", "train", "dev", "test"),
        help="Benchmark split to run.",
    )
    series_parser.add_argument(
        "--include-models",
        action="store_true",
        help="Also run optional sentence-transformers embedding and cross-encoder baselines.",
    )
    series_parser.add_argument(
        "--baseline-profile",
        action="append",
        choices=_baseline_profile_ids(),
        default=[],
        help="Expand a curated baseline profile.",
    )
    series_parser.add_argument(
        "--cross-encoder-model",
        action="append",
        default=[],
        help="Also run a sentence-transformers CrossEncoder-compatible reranker model.",
    )
    series_parser.add_argument(
        "--typed-decision-cross-encoder-model",
        action="append",
        default=[],
        help=(
            "Also run a CrossEncoder-compatible reranker for ordering while preserving "
            "typed first-stage support decisions."
        ),
    )
    series_parser.add_argument(
        "--reranker-model",
        action="append",
        default=[],
        help="Also run a Hugging Face transformers sequence-classification reranker model.",
    )
    series_parser.add_argument(
        "--cohere-reranker-model",
        action="append",
        default=[],
        help="Also run a Cohere reranker model. Requires COHERE_API_KEY.",
    )
    series_parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Allow Hugging Face optional model baselines to execute custom model code.",
    )
    series_parser.add_argument(
        "--reranker-support-threshold",
        type=float,
        default=None,
        help="Optional support threshold for --reranker-model scores after sigmoid.",
    )
    _add_reranker_score_cache_args(series_parser)
    series_parser.add_argument(
        "--details",
        action="store_true",
        help="Include per-case diagnostics for failed or risky cases.",
    )
    series_parser.add_argument(
        "--calibrate-thresholds",
        action="store_true",
        help="Include support-threshold sweep results for each experiment.",
    )
    calibrated_series_parser = subcommands.add_parser(
        "benchmark-series-calibrated",
        help="Calibrate thresholds on a series dev split and report held-out test results.",
    )
    calibrated_series_parser.add_argument("--series", default="evidence_benchmark_v2")
    calibrated_series_parser.add_argument(
        "--include-models",
        action="store_true",
        help="Also run optional sentence-transformers embedding and cross-encoder baselines.",
    )
    calibrated_series_parser.add_argument(
        "--baseline-profile",
        action="append",
        choices=_baseline_profile_ids(),
        default=[],
        help="Expand a curated baseline profile.",
    )
    calibrated_series_parser.add_argument(
        "--cross-encoder-model",
        action="append",
        default=[],
        help="Also run a sentence-transformers CrossEncoder-compatible reranker model.",
    )
    calibrated_series_parser.add_argument(
        "--typed-decision-cross-encoder-model",
        action="append",
        default=[],
        help=(
            "Also run a CrossEncoder-compatible reranker for ordering while preserving "
            "dev-calibrated typed first-stage support decisions."
        ),
    )
    calibrated_series_parser.add_argument(
        "--reranker-model",
        action="append",
        default=[],
        help="Also run a Hugging Face transformers sequence-classification reranker model.",
    )
    calibrated_series_parser.add_argument(
        "--cohere-reranker-model",
        action="append",
        default=[],
        help="Also run a Cohere reranker model. Requires COHERE_API_KEY.",
    )
    calibrated_series_parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Allow Hugging Face optional model baselines to execute custom model code.",
    )
    _add_reranker_score_cache_args(calibrated_series_parser)
    calibrated_series_parser.add_argument(
        "--details",
        action="store_true",
        help="Include per-case diagnostics for failed or risky cases.",
    )
    audit_parser = subcommands.add_parser(
        "audit-benchmark-series",
        help="Audit benchmark split size, label coverage, and SOTA readiness gaps.",
    )
    audit_parser.add_argument("--series", default="evidence_benchmark_v2")
    competitor_parser = subcommands.add_parser(
        "benchmark-competitors",
        help="Run local competitor curves and optional hosted competitors.",
    )
    competitor_parser.add_argument("--series", default="domain_evidence_benchmark_v4")
    competitor_parser.add_argument(
        "--split",
        default="test",
        choices=("train", "dev", "test", "all"),
    )
    competitor_parser.add_argument(
        "--competitor",
        action="append",
        default=[],
        help="Competitor id to run. Defaults to the required local competitor set.",
    )
    competitor_parser.add_argument(
        "--include-hosted",
        action="store_true",
        help="Include hosted competitors when provider credentials are available.",
    )
    competitor_parser.add_argument(
        "--format",
        choices=("json", "markdown"),
        default="json",
    )
    competitor_parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Allow optional Hugging Face competitor models to execute custom model code.",
    )
    competitor_parser.add_argument(
        "--score-cache-jsonl",
        type=Path,
        default=None,
        help="Optional persistent JSONL cache for pair reranker scores.",
    )
    competitor_parser.add_argument(
        "--score-batch-size",
        type=int,
        default=None,
        help="Optional batch size for uncached pair reranker scoring calls.",
    )
    subcommands.add_parser(
        "list-readme-sources",
        help="List the source README catalog used for larger curation queues.",
    )
    fetch_parser = subcommands.add_parser(
        "fetch-readme-sources",
        help="Fetch raw README sources into a local curation directory.",
    )
    fetch_parser.add_argument("--output-dir", type=Path, required=True)
    fetch_parser.add_argument(
        "--source",
        action="append",
        default=[],
        help="README source id to fetch. Defaults to the full catalog.",
    )
    fetch_parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional maximum number of sources to fetch after source filtering.",
    )
    queue_parser = subcommands.add_parser(
        "prepare-curation-queue",
        help="Convert fetched README sources into unreviewed span-labeling JSONL.",
    )
    queue_parser.add_argument("--input-dir", type=Path, required=True)
    queue_parser.add_argument("--output-jsonl", type=Path, required=True)
    queue_parser.add_argument("--max-items-per-doc", type=int, default=40)
    local_queue_parser = subcommands.add_parser(
        "prepare-document-curation-queue",
        help="Convert local document files into unreviewed span-labeling JSONL.",
    )
    local_queue_parser.add_argument("--input", type=Path, required=True)
    local_queue_parser.add_argument("--output-jsonl", type=Path, required=True)
    local_queue_parser.add_argument(
        "--pattern",
        action="append",
        default=[],
        help="Glob pattern for recursive directory scans. Defaults to Markdown/RST/TXT.",
    )
    local_queue_parser.add_argument("--max-items-per-doc", type=int, default=40)
    local_queue_parser.add_argument(
        "--allow-empty-heading",
        action="store_true",
        help="Allow spans without heading context into the queue.",
    )
    queue_audit_parser = subcommands.add_parser(
        "audit-curation-queue",
        help="Audit curation queue diversity before reviewed-case promotion.",
    )
    queue_audit_parser.add_argument("--queue-jsonl", type=Path, required=True)
    queue_audit_parser.add_argument("--min-documents", type=int, default=10)
    queue_audit_parser.add_argument("--min-items", type=int, default=200)
    queue_audit_parser.add_argument("--max-document-share", type=float, default=0.25)
    template_parser = subcommands.add_parser(
        "prepare-reviewed-case-template",
        help="Convert curation queue rows into unreviewed reviewed-case templates.",
    )
    template_parser.add_argument("--queue-jsonl", type=Path, required=True)
    template_parser.add_argument("--output-jsonl", type=Path, required=True)
    template_parser.add_argument(
        "--split",
        default="train",
        choices=("train", "dev", "test"),
    )
    series_template_parser = subcommands.add_parser(
        "prepare-reviewed-series-template",
        help=(
            "Convert curation queue rows into unreviewed reviewed-case templates "
            "with deterministic document-level train/dev/test splits."
        ),
    )
    series_template_parser.add_argument("--queue-jsonl", type=Path, required=True)
    series_template_parser.add_argument("--output-jsonl", type=Path, required=True)
    series_template_parser.add_argument("--train-ratio", type=float, default=0.7)
    series_template_parser.add_argument("--dev-ratio", type=float, default=0.15)
    series_template_parser.add_argument("--test-ratio", type=float, default=0.15)
    series_template_parser.add_argument("--seed", default="reviewed-series-v1")
    validate_parser = subcommands.add_parser(
        "validate-reviewed-cases",
        help="Validate promoted reviewed-case JSONL against fetched README sources.",
    )
    validate_parser.add_argument("--source-dir", type=Path, required=True)
    validate_parser.add_argument("--reviewed-jsonl", type=Path, required=True)
    validate_parser.add_argument("--suite-id", default="reviewed_evidence_cases")
    validate_parser.add_argument("--suite-name", default=None)
    validate_series_parser = subcommands.add_parser(
        "validate-reviewed-series",
        help="Validate reviewed-case JSONL as a train/dev/test benchmark series.",
    )
    validate_series_parser.add_argument("--source-dir", type=Path, required=True)
    validate_series_parser.add_argument("--reviewed-jsonl", type=Path, required=True)
    validate_series_parser.add_argument("--series-id", default="evidence_benchmark_v3_seed")
    validate_series_parser.add_argument("--series-name", default=None)
    benchmark_reviewed_parser = subcommands.add_parser(
        "benchmark-reviewed-series",
        help="Run baselines against an external reviewed train/dev/test JSONL series.",
    )
    benchmark_reviewed_parser.add_argument("--source-dir", type=Path, required=True)
    benchmark_reviewed_parser.add_argument("--reviewed-jsonl", type=Path, required=True)
    benchmark_reviewed_parser.add_argument("--series-id", default="reviewed_external_series")
    benchmark_reviewed_parser.add_argument("--series-name", default=None)
    benchmark_reviewed_parser.add_argument(
        "--split",
        default="all",
        choices=("all", "train", "dev", "test"),
        help="Reviewed benchmark split to run.",
    )
    benchmark_reviewed_parser.add_argument(
        "--include-models",
        action="store_true",
        help="Also run optional sentence-transformers embedding and cross-encoder baselines.",
    )
    benchmark_reviewed_parser.add_argument(
        "--baseline-profile",
        action="append",
        choices=_baseline_profile_ids(),
        default=[],
        help="Expand a curated baseline profile.",
    )
    benchmark_reviewed_parser.add_argument("--cross-encoder-model", action="append", default=[])
    benchmark_reviewed_parser.add_argument(
        "--typed-decision-cross-encoder-model",
        action="append",
        default=[],
    )
    benchmark_reviewed_parser.add_argument("--reranker-model", action="append", default=[])
    benchmark_reviewed_parser.add_argument("--cohere-reranker-model", action="append", default=[])
    benchmark_reviewed_parser.add_argument("--trust-remote-code", action="store_true")
    benchmark_reviewed_parser.add_argument("--reranker-support-threshold", type=float, default=None)
    _add_reranker_score_cache_args(benchmark_reviewed_parser)
    benchmark_reviewed_parser.add_argument("--details", action="store_true")
    benchmark_reviewed_parser.add_argument("--calibrate-thresholds", action="store_true")
    calibrated_reviewed_parser = subcommands.add_parser(
        "benchmark-reviewed-series-calibrated",
        help=(
            "Calibrate thresholds on an external reviewed dev split and report held-out "
            "test results."
        ),
    )
    calibrated_reviewed_parser.add_argument("--source-dir", type=Path, required=True)
    calibrated_reviewed_parser.add_argument("--reviewed-jsonl", type=Path, required=True)
    calibrated_reviewed_parser.add_argument("--series-id", default="reviewed_external_series")
    calibrated_reviewed_parser.add_argument("--series-name", default=None)
    calibrated_reviewed_parser.add_argument(
        "--include-models",
        action="store_true",
        help="Also run optional sentence-transformers embedding and cross-encoder baselines.",
    )
    calibrated_reviewed_parser.add_argument(
        "--baseline-profile",
        action="append",
        choices=_baseline_profile_ids(),
        default=[],
        help="Expand a curated baseline profile.",
    )
    calibrated_reviewed_parser.add_argument("--cross-encoder-model", action="append", default=[])
    calibrated_reviewed_parser.add_argument(
        "--typed-decision-cross-encoder-model",
        action="append",
        default=[],
    )
    calibrated_reviewed_parser.add_argument("--reranker-model", action="append", default=[])
    calibrated_reviewed_parser.add_argument("--cohere-reranker-model", action="append", default=[])
    calibrated_reviewed_parser.add_argument("--trust-remote-code", action="store_true")
    _add_reranker_score_cache_args(calibrated_reviewed_parser)
    calibrated_reviewed_parser.add_argument("--details", action="store_true")
    explain_reviewed_parser = subcommands.add_parser(
        "explain-reviewed-case",
        help="Explain scored candidates for one external reviewed benchmark case.",
    )
    explain_reviewed_parser.add_argument("--source-dir", type=Path, required=True)
    explain_reviewed_parser.add_argument("--reviewed-jsonl", type=Path, required=True)
    explain_reviewed_parser.add_argument("--series-id", default="reviewed_external_series")
    explain_reviewed_parser.add_argument("--case-id", required=True)
    explain_reviewed_parser.add_argument("--support-threshold", type=float, default=None)
    explain_reviewed_parser.add_argument("--candidate-limit", type=int, default=10)
    failure_reviewed_parser = subcommands.add_parser(
        "failure-report-reviewed-series",
        help="Run reviewed-series extractors and print text-backed failure reports.",
    )
    failure_reviewed_parser.add_argument("--source-dir", type=Path, required=True)
    failure_reviewed_parser.add_argument("--reviewed-jsonl", type=Path, required=True)
    failure_reviewed_parser.add_argument("--series-id", default="reviewed_external_series")
    failure_reviewed_parser.add_argument("--series-name", default=None)
    failure_reviewed_parser.add_argument(
        "--split",
        default="test",
        choices=("train", "dev", "test"),
    )
    failure_reviewed_parser.add_argument("--include-models", action="store_true")
    failure_reviewed_parser.add_argument(
        "--baseline-profile",
        action="append",
        choices=_baseline_profile_ids(),
        default=[],
    )
    failure_reviewed_parser.add_argument("--cross-encoder-model", action="append", default=[])
    failure_reviewed_parser.add_argument(
        "--typed-decision-cross-encoder-model",
        action="append",
        default=[],
    )
    failure_reviewed_parser.add_argument("--reranker-model", action="append", default=[])
    failure_reviewed_parser.add_argument("--cohere-reranker-model", action="append", default=[])
    failure_reviewed_parser.add_argument("--trust-remote-code", action="store_true")
    failure_reviewed_parser.add_argument("--reranker-support-threshold", type=float, default=None)
    _add_reranker_score_cache_args(failure_reviewed_parser)
    failure_reviewed_parser.add_argument(
        "--apply-dev-calibration",
        action="store_true",
        help="Apply each experiment's dev-selected support threshold before reporting failures.",
    )
    failure_reviewed_parser.add_argument(
        "--experiment-name",
        action="append",
        default=[],
        help="Optional experiment name filter. May be passed multiple times.",
    )
    subcommands.add_parser(
        "model-judge-rubric",
        help="Print the typed model-judge rubric and expected response schema.",
    )
    judge_request_parser = subcommands.add_parser(
        "export-model-judge-requests",
        help="Export model-judge request JSONL from a built-in benchmark series.",
    )
    judge_request_parser.add_argument("--series", default="mixed_evidence_benchmark_v3")
    judge_request_parser.add_argument(
        "--split",
        default="all",
        choices=("all", "train", "dev", "test"),
    )
    judge_request_parser.add_argument("--output-jsonl", type=Path, required=True)
    judge_request_parser.add_argument("--max-requests", type=int, default=None)
    judge_request_parser.add_argument(
        "--skip-unlabeled-forbidden",
        action="store_true",
        help="Do not include forbidden spans that lack a specific negative label.",
    )
    judge_convert_parser = subcommands.add_parser(
        "convert-model-judge-responses",
        help="Attach judge metadata to raw model response JSONL.",
    )
    judge_convert_parser.add_argument("--request-jsonl", type=Path, required=True)
    judge_convert_parser.add_argument("--response-jsonl", type=Path, required=True)
    judge_convert_parser.add_argument("--output-jsonl", type=Path, required=True)
    judge_convert_parser.add_argument("--judge-model", required=True)
    judge_convert_parser.add_argument("--pass-id", required=True)
    judge_audit_parser = subcommands.add_parser(
        "audit-model-judge-decisions",
        help="Summarize repeated model-judge decisions and disagreements.",
    )
    judge_audit_parser.add_argument("--decisions-jsonl", type=Path, required=True)
    judge_audit_parser.add_argument("--min-passes", type=int, default=2)
    export_reviewed_parser = subcommands.add_parser(
        "export-reviewed-training-jsonl",
        help="Export support and hard-negative training pairs from an external reviewed series.",
    )
    export_reviewed_parser.add_argument("--source-dir", type=Path, required=True)
    export_reviewed_parser.add_argument("--reviewed-jsonl", type=Path, required=True)
    export_reviewed_parser.add_argument("--series-id", default="reviewed_external_series")
    export_reviewed_parser.add_argument(
        "--split",
        default="train",
        choices=("all", "train", "dev", "test"),
    )
    export_reviewed_parser.add_argument("--negatives-per-case", type=int, default=3)
    subcommands.add_parser(
        "modal-reranker-probes",
        help="List packaged Modal reranker training probe presets.",
    )
    export_parser = subcommands.add_parser(
        "export-training-jsonl",
        help="Export bundled README benchmark support and hard-negative pairs.",
    )
    export_parser.add_argument(
        "--suite",
        default="popular_readme_v1",
        choices=(
            "popular_readme_v1",
            "hard_readme_v1",
            "adversarial_readme_v1",
            "relation_readme_v1",
            "all",
        ),
        help="README benchmark suite to export.",
    )
    export_parser.add_argument(
        "--series",
        default=None,
        help="Benchmark series to export instead of --suite.",
    )
    export_parser.add_argument(
        "--split",
        default="train",
        choices=("all", "train", "dev", "test"),
        help="Benchmark series split to export when --series is set.",
    )
    export_parser.add_argument("--negatives-per-case", type=int, default=3)
    args = parser.parse_args()

    if args.command == "benchmark-readmes":
        _print_readme_benchmarks(
            include_models=args.include_models,
            baseline_profile_ids=tuple(args.baseline_profile),
            cross_encoder_models=tuple(args.cross_encoder_model),
            typed_decision_cross_encoder_models=tuple(args.typed_decision_cross_encoder_model),
            reranker_models=tuple(args.reranker_model),
            cohere_reranker_models=tuple(args.cohere_reranker_model),
            trust_remote_code=args.trust_remote_code,
            reranker_support_threshold=args.reranker_support_threshold,
            reranker_score_cache_jsonl=args.reranker_score_cache_jsonl,
            reranker_score_batch_size=args.reranker_score_batch_size,
            suite_id=args.suite,
            include_details=args.details,
            include_calibration=args.calibrate_thresholds,
        )
        return
    if args.command == "benchmark-series":
        _print_benchmark_series(
            series_id=args.series,
            split=args.split,
            include_models=args.include_models,
            baseline_profile_ids=tuple(args.baseline_profile),
            cross_encoder_models=tuple(args.cross_encoder_model),
            typed_decision_cross_encoder_models=tuple(args.typed_decision_cross_encoder_model),
            reranker_models=tuple(args.reranker_model),
            cohere_reranker_models=tuple(args.cohere_reranker_model),
            trust_remote_code=args.trust_remote_code,
            reranker_support_threshold=args.reranker_support_threshold,
            reranker_score_cache_jsonl=args.reranker_score_cache_jsonl,
            reranker_score_batch_size=args.reranker_score_batch_size,
            include_details=args.details,
            include_calibration=args.calibrate_thresholds,
        )
        return
    if args.command == "benchmark-series-calibrated":
        _print_benchmark_series_calibrated(
            series_id=args.series,
            include_models=args.include_models,
            baseline_profile_ids=tuple(args.baseline_profile),
            cross_encoder_models=tuple(args.cross_encoder_model),
            typed_decision_cross_encoder_models=tuple(args.typed_decision_cross_encoder_model),
            reranker_models=tuple(args.reranker_model),
            cohere_reranker_models=tuple(args.cohere_reranker_model),
            trust_remote_code=args.trust_remote_code,
            reranker_score_cache_jsonl=args.reranker_score_cache_jsonl,
            reranker_score_batch_size=args.reranker_score_batch_size,
            include_details=args.details,
        )
        return
    if args.command == "export-training-jsonl":
        _print_training_jsonl(
            suite_id=args.suite,
            series_id=args.series,
            split=args.split,
            negatives_per_case=args.negatives_per_case,
        )
        return
    if args.command == "audit-benchmark-series":
        _print_benchmark_series_audit(series_id=args.series)
        return
    if args.command == "benchmark-competitors":
        _print_benchmark_competitors(
            series_id=args.series,
            split=args.split,
            competitor_ids=tuple(args.competitor),
            include_hosted=args.include_hosted,
            output_format=args.format,
            trust_remote_code=args.trust_remote_code,
            score_cache_jsonl=args.score_cache_jsonl,
            score_batch_size=args.score_batch_size,
        )
        return
    if args.command == "list-readme-sources":
        _print_readme_sources()
        return
    if args.command == "fetch-readme-sources":
        _print_fetch_readme_sources(
            output_dir=args.output_dir,
            source_ids=tuple(args.source),
            limit=args.limit,
        )
        return
    if args.command == "prepare-curation-queue":
        _print_prepare_curation_queue(
            input_dir=args.input_dir,
            output_jsonl=args.output_jsonl,
            max_items_per_doc=args.max_items_per_doc,
        )
        return
    if args.command == "prepare-document-curation-queue":
        _print_prepare_document_curation_queue(
            input_path=args.input,
            output_jsonl=args.output_jsonl,
            patterns=tuple(args.pattern),
            max_items_per_doc=args.max_items_per_doc,
            require_heading=not args.allow_empty_heading,
        )
        return
    if args.command == "audit-curation-queue":
        _print_audit_curation_queue(
            queue_jsonl=args.queue_jsonl,
            min_documents=args.min_documents,
            min_items=args.min_items,
            max_document_share=args.max_document_share,
        )
        return
    if args.command == "prepare-reviewed-case-template":
        _print_prepare_reviewed_case_template(
            queue_jsonl=args.queue_jsonl,
            output_jsonl=args.output_jsonl,
            split=BenchmarkSplit(args.split),
        )
        return
    if args.command == "prepare-reviewed-series-template":
        _print_prepare_reviewed_series_template(
            queue_jsonl=args.queue_jsonl,
            output_jsonl=args.output_jsonl,
            train_ratio=args.train_ratio,
            dev_ratio=args.dev_ratio,
            test_ratio=args.test_ratio,
            seed=args.seed,
        )
        return
    if args.command == "validate-reviewed-cases":
        _print_validate_reviewed_cases(
            source_dir=args.source_dir,
            reviewed_jsonl=args.reviewed_jsonl,
            suite_id=args.suite_id,
            suite_name=args.suite_name,
        )
        return
    if args.command == "validate-reviewed-series":
        _print_validate_reviewed_series(
            source_dir=args.source_dir,
            reviewed_jsonl=args.reviewed_jsonl,
            series_id=args.series_id,
            series_name=args.series_name,
        )
        return
    if args.command == "benchmark-reviewed-series":
        _print_benchmark_reviewed_series(
            source_dir=args.source_dir,
            reviewed_jsonl=args.reviewed_jsonl,
            series_id=args.series_id,
            series_name=args.series_name,
            split=args.split,
            include_models=args.include_models,
            baseline_profile_ids=tuple(args.baseline_profile),
            cross_encoder_models=tuple(args.cross_encoder_model),
            typed_decision_cross_encoder_models=tuple(args.typed_decision_cross_encoder_model),
            reranker_models=tuple(args.reranker_model),
            cohere_reranker_models=tuple(args.cohere_reranker_model),
            trust_remote_code=args.trust_remote_code,
            reranker_support_threshold=args.reranker_support_threshold,
            reranker_score_cache_jsonl=args.reranker_score_cache_jsonl,
            reranker_score_batch_size=args.reranker_score_batch_size,
            include_details=args.details,
            include_calibration=args.calibrate_thresholds,
        )
        return
    if args.command == "benchmark-reviewed-series-calibrated":
        _print_benchmark_reviewed_series_calibrated(
            source_dir=args.source_dir,
            reviewed_jsonl=args.reviewed_jsonl,
            series_id=args.series_id,
            series_name=args.series_name,
            include_models=args.include_models,
            baseline_profile_ids=tuple(args.baseline_profile),
            cross_encoder_models=tuple(args.cross_encoder_model),
            typed_decision_cross_encoder_models=tuple(args.typed_decision_cross_encoder_model),
            reranker_models=tuple(args.reranker_model),
            cohere_reranker_models=tuple(args.cohere_reranker_model),
            trust_remote_code=args.trust_remote_code,
            reranker_score_cache_jsonl=args.reranker_score_cache_jsonl,
            reranker_score_batch_size=args.reranker_score_batch_size,
            include_details=args.details,
        )
        return
    if args.command == "explain-reviewed-case":
        _print_explain_reviewed_case(
            source_dir=args.source_dir,
            reviewed_jsonl=args.reviewed_jsonl,
            series_id=args.series_id,
            case_id=args.case_id,
            support_threshold=args.support_threshold,
            candidate_limit=args.candidate_limit,
        )
        return
    if args.command == "failure-report-reviewed-series":
        _print_failure_report_reviewed_series(
            source_dir=args.source_dir,
            reviewed_jsonl=args.reviewed_jsonl,
            series_id=args.series_id,
            series_name=args.series_name,
            split=args.split,
            include_models=args.include_models,
            baseline_profile_ids=tuple(args.baseline_profile),
            cross_encoder_models=tuple(args.cross_encoder_model),
            typed_decision_cross_encoder_models=tuple(args.typed_decision_cross_encoder_model),
            reranker_models=tuple(args.reranker_model),
            cohere_reranker_models=tuple(args.cohere_reranker_model),
            trust_remote_code=args.trust_remote_code,
            reranker_support_threshold=args.reranker_support_threshold,
            reranker_score_cache_jsonl=args.reranker_score_cache_jsonl,
            reranker_score_batch_size=args.reranker_score_batch_size,
            apply_dev_calibration=args.apply_dev_calibration,
            experiment_names=tuple(args.experiment_name),
        )
        return
    if args.command == "model-judge-rubric":
        _print_model_judge_rubric()
        return
    if args.command == "export-model-judge-requests":
        _print_export_model_judge_requests(
            series_id=args.series,
            split=args.split,
            output_jsonl=args.output_jsonl,
            max_requests=args.max_requests,
            include_unlabeled_forbidden=not args.skip_unlabeled_forbidden,
        )
        return
    if args.command == "convert-model-judge-responses":
        _print_convert_model_judge_responses(
            request_jsonl=args.request_jsonl,
            response_jsonl=args.response_jsonl,
            output_jsonl=args.output_jsonl,
            judge_model=args.judge_model,
            pass_id=args.pass_id,
        )
        return
    if args.command == "audit-model-judge-decisions":
        _print_audit_model_judge_decisions(
            decisions_jsonl=args.decisions_jsonl,
            min_passes=args.min_passes,
        )
        return
    if args.command == "export-reviewed-training-jsonl":
        _print_reviewed_training_jsonl(
            source_dir=args.source_dir,
            reviewed_jsonl=args.reviewed_jsonl,
            series_id=args.series_id,
            split=args.split,
            negatives_per_case=args.negatives_per_case,
        )
        return
    if args.command == "modal-reranker-probes":
        _print_modal_reranker_probes()
        return

    document_text = args.document.read_text(encoding="utf-8")
    document = MarkdownSpanParser().parse(document_text, document_id=args.document.stem)
    intent = _extract_intent(args, parser)
    result = EvidenceExtractor.default().extract(intent, document, max_matches=args.max_matches)
    payload: dict[str, object] = {
        "abstained": result.abstained,
        "matches": [
            {
                "span_id": match.span.id,
                "kind": match.span.kind,
                "score": match.score,
                "text": match.span.text,
                "features": dict(match.features),
                "reasons": list(match.reasons),
            }
            for match in result.matches
        ],
        "trace": list(result.trace),
    }
    print(json.dumps(payload, indent=2))


def _extract_intent(args: argparse.Namespace, parser: argparse.ArgumentParser) -> IntentSpec:
    positive_examples = tuple(args.positive_example)
    if args.claim:
        return compile_intent(
            intent_id=args.intent_id,
            claim=args.claim,
            positive_examples=positive_examples,
        )
    if not args.label or not args.description:
        parser.error("extract requires either --claim or both --label and --description")
    return IntentSpec(
        id=args.intent_id,
        label=args.label,
        description=args.description,
        positive_examples=positive_examples,
    )


def _print_readme_benchmarks(
    *,
    include_models: bool,
    baseline_profile_ids: tuple[str, ...],
    cross_encoder_models: tuple[str, ...],
    typed_decision_cross_encoder_models: tuple[str, ...],
    reranker_models: tuple[str, ...],
    cohere_reranker_models: tuple[str, ...],
    trust_remote_code: bool,
    reranker_support_threshold: float | None,
    reranker_score_cache_jsonl: Path | None,
    reranker_score_batch_size: int | None,
    suite_id: str,
    include_details: bool,
    include_calibration: bool,
) -> None:
    payload: dict[str, object]
    if suite_id == "all":
        payload = {
            suite.id: _readme_suite_payload(
                suite_id=suite.id,
                include_models=include_models,
                baseline_profile_ids=baseline_profile_ids,
                cross_encoder_models=cross_encoder_models,
                typed_decision_cross_encoder_models=typed_decision_cross_encoder_models,
                reranker_models=reranker_models,
                cohere_reranker_models=cohere_reranker_models,
                trust_remote_code=trust_remote_code,
                reranker_support_threshold=reranker_support_threshold,
                reranker_score_cache_jsonl=reranker_score_cache_jsonl,
                reranker_score_batch_size=reranker_score_batch_size,
                include_details=include_details,
                include_calibration=include_calibration,
            )
            for suite in readme_benchmark_suites()
        }
    else:
        payload = _readme_suite_payload(
            suite_id=suite_id,
            include_models=include_models,
            baseline_profile_ids=baseline_profile_ids,
            cross_encoder_models=cross_encoder_models,
            typed_decision_cross_encoder_models=typed_decision_cross_encoder_models,
            reranker_models=reranker_models,
            cohere_reranker_models=cohere_reranker_models,
            trust_remote_code=trust_remote_code,
            reranker_support_threshold=reranker_support_threshold,
            reranker_score_cache_jsonl=reranker_score_cache_jsonl,
            reranker_score_batch_size=reranker_score_batch_size,
            include_details=include_details,
            include_calibration=include_calibration,
        )
    print(json.dumps(payload, indent=2))


def _readme_suite_payload(
    *,
    suite_id: str,
    include_models: bool,
    baseline_profile_ids: tuple[str, ...],
    cross_encoder_models: tuple[str, ...],
    typed_decision_cross_encoder_models: tuple[str, ...],
    reranker_models: tuple[str, ...],
    cohere_reranker_models: tuple[str, ...],
    trust_remote_code: bool,
    reranker_support_threshold: float | None,
    reranker_score_cache_jsonl: Path | None,
    reranker_score_batch_size: int | None,
    include_details: bool,
    include_calibration: bool,
) -> dict[str, object]:
    suite = readme_benchmark_suite_by_id(suite_id)
    return _suite_payload(
        suite=suite,
        include_models=include_models,
        baseline_profile_ids=baseline_profile_ids,
        cross_encoder_models=cross_encoder_models,
        typed_decision_cross_encoder_models=typed_decision_cross_encoder_models,
        reranker_models=reranker_models,
        cohere_reranker_models=cohere_reranker_models,
        trust_remote_code=trust_remote_code,
        reranker_support_threshold=reranker_support_threshold,
        reranker_score_cache_jsonl=reranker_score_cache_jsonl,
        reranker_score_batch_size=reranker_score_batch_size,
        include_details=include_details,
        include_calibration=include_calibration,
    )


def _print_benchmark_series(
    *,
    series_id: str,
    split: str,
    include_models: bool,
    baseline_profile_ids: tuple[str, ...],
    cross_encoder_models: tuple[str, ...],
    typed_decision_cross_encoder_models: tuple[str, ...],
    reranker_models: tuple[str, ...],
    cohere_reranker_models: tuple[str, ...],
    trust_remote_code: bool,
    reranker_support_threshold: float | None,
    reranker_score_cache_jsonl: Path | None,
    reranker_score_batch_size: int | None,
    include_details: bool,
    include_calibration: bool,
) -> None:
    series = benchmark_series_by_id(series_id)
    selected_split = None if split == "all" else BenchmarkSplit(split)
    suites = benchmark_series_suites(series, split=selected_split)
    payload: dict[str, object] = {
        "series_id": series.id,
        "name": series.name,
        "metadata": dict(series.metadata),
        "splits": {
            suite.metadata["split"]: _suite_payload(
                suite=suite,
                include_models=include_models,
                baseline_profile_ids=baseline_profile_ids,
                cross_encoder_models=cross_encoder_models,
                typed_decision_cross_encoder_models=typed_decision_cross_encoder_models,
                reranker_models=reranker_models,
                cohere_reranker_models=cohere_reranker_models,
                trust_remote_code=trust_remote_code,
                reranker_support_threshold=reranker_support_threshold,
                reranker_score_cache_jsonl=reranker_score_cache_jsonl,
                reranker_score_batch_size=reranker_score_batch_size,
                include_details=include_details,
                include_calibration=include_calibration,
            )
            for suite in suites
        },
    }
    print(json.dumps(payload, indent=2))


def _print_benchmark_series_calibrated(
    *,
    series_id: str,
    include_models: bool,
    baseline_profile_ids: tuple[str, ...],
    cross_encoder_models: tuple[str, ...],
    typed_decision_cross_encoder_models: tuple[str, ...],
    reranker_models: tuple[str, ...],
    cohere_reranker_models: tuple[str, ...],
    trust_remote_code: bool,
    reranker_score_cache_jsonl: Path | None,
    reranker_score_batch_size: int | None,
    include_details: bool,
) -> None:
    series = benchmark_series_by_id(series_id)
    payload = _calibrated_series_payload(
        series=series,
        include_models=include_models,
        baseline_profile_ids=baseline_profile_ids,
        cross_encoder_models=cross_encoder_models,
        typed_decision_cross_encoder_models=typed_decision_cross_encoder_models,
        reranker_models=reranker_models,
        cohere_reranker_models=cohere_reranker_models,
        trust_remote_code=trust_remote_code,
        reranker_score_cache_jsonl=reranker_score_cache_jsonl,
        reranker_score_batch_size=reranker_score_batch_size,
        include_details=include_details,
    )
    print(json.dumps(payload, indent=2))


def _print_benchmark_reviewed_series(
    *,
    source_dir: Path,
    reviewed_jsonl: Path,
    series_id: str,
    series_name: str | None,
    split: str,
    include_models: bool,
    baseline_profile_ids: tuple[str, ...],
    cross_encoder_models: tuple[str, ...],
    typed_decision_cross_encoder_models: tuple[str, ...],
    reranker_models: tuple[str, ...],
    cohere_reranker_models: tuple[str, ...],
    trust_remote_code: bool,
    reranker_support_threshold: float | None,
    reranker_score_cache_jsonl: Path | None,
    reranker_score_batch_size: int | None,
    include_details: bool,
    include_calibration: bool,
) -> None:
    result = load_reviewed_series_jsonl(
        reviewed_jsonl=reviewed_jsonl,
        source_dir=source_dir,
        series_id=series_id,
        series_name=series_name,
    )
    selected_split = None if split == "all" else BenchmarkSplit(split)
    suites = benchmark_series_suites(result.series, split=selected_split)
    audit = audit_benchmark_series(result.series)
    payload: dict[str, object] = {
        "series_id": result.series.id,
        "name": result.series.name,
        "metadata": dict(result.series.metadata),
        "split_counts": result.split_counts,
        "document_count": result.document_count,
        "audit": _benchmark_series_audit_payload(audit),
        "splits": {
            suite.metadata["split"]: _suite_payload(
                suite=suite,
                include_models=include_models,
                baseline_profile_ids=baseline_profile_ids,
                cross_encoder_models=cross_encoder_models,
                typed_decision_cross_encoder_models=typed_decision_cross_encoder_models,
                reranker_models=reranker_models,
                cohere_reranker_models=cohere_reranker_models,
                trust_remote_code=trust_remote_code,
                reranker_support_threshold=reranker_support_threshold,
                reranker_score_cache_jsonl=reranker_score_cache_jsonl,
                reranker_score_batch_size=reranker_score_batch_size,
                include_details=include_details,
                include_calibration=include_calibration,
            )
            for suite in suites
        },
    }
    print(json.dumps(payload, indent=2))


def _print_benchmark_reviewed_series_calibrated(
    *,
    source_dir: Path,
    reviewed_jsonl: Path,
    series_id: str,
    series_name: str | None,
    include_models: bool,
    baseline_profile_ids: tuple[str, ...],
    cross_encoder_models: tuple[str, ...],
    typed_decision_cross_encoder_models: tuple[str, ...],
    reranker_models: tuple[str, ...],
    cohere_reranker_models: tuple[str, ...],
    trust_remote_code: bool,
    reranker_score_cache_jsonl: Path | None,
    reranker_score_batch_size: int | None,
    include_details: bool,
) -> None:
    result = load_reviewed_series_jsonl(
        reviewed_jsonl=reviewed_jsonl,
        source_dir=source_dir,
        series_id=series_id,
        series_name=series_name,
    )
    payload = _calibrated_series_payload(
        series=result.series,
        include_models=include_models,
        baseline_profile_ids=baseline_profile_ids,
        cross_encoder_models=cross_encoder_models,
        typed_decision_cross_encoder_models=typed_decision_cross_encoder_models,
        reranker_models=reranker_models,
        cohere_reranker_models=cohere_reranker_models,
        trust_remote_code=trust_remote_code,
        reranker_score_cache_jsonl=reranker_score_cache_jsonl,
        reranker_score_batch_size=reranker_score_batch_size,
        include_details=include_details,
    )
    payload["split_counts"] = result.split_counts
    payload["document_count"] = result.document_count
    print(json.dumps(payload, indent=2))


def _calibrated_series_payload(
    *,
    series: BenchmarkSeries,
    include_models: bool,
    baseline_profile_ids: tuple[str, ...],
    cross_encoder_models: tuple[str, ...],
    typed_decision_cross_encoder_models: tuple[str, ...],
    reranker_models: tuple[str, ...],
    cohere_reranker_models: tuple[str, ...],
    trust_remote_code: bool,
    reranker_score_cache_jsonl: Path | None,
    reranker_score_batch_size: int | None,
    include_details: bool,
) -> dict[str, object]:
    dev_suite = series.splits[BenchmarkSplit.DEV]
    test_suite = series.splits[BenchmarkSplit.TEST]
    typed_first_stage_support_threshold = (
        calibrate_default_first_stage_support_threshold(dev_suite)
        if typed_decision_cross_encoder_models
        else None
    )
    specs, unavailable = _experiment_specs_for_options(
        include_models=include_models,
        baseline_profile_ids=baseline_profile_ids,
        cross_encoder_models=cross_encoder_models,
        typed_decision_cross_encoder_models=typed_decision_cross_encoder_models,
        reranker_models=reranker_models,
        cohere_reranker_models=cohere_reranker_models,
        trust_remote_code=trust_remote_code,
        reranker_support_threshold=None,
        reranker_score_cache_jsonl=reranker_score_cache_jsonl,
        reranker_score_batch_size=reranker_score_batch_size,
        typed_decision_first_stage_support_threshold=typed_first_stage_support_threshold,
    )
    experiments = run_calibrated_holdout(
        specs,
        dev_suite=dev_suite,
        test_suite=test_suite,
    )
    audit = audit_benchmark_series(series)
    payload: dict[str, object] = {
        "series_id": series.id,
        "name": series.name,
        "metadata": dict(series.metadata),
        "calibration_split": BenchmarkSplit.DEV.value,
        "evaluation_split": BenchmarkSplit.TEST.value,
        "audit": _benchmark_series_audit_payload(audit),
        "experiments": {
            experiment.name: _calibrated_experiment_payload(
                experiment,
                include_details=include_details,
            )
            for experiment in experiments
        },
    }
    if typed_first_stage_support_threshold is not None:
        payload["typed_first_stage_support_threshold"] = typed_first_stage_support_threshold
    if unavailable:
        payload["unavailable"] = unavailable
    return payload


def _suite_payload(
    *,
    suite: BenchmarkSuite,
    include_models: bool,
    baseline_profile_ids: tuple[str, ...],
    cross_encoder_models: tuple[str, ...],
    typed_decision_cross_encoder_models: tuple[str, ...],
    reranker_models: tuple[str, ...],
    cohere_reranker_models: tuple[str, ...],
    trust_remote_code: bool,
    reranker_support_threshold: float | None,
    reranker_score_cache_jsonl: Path | None,
    reranker_score_batch_size: int | None,
    include_details: bool,
    include_calibration: bool,
) -> dict[str, object]:
    (
        include_models,
        cross_encoder_models,
        reranker_models,
        cohere_reranker_models,
    ) = _resolve_baseline_options(
        include_models=include_models,
        baseline_profile_ids=baseline_profile_ids,
        cross_encoder_models=cross_encoder_models,
        reranker_models=reranker_models,
        cohere_reranker_models=cohere_reranker_models,
    )
    typed_decision_cross_encoder_models = _dedupe(typed_decision_cross_encoder_models)
    experiments = list(run_readme_baselines(suite, calibrate=include_calibration))
    unavailable: dict[str, str] = {}
    if include_models:
        try:
            experiments.extend(
                run_readme_sentence_transformer_baselines(
                    suite,
                    calibrate=include_calibration,
                    score_cache_path=reranker_score_cache_jsonl,
                    score_batch_size=reranker_score_batch_size,
                )
            )
        except RuntimeError as exc:
            unavailable["sentence_transformers"] = str(exc)
    for model_name in cross_encoder_models:
        try:
            experiments.append(
                run_readme_cross_encoder_baseline(
                    suite,
                    model_name=model_name,
                    support_threshold=reranker_support_threshold,
                    trust_remote_code=trust_remote_code,
                    calibrate=include_calibration,
                    score_cache_path=reranker_score_cache_jsonl,
                    score_batch_size=reranker_score_batch_size,
                )
            )
        except RuntimeError as exc:
            unavailable[f"sentence_transformer_cross_encoder:{model_name}"] = str(exc)
    for model_name in typed_decision_cross_encoder_models:
        try:
            experiments.append(
                run_experiment_spec(
                    typed_decision_cross_encoder_spec(
                        model_name=model_name,
                        trust_remote_code=trust_remote_code,
                        score_cache_path=reranker_score_cache_jsonl,
                        score_batch_size=reranker_score_batch_size,
                    ),
                    suite,
                    calibrate=include_calibration,
                )
            )
        except RuntimeError as exc:
            unavailable[f"typed_decision_cross_encoder:{model_name}"] = str(exc)
    for model_name in reranker_models:
        try:
            experiments.append(
                run_readme_transformers_reranker_baseline(
                    suite,
                    model_name=model_name,
                    support_threshold=reranker_support_threshold,
                    trust_remote_code=trust_remote_code,
                    calibrate=include_calibration,
                    score_cache_path=reranker_score_cache_jsonl,
                    score_batch_size=reranker_score_batch_size,
                )
            )
        except RuntimeError as exc:
            unavailable[f"transformers_reranker:{model_name}"] = str(exc)
    for model_name in cohere_reranker_models:
        try:
            experiments.append(
                run_readme_cohere_reranker_baseline(
                    suite,
                    model_name=model_name,
                    support_threshold=reranker_support_threshold,
                    calibrate=include_calibration,
                )
            )
        except RuntimeError as exc:
            unavailable[f"external_reranker:cohere:{model_name}"] = str(exc)
    payload: dict[str, object] = {
        experiment.name: _experiment_payload(
            experiment,
            include_details=include_details,
            include_calibration=include_calibration,
        )
        for experiment in experiments
    }
    if unavailable:
        payload["unavailable"] = unavailable
    return payload


def _experiment_specs_for_options(
    *,
    include_models: bool,
    baseline_profile_ids: tuple[str, ...],
    cross_encoder_models: tuple[str, ...],
    typed_decision_cross_encoder_models: tuple[str, ...],
    reranker_models: tuple[str, ...],
    cohere_reranker_models: tuple[str, ...],
    trust_remote_code: bool,
    reranker_support_threshold: float | None,
    reranker_score_cache_jsonl: Path | None,
    reranker_score_batch_size: int | None,
    typed_decision_first_stage_support_threshold: float | None = None,
) -> tuple[tuple[ExtractorExperimentSpec, ...], dict[str, str]]:
    (
        include_models,
        cross_encoder_models,
        reranker_models,
        cohere_reranker_models,
    ) = _resolve_baseline_options(
        include_models=include_models,
        baseline_profile_ids=baseline_profile_ids,
        cross_encoder_models=cross_encoder_models,
        reranker_models=reranker_models,
        cohere_reranker_models=cohere_reranker_models,
    )
    typed_decision_cross_encoder_models = _dedupe(typed_decision_cross_encoder_models)
    specs = list(readme_baseline_specs())
    unavailable: dict[str, str] = {}
    if include_models:
        try:
            specs.extend(
                sentence_transformer_baseline_specs(
                    score_cache_path=reranker_score_cache_jsonl,
                    score_batch_size=reranker_score_batch_size,
                )
            )
        except RuntimeError as exc:
            unavailable["sentence_transformers"] = str(exc)
    for model_name in cross_encoder_models:
        try:
            specs.append(
                cross_encoder_baseline_spec(
                    model_name=model_name,
                    support_threshold=reranker_support_threshold,
                    trust_remote_code=trust_remote_code,
                    score_cache_path=reranker_score_cache_jsonl,
                    score_batch_size=reranker_score_batch_size,
                )
            )
        except RuntimeError as exc:
            unavailable[f"sentence_transformer_cross_encoder:{model_name}"] = str(exc)
    for model_name in typed_decision_cross_encoder_models:
        try:
            specs.append(
                typed_decision_cross_encoder_spec(
                    model_name=model_name,
                    first_stage_support_threshold=(typed_decision_first_stage_support_threshold),
                    trust_remote_code=trust_remote_code,
                    score_cache_path=reranker_score_cache_jsonl,
                    score_batch_size=reranker_score_batch_size,
                )
            )
        except RuntimeError as exc:
            unavailable[f"typed_decision_cross_encoder:{model_name}"] = str(exc)
    for model_name in reranker_models:
        try:
            specs.append(
                transformers_reranker_baseline_spec(
                    model_name=model_name,
                    support_threshold=reranker_support_threshold,
                    trust_remote_code=trust_remote_code,
                    score_cache_path=reranker_score_cache_jsonl,
                    score_batch_size=reranker_score_batch_size,
                )
            )
        except RuntimeError as exc:
            unavailable[f"transformers_reranker:{model_name}"] = str(exc)
    for model_name in cohere_reranker_models:
        try:
            specs.append(
                cohere_reranker_baseline_spec(
                    model_name=model_name,
                    support_threshold=reranker_support_threshold,
                )
            )
        except RuntimeError as exc:
            unavailable[f"external_reranker:cohere:{model_name}"] = str(exc)
    return tuple(specs), unavailable


def _resolve_baseline_options(
    *,
    include_models: bool,
    baseline_profile_ids: tuple[str, ...],
    cross_encoder_models: tuple[str, ...],
    reranker_models: tuple[str, ...],
    cohere_reranker_models: tuple[str, ...],
) -> tuple[bool, tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    profiles = tuple(baseline_profile_by_id(profile_id) for profile_id in baseline_profile_ids)
    include_sentence_transformers = include_models or any(
        profile.include_sentence_transformer_baselines for profile in profiles
    )
    profile_cross_encoders = tuple(
        model for profile in profiles for model in profile.cross_encoder_models
    )
    profile_transformers_rerankers = tuple(
        model for profile in profiles for model in profile.transformers_reranker_models
    )
    profile_cohere_rerankers = tuple(
        model for profile in profiles for model in profile.cohere_reranker_models
    )
    return (
        include_sentence_transformers,
        _dedupe((*cross_encoder_models, *profile_cross_encoders)),
        _dedupe((*reranker_models, *profile_transformers_rerankers)),
        _dedupe((*cohere_reranker_models, *profile_cohere_rerankers)),
    )


def _baseline_profile_ids() -> tuple[str, ...]:
    return tuple(profile.id for profile in baseline_profiles())


def _add_reranker_score_cache_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--reranker-score-cache-jsonl",
        type=Path,
        default=None,
        help="Optional persistent JSONL cache for raw pair reranker scores.",
    )
    parser.add_argument(
        "--reranker-score-batch-size",
        type=int,
        default=None,
        help="Optional batch size for uncached pair reranker scoring calls.",
    )


def _dedupe(values: tuple[str, ...]) -> tuple[str, ...]:
    seen: set[str] = set()
    deduped: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        deduped.append(value)
    return tuple(deduped)


def _experiment_payload(
    experiment: RankerExperiment,
    *,
    include_details: bool,
    include_calibration: bool,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "case_count": experiment.report.case_count,
        "support_case_count": experiment.report.support_case_count,
        "abstain_case_count": experiment.report.abstain_case_count,
        "mean_reciprocal_rank": experiment.report.mean_reciprocal_rank,
        "recall_at_1": experiment.report.recall_at_1,
        "recall_at_3": experiment.report.recall_at_3,
        "recall_at_5": experiment.report.recall_at_5,
        "top1_support_accuracy": experiment.report.top1_support_accuracy,
        "abstain_accuracy": experiment.report.abstain_accuracy,
        "decision_accuracy": experiment.report.decision_accuracy,
        "diagnostic_top1_accuracy": experiment.report.diagnostic_top1_accuracy,
        "abstain_diagnostic_top1_accuracy": (
            experiment.report.abstain_diagnostic_top1_accuracy
        ),
        "forbidden_top1_rate": experiment.report.forbidden_top1_rate,
        "forbidden_supported_top1_rate": experiment.report.forbidden_supported_top1_rate,
        "negative_label_reports": _negative_label_reports_payload(
            experiment.report.negative_label_reports
        ),
        "elapsed_ms": experiment.elapsed_ms,
    }
    if include_details:
        payload["diagnostics"] = [
            _case_evaluation_payload(case)
            for case in experiment.cases
            if _should_include_case_diagnostic(case)
        ]
    if include_calibration and experiment.calibration is not None:
        payload["calibration"] = {
            "best": _calibration_point_payload(experiment.calibration.best),
            "points": [
                _calibration_point_payload(point) for point in experiment.calibration.points
            ],
        }
    return payload


def _calibrated_experiment_payload(
    experiment: CalibratedHoldoutExperiment,
    *,
    include_details: bool,
) -> dict[str, object]:
    if experiment.dev.calibration is None:
        raise ValueError("calibrated holdout experiment must include dev calibration")
    return {
        "decision_policy": experiment.decision_policy.value,
        "selected_threshold": experiment.selected_threshold,
        "dev_selected": _calibration_point_payload(experiment.dev.calibration.best),
        "dev": _experiment_payload(
            experiment.dev,
            include_details=include_details,
            include_calibration=True,
        ),
        "test": _experiment_payload(
            experiment.test,
            include_details=include_details,
            include_calibration=False,
        ),
    }


def _should_include_case_diagnostic(case: CaseEvaluation) -> bool:
    return (
        not case.decision_correct
        or case.forbidden_supported_top1
        or any(evaluation.supported_top1 for evaluation in case.negative_label_evaluations)
        or (bool(case.support_span_ids) and not case.top1_is_support)
    )


def _case_evaluation_payload(case: CaseEvaluation) -> dict[str, object]:
    return {
        "case_id": case.case_id,
        "document_id": case.document_id,
        "expect_abstain": case.expect_abstain,
        "support_span_ids": list(case.support_span_ids),
        "forbidden_span_ids": list(case.forbidden_span_ids),
        "matched_span_ids": list(case.matched_span_ids),
        "top_span_id": case.top_span_id,
        "top_label": case.top_label,
        "top_score": case.top_score,
        "first_support_rank": case.first_support_rank,
        "decision_correct": case.decision_correct,
        "diagnostic_top1": case.diagnostic_top1,
        "forbidden_top1": case.forbidden_top1,
        "forbidden_supported_top1": case.forbidden_supported_top1,
        "negative_label_evaluations": [
            {
                "label": evaluation.label.value,
                "span_ids": list(evaluation.span_ids),
                "top1": evaluation.top1,
                "labeled_top1": evaluation.labeled_top1,
                "supported_top1": evaluation.supported_top1,
            }
            for evaluation in case.negative_label_evaluations
            if evaluation.span_ids
        ],
        "top5_span_ids": list(case.ranked_span_ids[:5]),
    }


def _calibration_point_payload(point: CalibrationPoint) -> dict[str, object]:
    return {
        "threshold": point.threshold,
        "case_count": point.report.case_count,
        "top1_support_accuracy": point.report.top1_support_accuracy,
        "abstain_accuracy": point.report.abstain_accuracy,
        "decision_accuracy": point.report.decision_accuracy,
        "diagnostic_top1_accuracy": point.report.diagnostic_top1_accuracy,
        "abstain_diagnostic_top1_accuracy": point.report.abstain_diagnostic_top1_accuracy,
        "forbidden_supported_top1_rate": point.report.forbidden_supported_top1_rate,
        "negative_label_reports": _negative_label_reports_payload(
            point.report.negative_label_reports
        ),
    }


def _negative_label_reports_payload(
    reports: tuple[NegativeLabelReport, ...],
) -> dict[str, object]:
    return {
        report.label.value: {
            "case_count": report.case_count,
            "span_label_count": report.span_label_count,
            "top1_rate": report.top1_rate,
            "labeled_top1_rate": report.labeled_top1_rate,
            "supported_top1_rate": report.supported_top1_rate,
        }
        for report in reports
    }


def _print_benchmark_series_audit(*, series_id: str) -> None:
    series = benchmark_series_by_id(series_id)
    audit = (
        audit_expanded_domain_series(series)
        if series.id == "domain_evidence_benchmark_v4"
        else audit_benchmark_series(series)
    )
    print(json.dumps(_benchmark_series_audit_payload(audit), indent=2))


def _print_benchmark_competitors(
    *,
    series_id: str,
    split: str,
    competitor_ids: tuple[str, ...],
    include_hosted: bool,
    output_format: str,
    trust_remote_code: bool,
    score_cache_jsonl: Path | None,
    score_batch_size: int | None,
) -> None:
    payload = competitor_evaluation_payload(
        series=benchmark_series_by_id(series_id),
        split=split,
        competitor_ids=competitor_ids,
        include_hosted=include_hosted,
        registry=competitor_registry(),
        trust_remote_code=trust_remote_code,
        score_cache_path=score_cache_jsonl,
        score_batch_size=score_batch_size,
    )
    if output_format == "markdown":
        print(competitor_markdown_report(payload))
    else:
        print(json.dumps(payload, indent=2))


def _print_readme_sources() -> None:
    print(
        json.dumps(
            {
                "source_count": len(readme_source_catalog()),
                "sources": [_readme_source_payload(source) for source in readme_source_catalog()],
            },
            indent=2,
        )
    )


def _print_fetch_readme_sources(
    *,
    output_dir: Path,
    source_ids: tuple[str, ...],
    limit: int | None,
) -> None:
    sources = _selected_readme_sources(source_ids=source_ids, limit=limit)
    fetched = fetch_readme_sources(sources)
    manifest = write_fetched_readmes(fetched, output_dir)
    print(json.dumps(manifest, indent=2, sort_keys=True))


def _print_prepare_curation_queue(
    *,
    input_dir: Path,
    output_jsonl: Path,
    max_items_per_doc: int,
) -> None:
    items = curation_queue_from_fetched_readmes(
        input_dir,
        max_items_per_doc=max_items_per_doc,
    )
    write_curation_queue_jsonl(items, output_jsonl)
    print(
        json.dumps(
            {
                "input_dir": str(input_dir),
                "output_jsonl": str(output_jsonl),
                "document_count": len({item.document_id for item in items}),
                "item_count": len(items),
                "review_status": "unreviewed",
            },
            indent=2,
        )
    )


def _print_prepare_document_curation_queue(
    *,
    input_path: Path,
    output_jsonl: Path,
    patterns: tuple[str, ...],
    max_items_per_doc: int,
    require_heading: bool,
) -> None:
    items = curation_queue_from_local_documents(
        input_path,
        patterns=patterns or ("*.md", "*.markdown", "*.rst", "*.txt"),
        max_items_per_doc=max_items_per_doc,
        require_heading=require_heading,
    )
    write_curation_queue_jsonl(items, output_jsonl)
    print(
        json.dumps(
            {
                "input": str(input_path),
                "output_jsonl": str(output_jsonl),
                "document_count": len({item.document_id for item in items}),
                "item_count": len(items),
                "review_status": "unreviewed",
            },
            indent=2,
        )
    )


def _print_audit_curation_queue(
    *,
    queue_jsonl: Path,
    min_documents: int,
    min_items: int,
    max_document_share: float,
) -> None:
    audit = audit_curation_queue_jsonl(
        queue_jsonl=queue_jsonl,
        min_documents=min_documents,
        min_items=min_items,
        max_document_share=max_document_share,
    )
    print(json.dumps(_curation_queue_audit_payload(audit), indent=2))


def _print_prepare_reviewed_case_template(
    *,
    queue_jsonl: Path,
    output_jsonl: Path,
    split: BenchmarkSplit,
) -> None:
    row_count = write_reviewed_case_template_jsonl(
        queue_jsonl=queue_jsonl,
        output_jsonl=output_jsonl,
        split=split,
    )
    print(
        json.dumps(
            {
                "queue_jsonl": str(queue_jsonl),
                "output_jsonl": str(output_jsonl),
                "row_count": row_count,
                "review_status": "unreviewed",
            },
            indent=2,
        )
    )


def _print_prepare_reviewed_series_template(
    *,
    queue_jsonl: Path,
    output_jsonl: Path,
    train_ratio: float,
    dev_ratio: float,
    test_ratio: float,
    seed: str,
) -> None:
    split_counts = write_reviewed_series_template_jsonl(
        queue_jsonl=queue_jsonl,
        output_jsonl=output_jsonl,
        train_ratio=train_ratio,
        dev_ratio=dev_ratio,
        test_ratio=test_ratio,
        seed=seed,
    )
    print(
        json.dumps(
            {
                "queue_jsonl": str(queue_jsonl),
                "output_jsonl": str(output_jsonl),
                "split_counts": split_counts,
                "split_unit": "document",
                "review_status": "unreviewed",
            },
            indent=2,
        )
    )


def _print_validate_reviewed_cases(
    *,
    source_dir: Path,
    reviewed_jsonl: Path,
    suite_id: str,
    suite_name: str | None,
) -> None:
    result = load_reviewed_cases_jsonl(
        reviewed_jsonl=reviewed_jsonl,
        source_dir=source_dir,
        suite_id=suite_id,
        suite_name=suite_name,
    )
    audit = audit_benchmark_suite(result.suite)
    payload = _benchmark_suite_audit_payload(audit)
    payload["split_counts"] = result.split_counts
    payload["document_count"] = result.document_count
    print(json.dumps(payload, indent=2))


def _print_validate_reviewed_series(
    *,
    source_dir: Path,
    reviewed_jsonl: Path,
    series_id: str,
    series_name: str | None,
) -> None:
    result = load_reviewed_series_jsonl(
        reviewed_jsonl=reviewed_jsonl,
        source_dir=source_dir,
        series_id=series_id,
        series_name=series_name,
    )
    audit = audit_benchmark_series(result.series)
    payload = _benchmark_series_audit_payload(audit)
    payload["split_counts"] = result.split_counts
    payload["document_count"] = result.document_count
    print(json.dumps(payload, indent=2))


def _print_explain_reviewed_case(
    *,
    source_dir: Path,
    reviewed_jsonl: Path,
    series_id: str,
    case_id: str,
    support_threshold: float | None,
    candidate_limit: int,
) -> None:
    if candidate_limit < 1:
        raise ValueError("candidate-limit must be positive")
    result = load_reviewed_series_jsonl(
        reviewed_jsonl=reviewed_jsonl,
        source_dir=source_dir,
        series_id=series_id,
    )
    cases = {case.id: case for case in result.series.cases}
    try:
        case = cases[case_id]
    except KeyError as exc:
        raise ValueError(f"unknown reviewed case id {case_id!r}") from exc
    intent = (
        case.intent
        if support_threshold is None
        else replace(case.intent, min_support_score=support_threshold)
    )
    extraction = EvidenceExtractor.default().extract(
        intent,
        case.document,
        candidate_limit=candidate_limit,
    )
    payload = {
        "case_id": case.id,
        "document_id": case.document.id,
        "expect_abstain": case.expect_abstain,
        "intent": {
            "label": case.intent.label,
            "description": case.intent.description,
            "positive_examples": list(case.intent.positive_examples),
            "min_support_score": intent.min_support_score,
        },
        "gold": {
            "support_span_ids": list(case.support_span_ids),
            "near_miss_span_ids": list(case.near_miss_span_ids),
            "contradiction_span_ids": list(case.contradiction_span_ids),
            "insufficient_context_span_ids": list(case.insufficient_context_span_ids),
            "forbidden_span_ids": list(case.forbidden_span_ids),
        },
        "abstained": extraction.abstained,
        "matched_span_ids": [match.span.id for match in extraction.matches],
        "trace": list(extraction.trace),
        "candidates": [
            {
                "span_id": match.span.id,
                "kind": match.span.kind.value,
                "heading_path": list(match.span.heading_path),
                "score": match.score,
                "label": match.label.value,
                "selected": any(
                    selected.span.id == match.span.id for selected in extraction.matches
                ),
                "gold_labels": _gold_labels_for_span(case, match.span.id),
                "features": dict(match.features),
                "reasons": list(match.reasons),
                "text": match.span.text,
            }
            for match in extraction.candidates
        ],
    }
    print(json.dumps(payload, indent=2))


def _print_failure_report_reviewed_series(
    *,
    source_dir: Path,
    reviewed_jsonl: Path,
    series_id: str,
    series_name: str | None,
    split: str,
    include_models: bool,
    baseline_profile_ids: tuple[str, ...],
    cross_encoder_models: tuple[str, ...],
    typed_decision_cross_encoder_models: tuple[str, ...],
    reranker_models: tuple[str, ...],
    cohere_reranker_models: tuple[str, ...],
    trust_remote_code: bool,
    reranker_support_threshold: float | None,
    reranker_score_cache_jsonl: Path | None,
    reranker_score_batch_size: int | None,
    apply_dev_calibration: bool,
    experiment_names: tuple[str, ...],
) -> None:
    result = load_reviewed_series_jsonl(
        reviewed_jsonl=reviewed_jsonl,
        source_dir=source_dir,
        series_id=series_id,
        series_name=series_name,
    )
    suite = result.series.splits[BenchmarkSplit(split)]
    typed_first_stage_support_threshold = (
        calibrate_default_first_stage_support_threshold(result.series.splits[BenchmarkSplit.DEV])
        if apply_dev_calibration and typed_decision_cross_encoder_models
        else None
    )
    specs, unavailable = _experiment_specs_for_options(
        include_models=include_models,
        baseline_profile_ids=baseline_profile_ids,
        cross_encoder_models=cross_encoder_models,
        typed_decision_cross_encoder_models=typed_decision_cross_encoder_models,
        reranker_models=reranker_models,
        cohere_reranker_models=cohere_reranker_models,
        trust_remote_code=trust_remote_code,
        reranker_support_threshold=reranker_support_threshold,
        reranker_score_cache_jsonl=reranker_score_cache_jsonl,
        reranker_score_batch_size=reranker_score_batch_size,
        typed_decision_first_stage_support_threshold=typed_first_stage_support_threshold,
    )
    specs = _filter_experiment_specs(specs, experiment_names)
    if apply_dev_calibration:
        holdouts = run_calibrated_holdout(
            specs,
            dev_suite=result.series.splits[BenchmarkSplit.DEV],
            test_suite=suite,
        )
        experiments = tuple(holdout.test for holdout in holdouts)
        experiment_metadata = {
            holdout.name: {
                "decision_policy": holdout.decision_policy.value,
                "selected_threshold": holdout.selected_threshold,
            }
            for holdout in holdouts
        }
    else:
        experiments = run_experiment_specs(specs, suite, calibrate=False)
        experiment_metadata = {
            experiment.name: {
                "decision_policy": spec.decision_policy.value,
                "selected_threshold": None,
            }
            for experiment, spec in zip(experiments, specs, strict=True)
        }
    payload: dict[str, object] = {
        "series_id": result.series.id,
        "name": result.series.name,
        "split": split,
        "dev_calibrated": apply_dev_calibration,
        "typed_first_stage_support_threshold": typed_first_stage_support_threshold,
        "experiments": {
            experiment.name: {
                **experiment_metadata[experiment.name],
                "summary": _experiment_payload(
                    experiment,
                    include_details=False,
                    include_calibration=False,
                ),
                "failure_report": build_failure_report(
                    suite,
                    experiment.cases,
                ).to_json_dict(),
            }
            for experiment in experiments
        },
    }
    if unavailable:
        payload["unavailable"] = unavailable
    print(json.dumps(payload, indent=2))


def _filter_experiment_specs(
    specs: tuple[ExtractorExperimentSpec, ...],
    experiment_names: tuple[str, ...],
) -> tuple[ExtractorExperimentSpec, ...]:
    if not experiment_names:
        return specs
    requested = set(experiment_names)
    selected = tuple(spec for spec in specs if spec.name in requested)
    missing = requested - {spec.name for spec in selected}
    if missing:
        known = ", ".join(spec.name for spec in specs)
        raise ValueError(
            f"unknown experiment name(s): {', '.join(sorted(missing))}. Known: {known}"
        )
    return selected


def _print_model_judge_rubric() -> None:
    print(
        json.dumps(
            {
                "rubric": DEFAULT_MODEL_JUDGE_RUBRIC.to_json_dict(),
                "response_schema": model_judge_response_schema(),
            },
            indent=2,
            sort_keys=True,
        )
    )


def _print_export_model_judge_requests(
    *,
    series_id: str,
    split: str,
    output_jsonl: Path,
    max_requests: int | None,
    include_unlabeled_forbidden: bool,
) -> None:
    if max_requests is not None and max_requests < 1:
        raise ValueError("max-requests must be positive when provided")
    series = benchmark_series_by_id(series_id)
    selected_split = None if split == "all" else BenchmarkSplit(split)
    suites = benchmark_series_suites(series, split=selected_split)
    requests = tuple(
        request
        for suite in suites
        for request in model_judge_requests_from_suite(
            suite,
            include_unlabeled_forbidden=include_unlabeled_forbidden,
        )
    )
    requests = requests if max_requests is None else requests[:max_requests]
    write_model_judge_requests_jsonl(requests, output_jsonl)
    print(
        json.dumps(
            {
                "series_id": series.id,
                "split": split,
                "output_jsonl": str(output_jsonl),
                "request_count": len(requests),
                "rubric": DEFAULT_MODEL_JUDGE_RUBRIC.to_json_dict(),
            },
            indent=2,
            sort_keys=True,
        )
    )


def _print_convert_model_judge_responses(
    *,
    request_jsonl: Path,
    response_jsonl: Path,
    output_jsonl: Path,
    judge_model: str,
    pass_id: str,
) -> None:
    decisions = decisions_from_model_response_jsonl(
        response_jsonl=response_jsonl,
        request_jsonl=request_jsonl,
        judge_model=judge_model,
        pass_id=pass_id,
    )
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    output_jsonl.write_text(
        "\n".join(json.dumps(decision.to_json_dict(), sort_keys=True) for decision in decisions)
        + ("\n" if decisions else ""),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "request_jsonl": str(request_jsonl),
                "response_jsonl": str(response_jsonl),
                "output_jsonl": str(output_jsonl),
                "decision_count": len(decisions),
                "judge_model": judge_model,
                "pass_id": pass_id,
            },
            indent=2,
            sort_keys=True,
        )
    )


def _print_audit_model_judge_decisions(
    *,
    decisions_jsonl: Path,
    min_passes: int,
) -> None:
    decisions = read_model_judge_decisions_jsonl(decisions_jsonl)
    report = model_judge_agreement_report(decisions, min_passes=min_passes)
    print(json.dumps(report.to_json_dict(), indent=2, sort_keys=True))


def _print_modal_reranker_probes() -> None:
    print(
        json.dumps(
            {
                "probes": [probe.to_json_dict() for probe in modal_reranker_probes()],
            },
            indent=2,
            sort_keys=True,
        )
    )


def _gold_labels_for_span(case: BenchmarkCase, span_id: str) -> list[str]:
    labels: list[str] = []
    if span_id in case.support_span_ids:
        labels.append("support")
    if span_id in case.near_miss_span_ids:
        labels.append("near_miss")
    if span_id in case.contradiction_span_ids:
        labels.append("contradiction")
    if span_id in case.insufficient_context_span_ids:
        labels.append("insufficient_context")
    if span_id in case.forbidden_span_ids:
        labels.append("forbidden")
    return labels


def _selected_readme_sources(
    *,
    source_ids: tuple[str, ...],
    limit: int | None,
) -> tuple[ReadmeSource, ...]:
    if limit is not None and limit < 1:
        raise ValueError("limit must be positive when provided")
    sources = (
        tuple(readme_source_by_id(source_id) for source_id in source_ids)
        if source_ids
        else readme_source_catalog()
    )
    return sources if limit is None else sources[:limit]


def _readme_source_payload(source: ReadmeSource) -> dict[str, str]:
    return {
        "id": source.id,
        "repository": source.repository,
        "raw_url": source.raw_url,
        "license_name": source.license_name,
        "focus": source.focus,
    }


def _benchmark_series_audit_payload(audit: BenchmarkSeriesAudit) -> dict[str, object]:
    return {
        "series_id": audit.series_id,
        "case_count": audit.case_count,
        "reviewed_case_count": audit.reviewed_case_count,
        "train_case_count": audit.train_case_count,
        "dev_case_count": audit.dev_case_count,
        "test_case_count": audit.test_case_count,
        "label_counts": audit.label_counts,
        "curation_source_counts": audit.curation_source_counts,
        "difficulty_counts": audit.difficulty_counts,
        "phenomenon_counts": audit.phenomenon_counts,
        "domain_counts": audit.domain_counts,
        "genre_counts": audit.genre_counts,
        "splits": {
            split: _benchmark_suite_audit_payload(split_audit)
            for split, split_audit in audit.split_audits.items()
        },
        "warnings": list(audit.warnings),
        "sota_ready": not audit.warnings,
    }


def _benchmark_suite_audit_payload(audit: BenchmarkSuiteAudit) -> dict[str, object]:
    return {
        "suite_id": audit.suite_id,
        "case_count": audit.case_count,
        "reviewed_case_count": audit.reviewed_case_count,
        "unreviewed_case_count": audit.unreviewed_case_count,
        "support_case_count": audit.support_case_count,
        "abstain_case_count": audit.abstain_case_count,
        "relation_case_count": audit.relation_case_count,
        "support_label_count": audit.support_label_count,
        "near_miss_label_count": audit.near_miss_label_count,
        "contradiction_label_count": audit.contradiction_label_count,
        "insufficient_context_label_count": audit.insufficient_context_label_count,
        "forbidden_label_count": audit.forbidden_label_count,
        "document_count": audit.document_count,
        "curation_source_counts": audit.curation_source_counts,
        "difficulty_counts": audit.difficulty_counts,
        "phenomenon_counts": audit.phenomenon_counts,
        "warnings": list(audit.warnings),
    }


def _curation_queue_audit_payload(audit: CurationQueueAudit) -> dict[str, object]:
    return {
        "item_count": audit.item_count,
        "document_count": audit.document_count,
        "kind_counts": audit.kind_counts,
        "document_counts": audit.document_counts,
        "empty_heading_count": audit.empty_heading_count,
        "duplicate_text_count": audit.duplicate_text_count,
        "min_text_chars": audit.min_text_chars,
        "max_text_chars": audit.max_text_chars,
        "average_text_chars": audit.average_text_chars,
        "warnings": list(audit.warnings),
        "review_queue_ready": not audit.warnings,
    }


def _print_training_jsonl(
    *,
    suite_id: str,
    series_id: str | None,
    split: str,
    negatives_per_case: int,
) -> None:
    if negatives_per_case < 0:
        raise ValueError("negatives-per-case must be non-negative")
    if series_id is not None:
        series = benchmark_series_by_id(series_id)
        selected_split = None if split == "all" else BenchmarkSplit(split)
        suites = benchmark_series_suites(series, split=selected_split)
    else:
        suites = (
            readme_benchmark_suites()
            if suite_id == "all"
            else (readme_benchmark_suite_by_id(suite_id),)
        )
    pairs = tuple(
        pair
        for suite in suites
        for pair in training_pairs_from_suite(
            EvidenceExtractor.default(),
            suite,
            negatives_per_case=negatives_per_case,
        )
    )
    print(training_pairs_jsonl(pairs))


def _print_reviewed_training_jsonl(
    *,
    source_dir: Path,
    reviewed_jsonl: Path,
    series_id: str,
    split: str,
    negatives_per_case: int,
) -> None:
    if negatives_per_case < 0:
        raise ValueError("negatives-per-case must be non-negative")
    result = load_reviewed_series_jsonl(
        reviewed_jsonl=reviewed_jsonl,
        source_dir=source_dir,
        series_id=series_id,
    )
    selected_split = None if split == "all" else BenchmarkSplit(split)
    suites = benchmark_series_suites(result.series, split=selected_split)
    pairs = tuple(
        pair
        for suite in suites
        for pair in training_pairs_from_suite(
            EvidenceExtractor.default(),
            suite,
            negatives_per_case=negatives_per_case,
        )
    )
    print(training_pairs_jsonl(pairs))
