# SOTA Implementation Event Log

Date: 2026-05-08
Repository: `/Users/yahorbarkouski/travercedb`
Implementation report: `sota-implementation-2026-05-08.md`

## Resume State

- Current phase: implementation complete; final quality gates and git review
  pending.
- Artifact paths:
  - `docs/sota-implementation-2026-05-08.md`
  - `docs/sota-implementation-2026-05-08-event-log.md`
- Last completed dimension: Modal larger-model probes.
- Next planned checks:
  1. Run full package quality gates.
  2. Review package-scoped diff.
  3. Commit evidence package changes without staging unrelated root files.
- Known blockers:
  - Existing unrelated dirty files outside `packages/gia-evidence-finder` must not
    be staged or reverted.
  - Jina-style models require explicit `trust_remote_code=True`; this pass may
    evaluate them only with a conscious command flag.
  - Public SOTA cannot be honestly claimed until non-README benchmark and model
    adjudication evidence is stronger.

## Timeline

### Step 001 - Initialize Scope

- Action: inspected root git status and evidence package file layout.
- Reason: avoid overwriting unrelated user changes and identify current
  package boundaries.
- Evidence inspected:
  - `git status --short`
  - `packages/gia-evidence-finder/src/gia_evidence_finder/*`
  - `packages/gia-evidence-finder/tests/*`
- Observations:
  - Root storage/semantic files are dirty from unrelated work.
  - `packages/gia-evidence-finder` is the intended write scope.
- Outcome: proceed with package-scoped implementation.

### Step 002 - Create Implementation Artifacts

- Action: created this event log and the companion implementation report.
- Reason: preserve exact goal, resume state, commands, and findings across
  context compaction.
- Evidence inspected:
  - Audit skill report/event-log templates.
- Outcome: future steps will append command results and implementation notes
  here.

### Step 003 - Add Non-README Benchmarks

- Action: added curated non-README document benchmark support.
- Reason: README-only evidence benchmarks are too narrow for SOTA claims.
- Files changed:
  - `src/gia_evidence_finder/document_benchmarks.py`
  - `src/gia_evidence_finder/benchmark_series.py`
  - `src/gia_evidence_finder/__init__.py`
  - `tests/test_document_benchmarks.py`
- Observations:
  - `non_readme_docs_v1` has 15 reviewed curated cases across 5 document
    genres: spec, runbook, release notes, and issue discussion.
  - `mixed_evidence_benchmark_v3` combines the existing README v2 split with
    non-README train/dev/test cases.
- Outcome: targeted package checks passed for the new benchmark and registry.

### Step 004 - Build Model Judgment Contract

- Action: added typed model-judge requests, rubric, response conversion, and
  agreement audit support.
- Reason: benchmark labels need reproducible adjudication instead of informal
  one-off LLM judgments.
- Files changed:
  - `src/gia_evidence_finder/model_judgment.py`
  - `src/gia_evidence_finder/cli.py`
  - `src/gia_evidence_finder/__init__.py`
  - `tests/test_model_judgment.py`
- Observations:
  - `export-model-judge-requests` produced 32 dev-split requests for
    `mixed_evidence_benchmark_v3`.
  - The rubric is typed and versioned as
    `gia_evidence_finder_span_judge` / `2026-05-08`.
- Outcome: targeted model-judgment tests passed.

### Step 005 - Add Cached And Batched Reranking

- Action: added persistent JSONL pair-score cache and cached reranker wrapper.
- Reason: larger reranker probes should not rescore the same query/span pairs
  across calibration and holdout variants.
- Files changed:
  - `src/gia_evidence_finder/reranker_cache.py`
  - `src/gia_evidence_finder/experiments.py`
  - `src/gia_evidence_finder/cli.py`
  - `scripts/modal_train_reranker.py`
  - `tests/test_reranker_cache.py`
- Observations:
  - Cache keys include the reranker model name plus query/document SHA-256
    hashes.
  - Modal holdout evaluation for each larger probe saw 16,898 cache hits,
    6,722 misses, 314 model calls, and 6,709 persisted scores while processing
    23,620 pairs.
- Outcome: cache contract tests, model baseline tests, ruff, and mypy passed.

### Step 006 - Promote Typed-Decision Reranking

- Action: exposed typed-decision cross-encoder reranking through CLI options
  and shared dev-calibrated first-stage threshold helper.
- Reason: the strongest architecture is deterministic typed support decisions
  plus learned ordering, so it must be runnable outside the Modal script.
- Files changed:
  - `src/gia_evidence_finder/experiments.py`
  - `src/gia_evidence_finder/cli.py`
  - `scripts/modal_train_reranker.py`
- Observations:
  - `benchmark-series-calibrated` can now run
    `--typed-decision-cross-encoder-model`.
  - Held-out typed-decision runs use the dev-calibrated first-stage threshold
    when invoked through calibrated series commands.
- Outcome: targeted extraction/training tests passed.

### Step 007 - Package Trainer Probes

- Action: added named Modal reranker probe presets and CLI listing.
- Reason: larger model experiments should be reproducible and comparable by
  probe id, not shell-history flags.
- Files changed:
  - `src/gia_evidence_finder/training_runs.py`
  - `scripts/modal_train_reranker.py`
  - `src/gia_evidence_finder/cli.py`
  - `tests/test_training_runs.py`
- Observations:
  - Packaged probes: `minilm_l6_reference`, `minilm_l12_larger`,
    `electra_base_larger`.
  - `uv run --extra training modal run scripts/modal_train_reranker.py
    --list-probes` completed successfully.
- Outcome: probe package tests passed.

### Step 008 - Run Mixed Benchmark Checks

- Action: audited and benchmarked `mixed_evidence_benchmark_v3`.
- Reason: the new non-README slice must prove it broadens coverage without
  silently becoming an easy benchmark.
- Observations:
  - Audit: 85 reviewed cases, 15 non-README cases, 47/19/19 train/dev/test.
  - Audit warnings remain: total reviewed cases below 200 and dev/test below
    50.
  - Local calibrated deterministic test result: intent-aware default reached
    MRR/R@1/R@3/top-1 support of 1.0, decision 0.9474, abstain 0.8750, and
    forbidden-supported 0.0526.
- Outcome: useful regression suite; not sufficient for public SOTA proof.

### Step 009 - Run Larger Modal Probes

- Action: trained and evaluated MiniLM-L12 and Electra-base on Modal H200.
- Reason: test whether a larger reranker improves the current calibrated
  typed-decision architecture.
- Artifacts:
  - `.tmp/modal-train-reranker-minilm-l12-larger-20260508-metrics.json`
  - `.tmp/modal-train-reranker-electra-base-20260508-metrics.json`
- Results:
  - MiniLM-L12 calibrated typed-decision test: MRR 0.9877, R@1 0.9753, R@3
    1.0000, top-1 support 0.9630, abstain 0.8571, decision 0.9886,
    forbidden-supported 0.0000.
  - Electra-base calibrated typed-decision test: MRR 0.9051, R@1 0.8272, R@3
    0.9877, top-1 support 0.8148, abstain 0.8571, decision 0.9886,
    forbidden-supported 0.0000.
- Outcome: MiniLM-L12 is the best current candidate; Electra-base is not.

## Command Log

| Step | Command | Purpose | Result | Notes |
| --- | --- | --- | --- | --- |
| 001 | `git status --short` | Inspect dirty worktree | Passed | Unrelated dirty root files only. |
| 003 | `uv run ruff check . && uv run mypy && uv run pytest -q tests/test_document_benchmarks.py tests/test_benchmark_series.py tests/test_benchmark_audit.py` | Validate non-README benchmark implementation | Passed | 8 targeted tests passed. |
| 004 | `uv run ruff check . && uv run mypy && uv run pytest -q tests/test_model_judgment.py tests/test_document_benchmarks.py tests/test_benchmark_series.py tests/test_benchmark_audit.py` | Validate model judgment and benchmark changes | Passed | 11 targeted tests passed. |
| 005 | `uv run pytest -q tests/test_reranker_cache.py tests/test_model_baselines.py tests/test_model_judgment.py tests/test_document_benchmarks.py tests/test_benchmark_series.py` | Validate pair-score cache and model baseline integration | Passed | 22 targeted tests passed. |
| 007 | `uv run ruff check . --fix && uv run mypy && uv run pytest -q tests/test_training_runs.py tests/test_reranker_cache.py tests/test_evaluation_training.py tests/test_model_baselines.py` | Validate trainer probe packaging | Passed | 27 targeted tests passed. |
| 008 | `uv run gia-evidence-finder audit-benchmark-series --series mixed_evidence_benchmark_v3` | Audit mixed README/non-README benchmark | Passed | Warnings remain for benchmark scale. |
| 008 | `uv run gia-evidence-finder benchmark-series-calibrated --series mixed_evidence_benchmark_v3` | Measure mixed benchmark calibrated baselines | Passed | Intent-aware default decision 0.9474 on test. |
| 004 | `uv run gia-evidence-finder export-model-judge-requests --series mixed_evidence_benchmark_v3 --split dev --output-jsonl .tmp/mixed-dev-model-judge-requests.jsonl` | Exercise model-judge export | Passed | 32 requests exported. |
| 009 | `uv run --extra training modal run scripts/modal_train_reranker.py --probe-id minilm_l12_larger ...` | Train/evaluate larger MiniLM probe on Modal H200 | Passed | Best current candidate. |
| 009 | `uv run --extra training modal run scripts/modal_train_reranker.py --probe-id electra_base_larger ...` | Train/evaluate Electra-base probe on Modal H200 | Passed | Worse ranking than MiniLM-L12. |
| 010 | `uv run ruff check .` | Final package lint | Passed | Package root. |
| 010 | `uv run mypy` | Final package type check | Passed | 46 source files. |
| 010 | `uv run pytest -q` | Final package test suite | Passed | 85 tests passed. |
| 010 | `uv run ruff check packages/gia-evidence-finder` | Root-invoked package lint | Passed | Confirms root uv can lint package scope. |

## Hypothesis Log

| ID | Hypothesis | Evidence For | Evidence Against | Status | Related Finding |
| --- | --- | --- | --- | --- | --- |
| H-001 | Current best architecture remains calibrated typed decision plus learned ordering. | MiniLM-L12 improves reviewed holdout ranking while preserving decision 0.9886 and forbidden-supported 0.0000. | Mixed v3 still shows benchmark-scale weakness and one forbidden-supported deterministic miss. | Confirmed | MiniLM-L12 is current candidate. |
| H-002 | Larger reranker probes will not help unless evaluation uses batched/cached scoring. | Modal holdout processed 23,620 pairs with 16,898 cache hits and only 314 model calls. | None. | Confirmed | Cache is necessary for iteration speed. |
| H-003 | Non-README benchmark will expose new failure modes around tables, specs, policies, and operational docs. | Mixed v3 adds issue/runbook/spec/release-note coverage and still shows a forbidden-supported miss after calibration. | The suite is too small to fully characterize failures. | Partially confirmed | Expand non-README reviewed cases. |
| H-004 | The deterministic typed ranker can handle curated non-README direct claims but abstention remains the hard part. | Targeted test shows recall/top-1 support clears the suite; abstention was initially sensitive to contradiction wording. | Larger reviewed non-README split not yet built. | Confirmed | TBD |

## Files and Docs Inspected

- `AGENTS.md`: repository rules, uv usage, quality gates, git discipline.
- `README.md`, `docs/README.md`: repository navigation and quality commands.
- `packages/gia-evidence-finder/src/gia_evidence_finder/model_baselines.py`:
  reranker and hybrid extraction boundaries.
- `packages/gia-evidence-finder/src/gia_evidence_finder/training.py`:
  training-pair export and label score mapping.
- `packages/gia-evidence-finder/scripts/modal_train_reranker.py`:
  Modal training/evaluation path.
- `packages/gia-evidence-finder/src/gia_evidence_finder/benchmark_series.py`:
  built-in benchmark series registry.
- `packages/gia-evidence-finder/src/gia_evidence_finder/readme_benchmarks.py`:
  current README-only curated benchmark pattern.
- `packages/gia-evidence-finder/src/gia_evidence_finder/reviewed_cases.py`:
  external reviewed JSONL loading and validation.

## Blockers and Deferred Checks

- Public SOTA is still blocked by benchmark scale: `mixed_evidence_benchmark_v3`
  has only 85 reviewed cases and small dev/test splits.
- Model-judge requests are implemented and exported, but no multi-model judge
  decision set has been run and promoted yet.
