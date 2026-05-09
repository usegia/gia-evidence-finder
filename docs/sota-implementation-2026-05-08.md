# SOTA Implementation Report

Date: 2026-05-08
Repository: `/Users/yahorbarkouski/travercedb`

## Executive Summary

Goal: turn gia evidence finder from a strong README benchmark prototype into a
release-candidate evidence extraction tool with broader benchmarks,
reproducible model adjudication, cached/batched reranking, packaged training,
and larger Modal model probes.

Current product candidate before this pass:

```txt
typed intent -> dev-calibrated deterministic support labels -> trained reranker ordering
```

The known best measured reviewed-README result before this pass is:

| System | MRR | R@1 | R@3 | Top-1 support | Decision | Forbidden supported |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Deterministic typed default | 0.9536 | 0.9259 | 0.9877 | 0.9259 | 0.9886 | 0.0000 |
| Fine-tuned MiniLM + calibrated typed decision | 0.9774 | 0.9630 | 1.0000 | 0.9506 | 0.9886 | 0.0000 |

This is not yet broad industry SOTA. The immediate target is a stronger
internal SOTA candidate whose quality claim is reproducible and genre-diverse.

## Scope

- Included:
  - Non-README benchmark support and at least one curated non-README suite.
  - Reproducible model-judge curation/adjudication contracts.
  - Cached and batched reranker scoring.
  - Reranking package/runtime improvements.
  - Modal training/evaluation packaging and larger model probes where practical.
  - Tests, docs, and quality gates.
- Excluded:
  - Root TraverceDB storage/semantic dirty files already present before this
    pass.
  - Public SOTA marketing claims unless the benchmark evidence genuinely
    supports them.

## System Map

- `contracts.py`: typed span, intent, benchmark, and training contracts.
- `parsing.py`: Markdown/RST-ish span parser with stable span IDs.
- `extractor.py` and `ranking.py`: deterministic typed support decision path.
- `model_baselines.py`: BM25, embeddings, cross-encoder, external reranker,
  and typed-decision hybrid boundaries.
- `experiments.py` and `calibration.py`: benchmark and dev/test calibration.
- `reviewed_cases.py`: external reviewed JSONL loading into train/dev/test
  benchmark series.
- `training.py` and `scripts/modal_train_reranker.py`: pair export, fine-tune,
  saved-model evaluation.
- `cli.py`: benchmark, curation, explanation, failure-report, and training
  export commands.

## Planned Work

1. Add non-README benchmarks.
2. Build reproducible model judgment.
3. Add batched/cached reranking.
4. Improve reranking and typed-decision packaging.
5. Package training/evaluation and run larger Modal probes.

## Findings And Outcomes

This section is append-only during implementation.

### 2026-05-08 Implementation Pass

Implemented all five requested tracks:

1. Added `non_readme_docs_v1`, a curated 15-case suite across specs, runbooks,
   release notes, and issue discussions.
2. Added typed model-judge request, response, conversion, and agreement
   contracts plus CLI commands for export/audit.
3. Added persistent JSONL pair-score caching and miss batching for reranker
   models, including Modal holdout evaluation cache stats.
4. Promoted typed-decision cross-encoder reranking to a first-class CLI path
   with dev-calibrated first-stage thresholds in held-out commands.
5. Packaged Modal reranker probes and ran two larger H200 probes.

`mixed_evidence_benchmark_v3` now combines README and non-README checks:

| Metric | Value |
| --- | ---: |
| Reviewed cases | 85 |
| Train/dev/test | 47 / 19 / 19 |
| Non-README cases | 15 |
| Support labels | 63 |
| Abstain cases | 23 |
| Relation cases | 16 |

This mixed benchmark is useful for regression but still not a public SOTA proof.
The audit warns that the full series is below 200 reviewed cases and both
dev/test splits are below 50 cases.

Local calibrated deterministic result on `mixed_evidence_benchmark_v3` test:

| System | MRR | R@1 | R@3 | Top-1 support | Abstain | Decision | Forbidden supported |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Intent-aware default | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.8750 | 0.9474 | 0.0526 |
| Keyword overlap | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.3750 | 0.7368 | 0.2632 |
| BM25 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 0.5789 | 0.4211 |

Reviewed external holdout, calibrated typed-decision reranker:

| Probe | MRR | R@1 | R@3 | Top-1 support | Abstain | Decision | Forbidden supported |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Prior MiniLM-L6 | 0.9774 | 0.9630 | 1.0000 | 0.9506 | 0.8571 | 0.9886 | 0.0000 |
| MiniLM-L12 larger | 0.9877 | 0.9753 | 1.0000 | 0.9630 | 0.8571 | 0.9886 | 0.0000 |
| Electra-base larger | 0.9051 | 0.8272 | 0.9877 | 0.8148 | 0.8571 | 0.9886 | 0.0000 |

MiniLM-L12 is the best current candidate. It improves ranking quality over the
previous MiniLM-L6 run while preserving the strongest final decision and
forbidden-support metrics. Electra-base is a useful negative result: larger
capacity alone did not improve extraction quality.

The Modal extraction evaluator scored `23,620` query/span pairs per larger run.
The new cache reduced model calls to `314` and recorded `16,898` cache hits,
`6,722` misses, and `6,709` persisted scores, confirming the cached/batched
path is materially useful for future larger probes.

Current status: this is a strong internal SOTA candidate, not yet an honest
industry/public SOTA claim. The next proof gap is benchmark scale and genre
coverage, not another small model tweak.
