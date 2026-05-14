# Benchmark Series

gia evidence finder benchmarks are versioned. Do not mutate old labels to improve
results. Add a new suite when the task shape changes.

## Current Suites

| Suite | Documents | Cases | Purpose |
| --- | ---: | ---: | --- |
| `popular_readme_v1` | 8 | 22 supported | Broad popular-README smoke benchmark with explicit technical anchors. |
| `hard_readme_v1` | 8 | 11 supported, 5 unsupported | Paraphrases, nearby hard negatives, and no-evidence abstention cases. |
| `adversarial_readme_v1` | 8 | 8 supported, 8 unsupported | Diagnostic polarity, unsupported-claim, and hard-near-miss cases. |
| `relation_readme_v1` | 8 | 8 supported, 8 unsupported | Diagnostic subject/predicate/modifier binding cases where nearby true facts can form false claims. |
| `non_readme_docs_v1` | 5 | 13 supported, 2 unsupported | Curated specs, runbooks, release notes, and issue-discussion excerpts that exercise non-README evidence extraction. |
| `polarity_negation_v1` | 5 | 12 supported, 8 unsupported | Focused polarity suite for positive claims contradicted by negated spans and negative claims supported by negated spans. |
| `quantifier_numeric_date_v1` | 5 | 20+ reviewed | Value-level exact numeric, date, currency, percentage, duration, and version regression suite. |
| `quantifier_binding_v2` | 1 | 24 reviewed | Role-bound date, metric, money, apartment, count, version, and range cases that reject swapped values. |

The README suites use source-attributed excerpts from React, Kubernetes, Django,
FastAPI, PyTorch, Rust, VS Code, and Ansible READMEs. The document and polarity
suites add smaller source-attributed specs, runbooks, design notes, and focused
negative-capability examples.

## Current Series

| Series | Train | Dev | Test | Purpose |
| --- | ---: | ---: | ---: | --- |
| `evidence_benchmark_v2` | 38 | 16 | 16 | Starter frozen split for calibration, model selection, and final no-retune reporting. |
| `mixed_evidence_benchmark_v3` | 47 | 19 | 19 | README plus non-README frozen split for broader local regression checks. |
| `polarity_evidence_benchmark_v1` | 9 | 8 | 3 | Focused negation, contradiction, `without`, `rejected`, and `not just` regression split. |
| `quantifier_evidence_benchmark_v1` | 9 | 6 | 5 | Focused value-level numeric and date regression split. |
| `quantifier_binding_evidence_benchmark_v2` | 8 | 8 | 8 | Focused role-bound quantifier split for swapped-role and insufficient-context regressions. |

`evidence_benchmark_v2` uses `popular_readme_v1` plus `hard_readme_v1` as the
train split, `adversarial_readme_v1` as the dev split, and
`relation_readme_v1` as the test split. The train split is currently only for
training-data export and small model experiments; the dev split is where
thresholds should be calibrated; the test split should be reported without
retuning.

`mixed_evidence_benchmark_v3` adds `non_readme_docs_v1` cases into each split.
It is a stronger regression benchmark than v2 but still not a public SOTA proof:
`audit-benchmark-series --series mixed_evidence_benchmark_v3` currently warns
that it has only 85 reviewed cases and dev/test splits below 50 cases.

## Commands

```sh
uv run gia-evidence-finder benchmark-readmes
uv run gia-evidence-finder benchmark-readmes --suite hard_readme_v1
uv run gia-evidence-finder benchmark-readmes --suite adversarial_readme_v1 --details
uv run gia-evidence-finder benchmark-readmes --suite relation_readme_v1 --details
uv run gia-evidence-finder benchmark-readmes --suite adversarial_readme_v1 --calibrate-thresholds
uv run gia-evidence-finder benchmark-readmes --suite all
uv run gia-evidence-finder benchmark-series --split all --calibrate-thresholds
uv run gia-evidence-finder benchmark-series-calibrated
uv run gia-evidence-finder benchmark-series-calibrated --series mixed_evidence_benchmark_v3
uv run gia-evidence-finder benchmark-series --series polarity_evidence_benchmark_v1 --split all
uv run gia-evidence-finder benchmark-series-calibrated --series polarity_evidence_benchmark_v1
uv run gia-evidence-finder benchmark-series --series quantifier_binding_evidence_benchmark_v2 --split all
uv run gia-evidence-finder audit-benchmark-series
uv run gia-evidence-finder audit-benchmark-series --series mixed_evidence_benchmark_v3
uv run gia-evidence-finder list-readme-sources
uv run gia-evidence-finder fetch-readme-sources --output-dir .tmp/readme-sources
uv run gia-evidence-finder prepare-curation-queue --input-dir .tmp/readme-sources --output-jsonl .tmp/readme-curation.jsonl
uv run gia-evidence-finder audit-curation-queue --queue-jsonl .tmp/readme-curation.jsonl
uv run gia-evidence-finder prepare-reviewed-case-template --queue-jsonl .tmp/readme-curation.jsonl --output-jsonl .tmp/reviewed-case-template.jsonl
uv run gia-evidence-finder prepare-reviewed-series-template --queue-jsonl .tmp/readme-curation.jsonl --output-jsonl .tmp/reviewed-series-template.jsonl
uv run gia-evidence-finder validate-reviewed-cases --source-dir .tmp/readme-sources --reviewed-jsonl .tmp/reviewed-cases.jsonl --suite-id evidence_benchmark_v3_seed
uv run gia-evidence-finder validate-reviewed-series --source-dir .tmp/readme-sources --reviewed-jsonl .tmp/reviewed-cases.jsonl --series-id evidence_benchmark_v3
uv run gia-evidence-finder benchmark-reviewed-series --source-dir .tmp/readme-sources --reviewed-jsonl .tmp/reviewed-cases.jsonl --series-id evidence_benchmark_v3 --split test
uv run gia-evidence-finder benchmark-reviewed-series-calibrated --source-dir .tmp/readme-sources --reviewed-jsonl .tmp/reviewed-cases.jsonl --series-id evidence_benchmark_v3
uv run gia-evidence-finder explain-reviewed-case --source-dir .tmp/readme-sources --reviewed-jsonl .tmp/reviewed-cases.jsonl --series-id evidence_benchmark_v3 --case-id CASE_ID
uv run --extra models gia-evidence-finder benchmark-readmes --suite hard_readme_v1 --include-models
uv run --extra models gia-evidence-finder benchmark-series --split test --baseline-profile local_smoke
uv run --extra models gia-evidence-finder benchmark-series-calibrated --series mixed_evidence_benchmark_v3 --typed-decision-cross-encoder-model cross-encoder/ms-marco-MiniLM-L-12-v2 --reranker-score-cache-jsonl .tmp/minilm-l12-pair-scores.jsonl --reranker-score-batch-size 64
uv run --extra models gia-evidence-finder benchmark-series --split dev --baseline-profile open_reranker_probe
COHERE_API_KEY=... uv run gia-evidence-finder benchmark-series --split dev --baseline-profile api_reranker_probe
uv run --extra models gia-evidence-finder benchmark-readmes --suite adversarial_readme_v1 --reranker-model BAAI/bge-reranker-v2-m3 --details
uv run gia-evidence-finder export-training-jsonl --suite all --negatives-per-case 5
uv run gia-evidence-finder export-training-jsonl --series evidence_benchmark_v2 --split train --negatives-per-case 5
uv run gia-evidence-finder export-reviewed-training-jsonl --source-dir .tmp/readme-sources --reviewed-jsonl .tmp/reviewed-cases.jsonl --series-id evidence_benchmark_v3 --split train --negatives-per-case 5
```

## Metric Semantics

- `MRR`, `Recall@K`, and `Top-1 support accuracy` are computed over cases with
  labeled support spans.
- `Abstain accuracy` is computed over unsupported cases that should return no
  evidence.
- `Decision accuracy` is computed over all cases and measures only the routing
  decision: abstain versus return support. It catches false abstains on
  supported cases and false supports on unsupported cases, but it does not prove
  that the returned support span is the reviewed gold evidence.
- `Evidence decision accuracy` is the stricter product metric. Supported cases
  pass only when the first selected support match is a reviewed support span;
  unsupported cases pass only when the extractor abstains. Public readiness
  claims should use this metric before `Decision accuracy`.
- `Forbidden top-1 rate` is computed over all cases and flags when a known
  hard negative is ranked first.
- `Forbidden supported top-1 rate` is stricter: it flags only when a known hard
  negative is ranked first and labeled as support. This is the more important
  false-support metric.
- Training labels distinguish `supports`, `near_miss`, `contradicts`,
  `insufficient_context`, and `reject`. This gives future fine-tuning more
  useful supervision than a flat positive/negative pair set.
- Experiment output includes per-label negative diagnostics for near misses,
  contradictions, and insufficient context. The key SOTA risk metric is the
  label's `supported_top1_rate`: a span may be retrieved for analysis, but it
  must not be accepted as supporting evidence for the wrong claim type.
- `--calibrate-thresholds` replays the same candidate scores across support
  thresholds and reports a best balanced point plus the full threshold table.
  The objective prioritizes `Evidence decision accuracy`, then routing
  `Decision accuracy`, support top-1, abstention, false-support safety, MRR, and
  lower-threshold tie-breaks. This is a diagnostic for selecting future dev-set
  thresholds; it does not alter default extractor behavior.
- The intent-aware default extractor reports its own typed labels in calibrated
  holdout runs. Threshold replay can promote ordinary score candidates for
  baselines, but it must never turn `contradicts` into support.

Benchmark cases must not leak gold labels into the intent. A forbidden span can
be listed for evaluation and training export, but its text must not be copied
into `IntentSpec.negative_examples` unless the case is explicitly modeling a
user-provided negative example.

## Current Gate

The deterministic intent-aware ranker must:

- beat the keyword overlap baseline on `MRR` and top-1 support accuracy for
  `popular_readme_v1`;
- reach at least `0.95` support-case `Recall@3` and top-1 support accuracy on
  `hard_readme_v1`;
- reach `1.0` abstain accuracy on `hard_readme_v1`;
- keep forbidden supported top-1 rate at `0.0`.
- clear `polarity_evidence_benchmark_v1` with `1.0` evidence decision accuracy,
  `1.0` routing decision accuracy, `1.0` abstain accuracy, and `0.0`
  contradiction supported top-1 rate for the intent-aware default extractor.

`adversarial_readme_v1` is not a pass/fail release gate yet. It is a diagnostic
suite. Current expected behavior is to show at least one real gap so that
ranking, calibration, and typed-intent improvements have a measurable target.
`relation_readme_v1` now exercises typed `EvidenceRelation` constraints. Its
purpose is to catch false supports created by binding a true attribute to the
wrong subject, product variant, dependency, or tool. Relation constraints are
generic typed input, not benchmark-specific ranker logic.

Baseline profiles are reproducible model lists:

- `local_smoke`: MiniLM embedding retrieval plus MiniLM cross-encoder rerank.
- `open_reranker_probe`: compact open probes such as Qwen3 0.6B, Jina v2, and
  BGE v2 m3.
- `api_reranker_probe`: managed API probes such as Cohere `rerank-v4.0-pro`;
  this requires provider credentials such as `COHERE_API_KEY`.
- `open_sota_probe`: larger Qwen3 4B/8B probes intended for GPU-backed runs.

`audit-benchmark-series` reports split size, label coverage, relation coverage,
typed curation coverage, and warnings that block a benchmark from being treated
as SOTA-ready. Cases must carry reviewed curation metadata to count toward the
reviewed proof threshold, so generated or model-proposed cases can be included
for exploration without inflating quality claims. The current starter series
intentionally warns that it has fewer than 200 reviewed cases and small dev/test
splits.

`list-readme-sources` and `fetch-readme-sources` are the acquisition path for
the next dataset expansion. They fetch raw READMEs into a local curation
directory with a manifest containing repository, license, checksum, line count,
and parsed span count. Raw upstream documents should stay out of commits unless
their license and product need are explicitly reviewed.
`prepare-curation-queue` converts those fetched documents into unreviewed JSONL
span candidates. Those rows are review work, not gold labels; they should become
`BenchmarkCase` objects only after a person or adjudication workflow confirms
support, near-miss, contradiction, insufficient-context, and forbidden spans.
`audit-curation-queue` checks whether the review queue has enough source
diversity and flags obvious quality risks such as one document dominating,
missing heading context, or duplicated candidate text.
`prepare-reviewed-case-template` creates an intentionally unreviewed case draft
from queue rows. `prepare-reviewed-series-template` uses the same unreviewed
draft contract but assigns whole documents, not individual rows, to
train/dev/test so a promoted benchmark does not leak one README across splits.
`validate-reviewed-cases` is the promotion gate that rejects templates until
their curation metadata is marked reviewed and all labeled span ids resolve
against the fetched source manifest.
`validate-reviewed-series` adds the train/dev/test series requirement and
returns the same audit payload as bundled benchmark series, so v3 cannot be
used for model selection until all splits exist.

## Next Suites

1. Long-document suite: larger READMEs and docs where paragraph boundaries
   matter.
2. Dependency suite: cases that require extracting a sentence plus the heading,
   previous sentence, or table row context.
3. Near-miss suite: many semantically related but non-supporting spans, built
   from hard negatives exported by `hard_negative_pairs`.
4. Fine-tune split: train/dev/test JSONL exported from reviewed benchmark cases.
5. Relation-sensitive suite: subject/predicate cases where a related sentence
   mentions the same entities but does not support the exact claim. The first
   frozen slice is `relation_readme_v1`; the next version should expand it with
   longer spans, tables, and multi-sentence context.

## Reviewed-Series Holdout Gate

Use `benchmark-reviewed-series-calibrated` for fair model selection on external
reviewed JSONL. It calibrates thresholds on the dev split and reports the test
split without retuning. Do not compare uncalibrated model scores against
calibrated deterministic scores.

The 2026-05-08 reviewed series
`evidence_benchmark_v3_model_adjudicated_final` has 290 cases and no audit
warnings. Current best deterministic test result is `Decision 0.9886`,
`MRR 0.9536`, and `Forbidden Supported 0.0000`. Current best trained ranking
result is seeded Modal MiniLM with typed first-stage candidates:
`MRR 0.9774`, `R@1 0.9630`, `R@3 1.0000`, `Decision 0.9773`, and
`Forbidden Supported 0.0000`.

These historical numbers predate the explicit `Evidence decision accuracy`
field. Re-run the reviewed holdout with the current reporter before using them
as release or marketing evidence.

The packaged Modal larger-model probes are listed with
`gia-evidence-finder modal-reranker-probes`. On the same reviewed holdout,
`minilm_l12_larger` improved the calibrated typed-decision ranking result to
`MRR 0.9877`, `R@1 0.9753`, `R@3 1.0000`, `Top-1 support 0.9630`,
`Decision 0.9886`, and `Forbidden Supported 0.0000`. The
`electra_base_larger` probe kept `Decision 0.9886` and `Forbidden Supported
0.0000`, but dropped to `R@1 0.8272`, so it is not the recommended default.
