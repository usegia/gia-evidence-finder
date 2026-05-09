# gia evidence finder Quality Plan

The product goal is not generic semantic search. The goal is typed evidence
selection:

```txt
Given a claim intent, return the smallest document spans that directly support
it, reject near misses, abstain when support is absent, and provide a stable
trace.
```

## Baselines

The package keeps baselines for measurement, not as the product design:

1. Keyword overlap over stable spans.
2. Embedding-only retrieval.
3. Retrieve plus rerank.
4. External reranker or LLM verifier as a quality ceiling.

The default package path is intentionally stronger than keyword overlap. It uses
typed intent examples, positive and negative examples, required facets, excluded
facets, typed relation constraints, span kind priors, heading context, and soft
lexical similarity.

Typed relations represent subject/predicate/object/modifier binding and
forbidden bridge phrases such as dependency or distribution language. They make
the product contract explicit when a claim asks whether an attribute belongs to
one product, project, dependency, distribution, or tool rather than a nearby
related entity.

Raw claims can be converted into a typed `IntentSpec` with the deterministic
claim compiler. The compiler extracts stable technical anchors and conservative
excluded facets for obvious negation or "itself" claims. It is intentionally
small; complex intent typing should become explicit data or a trained model, not
hidden prompt behavior.

## Starter Gates

The starter deterministic ranker should clear these gates on local fixtures:

- `Recall@3 >= 0.85`
- `Top-1 support accuracy >= 0.75` on supported cases
- `Forbidden top-1 rate == 0`
- `Abstain accuracy == 1.0` on explicit no-evidence cases

The bundled README suites are starter benchmarks, not final quality proof. Their
purpose is to catch regressions and force the package to beat weak baselines
before heavier reranker experiments are added.

- `popular_readme_v1`: 8 source-attributed README excerpts and 22 supported
  evidence cases.
- `hard_readme_v1`: the same source documents with 11 supported hard cases and
  5 unsupported abstention cases.
- `adversarial_readme_v1`: the same source documents with 8 supported and 8
  unsupported diagnostic cases for polarity, unsupported claims, and hard
  near-misses.
- `relation_readme_v1`: the same source documents with 8 supported and 8
  unsupported diagnostic cases for subject, predicate, and modifier binding.

Retrieval metrics (`MRR`, `Recall@K`, and top-1 support accuracy) are computed
over supported cases. `Abstain accuracy` is computed over unsupported cases.
`Decision accuracy` is computed over all cases. `Forbidden supported top-1
rate` is the strict false-support metric for known hard negatives. Per-label
negative reports additionally track when near-miss, contradiction, or
insufficient-context spans are ranked first and incorrectly accepted as support.

Benchmark labels must not leak into the evaluated intent. Forbidden span text is
used for evaluation and training export, not automatically as negative examples.

Run it with:

```sh
uv run gia-evidence-finder benchmark-readmes
uv run gia-evidence-finder benchmark-readmes --suite hard_readme_v1
uv run gia-evidence-finder benchmark-readmes --suite adversarial_readme_v1 --details
uv run gia-evidence-finder benchmark-readmes --suite relation_readme_v1 --details
uv run gia-evidence-finder benchmark-readmes --suite all
uv run gia-evidence-finder benchmark-readmes --suite adversarial_readme_v1 --calibrate-thresholds
uv run gia-evidence-finder benchmark-series --split all --calibrate-thresholds
uv run gia-evidence-finder benchmark-series-calibrated
uv run gia-evidence-finder audit-benchmark-series
```

Build the next local curation queue with:

```sh
uv run gia-evidence-finder list-readme-sources
uv run gia-evidence-finder fetch-readme-sources --output-dir .tmp/readme-sources
uv run gia-evidence-finder prepare-curation-queue --input-dir .tmp/readme-sources --output-jsonl .tmp/readme-curation.jsonl
uv run gia-evidence-finder prepare-document-curation-queue --input docs --output-jsonl .tmp/docs-curation.jsonl
uv run gia-evidence-finder audit-curation-queue --queue-jsonl .tmp/readme-curation.jsonl
uv run gia-evidence-finder prepare-reviewed-case-template --queue-jsonl .tmp/readme-curation.jsonl --output-jsonl .tmp/reviewed-case-template.jsonl
uv run gia-evidence-finder prepare-reviewed-series-template --queue-jsonl .tmp/readme-curation.jsonl --output-jsonl .tmp/reviewed-series-template.jsonl
uv run gia-evidence-finder validate-reviewed-cases --source-dir .tmp/readme-sources --reviewed-jsonl .tmp/reviewed-cases.jsonl --suite-id evidence_benchmark_v3_seed
uv run gia-evidence-finder validate-reviewed-series --source-dir .tmp/readme-sources --reviewed-jsonl .tmp/reviewed-cases.jsonl --series-id evidence_benchmark_v3
uv run gia-evidence-finder benchmark-reviewed-series-calibrated --source-dir .tmp/readme-sources --reviewed-jsonl .tmp/reviewed-cases.jsonl --series-id evidence_benchmark_v3
uv run gia-evidence-finder failure-report-reviewed-series --source-dir .tmp/readme-sources --reviewed-jsonl .tmp/reviewed-cases.jsonl --series-id evidence_benchmark_v3 --split test --apply-dev-calibration
```

Export fine-tuning candidates with:

```sh
uv run gia-evidence-finder export-training-jsonl --suite all --negatives-per-case 5
uv run gia-evidence-finder export-training-jsonl --series evidence_benchmark_v2 --split train --negatives-per-case 5
uv run gia-evidence-finder export-reviewed-training-jsonl --source-dir .tmp/readme-sources --reviewed-jsonl .tmp/reviewed-cases.jsonl --series-id evidence_benchmark_v3 --split train --negatives-per-case 5
```

Train a small reranker on Modal with:

```sh
uv run --extra training modal run scripts/modal_train_reranker.py \
  --train-jsonl-path .tmp/reviewed-train-pairs.jsonl \
  --dev-jsonl-path .tmp/reviewed-dev-pairs.jsonl \
  --reviewed-jsonl-path .tmp/reviewed-cases-final.jsonl \
  --source-dir .tmp/readme-sources \
  --series-id evidence_benchmark_v3_model_adjudicated_final \
  --epochs 4 \
  --seed 13
```

Re-evaluate a saved Modal model without retraining:

```sh
uv run --extra training modal run scripts/modal_train_reranker.py \
  --evaluate-run-name reviewed-msmarco-minilm-20260508-e4-seed13 \
  --reviewed-jsonl-path .tmp/reviewed-cases-final.jsonl \
  --source-dir .tmp/readme-sources \
  --series-id evidence_benchmark_v3_model_adjudicated_final
```

Run optional local model baselines with:

```sh
uv run --extra models gia-evidence-finder benchmark-readmes --include-models
uv run --extra models gia-evidence-finder benchmark-readmes --suite adversarial_readme_v1 --reranker-model BAAI/bge-reranker-v2-m3 --details
uv run --extra models gia-evidence-finder benchmark-series --split dev --baseline-profile open_reranker_probe
COHERE_API_KEY=... uv run gia-evidence-finder benchmark-series --split dev --baseline-profile api_reranker_probe
```

Record material model runs in `docs/model-baseline-results.md`.

`evidence_benchmark_v2` is the current starter train/dev/test split. Threshold
calibration should use its dev split and the test split should be reported
without retuning. The sweep replays recorded candidate scores at multiple
support thresholds, then selects the best balanced point by decision accuracy,
top-1 support accuracy, abstention accuracy, forbidden supported top-1 rate,
MRR, and lower-threshold tie-break. Each point also reports per-label supported
top-1 rates for negative labels. It does not mutate benchmark labels or change
the extractor's runtime default. Use `SupportThresholdOverrideExtractor` when a
dev-selected threshold must be applied to a held-out split.

`audit-benchmark-series` is the readiness gate for dataset quality. A series
must not be called SOTA-ready while it has too few reviewed cases, lacks
contradiction or insufficient-context labels, has weak dev/test splits, or uses
unreviewed generated cases as proof. Model-proposed cases are allowed for
exploration only after their curation metadata keeps them out of reviewed counts.

These are only starter gates. A real dataset should target:

- `Recall@5 >= 0.95`
- `Recall@3 >= 0.90`
- `MRR >= 0.90`
- `Top-1 support accuracy >= 0.85`
- Near-miss false positives down 30-50% versus a strong local reranker
- `Forbidden supported top-1 rate == 0.0` on frozen adversarial tests
- `Decision accuracy >= 0.95` on frozen relation-binding tests before any
  trained model is considered ready for the hot path

## Training Roadmap

1. Expand beyond README sources into TraverceDB docs, specs, plans, benchmark
   reports, customer-style documents, changelogs, API docs, tables, and issue
   discussions.
2. Label support, near miss, reject, contradiction, and insufficient-context
   spans with reproducible model adjudication: fixed rubric, source hashes,
   repeat passes, disagreement tracking, and explicit `model_adjudicated_*`
   curation sources.
3. Export hard-negative pairs with `hard_negative_pairs`.
4. Fine-tune small cross-encoders first, then compare larger Modal GPU
   rerankers only after the dataset grows.
5. Preserve dev-calibrated typed support/abstain decisions unless a trained
   model proves it can beat them on frozen decision accuracy. Raw first-stage
   labels are too conservative; the current best hybrid uses the deterministic
   dev threshold (`0.30` on the reviewed README split) and learned ordering.
6. Keep LLM verifiers out of the hot path. Use them for label proposals,
   benchmark audits, and low-confidence adjudication.

## Why A Dedicated Library

RAG frameworks retrieve chunks and often answer with generated prose. Traverce
Evidence returns typed, minimal, source-backed spans with deterministic
score features and abstention. That contract is the part worth owning.
