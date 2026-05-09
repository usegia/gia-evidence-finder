# Model Baseline Results

Date: 2026-05-08

Command:

```sh
uv run --extra models gia-evidence-finder benchmark-readmes --include-models
uv run --extra models gia-evidence-finder benchmark-readmes --suite adversarial_readme_v1 --reranker-model BAAI/bge-reranker-v2-m3 --details
uv run gia-evidence-finder benchmark-readmes --suite adversarial_readme_v1 --calibrate-thresholds
uv run --extra models gia-evidence-finder benchmark-readmes --suite relation_readme_v1 --include-models --calibrate-thresholds
uv run --extra models gia-evidence-finder benchmark-readmes --suite relation_readme_v1 --reranker-model BAAI/bge-reranker-v2-m3 --calibrate-thresholds
uv run gia-evidence-finder benchmark-series --split test --calibrate-thresholds
```

Suites:

- `popular_readme_v1`: 8 source-attributed README excerpts, 22 supported cases
- `hard_readme_v1`: same source documents, 11 supported cases, 5 unsupported
  abstention cases
- `adversarial_readme_v1`: same source documents, 8 supported cases, 8
  unsupported diagnostic cases
- `relation_readme_v1`: same source documents, 8 supported cases, 8
  unsupported subject/predicate/modifier binding cases

Retrieval metrics are computed over supported cases. Abstain accuracy is
computed over unsupported cases. `Forbidden supported top-1` is the stricter
false-support metric for known hard negatives.

## `popular_readme_v1`

| Experiment | MRR | R@1 | R@3 | R@5 | Top-1 Support | Forbidden Top-1 | Elapsed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| intent-aware default | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 152 ms |
| keyword overlap baseline | 0.9545 | 0.9091 | 1.0000 | 1.0000 | 0.9091 | 0.0000 | 6 ms |
| MiniLM embedding retrieval | 0.9470 | 0.9091 | 1.0000 | 1.0000 | 0.9091 | 0.0000 | 643 ms |
| MiniLM cross-encoder rerank | 0.9091 | 0.8182 | 1.0000 | 1.0000 | 0.8182 | 0.0000 | 901 ms |

## `hard_readme_v1`

| Experiment | MRR | R@1 | R@3 | R@5 | Top-1 Support | Abstain | Decision | Forbidden Top-1 | Elapsed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| intent-aware default | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 124 ms |
| keyword overlap baseline | 0.9545 | 0.9091 | 1.0000 | 1.0000 | 0.9091 | 1.0000 | 0.9375 | 0.0000 | 4 ms |
| MiniLM embedding retrieval | 0.9545 | 0.9091 | 1.0000 | 1.0000 | 0.9091 | 1.0000 | 1.0000 | 0.1250 | 699 ms |
| MiniLM cross-encoder rerank | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 681 ms |

## `adversarial_readme_v1`

| Experiment | MRR | R@1 | R@3 | R@5 | Top-1 Support | Abstain | Decision | Forbidden Supported Top-1 | Elapsed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| intent-aware default | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.8750 | 0.9375 | 0.0625 | 106 ms |
| keyword overlap baseline | 0.9167 | 0.8750 | 1.0000 | 1.0000 | 0.8750 | 0.7500 | 0.8750 | 0.1250 | 4 ms |
| MiniLM embedding retrieval | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.2500 | 0.6250 | 0.3750 | 648 ms |
| MiniLM cross-encoder rerank | 0.8125 | 0.6250 | 1.0000 | 1.0000 | 0.6250 | 0.1250 | 0.5625 | 0.4375 | 684 ms |
| BGE reranker v2 m3 | 0.7917 | 0.6250 | 1.0000 | 1.0000 | 0.6250 | 0.0000 | 0.5000 | 0.1250 | 4098 ms |

Elapsed times are from a warm local model cache and should be treated as
environment-dependent smoke numbers, not a benchmarked latency target.

Models:

- `sentence-transformers/all-MiniLM-L6-v2`
- `cross-encoder/ms-marco-MiniLM-L-6-v2`
- `BAAI/bge-reranker-v2-m3`

## Interpretation

The current deterministic intent-aware ranker beats the first local model
baselines on `popular_readme_v1`, ties the MiniLM cross-encoder on
`hard_readme_v1`, and beats MiniLM/BGE baselines on adversarial abstention.
That is useful, but it is not a final quality claim:

- The suite is still small.
- The README excerpts are short.
- Many cases contain explicit technical anchors.
- The cross-encoder is a general MS MARCO reranker, not fine-tuned for typed
  claim evidence extraction.
- First-stage candidate quality affects cross-encoder output.
- The MiniLM embedding retriever puts known forbidden spans first on some hard
  cases even when its final abstain/support decision is correct.
- Generic rerankers need calibration; raw cross-encoder scores are not reliable
  support probabilities.
- The deterministic ranker still falsely supports
  `adversarial.vscode_code_oss_is_product_distribution`, which needs
  relation-sensitive claim understanding rather than more lexical overlap.
- The deterministic claim compiler can reject that specific raw "itself" claim
  by adding conservative excluded indirection facets, but the benchmark keeps
  the manually typed case as a reminder that typed-intent construction matters.

The result supports the current direction: typed facets, negative examples,
minimal spans, and abstention are real product advantages over generic retrieval.
It does not prove superiority over strong modern rerankers such as BGE, Qwen,
Jina, Cohere, or ColBERT-style late interaction.

## Threshold Calibration Smoke

The first deterministic calibration sweep on `adversarial_readme_v1` selected
threshold `0.45` for the intent-aware default:

| Experiment | Best threshold | Top-1 Support | Abstain | Decision | Forbidden Supported Top-1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| intent-aware default | 0.45 | 1.0000 | 0.8750 | 0.9375 | 0.0625 |
| keyword overlap baseline | 0.45 | 0.8750 | 0.6250 | 0.8125 | 0.1875 |

The sweep is useful because it shows the current trade-off directly: raising
the default ranker threshold to `0.65` removes known forbidden support on this
suite, but drops top-1 support accuracy to `0.3750`. That means the remaining
gap is not just a threshold problem. We need better relation-sensitive intent
representation and trained support-vs-near-miss discrimination.

## `relation_readme_v1`

This diagnostic suite was added after the adversarial sweep showed that the
remaining deterministic failures were relation-binding failures, not simple
threshold failures. The suite now uses typed `EvidenceRelation` constraints for
subject/predicate/object/modifier binding and forbidden bridge phrases.

| Experiment | MRR | R@1 | R@3 | R@5 | Top-1 Support | Abstain | Decision | Forbidden Supported Top-1 | Best threshold |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| intent-aware default with relation constraints | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 0.55 |
| keyword overlap baseline | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.7500 | 0.8750 | 0.1250 | 0.60 |
| MiniLM embedding retrieval | 0.9375 | 0.8750 | 1.0000 | 1.0000 | 0.8750 | 0.1250 | 0.5625 | 0.4375 | 0.75 |
| MiniLM cross-encoder rerank | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 0.5000 | 0.4375 | 0.95 |
| BGE reranker v2 m3 | 0.7604 | 0.6250 | 0.8750 | 1.0000 | 0.6250 | 0.0000 | 0.5000 | 0.3750 | 0.95 |

Before relation constraints, the default failed two unsupported claims:

- `relation.fastapi_pydantic_matches_node_go`: transfers FastAPI's
  performance claim to Pydantic itself.
- `relation.vscode_code_oss_has_distribution_customizations`: transfers the
  Visual Studio Code distribution's Microsoft-specific customizations to
  Code - OSS itself.

Typed relations fix those two failures without adding document-specific ranker
logic. This is the strongest signal so far that the product advantage is the
typed evidence contract plus calibrated support judgment, not just a stronger
semantic reranker. The generic rerankers remain useful baselines and possible
model backbones, but they retrieve plausible semantic matches without a
first-class support/abstain contract.

## `evidence_benchmark_v2`

`evidence_benchmark_v2` is the current starter frozen split:

- train: `popular_readme_v1` + `hard_readme_v1`, 38 cases
- dev: `adversarial_readme_v1`, 16 cases
- test: `relation_readme_v1`, 16 cases

The current deterministic test result is `Decision 1.0000`,
`Top-1 Support 1.0000`, `Abstain 1.0000`, and
`Forbidden Supported Top-1 0.0000`. This should be treated as a starter
benchmark result, not final proof of industry superiority. The next proof step
is a 200-500 case reviewed split with long documents and more relation traps.
`audit-benchmark-series` intentionally keeps that gap visible by warning when
the series has too few reviewed cases or lacks label coverage needed for
training.

Current deterministic split summary:

| Split | Default Decision | Default Forbidden Supported | Keyword Decision | Keyword Forbidden Supported |
| --- | ---: | ---: | ---: | ---: |
| train | 1.0000 | 0.0000 | 0.9737 | 0.0000 |
| dev | 0.9375 | 0.0625 | 0.8750 | 0.1250 |
| test | 1.0000 | 0.0000 | 0.8750 | 0.1250 |

Current per-label negative risk summary:

| Split | Default negative supported top-1 | Keyword negative supported top-1 |
| --- | --- | --- |
| train | near_miss 0.0000, insufficient_context 0.0000 | near_miss 0.0000, insufficient_context 0.0000 |
| dev | near_miss 0.0000, contradicts 0.0000 | near_miss 0.0000, contradicts 0.2000 |
| test | insufficient_context 0.0000 | insufficient_context 1.0000 |

Current audit summary: 70 reviewed cases across 8 benchmark README excerpts,
with 23 hard cases, 29 medium cases, and 18 standard cases. The source catalog
now tracks 19 README projects for expansion. The benchmark remains
intentionally not SOTA-ready because the frozen reviewed set is below 200 cases
and the dev/test splits are still 16 cases each.

Source acquisition and curation queue commands now exist for the next expansion:

```sh
uv run gia-evidence-finder fetch-readme-sources --output-dir .tmp/readme-sources
uv run gia-evidence-finder prepare-curation-queue --input-dir .tmp/readme-sources --output-jsonl .tmp/readme-curation.jsonl
uv run gia-evidence-finder audit-curation-queue --queue-jsonl .tmp/readme-curation.jsonl
uv run gia-evidence-finder prepare-reviewed-series-template --queue-jsonl .tmp/readme-curation.jsonl --output-jsonl .tmp/reviewed-series-template.jsonl
```

## Next Experiments

1. Run `open_reranker_probe`, `open_sota_probe`, and `api_reranker_probe` on
   GPU-backed hardware or with provider credentials.
2. Add more managed API rerankers through the external reranker boundary.
3. Expand the relation suite into a frozen dev/test split with at least 200
   reviewed cases.
4. Review the generated curation queue and promote confirmed rows into
   `BenchmarkCase` objects with support, near-miss, contradiction,
   insufficient-context, and forbidden labels.
5. Export hard-negative pairs and fine-tune a small cross-encoder.
6. Improve the raw-claim compiler so it can produce relation constraints for
   more subject/attribute-transfer claims.
7. Calibrate support thresholds only on the dev split, then report the frozen
   test split without retuning.

## Reviewed Series v3 and First Modal Fine-Tune

The larger local reviewed series is
`evidence_benchmark_v3_model_adjudicated_final`: 290 reviewed cases across 16
README documents, with 132 train, 70 dev, and 88 test cases. It has no audit
warnings and is the current best engineering benchmark.

Command:

```sh
uv run gia-evidence-finder benchmark-reviewed-series-calibrated \
  --source-dir .tmp/readme-sources \
  --reviewed-jsonl .tmp/reviewed-cases-final.jsonl \
  --series-id evidence_benchmark_v3_model_adjudicated_final
```

Held-out test results after dev threshold selection:

| Experiment | MRR | R@1 | R@3 | Top-1 Support | Abstain | Decision | Forbidden Supported |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| intent-aware default | 0.9536 | 0.9259 | 0.9877 | 0.9259 | 0.8571 | 0.9886 | 0.0000 |
| keyword overlap baseline | 0.7881 | 0.6049 | 0.9630 | 0.6049 | 0.0000 | 0.9205 | 0.0114 |
| BM25 baseline | 0.9362 | 0.9012 | 0.9630 | 0.9012 | 0.0000 | 0.9205 | 0.0000 |
| MiniLM embedding | 0.7546 | 0.6420 | 0.8642 | 0.6420 | 0.0000 | 0.9205 | 0.0000 |
| MiniLM cross-encoder | 0.7449 | 0.5185 | 0.9877 | 0.5185 | 0.0000 | 0.9091 | 0.0114 |
| Qwen3 reranker 0.6B | 0.8395 | 0.6914 | 0.9877 | 0.6914 | 0.8571 | 0.9886 | 0.0000 |

Modal fine-tuning used:

```sh
uv run --extra training modal run scripts/modal_train_reranker.py \
  --train-jsonl-path .tmp/reviewed-train-pairs.jsonl \
  --dev-jsonl-path .tmp/reviewed-dev-pairs.jsonl \
  --reviewed-jsonl-path .tmp/reviewed-cases-final.jsonl \
  --source-dir .tmp/readme-sources \
  --series-id evidence_benchmark_v3_model_adjudicated_final \
  --run-name reviewed-msmarco-minilm-20260508-e4-seed13 \
  --epochs 4 \
  --batch-size 16 \
  --learning-rate 2e-5 \
  --warmup-ratio 0.1 \
  --seed 13
```

Seeded fine-tuned test results:

| Experiment | MRR | R@1 | R@3 | Top-1 Support | Abstain | Decision | Forbidden Supported |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| trained MiniLM, keyword first-stage | 0.9733 | 0.9630 | 0.9877 | 0.9630 | 0.8571 | 0.9773 | 0.0000 |
| trained MiniLM, typed first-stage | 0.9774 | 0.9630 | 1.0000 | 0.9506 | 0.8571 | 0.9773 | 0.0000 |
| trained MiniLM, typed first-stage gated | 0.9774 | 0.9630 | 1.0000 | 0.9506 | 0.8571 | 0.9773 | 0.0000 |
| trained MiniLM, raw preserve typed labels | 0.9774 | 0.9630 | 1.0000 | 0.5926 | 1.0000 | 0.6364 | 0.0000 |
| trained MiniLM, calibrated typed decision | 0.9774 | 0.9630 | 1.0000 | 0.9506 | 0.8571 | 0.9886 | 0.0000 |

This is the first result where a learned model beats the deterministic default
on ranking. After fixing the holdout evaluation policy, raw label preservation
proved too conservative because the default first-stage threshold leaves many
true supports as near misses. The current best architecture is:

```txt
dev-calibrated typed first-stage labels + trained reranker ordering
```

That hybrid matches the deterministic default on decision accuracy (`0.9886`)
while improving ranking (`MRR 0.9774` vs `0.9536`, `R@1 0.9630` vs `0.9259`,
`R@3 1.0000` vs `0.9877`).
