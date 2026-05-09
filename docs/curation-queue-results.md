# README Curation Queue Results

Last updated: 2026-05-08.

This report records aggregate metrics for local curation artifacts only. Raw
upstream README files and generated JSONL queues stay in `.tmp/` and are not
committed.

## 2026-05-08 Queue

Commands:

```sh
uv run gia-evidence-finder fetch-readme-sources --output-dir .tmp/readme-sources
uv run gia-evidence-finder prepare-curation-queue --input-dir .tmp/readme-sources --output-jsonl .tmp/readme-curation.jsonl --max-items-per-doc 40
uv run gia-evidence-finder audit-curation-queue --queue-jsonl .tmp/readme-curation.jsonl
uv run gia-evidence-finder prepare-reviewed-series-template --queue-jsonl .tmp/readme-curation.jsonl --output-jsonl .tmp/reviewed-series-template.jsonl
```

Source acquisition fetched 19 README projects. The stricter curation queue kept
430 review candidates from 16 documents after requiring heading context and
deduplicating normalized text.

Queue audit:

- `item_count`: 430
- `document_count`: 16
- `kind_counts`: 173 bullets, 249 sentences, 8 table rows
- `empty_heading_count`: 0
- `duplicate_text_count`: 0
- `review_queue_ready`: true

Document-level template split:

- `train`: 294 candidates
- `dev`: 91 candidates
- `test`: 45 candidates

The generated templates remain unreviewed. They are not benchmark truth until a
reviewer or adjudication workflow confirms support, near-miss, contradiction,
insufficient-context, and forbidden span labels and flips curation metadata to
`reviewed: true`.

## 2026-05-08 Model-Adjudicated Promotion

Local promoted artifact:

```sh
.tmp/reviewed-cases-final.jsonl
```

This artifact is intentionally local because it depends on fetched upstream
README source files under `.tmp/readme-sources/`. The promoted JSONL does not
commit raw README content. Each case records the fetched source checksum in
curation notes.

Promotion method:

- started from the document-level train/dev/test template
- accepted support rows only when the claim could be restated from a resolved
  support span
- paraphrased support intents through `openai/gpt-5.5` on OpenRouter, then
  filtered low-signal project-process rows
- added abstention cases and explicit contradiction labels
- validated every promoted span id against the fetched source manifest

Final reviewed-series audit:

- `case_count`: 290
- `reviewed_case_count`: 290
- `support_case_count`: 268
- `abstain_case_count`: 22
- `split_counts`: 132 train, 70 dev, 88 test
- `document_count`: 16
- `near_miss` labels: 268
- `contradiction` labels: 4
- `insufficient_context` labels: 286
- `forbidden` labels: 594
- `relation_case` count: 14
- audit warnings: none

Validation command:

```sh
uv run gia-evidence-finder validate-reviewed-series --source-dir .tmp/readme-sources --reviewed-jsonl .tmp/reviewed-cases-final.jsonl --series-id evidence_benchmark_v3_model_adjudicated_final
```

The uncalibrated paraphrased benchmark is intentionally harder than the first
exact-span promotion. On the final test split:

| Extractor | MRR | Recall@1 | Recall@3 | Top-1 support accuracy | Decision accuracy |
| --- | ---: | ---: | ---: | ---: | ---: |
| Intent-aware default | 0.9536 | 0.9259 | 0.9877 | 0.5926 | 0.6364 |
| Keyword overlap | 0.7881 | 0.6049 | 0.9630 | 0.0864 | 0.2159 |

After selecting the support threshold on dev only and applying it to test:

| Extractor | Dev threshold | Test MRR | Test Recall@1 | Test Recall@3 | Test top-1 support accuracy | Test decision accuracy | Test forbidden supported top-1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Intent-aware default | 0.30 | 0.9536 | 0.9259 | 0.9877 | 0.9259 | 0.9886 | 0.0000 |
| Keyword overlap | 0.05 | 0.7881 | 0.6049 | 0.9630 | 0.6049 | 0.9205 | 0.0114 |

The result is now a useful benchmark signal: keyword overlap remains competitive
on broad recall but loses sharply on top-1 evidence selection once the intents
are paraphrased and adjudicated.
