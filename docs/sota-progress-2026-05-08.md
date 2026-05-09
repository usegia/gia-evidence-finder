# SOTA Progress Report: Typed Evidence Extraction

Date: 2026-05-08

## Current Verdict

gia evidence finder is now useful as a real extraction library, not just a
prototype. The strongest measured direction is:

```txt
typed intent + dev-calibrated deterministic support labels + trained reranker ordering
```

On the reviewed README benchmark (`290` reviewed cases, `16` documents,
document-level train/dev/test split), the calibrated hybrid now beats every
measured baseline on ranking quality while tying the deterministic extractor's
best held-out decision accuracy. The important correction is that "preserve
typed labels" is only valid when the typed first-stage support threshold is
itself selected on dev.

## Reviewed Benchmark

- Series id: `evidence_benchmark_v3_model_adjudicated_final`
- Train/dev/test: `132 / 70 / 88`
- Support/abstain cases: `268 / 22`
- Special negatives: `268` near miss, `4` contradiction,
  `286` insufficient context, `594` forbidden span labels
- Relation cases: `14`
- Audit warnings: none

The source READMEs and reviewed JSONL are local `.tmp` artifacts and are not
committed. Raw upstream docs should stay out of git unless their license and
product need are reviewed.

## Held-Out Test Results

All rows below use dev-selected thresholds and report the frozen test split.

| System | MRR | R@1 | R@3 | Top-1 support | Abstain | Decision | Forbidden supported |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Keyword overlap | 0.7881 | 0.6049 | 0.9630 | 0.6049 | 0.0000 | 0.9205 | 0.0114 |
| BM25 | 0.9362 | 0.9012 | 0.9630 | 0.9012 | 0.0000 | 0.9205 | 0.0000 |
| MiniLM embedding | 0.7546 | 0.6420 | 0.8642 | 0.6420 | 0.0000 | 0.9205 | 0.0000 |
| MiniLM cross-encoder | 0.7449 | 0.5185 | 0.9877 | 0.5185 | 0.0000 | 0.9091 | 0.0114 |
| Qwen3 reranker 0.6B | 0.8395 | 0.6914 | 0.9877 | 0.6914 | 0.8571 | 0.9886 | 0.0000 |
| Deterministic typed default | 0.9536 | 0.9259 | 0.9877 | 0.9259 | 0.8571 | 0.9886 | 0.0000 |
| Fine-tuned MiniLM, keyword first-stage | 0.9733 | 0.9630 | 0.9877 | 0.9630 | 0.8571 | 0.9773 | 0.0000 |
| Fine-tuned MiniLM, typed first-stage | 0.9774 | 0.9630 | 1.0000 | 0.9506 | 0.8571 | 0.9773 | 0.0000 |
| Fine-tuned MiniLM, raw typed-label preserve | 0.9774 | 0.9630 | 1.0000 | 0.5926 | 1.0000 | 0.6364 | 0.0000 |
| Fine-tuned MiniLM-L6, calibrated typed decision | 0.9774 | 0.9630 | 1.0000 | 0.9506 | 0.8571 | 0.9886 | 0.0000 |
| Fine-tuned MiniLM-L12, calibrated typed decision | 0.9877 | 0.9753 | 1.0000 | 0.9630 | 0.8571 | 0.9886 | 0.0000 |
| Fine-tuned Electra-base, calibrated typed decision | 0.9051 | 0.8272 | 0.9877 | 0.8148 | 0.8571 | 0.9886 | 0.0000 |

Interpretation:

- BM25 is a strong competitor for lexical README claims and is now included as
  a first-class baseline.
- Generic off-the-shelf rerankers do not beat the typed extractor on this task.
- Fine-tuning helps materially. The trained reranker beats the deterministic
  scorer on ranking (`MRR`, `R@1`, `R@3`).
- Raw label preservation is not enough. The typed first-stage labels must use
  the dev-selected support threshold (`0.30` in this run), otherwise many true
  supports remain near misses after reranking.
- The current best architecture is calibrated typed support/abstain judgment
  plus learned ordering. MiniLM-L12 is the best measured reranker so far;
  Electra-base is larger but worse on extraction ranking.

## What Works Internally

The library is not trying to answer questions. It returns typed source spans.
The current stack has these layers:

1. `MarkdownSpanParser` creates stable spans with IDs, heading context, offsets,
   parent/neighbor links, and span kinds.
2. `IntentSpec` carries typed search intent: label, description, examples,
   required/excluded facets, preferred span kinds, and relation constraints.
3. `IntentAwareRanker` scores spans with deterministic features:
   facet coverage, relation binding, positive/negative example similarity,
   heading overlap, exact anchors, span-kind priors, and exclusion penalties.
4. `EvidenceExtractor` labels candidates as support, near miss, or reject and
   returns minimal non-overlapping matches plus a trace.
5. Baseline/model boundaries compare keyword overlap, BM25, embeddings,
   cross-encoders, transformers rerankers, and external rerankers through the
   same evaluation contract.
6. Reviewed-series calibration now makes the held-out decision policy explicit:
   score-threshold relabeling or extractor-owned labels.
7. `TypedDecisionRerankExtractor` uses a trained reranker for ordering while
   preserving typed first-stage labels, optionally with a calibrated first-stage
   support threshold.
8. Modal training exports reviewed hard-negative pairs, fine-tunes a
   cross-encoder, evaluates extraction-level dev/test quality, and uses a
   persistent pair-score cache for repeated reranker variants.

## Example Behavior

For intent:

```txt
Find evidence that React lets developers build encapsulated components that
manage their own state.
```

The extractor ranks the direct React sentence first, marks it as support, and
records reasons such as:

- all required facets covered
- similar to positive examples
- contains exact intent anchors

It also keeps nearby React sentences as candidates but does not need them as the
minimal answer. This is the product shape we want: source-backed extraction with
traceable score features, not generated prose.

## Current Competitor Position

We are already better than naive keyword overlap, BM25, embedding retrieval,
MiniLM cross-encoder reranking, BGE v2 m3 smoke, Qwen3 0.6B, and the larger
Electra-base fine-tune on the reviewed benchmark's ranking axis. The
MiniLM-L12 calibrated typed-decision hybrid also matches the best measured
decision accuracy and keeps forbidden supported top-1 at zero.

We should not claim broad industry SOTA yet. The reviewed set is strong enough
to guide development, but still too small and too README-shaped. The new
`mixed_evidence_benchmark_v3` adds specs, runbooks, release notes, and issue
discussion cases, but it has only 85 reviewed cases and small dev/test splits.
The next proof requires more document genres at reviewed-series scale and
reproducible model-as-judge adjudication with disagreement tracking.

## Next Plan

1. Keep the calibrated typed-decision hybrid as the product candidate:
   deterministic support/abstain first-stage threshold from dev, trained
   reranker ordering second.
2. Expand the non-README suite beyond the first 15 curated cases: specs,
   architecture docs, issue/PR discussions, changelog entries, API docs, and
   tabular sections.
3. Add deterministic failure reports for the one or two remaining decision
   errors so we know whether the gap is labeling, first-stage intent typing, or
   learned score calibration.
4. Train larger rerankers on Modal only after benchmark scale improves. The
   MiniLM-L12 probe helped; the Electra-base probe did not.
5. Replace human-audit gating with a reproducible model-adjudication protocol:
   fixed rubric, repeated passes, source hashes, disagreement tracking, and
   explicit `model_adjudicated_*` curation sources.
6. Expand from pair-level training to listwise/ranking training, because the
   extraction metric is about choosing the best minimal span, not just binary
   support classification.
7. Keep pair-score caching on for all expensive reranker evaluations. The
   larger probes processed 23,620 pair scores each while reusing 16,898 cached
   scores across variants.
