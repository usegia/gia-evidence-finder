# Product Readiness Assessment

Date: 2026-05-09

## Verdict

`gia evidence finder` is worth shipping as a narrow source-grounding primitive:

```text
claim or typed intent + caller-provided source spans -> support / contradiction / abstention spans
```

It should not be positioned as a corpus search engine, PDF ingestion product, or
industry-wide SOTA system. The useful product is the strict evidence decision
layer that sits after retrieval or ingestion and refuses semantically similar
but unsupported spans.

## Corrections Made

- The quick-start API now accepts `compile_intent("claim text")`, matching the
  README.
- The CLI quick-start now uses a claim that the default extractor actually
  supports.
- `evidence_decision_accuracy` now distinguishes correct evidence selection
  from the weaker abstain-versus-support routing metric.
- Threshold calibration now prioritizes evidence decision accuracy before the
  legacy routing metric.
- CLI and competitor reports expose the stricter metric and include failed
  evidence decisions in diagnostics.
- Public docs now state that non-Markdown sources must be converted into
  `EvidenceDocument` spans before extraction.

## Model Decision

Do not ship the fine-tuned model as the default runtime yet.

The current best shape is deterministic typed support/abstain judgment by
default, with an optional learned reranker only after frozen holdout validation.
A model artifact is ready to publish only when it has:

- a model card and license review;
- a content-addressed artifact hash;
- frozen train/dev/test provenance;
- exact reproduction commands;
- `evidence_decision_accuracy` reported beside `MRR`, `Recall@K`, abstention,
  and false-support safety;
- no regression in `forbidden_supported_top1_rate`.

## Remaining Maturity Work

1. Move built-in synonym groups and quantifier role aliases toward configurable
   profiles. The current defaults are practical starter heuristics, but they
   still mix generic evidence logic with product/domain vocabulary.
2. Add ingestion adapters or examples for PDFs, HTML, tickets, and record fields
   only when they produce stable `EvidenceDocument` spans with source offsets.
3. Expand reviewed frozen holdouts and re-run hosted competitors before making
   public market-wide claims.
4. Treat model training as an optional enhancement to ranking, not a substitute
   for typed evidence contracts and abstention tests.
