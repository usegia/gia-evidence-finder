# Polarity And Negation Implementation Report

Date: 2026-05-08

## Scope

This pass focused only on `gia-evidence-finder`. The goal was to close the false
support class where a span contains the right anchors but negates the user's
claim, for example selecting `No evaluation, no search, no opening book.` as
support for `includes evaluation, search, and an opening book`.

## Result

The package now has a deterministic polarity layer and a focused benchmark:

- `polarity_negation_v1`: 20 reviewed cases across 5 source-attributed excerpts.
- `polarity_evidence_benchmark_v1`: train/dev/test split with 9 / 8 / 3 cases.
- Covered phenomena: positive support, positive claims contradicted by negated
  spans, negative claims supported by negated spans, `without`, `must not`,
  `rejected`, `instead of`, and `not just`.

Current focused-series result for the intent-aware default extractor:

| Split | Decision | Top-1 Support | Abstain | Forbidden Supported | Contradiction Supported |
| --- | ---: | ---: | ---: | ---: | ---: |
| Train | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| Dev | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| Test | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 0.0000 |

The same focused series exposes the cheap lexical baselines:

- Keyword overlap: false-supports contradiction cases on train/dev; test is too
  small to be a stable proof.
- BM25: false-supports contradiction cases, including `not just` exclusivity.

The broader `mixed_evidence_benchmark_v3` remains healthy after the change:

| Split | Decision | Top-1 Support | Abstain | Forbidden Supported |
| --- | ---: | ---: | ---: | ---: |
| Train | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| Dev | 0.9474 | 1.0000 | 0.8750 | 0.0526 |
| Test | 1.0000 | 1.0000 | 1.0000 | 0.0000 |

The remaining mixed dev miss predates this focused work and is not from the new
polarity benchmark.

## Design

The new `polarity.py` module is a pure deterministic analyzer. It compares
shared content anchors between intent text and candidate span text, then reports:

- polarity alignment;
- polarity mismatch;
- contradiction score;
- intent negated-anchor coverage;
- span negated-anchor coverage;
- shared anchor count.

The ranker consumes these values as score features. The extractor then maps
strong polarity flips to `SupportLabel.CONTRADICTS`, which keeps contradicted
spans visible as diagnostic candidates while preventing them from entering
`matches`.

Important implementation details:

- `not just Python` negates exclusivity, not `Python`.
- `No evaluation, no search, no opening book` negates all three anchors.
- `Chess960 ... are not supported` handles postposed negation over a list.
- `Avoid custom agents; be agentless by SSH` does not let `avoid` leak across
  the semicolon into the positive `agentless` explanation.
- `rejected adding an LLM answer generator` supports the negative claim that the
  LLM generator was rejected, and contradicts the positive claim that it was
  added.

## Contract Changes

- `CandidateScore` now carries the candidate support label as well as score.
- Calibration is label-aware: threshold replay may promote ordinary low-score
  candidates, but it must not promote a `contradicts` candidate into support.
- The intent-aware default extractor uses extractor labels in calibrated
  holdout runs. Keyword and BM25 baselines remain score-threshold calibrated.
- `compile_intent` now extracts lower-case lexical anchors from raw claims, so
  non-capitalized claims such as `includes evaluation, search, and an opening
  book` have real required facets.
- `compile_intent` no longer treats every negated phrase in the user claim as an
  excluded facet. A negative claim like `has no evaluation` is now a claim to
  support, not a reason to penalize the matching span.

## Verification

Commands run:

```sh
uv run pytest -q tests/test_polarity_benchmarks.py tests/test_intent_compiler.py tests/test_evaluation_training.py tests/test_extraction.py tests/test_document_benchmarks.py tests/test_readme_benchmarks.py tests/test_benchmark_series.py
uv run gia-evidence-finder benchmark-series --series polarity_evidence_benchmark_v1 --split all
uv run python - <<'PY'
from gia_evidence_finder import *
for series_id in ('polarity_evidence_benchmark_v1','mixed_evidence_benchmark_v3'):
    series=benchmark_series_by_id(series_id)
    print(series_id)
    for split in BenchmarkSplit:
        suite=benchmark_series_suites(series, split=split)[0]
        rep=evaluate_suite(EvidenceExtractor.default(), suite.cases)
        print(split.value, rep.decision_accuracy, rep.top1_support_accuracy, rep.abstain_accuracy, rep.forbidden_supported_top1_rate)
PY
```

## Remaining Work

This is a strong deterministic fix for common negation and contradiction
patterns, not a full natural-language inference engine. Next benchmark growth
should add:

- longer multi-sentence negation scopes;
- table rows with absent/present support;
- double negation;
- temporal negation such as `no longer`;
- comparative numeric contradictions such as `55x` vs `5x`;
- model-judge reviewed cases for ambiguous rejection and exception language.
