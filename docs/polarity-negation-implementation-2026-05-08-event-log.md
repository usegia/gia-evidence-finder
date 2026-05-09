# Polarity And Negation Implementation Event Log

Date: 2026-05-08

## Resume State

- Phase: implementation complete; full package quality gate still pending.
- Scope: `packages/gia-evidence-finder` only.
- Primary files changed:
  - `src/gia_evidence_finder/polarity.py`
  - `src/gia_evidence_finder/ranking.py`
  - `src/gia_evidence_finder/extractor.py`
  - `src/gia_evidence_finder/polarity_benchmarks.py`
  - `src/gia_evidence_finder/benchmark_series.py`
  - `src/gia_evidence_finder/calibration.py`
  - `src/gia_evidence_finder/intent_compiler.py`
  - `tests/test_polarity_benchmarks.py`
  - `tests/test_intent_compiler.py`
  - `docs/benchmark-series.md`
  - `README.md`
- Next check: run `uv run ruff check .`, `uv run mypy`, and `uv run pytest`
  from `packages/gia-evidence-finder`.

## Step 001 - Establish Scope

- Read repository guidance, root README/docs map, current package contracts, and
  existing evidence benchmark docs.
- Confirmed unrelated dirty files exist in root TraverceDB code and docs; this
  pass intentionally avoids them.
- Scope narrowed to contradiction and negation handling in the side package.

## Step 002 - Reproduce The Failure

- Ran the current extractor on the `ultrachess` README.
- Confirmed the false-support pattern: a positive claim about evaluation/search
  could select a negated limitation span.
- Also confirmed positive support still worked for normal capability claims.

## Step 003 - Inspect Existing Contracts

- Reviewed:
  - `contracts.py`
  - `ranking.py`
  - `extractor.py`
  - `intent_compiler.py`
  - `evaluation.py`
  - `calibration.py`
  - README, non-README, and relation benchmark suites.
- Observation: contradiction labels existed in benchmark contracts, but the
  default extractor had no generic polarity feature and labels were assigned
  only by score thresholds.

## Step 004 - Add Focused Benchmark

- Added `polarity_benchmarks.py`.
- Added `polarity_negation_v1` with 20 reviewed cases over 5 excerpts.
- Added `polarity_evidence_benchmark_v1` with train/dev/test splits.
- Added direct tests for:
  - positive claim contradicted by `No ...`;
  - negative claim supported by `No ...`;
  - `not supported`;
  - `without`;
  - `must not`;
  - `instead of`;
  - `not prose`;
  - `rejected`;
  - `not just`.

## Step 005 - Implement Polarity Analyzer

- Added `polarity.py` as a pure deterministic component.
- Integrated polarity features into `IntentAwareRanker`.
- Added typed contradiction labeling in `EvidenceExtractor`.
- Preserved contradicted spans as candidates, but excluded them from support
  matches.

## Step 006 - Fix Compiler Behavior

- Updated `compile_intent` to extract lower-case lexical anchors, so raw claims
  with lowercase capability words get required facets.
- Removed automatic excluded facets for negated phrases in the user claim. A
  negative claim is now treated as the claim to support.

## Step 007 - Calibration Repair

- Added support labels to `CandidateScore`.
- Made threshold replay label-aware: a `contradicts` candidate cannot be
  promoted into support by a low threshold.
- Switched the intent-aware default holdout policy to extractor labels. Keyword
  and BM25 baselines still use score-threshold calibration.

## Step 008 - Iteration Notes

- Initial polarity windows over-penalized negative support cases.
- Added punctuation boundaries so `avoid custom agents; be agentless...` does
  not let `avoid` negate the later positive explanation.
- Added postposed negation handling for list forms like `Chess960 ... are not
  supported`.
- Added special handling for `not just` so exclusivity is negated, not the
  entity token itself.

## Step 009 - Targeted Verification

- Command:

```sh
uv run pytest -q tests/test_polarity_benchmarks.py tests/test_intent_compiler.py tests/test_evaluation_training.py tests/test_extraction.py tests/test_document_benchmarks.py tests/test_readme_benchmarks.py tests/test_benchmark_series.py
```

- Result: 44 passed.

## Step 010 - Benchmark Verification

- Command:

```sh
uv run gia-evidence-finder benchmark-series --series polarity_evidence_benchmark_v1 --split all
```

- Result: intent-aware default reached 1.0000 decision, top-1 support, abstain,
  and 0.0000 contradiction-supported rate on train/dev/test.
- Keyword and BM25 baselines still false-support contradiction spans on the
  focused suite.

## Step 011 - Mixed Regression Check

- Command:

```sh
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

- Result:
  - `polarity_evidence_benchmark_v1`: all splits 1.0000 decision/top-1/abstain,
    0.0000 forbidden-supported.
  - `mixed_evidence_benchmark_v3`: train 1.0000/1.0000/1.0000/0.0000; dev
    0.9474/1.0000/0.8750/0.0526; test 1.0000/1.0000/1.0000/0.0000.

## Open Follow-Up

- Add more reviewed examples before treating polarity as a public SOTA proof.
- Extend polarity coverage to double negation, temporal negation, table rows,
  and numeric/comparative contradiction.
