# gia evidence finder

Find source-backed evidence for claims.

`gia evidence finder` extracts the smallest source spans that directly support a
claim, rejects near misses, detects contradictions, handles numeric and date
mismatches, abstains when support is absent, and returns traceable evidence
instead of generated prose.

Most retrieval tools return semantically similar chunks. `gia evidence finder`
is stricter: it asks whether a span actually supports the claim.

## Install

```sh
pip install gia-evidence-finder
```

Optional model and training dependencies are split out:

```sh
pip install "gia-evidence-finder[models]"
pip install "gia-evidence-finder[training]"
```

## Quick Example

```python
from gia_evidence_finder import EvidenceExtractor, MarkdownSpanParser, compile_intent

document = MarkdownSpanParser().parse(
    """
# README

No evaluation, no search, no opening book.
UltraChess is a browser-based chess variant playground.
""",
    document_id="readme",
)

intent = compile_intent("includes evaluation, search, and an opening book")
result = EvidenceExtractor.default().extract(intent, document)

assert result.abstained
assert result.candidates[0].label == "contradicts"
```

For a supported claim:

```python
intent = compile_intent("UltraChess is browser-based")
result = EvidenceExtractor.default().extract(intent, document)

assert not result.abstained
print(result.matches[0].span.text)
```

## CLI

```sh
gia-evidence-finder extract README.md \
  --claim "the project supports browser-based play"
```

Run the deterministic benchmark gates:

```sh
gia-evidence-finder benchmark-series --split all --calibrate-thresholds
gia-evidence-finder benchmark-series --series polarity_evidence_benchmark_v1 --split all
gia-evidence-finder benchmark-series --series quantifier_binding_evidence_benchmark_v2 --split all
```

Run competitor reports:

```sh
gia-evidence-finder benchmark-competitors \
  --series domain_evidence_benchmark_v4 \
  --split test
```

## Core Concepts

- `EvidenceDocument`: parsed source text with stable spans.
- `DocumentSpan`: a sentence, paragraph, bullet, table row, heading, or code span
  with offsets and context.
- `IntentSpec`: typed evidence intent for a claim.
- `SpanMatch`: a candidate span with score, label, features, and reasons.
- `SupportLabel`: `supports`, `near_miss`, `contradicts`,
  `insufficient_context`, `reject`, or `abstain`.
- `ExtractionResult`: final matches, rejected/diagnostic candidates, and trace.

## What It Handles

- Direct source support.
- Near misses that mention the right topic but do not prove the claim.
- Negation and contradiction, including negative claims supported by negative
  source text.
- Relation binding, where a true attribute belongs to a nearby dependency,
  product variant, person, organization, or tool rather than the claim subject.
- Numeric, date, money, duration, percentage, version, and range mismatches.
- Role-bound quantities such as `started in 2024` and `ended in 2025`.
- Deterministic abstention when support is absent.
- Stable span ids, offsets, labels, feature breakdowns, and traceable decisions.

## Why This Exists

Applications that build claim graphs, semantic search systems, agents, or
review pipelines need a stricter primitive than chunk retrieval. A semantically
similar chunk can still be wrong evidence. `gia evidence finder` is designed to
answer the narrower question:

```text
Does this source span directly support this claim?
```

It can be used independently, or as the source-grounding layer for systems that
turn documents, comments, record fields, READMEs, PDFs, specs, and tickets into
reviewable claims.

## Benchmarks

The benchmark suite is versioned and intentionally reports support quality,
abstention quality, and false-support risk separately.

Current benchmark coverage includes:

- popular README excerpts;
- hard README paraphrases and near misses;
- relation-binding traps;
- non-README specs, runbooks, release notes, and issue discussions;
- project-management, people-search, apartment-search, and technical/product
  source artifacts;
- polarity and negation cases;
- numeric, date, money, version, duration, and role-bound quantity cases.

The strongest measured direction is:

```text
typed deterministic support/abstain judgment
+ optional trained reranker ordering
```

The current reports show the typed default and typed-plus-trained-reranker
hybrid beating keyword overlap, BM25, embedding retrieval, and several generic
reranker setups on the reviewed evidence task, especially around
support/abstention, relation binding, negation, and numeric/date safety.

This is not a broad public SOTA claim. Larger frozen suites and more hosted
competitor runs are still required before making industry-wide claims.

See:

- [`docs/benchmark-series.md`](docs/benchmark-series.md)
- [`docs/model-baseline-results.md`](docs/model-baseline-results.md)
- [`docs/domain-competitor-benchmarking.md`](docs/domain-competitor-benchmarking.md)
- [`docs/polarity-negation-implementation-2026-05-08.md`](docs/polarity-negation-implementation-2026-05-08.md)
- [`docs/quantifier-date-implementation-2026-05-09.md`](docs/quantifier-date-implementation-2026-05-09.md)
- [`docs/quantifier-binding-implementation-2026-05-09.md`](docs/quantifier-binding-implementation-2026-05-09.md)

## Development

This repository uses `uv`.

```sh
uv sync --dev
uv run ruff check .
uv run mypy
uv run pytest
```

Build the package:

```sh
uv build
```

Publish to TestPyPI first:

```sh
uv publish --publish-url https://test.pypi.org/legacy/
```

Then publish to PyPI:

```sh
uv publish
```

## Repository Hygiene

Do not commit local benchmark downloads, generated review files, model caches,
virtualenvs, or package build outputs. The benchmark acquisition commands write
to `.tmp/` by default, and `.tmp/` is intentionally ignored.

## License

Apache-2.0.
