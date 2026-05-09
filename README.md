# gia evidence finder

[![PyPI](https://img.shields.io/pypi/v/gia-evidence-finder.svg)](https://pypi.org/project/gia-evidence-finder/)

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
cat > sample.md <<'EOF'
# README

No evaluation, no search, no opening book.
UltraChess is a browser-based chess variant playground.
EOF

gia-evidence-finder extract sample.md \
  --claim "UltraChess is browser-based"
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

`gia evidence finder` is not a summarizer and not just semantic retrieval. It
takes a claim or typed intent, scores source spans, and decides whether each
candidate is direct support, a near miss, a contradiction, insufficient context,
or irrelevant.

It handles these evidence patterns:

- **Direct support**: the claim is actually stated by the source.
  - Claim: `UltraChess is browser-based`.
  - Source: `UltraChess is a browser-based chess variant playground.`
  - Result: returns the sentence as `supports`.

- **Smallest useful span extraction**: when a paragraph, section, or heading is
  related, the extractor still prefers the minimal sentence, bullet, code block,
  or table row that proves the claim.
  - Claim: `the quality gate runs ruff, mypy, and pytest`.
  - Source: `uv run ruff check . && uv run mypy && uv run pytest`.
  - Result: returns the code span, not the whole README section.

- **Near-miss refusal**: a span can share many words with the claim and still
  fail to prove it.
  - Claim: `FastAPI provides automatic interactive docs, not runtime speed`.
  - Source: `Very high performance, on par with NodeJS and Go.`
  - Result: labels the speed span as `near_miss`, not support for the docs
    claim.

- **Contradiction and negation**: the extractor distinguishes positive claims
  from negative source text, and negative claims from negative support.
  - Claim: `includes evaluation, search, and an opening book`.
  - Source: `No evaluation, no search, no opening book.`
  - Result: labels the span as `contradicts`.
  - Claim: `has no evaluation, no search, and no opening book`.
  - Source: `No evaluation, no search, no opening book.`
  - Result: returns the same span as `supports`.

- **Explicit counterclaims**: if the source says the claim is false and gives
  the opposite policy, it is a contradiction even when lexical overlap is high.
  - Claim: `PR discussion forbids open semantic gate diagnostics`.
  - Source: `PR discussion does not forbid open semantic gate diagnostics; it
    requires open semantic gate diagnostics.`
  - Result: labels the span as `contradicts`.

- **Insufficient-context refusal**: the extractor can return the best diagnostic
  span explaining why a claim is not proven.
  - Claim: `a dependent library itself provides open semantic gate diagnostics`.
  - Source: `A dependent library is mentioned near open semantic gate
    diagnostics, but this does not prove the dependency provides it.`
  - Result: labels the span as `insufficient_context`.

- **Relation and subject binding**: true facts do not get transferred across
  nearby entities.
  - Claim: `Pydantic itself is on par with NodeJS and Go`.
  - Source: `FastAPI is very high performance, on par with NodeJS and Go, thanks
    to Starlette and Pydantic.`
  - Result: abstains because the performance claim belongs to FastAPI, not
    Pydantic.

- **Source reliability conflicts**: a current or official source can support a
  claim while an old note, archived note, or outdated blurb is treated as a
  conflicting diagnostic span.
  - Claim: `the current spec says open semantic gate diagnostics is required`.
  - Source: `The current spec says open semantic gate diagnostics is required.`
  - Conflicting source: `An archived note incorrectly says open semantic gate
    diagnostics is optional.`
  - Result: returns the current spec as support and keeps the archived note as a
    contradiction candidate.

- **Numeric, date, money, duration, percentage, version, and range checks**:
  semantically similar spans are rejected when the value is wrong.
  - Claim: `OAuth hardening project was created in 2026`.
  - Source: `OAuth hardening project was created in 2025.`
  - Result: labels the source as `contradicts`.

- **Role-bound quantities**: matching numbers are not enough; the number must
  be attached to the right local role.
  - Claim: `Project started in 2024 and ended in 2025`.
  - Source: `Project started in 2025 and ended in 2024.`
  - Result: labels the swapped dates as `contradicts`.
  - Claim: `Rent is $3,200 and deposit is $1,000`.
  - Source: `Rent is $1,000 and deposit is $3,200.`
  - Result: labels the swapped money roles as `contradicts`.

- **Deterministic abstention**: when no span directly supports the claim, the
  extractor returns no support matches and keeps ranked diagnostic candidates
  for review.

- **Traceable outputs**: every `SpanMatch` includes a stable span id, source
  offsets, label, score, feature breakdown, and reasons so downstream systems
  can audit why a span was accepted or rejected.

## Why This Exists

Applications that build claim graphs, semantic search systems, agents, or
review pipelines need a stricter primitive than chunk retrieval. A semantically
similar chunk can still be wrong evidence. `gia evidence finder` is designed to
answer the narrower question:

```text
Does this source span directly support this claim?
```

It can be used independently, or as the source-grounding layer for systems that
turn documents, comments, record fields, READMEs, specs, tickets, or parsed PDFs
into reviewable claims.

## Boundaries

`gia evidence finder` is a per-document evidence verifier and extractor. It
does not crawl a corpus, choose which document to search, generate claims, or
perform PDF/OCR/HTML ingestion by itself. The package ships Markdown/plain-text
span parsing and typed contracts; callers should convert other source formats
into `EvidenceDocument` spans before extraction.

The default path is deterministic. Optional model dependencies are for
baselines, reranking experiments, and future hybrids, not for hidden runtime
behavior in the core package.

## Benchmarks

The benchmark suite is versioned and intentionally reports support quality,
abstention quality, evidence-decision quality, and false-support risk
separately.

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

## Model Policy

Do not ship a fine-tuned model as the default runtime until it beats the
deterministic path on frozen reviewed benchmarks without increasing
false-support risk. A model release should be a separate optional artifact with
a model card, license review, artifact hash, training data provenance, and
reported `evidence_decision_accuracy`, not just reranking `MRR`.

The current recommended product shape is:

```text
deterministic typed support/abstain judgment by default
+ optional learned reranker for ordering after frozen holdout validation
```

See:

- [`docs/benchmark-series.md`](docs/benchmark-series.md)
- [`docs/product-readiness-2026-05-09.md`](docs/product-readiness-2026-05-09.md)
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

Publishing is automated through GitHub Actions. To publish a new PyPI release,
bump `version` in `pyproject.toml` and merge to `main`; the `Publish` workflow
runs the quality gate, builds the package, checks whether that version already
exists on PyPI, publishes only new versions through PyPI Trusted Publishing, and
creates the matching GitHub tag and release.

Manual TestPyPI publishing is still useful for release candidates:

```sh
uv publish --publish-url https://test.pypi.org/legacy/
```

## Repository Hygiene

Do not commit local benchmark downloads, generated review files, model caches,
virtualenvs, or package build outputs. The benchmark acquisition commands write
to `.tmp/` by default, and `.tmp/` is intentionally ignored.

## License

Apache-2.0.
