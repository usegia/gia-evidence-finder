# Domain Competitor Benchmarking

Date: 2026-05-09

## What Changed

gia evidence finder now has a first-pass benchmark and reporting structure for
the three quality axes needed before broader SOTA claims:

1. Retrieval and ranking quality.
2. Support and abstention decision quality.
3. Safety/refusal quality for forbidden, near-miss, contradiction, and
   insufficient-context spans.

The new `domain_evidence_benchmark_v4` series expands beyond README excerpts
into project-management, people-search, apartment-search, and technical/product
source artifacts. It uses small curated excerpts rather than full upstream
documents.

## Benchmark Targets

The expanded suite has:

- 40 source documents.
- 320 reviewed cases.
- 4 domains.
- 22 genres.
- document-isolated train/dev/test splits.
- source hashes and review-source metadata on every document.

Run the audit with:

```sh
uv run gia-evidence-finder audit-benchmark-series --series domain_evidence_benchmark_v4
```

## Competitor Curves

Run local competitor curves with:

```sh
uv run gia-evidence-finder benchmark-competitors \
  --series domain_evidence_benchmark_v4 \
  --split test
```

The default competitor set is staged for reproducibility. Hosted competitors are
available only when explicitly requested:

```sh
uv run gia-evidence-finder benchmark-competitors \
  --series domain_evidence_benchmark_v4 \
  --split test \
  --include-hosted \
  --score-cache-jsonl .tmp/domain-competitor-reranker-cache.jsonl
```

Markdown output is available for reports:

```sh
uv run gia-evidence-finder benchmark-competitors \
  --series domain_evidence_benchmark_v4 \
  --split test \
  --competitor typed_default \
  --competitor bm25 \
  --format markdown
```

## Current Boundary

This implementation creates the benchmark and reporting protocol. Public SOTA
claims still require running the optional model and hosted competitors on a
frozen test split, recording unavailable systems, and comparing all systems
across the three axes rather than a single leaderboard.
