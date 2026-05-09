from __future__ import annotations

import pytest

from gia_evidence_finder import EvidenceDocument, MarkdownSpanParser

DOC_TEXT = """# TraverceDB

TraverceDB is a trainable semantic graph database. Search results carry evidence
that explains why each record matched.

## Semantic Search

Open semantic gates search uncurated customer text before curated concept packs
exist. Later, curated concept packs can promote repeated matches into stable
concept IDs.

- Production benchmark gates must use Qwen3 embeddings through OpenRouter.
  The deterministic provider is test-only and must never seed benchmark data.

## Local Quality

Run the full quality pass after major code or contract changes:

```sh
uv run ruff check . && uv run mypy && uv run pytest
```

Showcase checks use Bun and Next.js:

```sh
cd showcases/travercedb-showcase
bun run typecheck
bun run lint
```

## API Security

Production mutating routes require the x-traverce-admin-token header. Query and
status routes remain readable without that header.

## Fixtures

Seed the project-management benchmark example into Postgres:

```sh
curl -X POST http://localhost:8000/admin/examples/project-management/seed
```

Frontend requests can select the fixture by including
"schema_name": "project_management_benchmark" in /query, /query/plan, or
/query/structured payloads.

## Operations

Postgres migrations use a ledger with checksum verification and advisory-lock
protected upgrades.

| Capability | Status |
| --- | --- |
| evidence traces | supported |
| hallucinated answers | rejected |
"""


@pytest.fixture
def evidence_document() -> EvidenceDocument:
    return MarkdownSpanParser().parse(DOC_TEXT, document_id="repo_readme")
