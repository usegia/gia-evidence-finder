from __future__ import annotations

import json
from pathlib import Path

from gia_evidence_finder import (
    MarkdownSpanParser,
    curation_queue_from_document,
    curation_queue_from_fetched_readmes,
    curation_queue_from_local_documents,
    write_curation_queue_jsonl,
)


def test_curation_queue_from_document_exports_unreviewed_span_candidates() -> None:
    document = MarkdownSpanParser().parse(
        """# Project

## Features

- Extracts evidence spans from technical documents.
- Supports abstention when claims have no direct evidence.
""",
        document_id="fixture",
        source="https://example.test/README.md",
    )

    items = curation_queue_from_document(document, max_items=5)

    assert len(items) == 2
    assert all(item.review_status == "unreviewed" for item in items)
    assert items[0].heading_path == ("Project", "Features")
    assert "proof" in items[0].anchor_terms
    assert items[0].suggested_label.startswith("Project / Features")


def test_curation_queue_from_fetched_readme_manifest(tmp_path: Path) -> None:
    (tmp_path / "fixture.md").write_text(
        """# Fixture

Fixture parses source documents into reviewable spans. It keeps generated cases unreviewed.
""",
        encoding="utf-8",
    )
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "sources": [
                    {
                        "id": "fixture",
                        "filename": "fixture.md",
                        "raw_url": "https://example.test/README.md",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    items = curation_queue_from_fetched_readmes(tmp_path, max_items_per_doc=3)
    output_path = tmp_path / "queue.jsonl"
    write_curation_queue_jsonl(items, output_path)
    rows = [
        json.loads(line)
        for line in output_path.read_text(encoding="utf-8").splitlines()
    ]

    assert len(items) == 2
    assert rows[0]["document_id"] == "fixture"
    assert rows[0]["review_status"] == "unreviewed"
    assert rows[0]["source"] == "https://example.test/README.md"


def test_curation_queue_from_local_documents_scans_non_readme_docs(tmp_path: Path) -> None:
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    (docs_dir / "decision.md").write_text(
        """# Decision

## Storage

The storage contract preserves replayable write epochs for later verification.
""",
        encoding="utf-8",
    )
    (docs_dir / "ignored.py").write_text("print('ignored')", encoding="utf-8")

    items = curation_queue_from_local_documents(docs_dir, max_items_per_doc=5)

    assert len(items) == 1
    assert items[0].document_id == "decision"
    assert items[0].source == str(docs_dir / "decision.md")
    assert "replayable" in items[0].anchor_terms


def test_curation_queue_requires_heading_context_by_default() -> None:
    document = MarkdownSpanParser().parse(
        "- This bullet has enough words to be a candidate but no heading context.",
        document_id="fixture",
    )

    assert curation_queue_from_document(document, max_items=5) == ()
    assert len(
        curation_queue_from_document(document, max_items=5, require_heading=False)
    ) == 1


def test_curation_queue_deduplicates_normalized_text_by_default() -> None:
    document = MarkdownSpanParser().parse(
        """# Fixture

Repeated evidence candidate with enough context.
Repeated   evidence candidate with enough context.
Different evidence candidate with enough context.
""",
        document_id="fixture",
    )

    items = curation_queue_from_document(document, max_items=5)

    assert [item.text for item in items] == [
        "Repeated evidence candidate with enough context.",
        "Different evidence candidate with enough context.",
    ]
