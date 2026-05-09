from __future__ import annotations

import json
from pathlib import Path

import pytest

from gia_evidence_finder import audit_curation_queue_jsonl


def test_audit_curation_queue_jsonl_reports_ready_diverse_queue(tmp_path: Path) -> None:
    queue_jsonl = tmp_path / "queue.jsonl"
    queue_jsonl.write_text(
        "\n".join(
            json.dumps(
                {
                    "document_id": document_id,
                    "kind": "sentence",
                    "heading_path": [document_id.title()],
                    "text": f"{document_id} provides reviewed evidence candidate {index}.",
                },
                sort_keys=True,
            )
            for index, document_id in enumerate(("alpha", "beta", "gamma"), start=1)
        )
        + "\n",
        encoding="utf-8",
    )

    audit = audit_curation_queue_jsonl(
        queue_jsonl=queue_jsonl,
        min_documents=3,
        min_items=3,
        max_document_share=0.5,
    )

    assert audit.item_count == 3
    assert audit.document_count == 3
    assert audit.kind_counts == {"sentence": 3}
    assert audit.warnings == ()


def test_audit_curation_queue_jsonl_flags_quality_risks(tmp_path: Path) -> None:
    queue_jsonl = tmp_path / "queue.jsonl"
    queue_jsonl.write_text(
        "\n".join(
            json.dumps(row, sort_keys=True)
            for row in (
                {
                    "document_id": "alpha",
                    "kind": "sentence",
                    "heading_path": [],
                    "text": "Same evidence candidate.",
                },
                {
                    "document_id": "alpha",
                    "kind": "sentence",
                    "heading_path": [],
                    "text": "Same evidence candidate.",
                },
            )
        )
        + "\n",
        encoding="utf-8",
    )

    audit = audit_curation_queue_jsonl(
        queue_jsonl=queue_jsonl,
        min_documents=3,
        min_items=3,
        max_document_share=0.75,
    )

    assert audit.duplicate_text_count == 1
    assert audit.empty_heading_count == 2
    assert any("target is at least 3" in warning for warning in audit.warnings)
    assert any("maximum share is 75%" in warning for warning in audit.warnings)
    assert any("no heading context" in warning for warning in audit.warnings)


def test_audit_curation_queue_jsonl_rejects_empty_queue(tmp_path: Path) -> None:
    queue_jsonl = tmp_path / "empty.jsonl"
    queue_jsonl.write_text("", encoding="utf-8")

    with pytest.raises(ValueError, match="at least one item"):
        audit_curation_queue_jsonl(queue_jsonl=queue_jsonl)
