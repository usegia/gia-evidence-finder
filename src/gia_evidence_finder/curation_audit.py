from __future__ import annotations

import json
import re
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class CurationQueueAudit:
    item_count: int
    document_count: int
    kind_counts: dict[str, int]
    document_counts: dict[str, int]
    empty_heading_count: int
    duplicate_text_count: int
    min_text_chars: int
    max_text_chars: int
    average_text_chars: float
    warnings: tuple[str, ...]


def audit_curation_queue_jsonl(
    *,
    queue_jsonl: Path,
    min_documents: int = 10,
    min_items: int = 200,
    max_document_share: float = 0.25,
) -> CurationQueueAudit:
    if min_documents < 1:
        raise ValueError("min_documents must be positive")
    if min_items < 1:
        raise ValueError("min_items must be positive")
    if not 0 < max_document_share <= 1:
        raise ValueError("max_document_share must be between 0 and 1")

    rows = _queue_rows(queue_jsonl)
    document_counts: Counter[str] = Counter()
    kind_counts: Counter[str] = Counter()
    text_counts: Counter[str] = Counter()
    text_lengths: list[int] = []
    empty_heading_count = 0
    for row in rows:
        document_counts[_required_string(row, "document_id")] += 1
        kind_counts[_required_string(row, "kind")] += 1
        text = _required_string(row, "text")
        text_lengths.append(len(text))
        text_counts[_normalize_text(text)] += 1
        if not _string_list(row.get("heading_path")):
            empty_heading_count += 1

    warnings = _audit_warnings(
        item_count=len(rows),
        document_counts=document_counts,
        empty_heading_count=empty_heading_count,
        duplicate_text_count=sum(count - 1 for count in text_counts.values() if count > 1),
        min_documents=min_documents,
        min_items=min_items,
        max_document_share=max_document_share,
    )
    return CurationQueueAudit(
        item_count=len(rows),
        document_count=len(document_counts),
        kind_counts=dict(sorted(kind_counts.items())),
        document_counts=dict(sorted(document_counts.items())),
        empty_heading_count=empty_heading_count,
        duplicate_text_count=sum(count - 1 for count in text_counts.values() if count > 1),
        min_text_chars=min(text_lengths),
        max_text_chars=max(text_lengths),
        average_text_chars=sum(text_lengths) / len(text_lengths),
        warnings=warnings,
    )


def _queue_rows(queue_jsonl: Path) -> tuple[Mapping[str, object], ...]:
    rows: list[Mapping[str, object]] = []
    for line_number, line in enumerate(
        queue_jsonl.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(f"line {line_number}: queue row must be an object")
        rows.append(row)
    if not rows:
        raise ValueError("curation queue JSONL must include at least one item")
    return tuple(rows)


def _audit_warnings(
    *,
    item_count: int,
    document_counts: Counter[str],
    empty_heading_count: int,
    duplicate_text_count: int,
    min_documents: int,
    min_items: int,
    max_document_share: float,
) -> tuple[str, ...]:
    warnings: list[str] = []
    if item_count < min_items:
        warnings.append(f"queue has {item_count} items; target is at least {min_items}")
    if len(document_counts) < min_documents:
        warnings.append(
            f"queue has {len(document_counts)} documents; target is at least {min_documents}"
        )
    top_document, top_count = document_counts.most_common(1)[0]
    top_share = top_count / item_count
    if top_share > max_document_share:
        warnings.append(
            f"document {top_document!r} contributes {top_count}/{item_count} items; "
            f"maximum share is {max_document_share:.0%}"
        )
    if empty_heading_count:
        warnings.append(f"{empty_heading_count} items have no heading context")
    if duplicate_text_count:
        warnings.append(f"{duplicate_text_count} items duplicate normalized text")
    return tuple(warnings)


def _required_string(row: Mapping[str, object], key: str) -> str:
    value = row.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"queue row {key!r} must be a non-empty string")
    return value


def _string_list(value: object) -> tuple[str, ...]:
    if not isinstance(value, list):
        return ()
    return tuple(item for item in value if isinstance(item, str) and item.strip())


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().casefold()
