from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from gia_evidence_finder.contracts import DocumentSpan, EvidenceDocument, SpanKind
from gia_evidence_finder.parsing import MarkdownSpanParser
from gia_evidence_finder.text import tokenize

CURATION_SPAN_KINDS: tuple[SpanKind, ...] = (
    SpanKind.SENTENCE,
    SpanKind.BULLET,
    SpanKind.TABLE_ROW,
)
LOCAL_CURATION_SPAN_KINDS: tuple[SpanKind, ...] = (
    SpanKind.SENTENCE,
    SpanKind.PARAGRAPH,
    SpanKind.BULLET,
    SpanKind.TABLE_ROW,
)
LOCAL_DOCUMENT_PATTERNS: tuple[str, ...] = ("*.md", "*.markdown", "*.rst", "*.txt")
WHITESPACE_RE = re.compile(r"\s+")
DOCUMENT_ID_RE = re.compile(r"[^a-zA-Z0-9]+")

STOPWORDS: frozenset[str] = frozenset(
    {
        "about",
        "after",
        "also",
        "and",
        "are",
        "based",
        "been",
        "but",
        "can",
        "for",
        "from",
        "has",
        "have",
        "into",
        "its",
        "more",
        "not",
        "our",
        "that",
        "the",
        "their",
        "this",
        "through",
        "to",
        "use",
        "using",
        "with",
        "you",
        "your",
    }
)


@dataclass(frozen=True)
class CurationQueueItem:
    id: str
    document_id: str
    source: str | None
    span_id: str
    kind: SpanKind
    heading_path: tuple[str, ...]
    text: str
    anchor_terms: tuple[str, ...]
    suggested_label: str
    review_status: str = "unreviewed"

    def to_json_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "document_id": self.document_id,
            "source": self.source,
            "span_id": self.span_id,
            "kind": self.kind.value,
            "heading_path": list(self.heading_path),
            "text": self.text,
            "anchor_terms": list(self.anchor_terms),
            "suggested_label": self.suggested_label,
            "review_status": self.review_status,
        }


def curation_queue_from_document(
    document: EvidenceDocument,
    *,
    max_items: int = 40,
    min_chars: int = 30,
    max_chars: int = 500,
    span_kinds: tuple[SpanKind, ...] = CURATION_SPAN_KINDS,
    require_heading: bool = True,
    dedupe_text: bool = True,
) -> tuple[CurationQueueItem, ...]:
    if max_items < 1:
        raise ValueError("max_items must be positive")
    if not span_kinds:
        raise ValueError("span_kinds must not be empty")
    candidates = [
        span
        for span in document.spans
        if _is_curation_candidate(
            span,
            min_chars=min_chars,
            max_chars=max_chars,
            span_kinds=span_kinds,
            require_heading=require_heading,
        )
    ]
    candidates.sort(key=lambda span: _candidate_sort_key(span, span_kinds=span_kinds))
    selected = _dedupe_spans(candidates, limit=max_items) if dedupe_text else candidates[:max_items]
    return tuple(_queue_item(document, span) for span in selected)


def curation_queue_from_fetched_readmes(
    input_dir: Path,
    *,
    max_items_per_doc: int = 40,
    require_heading: bool = True,
    dedupe_text: bool = True,
) -> tuple[CurationQueueItem, ...]:
    manifest_path = input_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"manifest not found: {manifest_path}")
    manifest = cast(
        dict[str, object],
        json.loads(manifest_path.read_text(encoding="utf-8")),
    )
    entries = cast(list[dict[str, object]], manifest.get("sources", []))
    items: list[CurationQueueItem] = []
    parser = MarkdownSpanParser()
    for entry in entries:
        document_id = str(entry["id"])
        filename = str(entry["filename"])
        raw_url = str(entry["raw_url"])
        text = (input_dir / filename).read_text(encoding="utf-8")
        document = parser.parse(text, document_id=document_id, source=raw_url)
        items.extend(
            curation_queue_from_document(
                document,
                max_items=max_items_per_doc,
                require_heading=require_heading,
                dedupe_text=dedupe_text,
            )
        )
    if dedupe_text:
        items = list(_dedupe_queue_items(items))
    return tuple(items)


def curation_queue_from_local_documents(
    input_path: Path,
    *,
    patterns: tuple[str, ...] = LOCAL_DOCUMENT_PATTERNS,
    max_items_per_doc: int = 40,
    require_heading: bool = True,
    dedupe_text: bool = True,
) -> tuple[CurationQueueItem, ...]:
    paths = _local_document_paths(input_path, patterns=patterns)
    parser = MarkdownSpanParser()
    items: list[CurationQueueItem] = []
    document_ids = _document_ids_for_paths(paths, root=input_path if input_path.is_dir() else None)
    for path in paths:
        document = parser.parse(
            path.read_text(encoding="utf-8"),
            document_id=document_ids[path],
            source=str(path),
        )
        items.extend(
            curation_queue_from_document(
                document,
                max_items=max_items_per_doc,
                span_kinds=LOCAL_CURATION_SPAN_KINDS,
                require_heading=require_heading,
                dedupe_text=dedupe_text,
            )
        )
    if dedupe_text:
        items = list(_dedupe_queue_items(items))
    return tuple(items)


def write_curation_queue_jsonl(
    items: tuple[CurationQueueItem, ...],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        "\n".join(json.dumps(item.to_json_dict(), sort_keys=True) for item in items)
        + ("\n" if items else ""),
        encoding="utf-8",
    )


def _is_curation_candidate(
    span: DocumentSpan,
    *,
    min_chars: int,
    max_chars: int,
    span_kinds: tuple[SpanKind, ...],
    require_heading: bool,
) -> bool:
    text = span.text.strip()
    if span.kind not in span_kinds:
        return False
    if not min_chars <= len(text) <= max_chars:
        return False
    if require_heading and not span.heading_path:
        return False
    if text.startswith("[!") or text.startswith("!["):
        return False
    return bool(_anchor_terms(text))


def _candidate_sort_key(
    span: DocumentSpan,
    *,
    span_kinds: tuple[SpanKind, ...],
) -> tuple[int, int, int]:
    heading_bonus = 0 if span.heading_path else 1
    kind_rank = span_kinds.index(span.kind)
    return (heading_bonus, kind_rank, span.ordinal)


def _queue_item(document: EvidenceDocument, span: DocumentSpan) -> CurationQueueItem:
    anchor_terms = _anchor_terms(span.text)
    return CurationQueueItem(
        id=f"{span.id}:support_candidate",
        document_id=document.id,
        source=document.source,
        span_id=span.id,
        kind=span.kind,
        heading_path=span.heading_path,
        text=span.text,
        anchor_terms=anchor_terms,
        suggested_label=_suggested_label(span, anchor_terms),
    )


def _dedupe_spans(
    spans: list[DocumentSpan],
    *,
    limit: int,
) -> list[DocumentSpan]:
    selected: list[DocumentSpan] = []
    seen: set[str] = set()
    for span in spans:
        key = _normalized_text(span.text)
        if key in seen:
            continue
        seen.add(key)
        selected.append(span)
        if len(selected) >= limit:
            break
    return selected


def _dedupe_queue_items(
    items: list[CurationQueueItem],
) -> tuple[CurationQueueItem, ...]:
    selected: list[CurationQueueItem] = []
    seen: set[str] = set()
    for item in items:
        key = _normalized_text(item.text)
        if key in seen:
            continue
        seen.add(key)
        selected.append(item)
    return tuple(selected)


def _local_document_paths(input_path: Path, *, patterns: tuple[str, ...]) -> tuple[Path, ...]:
    if not patterns:
        raise ValueError("patterns must not be empty")
    if input_path.is_file():
        return (input_path,)
    if not input_path.is_dir():
        raise FileNotFoundError(f"input path not found: {input_path}")
    paths: set[Path] = set()
    for pattern in patterns:
        paths.update(path for path in input_path.rglob(pattern) if path.is_file())
    if not paths:
        raise ValueError(f"no local documents matched under {input_path}")
    return tuple(sorted(paths))


def _document_ids_for_paths(paths: tuple[Path, ...], *, root: Path | None) -> dict[Path, str]:
    counts: dict[str, int] = {}
    document_ids: dict[Path, str] = {}
    for path in paths:
        base_id = _document_id_for_path(path, root=root)
        count = counts.get(base_id, 0) + 1
        counts[base_id] = count
        document_ids[path] = base_id if count == 1 else f"{base_id}_{count}"
    return document_ids


def _document_id_for_path(path: Path, *, root: Path | None) -> str:
    relative = path.relative_to(root) if root is not None else Path(path.name)
    without_suffix = relative.with_suffix("").as_posix()
    document_id = DOCUMENT_ID_RE.sub("_", without_suffix).strip("_").lower()
    return document_id or "document"


def _normalized_text(text: str) -> str:
    return WHITESPACE_RE.sub(" ", text).strip().casefold()


def _anchor_terms(text: str, *, limit: int = 8) -> tuple[str, ...]:
    terms: list[str] = []
    seen: set[str] = set()
    for token in tokenize(text):
        if len(token) < 3 or token in STOPWORDS or token in seen:
            continue
        seen.add(token)
        terms.append(token)
        if len(terms) >= limit:
            break
    return tuple(terms)


def _suggested_label(span: DocumentSpan, anchor_terms: tuple[str, ...]) -> str:
    heading = " / ".join(span.heading_path)
    anchors = " ".join(anchor_terms[:4])
    if heading and anchors:
        return f"{heading}: {anchors}"
    return anchors or span.text[:80]
