from __future__ import annotations

import html
import re
from dataclasses import dataclass

from gia_evidence_finder.contracts import DocumentSpan, EvidenceDocument, SpanKind

SENTENCE_BOUNDARY_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9`])")
TABLE_SEPARATOR_RE = re.compile(r"^\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)+\|?$")
LINKED_IMAGE_RE = re.compile(r"\[!\[([^\]]*)\]\([^)]+\)\]\([^)]+\)")
IMAGE_RE = re.compile(r"!\[([^\]]*)\]\([^)]+\)")
LINK_RE = re.compile(r"\[([^\]]+)\]\([^)]+\)")
HTML_TAG_RE = re.compile(r"<[^>]+>")
WHITESPACE_RE = re.compile(r"\s+")


@dataclass(frozen=True)
class _Line:
    text: str
    start: int
    end: int


class MarkdownSpanParser:
    def __init__(self, *, include_sentence_spans: bool = True) -> None:
        self._include_sentence_spans = include_sentence_spans

    def parse(self, text: str, *, document_id: str, source: str | None = None) -> EvidenceDocument:
        spans: list[DocumentSpan] = []
        heading_path: list[str] = []
        pending: list[_Line] = []
        pending_kind = SpanKind.PARAGRAPH
        code_lines: list[_Line] = []
        in_code = False
        ordinal = 0

        def next_id() -> str:
            nonlocal ordinal
            ordinal += 1
            return f"{document_id}:s{ordinal:04d}"

        def append_span(
            *,
            kind: SpanKind,
            value: str,
            start: int,
            end: int,
            parent_id: str | None = None,
            heading_override: tuple[str, ...] | None = None,
        ) -> DocumentSpan:
            previous_id = spans[-1].id if spans else None
            active_heading = (
                heading_override if heading_override is not None else tuple(heading_path)
            )
            span = DocumentSpan(
                id=next_id(),
                document_id=document_id,
                kind=kind,
                text=value,
                ordinal=ordinal,
                heading_path=active_heading,
                parent_id=parent_id,
                previous_id=previous_id,
                char_start=start,
                char_end=end,
            )
            if spans:
                previous = spans[-1]
                spans[-1] = DocumentSpan(
                    id=previous.id,
                    document_id=previous.document_id,
                    kind=previous.kind,
                    text=previous.text,
                    ordinal=previous.ordinal,
                    heading_path=previous.heading_path,
                    parent_id=previous.parent_id,
                    previous_id=previous.previous_id,
                    next_id=span.id,
                    char_start=previous.char_start,
                    char_end=previous.char_end,
                    metadata=previous.metadata,
                )
            spans.append(span)
            return span

        def flush_pending() -> None:
            nonlocal pending, pending_kind
            if not pending:
                return
            raw = _join_lines(pending, pending_kind)
            if raw:
                parent = append_span(
                    kind=pending_kind,
                    value=raw,
                    start=pending[0].start,
                    end=pending[-1].end,
                )
                if self._include_sentence_spans and pending_kind in {
                    SpanKind.PARAGRAPH,
                    SpanKind.BULLET,
                    SpanKind.TABLE_ROW,
                }:
                    for sentence, local_start, local_end in split_sentences(raw):
                        if sentence != raw:
                            append_span(
                                kind=SpanKind.SENTENCE,
                                value=sentence,
                                start=parent.char_start + local_start,
                                end=parent.char_start + local_end,
                                parent_id=parent.id,
                            )
            pending = []
            pending_kind = SpanKind.PARAGRAPH

        def flush_code() -> None:
            nonlocal code_lines
            if not code_lines:
                return
            value = "\n".join(line.text.rstrip() for line in code_lines).strip()
            if value:
                append_span(
                    kind=SpanKind.CODE,
                    value=value,
                    start=code_lines[0].start,
                    end=code_lines[-1].end,
                )
            code_lines = []

        for line in _lines_with_offsets(text):
            stripped = line.text.strip()
            if stripped.startswith("```"):
                if in_code:
                    flush_code()
                    in_code = False
                else:
                    flush_pending()
                    in_code = True
                continue
            if in_code:
                code_lines.append(line)
                continue
            if not stripped:
                flush_pending()
                continue
            heading_level = _heading_level(stripped)
            if heading_level is not None:
                flush_pending()
                heading_text = normalize_markdown_inline_text(
                    stripped[heading_level:].strip()
                )
                heading_path = heading_path[: heading_level - 1]
                heading_path.append(heading_text)
                append_span(
                    kind=SpanKind.HEADING,
                    value=heading_text,
                    start=line.start,
                    end=line.end,
                    heading_override=tuple(heading_path[:-1]),
                )
                continue
            if TABLE_SEPARATOR_RE.match(stripped):
                flush_pending()
                continue
            if _is_table_row(stripped):
                flush_pending()
                pending_kind = SpanKind.TABLE_ROW
                pending = [line]
                flush_pending()
                continue
            if _is_bullet(stripped):
                flush_pending()
                pending_kind = SpanKind.BULLET
                pending = [line]
                continue
            pending.append(line)
        flush_pending()
        flush_code()
        return EvidenceDocument(id=document_id, spans=tuple(spans), source=source)


def split_sentences(text: str) -> tuple[tuple[str, int, int], ...]:
    parts: list[tuple[str, int, int]] = []
    start = 0
    for match in SENTENCE_BOUNDARY_RE.finditer(text):
        sentence = text[start : match.start()].strip()
        if sentence:
            local_start = text.find(sentence, start, match.start() + 1)
            parts.append((sentence, local_start, local_start + len(sentence)))
        start = match.end()
    sentence = text[start:].strip()
    if sentence:
        local_start = text.find(sentence, start)
        parts.append((sentence, local_start, local_start + len(sentence)))
    return tuple(parts)


def normalize_markdown_inline_text(text: str) -> str:
    normalized = _normalize_markdown_inline_text(text, linked_image_replacement="")
    if normalized:
        return normalized
    return _normalize_markdown_inline_text(text, linked_image_replacement=r"\1") or text


def _normalize_markdown_inline_text(
    text: str,
    *,
    linked_image_replacement: str,
) -> str:
    without_badges = LINKED_IMAGE_RE.sub(linked_image_replacement, text)
    without_images = IMAGE_RE.sub(r"\1", without_badges)
    without_links = LINK_RE.sub(r"\1", without_images)
    without_tags = HTML_TAG_RE.sub(" ", without_links)
    decoded = html.unescape(without_tags).replace("\u00b7", " ")
    return WHITESPACE_RE.sub(" ", decoded).strip(" -|")


def _lines_with_offsets(text: str) -> tuple[_Line, ...]:
    lines: list[_Line] = []
    offset = 0
    for raw in text.splitlines(keepends=True):
        value = raw.rstrip("\n")
        lines.append(_Line(text=value, start=offset, end=offset + len(value)))
        offset += len(raw)
    if not text:
        return ()
    return tuple(lines)


def _heading_level(stripped_line: str) -> int | None:
    if not stripped_line.startswith("#"):
        return None
    level = len(stripped_line) - len(stripped_line.lstrip("#"))
    if level < 1 or level > 6 or len(stripped_line) <= level or stripped_line[level] != " ":
        return None
    return level


def _is_bullet(stripped_line: str) -> bool:
    return bool(re.match(r"^[-*+]\s+", stripped_line))


def _is_table_row(stripped_line: str) -> bool:
    return (
        stripped_line.startswith("|")
        and stripped_line.endswith("|")
        and "|" in stripped_line[1:-1]
    )


def _join_lines(lines: list[_Line], kind: SpanKind) -> str:
    value = " ".join(line.text.strip() for line in lines).strip()
    if kind == SpanKind.BULLET:
        return re.sub(r"^[-*+]\s+", "", value).strip()
    if kind == SpanKind.TABLE_ROW:
        cells = [cell.strip() for cell in value.strip("|").split("|")]
        return " | ".join(cell for cell in cells if cell)
    return value
