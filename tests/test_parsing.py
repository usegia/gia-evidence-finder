from __future__ import annotations

from gia_evidence_finder import MarkdownSpanParser, SpanKind, split_sentences


def test_parser_preserves_wrapped_bullets_code_tables_and_headings() -> None:
    text = """# Root

## Setup

- First line of a bullet
  continues on the next line.

```sh
uv run pytest
```

| Name | Status |
| --- | --- |
| traces | supported |
"""
    document = MarkdownSpanParser().parse(text, document_id="doc")

    bullet = next(span for span in document.spans if span.kind == SpanKind.BULLET)
    code = next(span for span in document.spans if span.kind == SpanKind.CODE)
    row = next(span for span in document.spans if span.kind == SpanKind.TABLE_ROW)

    assert bullet.text == "First line of a bullet continues on the next line."
    assert bullet.heading_path == ("Root", "Setup")
    assert code.text == "uv run pytest"
    assert row.text == "Name | Status"
    assert all(span.char_end >= span.char_start for span in document.spans)
    assert all(span.previous_id is not None for span in document.spans[1:])


def test_parser_creates_sentence_children_for_minimal_evidence() -> None:
    document = MarkdownSpanParser().parse(
        "# Product\n\nFirst sentence explains setup. Second sentence proves tracing.",
        document_id="doc",
    )

    paragraph = next(span for span in document.spans if span.kind == SpanKind.PARAGRAPH)
    children = [span for span in document.spans if span.parent_id == paragraph.id]

    assert [child.text for child in children] == [
        "First sentence explains setup.",
        "Second sentence proves tracing.",
    ]
    assert all(child.kind == SpanKind.SENTENCE for child in children)


def test_parser_normalizes_markdown_badge_headings_for_context() -> None:
    text = """# [React](https://react.dev/) &middot; [![npm version](https://img.shields.io/npm/v/react.svg)](https://www.npmjs.com/package/react)

Declarative views make your code more predictable.
"""
    document = MarkdownSpanParser().parse(text, document_id="doc")

    heading = next(span for span in document.spans if span.kind == SpanKind.HEADING)
    paragraph = next(span for span in document.spans if span.kind == SpanKind.PARAGRAPH)

    assert heading.text == "React"
    assert paragraph.heading_path == ("React",)


def test_parser_keeps_linked_image_only_headings_non_empty() -> None:
    text = """# [![Build status](https://img.shields.io/build.svg)](https://example.test/build)

The build status badge is the only title content.
"""
    document = MarkdownSpanParser().parse(text, document_id="doc")

    heading = next(span for span in document.spans if span.kind == SpanKind.HEADING)

    assert heading.text == "Build status"


def test_split_sentences_keeps_offsets() -> None:
    assert split_sentences("Alpha works. Beta proves it.") == (
        ("Alpha works.", 0, 12),
        ("Beta proves it.", 13, 28),
    )
