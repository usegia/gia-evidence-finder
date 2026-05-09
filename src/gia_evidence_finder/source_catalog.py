from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import cast
from urllib.request import Request, urlopen

from gia_evidence_finder.parsing import MarkdownSpanParser

FetchText = Callable[[str], str]


@dataclass(frozen=True)
class ReadmeSource:
    id: str
    repository: str
    raw_url: str
    license_name: str
    focus: str


@dataclass(frozen=True)
class FetchedReadme:
    source: ReadmeSource
    text: str
    sha256: str
    byte_count: int
    line_count: int
    span_count: int

    @property
    def filename(self) -> str:
        suffix = ".rst" if self.source.raw_url.endswith(".rst") else ".md"
        return f"{self.source.id}{suffix}"

    def manifest_entry(self) -> dict[str, object]:
        return {
            "id": self.source.id,
            "repository": self.source.repository,
            "raw_url": self.source.raw_url,
            "license_name": self.source.license_name,
            "focus": self.source.focus,
            "filename": self.filename,
            "sha256": self.sha256,
            "byte_count": self.byte_count,
            "line_count": self.line_count,
            "span_count": self.span_count,
        }


README_SOURCE_CATALOG: tuple[ReadmeSource, ...] = (
    ReadmeSource(
        id="react",
        repository="facebook/react",
        raw_url="https://raw.githubusercontent.com/facebook/react/main/README.md",
        license_name="MIT",
        focus="frontend library",
    ),
    ReadmeSource(
        id="kubernetes",
        repository="kubernetes/kubernetes",
        raw_url="https://raw.githubusercontent.com/kubernetes/kubernetes/master/README.md",
        license_name="Apache-2.0",
        focus="container orchestration",
    ),
    ReadmeSource(
        id="django",
        repository="django/django",
        raw_url="https://raw.githubusercontent.com/django/django/main/README.rst",
        license_name="BSD-3-Clause",
        focus="web framework",
    ),
    ReadmeSource(
        id="fastapi",
        repository="fastapi/fastapi",
        raw_url="https://raw.githubusercontent.com/fastapi/fastapi/master/README.md",
        license_name="MIT",
        focus="api framework",
    ),
    ReadmeSource(
        id="pytorch",
        repository="pytorch/pytorch",
        raw_url="https://raw.githubusercontent.com/pytorch/pytorch/main/README.md",
        license_name="BSD-style",
        focus="machine learning framework",
    ),
    ReadmeSource(
        id="rust",
        repository="rust-lang/rust",
        raw_url="https://raw.githubusercontent.com/rust-lang/rust/master/README.md",
        license_name="MIT OR Apache-2.0",
        focus="programming language",
    ),
    ReadmeSource(
        id="vscode",
        repository="microsoft/vscode",
        raw_url="https://raw.githubusercontent.com/microsoft/vscode/main/README.md",
        license_name="MIT",
        focus="developer tool",
    ),
    ReadmeSource(
        id="ansible",
        repository="ansible/ansible",
        raw_url="https://raw.githubusercontent.com/ansible/ansible/devel/README.md",
        license_name="GPL-3.0",
        focus="automation platform",
    ),
    ReadmeSource(
        id="numpy",
        repository="numpy/numpy",
        raw_url="https://raw.githubusercontent.com/numpy/numpy/main/README.md",
        license_name="BSD-3-Clause",
        focus="numerical computing",
    ),
    ReadmeSource(
        id="pandas",
        repository="pandas-dev/pandas",
        raw_url="https://raw.githubusercontent.com/pandas-dev/pandas/main/README.md",
        license_name="BSD-3-Clause",
        focus="data analysis",
    ),
    ReadmeSource(
        id="scikit_learn",
        repository="scikit-learn/scikit-learn",
        raw_url="https://raw.githubusercontent.com/scikit-learn/scikit-learn/main/README.rst",
        license_name="BSD-3-Clause",
        focus="machine learning library",
    ),
    ReadmeSource(
        id="tensorflow",
        repository="tensorflow/tensorflow",
        raw_url="https://raw.githubusercontent.com/tensorflow/tensorflow/master/README.md",
        license_name="Apache-2.0",
        focus="machine learning framework",
    ),
    ReadmeSource(
        id="transformers",
        repository="huggingface/transformers",
        raw_url="https://raw.githubusercontent.com/huggingface/transformers/main/README.md",
        license_name="Apache-2.0",
        focus="model library",
    ),
    ReadmeSource(
        id="langchain",
        repository="langchain-ai/langchain",
        raw_url="https://raw.githubusercontent.com/langchain-ai/langchain/master/README.md",
        license_name="MIT",
        focus="llm application framework",
    ),
    ReadmeSource(
        id="node",
        repository="nodejs/node",
        raw_url="https://raw.githubusercontent.com/nodejs/node/main/README.md",
        license_name="MIT",
        focus="runtime",
    ),
    ReadmeSource(
        id="typescript",
        repository="microsoft/TypeScript",
        raw_url="https://raw.githubusercontent.com/microsoft/TypeScript/main/README.md",
        license_name="Apache-2.0",
        focus="programming language",
    ),
    ReadmeSource(
        id="vue",
        repository="vuejs/core",
        raw_url="https://raw.githubusercontent.com/vuejs/core/main/README.md",
        license_name="MIT",
        focus="frontend framework",
    ),
    ReadmeSource(
        id="go",
        repository="golang/go",
        raw_url="https://raw.githubusercontent.com/golang/go/master/README.md",
        license_name="BSD-3-Clause",
        focus="programming language",
    ),
    ReadmeSource(
        id="spark",
        repository="apache/spark",
        raw_url="https://raw.githubusercontent.com/apache/spark/master/README.md",
        license_name="Apache-2.0",
        focus="data processing engine",
    ),
)


def readme_source_catalog() -> tuple[ReadmeSource, ...]:
    return README_SOURCE_CATALOG


def readme_source_by_id(source_id: str) -> ReadmeSource:
    sources = {source.id: source for source in README_SOURCE_CATALOG}
    try:
        return sources[source_id]
    except KeyError as exc:
        raise ValueError(f"unknown README source {source_id!r}") from exc


def fetch_readme_source(
    source: ReadmeSource,
    *,
    fetch_text: FetchText | None = None,
) -> FetchedReadme:
    text = (fetch_text or _fetch_url_text)(source.raw_url)
    encoded = text.encode("utf-8")
    document = MarkdownSpanParser().parse(
        text,
        document_id=source.id,
        source=source.raw_url,
    )
    return FetchedReadme(
        source=source,
        text=text,
        sha256=hashlib.sha256(encoded).hexdigest(),
        byte_count=len(encoded),
        line_count=len(text.splitlines()),
        span_count=len(document.spans),
    )


def fetch_readme_sources(
    sources: Iterable[ReadmeSource],
    *,
    fetch_text: FetchText | None = None,
) -> tuple[FetchedReadme, ...]:
    return tuple(
        fetch_readme_source(source, fetch_text=fetch_text)
        for source in sources
    )


def write_fetched_readmes(
    fetched: Iterable[FetchedReadme],
    output_dir: Path,
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    entries: list[dict[str, object]] = []
    for item in fetched:
        (output_dir / item.filename).write_text(item.text, encoding="utf-8")
        entries.append(item.manifest_entry())
    manifest: dict[str, object] = {
        "source_count": len(entries),
        "sources": entries,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return manifest


def _fetch_url_text(url: str) -> str:
    request = Request(
        url,
        headers={"User-Agent": "gia-evidence-finder-source-fetcher"},
    )
    with urlopen(request, timeout=30) as response:
        charset = response.headers.get_content_charset() or "utf-8"
        body = cast(bytes, response.read())
        return body.decode(charset, errors="replace")
