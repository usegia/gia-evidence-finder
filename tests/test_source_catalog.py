from __future__ import annotations

import json
from pathlib import Path

from gia_evidence_finder import (
    fetch_readme_source,
    readme_source_by_id,
    readme_source_catalog,
    write_fetched_readmes,
)


def test_readme_source_catalog_covers_popular_project_families() -> None:
    catalog = readme_source_catalog()
    ids = {source.id for source in catalog}

    assert len(catalog) >= 18
    assert {
        "react",
        "kubernetes",
        "fastapi",
        "pytorch",
        "numpy",
        "pandas",
        "tensorflow",
        "typescript",
    } <= ids
    assert readme_source_by_id("react").repository == "facebook/react"


def test_fetch_readme_source_uses_injected_fetch_boundary() -> None:
    source = readme_source_by_id("react")

    fetched = fetch_readme_source(source, fetch_text=_fake_fetch)

    assert fetched.source == source
    assert fetched.filename == "react.md"
    assert fetched.byte_count > 0
    assert fetched.line_count == 3
    assert fetched.span_count >= 2
    assert fetched.manifest_entry()["sha256"] == fetched.sha256


def test_write_fetched_readmes_writes_manifest_and_raw_text(tmp_path: Path) -> None:
    source = readme_source_by_id("react")
    fetched = fetch_readme_source(source, fetch_text=_fake_fetch)

    manifest = write_fetched_readmes((fetched,), tmp_path)
    manifest_from_disk = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))

    assert manifest["source_count"] == 1
    assert manifest_from_disk == manifest
    assert (tmp_path / "react.md").read_text(encoding="utf-8") == fetched.text


def _fake_fetch(url: str) -> str:
    assert url
    return "# Fixture\n\nFixture documents evidence extraction."
