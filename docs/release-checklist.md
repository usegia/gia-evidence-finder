# Release Checklist

Use this checklist before publishing `gia-evidence-finder` to a package
registry.

## Pre-Release

1. Choose and add a project license.
2. Confirm the public repository URL and add `[project.urls]` to
   `pyproject.toml`.
3. Update the version in `pyproject.toml`.
4. Review benchmark reports for claims that are too broad for the current
   measured evidence.
5. Ensure `.tmp/`, model outputs, downloaded source documents, virtualenvs,
   caches, and build artifacts are not staged.

## Quality Gate

```sh
uv sync --dev
uv run ruff check .
uv run mypy
uv run pytest
uv build
```

## TestPyPI

```sh
uv publish --publish-url https://test.pypi.org/legacy/
```

Then test a clean install from TestPyPI in a temporary environment.

## PyPI

```sh
uv publish
```

## Git Tag

```sh
git tag v0.1.0
git push origin v0.1.0
```

## Downstream Integration

After publishing, downstream projects should depend on the released package:

```toml
gia-evidence-finder >= 0.1.0
```

Keep downstream claim ingestion, source attachment, and graph/index promotion
logic outside this package. This package owns the narrower contract:

```text
claim or typed intent + source text -> support / contradiction / abstention spans
```
