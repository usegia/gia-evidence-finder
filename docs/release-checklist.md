# Release Checklist

Use this checklist before publishing `gia-evidence-finder` to a package
registry.

## Pre-Release

1. Confirm the Apache-2.0 license is still the desired public license.
2. Confirm the public repository URL in `[project.urls]`.
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

Preferred path: configure PyPI trusted publishing for this GitHub repository,
then merge a version bump to `main`. The GitHub `Publish` workflow runs on
every push to `main`, executes the quality gate, builds the distributions,
checks whether the `pyproject.toml` version already exists on PyPI, and publishes
only when that version is new.

PyPI trusted publisher settings:

- PyPI project name: `gia-evidence-finder`
- Owner: `usegia`
- Repository name: `gia-evidence-finder`
- Workflow name: `publish.yml`
- Environment name: `pypi`

If the merged version is already present on PyPI, the workflow reports a skipped
publish instead of failing on a duplicate upload. To publish a new package, bump
the version in `pyproject.toml` before merging.

Manual token path:

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
