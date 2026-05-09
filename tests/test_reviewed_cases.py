from __future__ import annotations

import json
from pathlib import Path

import pytest

from gia_evidence_finder import (
    BenchmarkSplit,
    load_reviewed_cases_jsonl,
    load_reviewed_series_jsonl,
    write_reviewed_case_template_jsonl,
    write_reviewed_series_template_jsonl,
)


def test_load_reviewed_cases_jsonl_promotes_reviewed_case(tmp_path: Path) -> None:
    source_dir = _source_dir(tmp_path)
    reviewed_jsonl = tmp_path / "reviewed.jsonl"
    reviewed_jsonl.write_text(
        json.dumps(
            {
                "id": "fixture.supports_extraction",
                "split": "dev",
                "document_id": "fixture",
                "label": "Fixture extracts evidence spans",
                "description": "Find evidence that Fixture extracts evidence spans.",
                "positive_examples": ["extracts evidence spans"],
                "required_facets": [
                    {"name": "evidence", "phrases": ["evidence spans"]}
                ],
                "support_span_ids": ["fixture:s0004"],
                "near_miss_span_ids": [],
                "contradiction_span_ids": [],
                "insufficient_context_span_ids": [],
                "forbidden_span_ids": ["fixture:s0005"],
                "expect_abstain": False,
                "curation": {
                    "reviewed": True,
                    "source": "manual_review",
                    "difficulty": "medium",
                    "phenomena": ["direct_support", "hard_negative"],
                    "notes": ["Reviewed fixture case."],
                },
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    result = load_reviewed_cases_jsonl(
        reviewed_jsonl=reviewed_jsonl,
        source_dir=source_dir,
        suite_id="fixture_suite",
    )

    case = result.suite.cases[0]
    assert result.split_counts == {"dev": 1}
    assert result.document_count == 1
    assert case.curation.reviewed
    assert case.support_span_ids == ("fixture:s0004",)
    assert case.forbidden_span_ids == ("fixture:s0005",)


def test_load_reviewed_cases_jsonl_rejects_unreviewed_template(tmp_path: Path) -> None:
    source_dir = _source_dir(tmp_path)
    reviewed_jsonl = tmp_path / "unreviewed.jsonl"
    reviewed_jsonl.write_text(
        json.dumps(
            {
                "id": "fixture.unreviewed",
                "split": "train",
                "document_id": "fixture",
                "label": "Unreviewed fixture",
                "description": "Find unreviewed fixture evidence.",
                "positive_examples": ["fixture evidence"],
                "required_facets": [],
                "support_span_ids": ["fixture:s0004"],
                "expect_abstain": False,
                "curation": {
                    "reviewed": False,
                    "source": "queue_template",
                    "difficulty": "unreviewed",
                    "phenomena": ["direct_support"],
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="must be reviewed"):
        load_reviewed_cases_jsonl(
            reviewed_jsonl=reviewed_jsonl,
            source_dir=source_dir,
            suite_id="fixture_suite",
        )


def test_load_reviewed_series_jsonl_requires_all_splits(tmp_path: Path) -> None:
    source_dir = _source_dir(tmp_path)
    reviewed_jsonl = tmp_path / "series.jsonl"
    reviewed_jsonl.write_text(
        "\n".join(
            json.dumps(_reviewed_row(case_id=f"fixture.{split}", split=split))
            for split in ("train", "dev", "test")
        )
        + "\n",
        encoding="utf-8",
    )

    result = load_reviewed_series_jsonl(
        reviewed_jsonl=reviewed_jsonl,
        source_dir=source_dir,
        series_id="fixture_series",
    )

    assert result.series.id == "fixture_series"
    assert result.split_counts == {"dev": 1, "test": 1, "train": 1}
    assert result.series.splits[BenchmarkSplit.TRAIN].cases[0].id == "fixture.train"
    assert result.series.metadata["status"] == "reviewed_external"


def test_load_reviewed_series_jsonl_rejects_missing_split(tmp_path: Path) -> None:
    source_dir = _source_dir(tmp_path)
    reviewed_jsonl = tmp_path / "missing_test.jsonl"
    reviewed_jsonl.write_text(
        "\n".join(
            json.dumps(_reviewed_row(case_id=f"fixture.{split}", split=split))
            for split in ("train", "dev")
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="missing splits: test"):
        load_reviewed_series_jsonl(
            reviewed_jsonl=reviewed_jsonl,
            source_dir=source_dir,
            series_id="fixture_series",
        )


def test_write_reviewed_case_template_jsonl_marks_rows_unreviewed(tmp_path: Path) -> None:
    queue_jsonl = tmp_path / "queue.jsonl"
    output_jsonl = tmp_path / "template.jsonl"
    queue_jsonl.write_text(
        json.dumps(
            {
                "document_id": "fixture",
                "span_id": "fixture:s0004",
                "suggested_label": "Fixture extracts evidence spans",
                "anchor_terms": ["fixture", "extract", "evidence"],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    count = write_reviewed_case_template_jsonl(
        queue_jsonl=queue_jsonl,
        output_jsonl=output_jsonl,
        split=BenchmarkSplit.TEST,
    )
    row = json.loads(output_jsonl.read_text(encoding="utf-8"))

    assert count == 1
    assert row["split"] == "test"
    assert row["curation"]["reviewed"] is False
    assert row["support_span_ids"] == ["fixture:s0004"]


def test_write_reviewed_series_template_jsonl_splits_by_document(tmp_path: Path) -> None:
    queue_jsonl = tmp_path / "queue.jsonl"
    output_jsonl = tmp_path / "series_template.jsonl"
    queue_jsonl.write_text(
        "\n".join(
            json.dumps(
                {
                    "document_id": document_id,
                    "span_id": span_id,
                    "suggested_label": f"{document_id} extracts evidence",
                    "anchor_terms": [document_id, "evidence"],
                },
                sort_keys=True,
            )
            for document_id, span_id in (
                ("alpha", "alpha:s0001"),
                ("alpha", "alpha:s0002"),
                ("beta", "beta:s0001"),
                ("gamma", "gamma:s0001"),
            )
        )
        + "\n",
        encoding="utf-8",
    )

    split_counts = write_reviewed_series_template_jsonl(
        queue_jsonl=queue_jsonl,
        output_jsonl=output_jsonl,
    )
    rows = [
        json.loads(line)
        for line in output_jsonl.read_text(encoding="utf-8").splitlines()
    ]
    document_splits: dict[str, set[str]] = {}
    for row in rows:
        document_splits.setdefault(row["document_id"], set()).add(row["split"])

    assert set(split_counts) == {"dev", "test", "train"}
    assert sum(split_counts.values()) == 4
    assert {row["split"] for row in rows} == {"dev", "test", "train"}
    assert all(len(splits) == 1 for splits in document_splits.values())
    assert all(row["curation"]["reviewed"] is False for row in rows)


def test_write_reviewed_series_template_jsonl_requires_three_documents(
    tmp_path: Path,
) -> None:
    queue_jsonl = tmp_path / "queue.jsonl"
    queue_jsonl.write_text(
        "\n".join(
            json.dumps(
                {
                    "document_id": document_id,
                    "span_id": f"{document_id}:s0001",
                    "suggested_label": f"{document_id} extracts evidence",
                    "anchor_terms": [document_id, "evidence"],
                }
            )
            for document_id in ("alpha", "beta")
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="at least three documents"):
        write_reviewed_series_template_jsonl(
            queue_jsonl=queue_jsonl,
            output_jsonl=tmp_path / "series_template.jsonl",
        )


def _source_dir(tmp_path: Path) -> Path:
    source_dir = tmp_path / "sources"
    source_dir.mkdir()
    (source_dir / "fixture.md").write_text(
        (
            "# Fixture\n\n## Features\n\n"
            "Fixture extracts evidence spans. Fixture rejects false claims."
        ),
        encoding="utf-8",
    )
    (source_dir / "manifest.json").write_text(
        json.dumps(
            {
                "sources": [
                    {
                        "id": "fixture",
                        "filename": "fixture.md",
                        "raw_url": "https://example.test/fixture.md",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    return source_dir


def _reviewed_row(*, case_id: str, split: str) -> dict[str, object]:
    return {
        "id": case_id,
        "split": split,
        "document_id": "fixture",
        "label": "Fixture extracts evidence spans",
        "description": "Find evidence that Fixture extracts evidence spans.",
        "positive_examples": ["extracts evidence spans"],
        "required_facets": [{"name": "evidence", "phrases": ["evidence spans"]}],
        "support_span_ids": ["fixture:s0004"],
        "near_miss_span_ids": [],
        "contradiction_span_ids": [],
        "insufficient_context_span_ids": [],
        "forbidden_span_ids": ["fixture:s0005"],
        "expect_abstain": False,
        "curation": {
            "reviewed": True,
            "source": "manual_review",
            "difficulty": "medium",
            "phenomena": ["direct_support", "hard_negative"],
        },
    }
