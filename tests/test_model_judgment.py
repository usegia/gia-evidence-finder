from __future__ import annotations

import json
from pathlib import Path

from gia_evidence_finder import (
    DEFAULT_MODEL_JUDGE_RUBRIC,
    ModelJudgeDecision,
    SupportLabel,
    decisions_from_model_response_jsonl,
    model_judge_agreement_report,
    model_judge_requests_from_suite,
    model_judge_response_schema,
    non_readme_benchmark_suite,
    read_model_judge_decisions_jsonl,
    read_model_judge_requests_jsonl,
    write_model_judge_requests_jsonl,
)


def test_model_judge_requests_include_source_hash_and_expected_labels() -> None:
    suite = non_readme_benchmark_suite()

    requests = model_judge_requests_from_suite(suite)

    assert requests
    assert all(request.source_hash for request in requests)
    assert {request.expected_label for request in requests} >= {
        SupportLabel.SUPPORTS,
        SupportLabel.CONTRADICTS,
        SupportLabel.REJECT,
    }
    assert "Return only JSON" in requests[0].prompt(DEFAULT_MODEL_JUDGE_RUBRIC)
    assert model_judge_response_schema()["required"] == ["label", "confidence", "rationale"]


def test_model_judge_request_jsonl_round_trips(tmp_path: Path) -> None:
    requests = model_judge_requests_from_suite(non_readme_benchmark_suite())[:3]
    output_path = tmp_path / "requests.jsonl"

    write_model_judge_requests_jsonl(requests, output_path)
    loaded = read_model_judge_requests_jsonl(output_path)

    assert loaded == requests


def test_model_judge_response_conversion_and_agreement_report(tmp_path: Path) -> None:
    request = model_judge_requests_from_suite(non_readme_benchmark_suite())[0]
    request_jsonl = tmp_path / "requests.jsonl"
    response_jsonl = tmp_path / "responses.jsonl"
    write_model_judge_requests_jsonl((request,), request_jsonl)
    response_jsonl.write_text(
        json.dumps(
            {
                "request_id": request.id,
                "label": SupportLabel.SUPPORTS.value,
                "confidence": 0.91,
                "rationale": "The span directly proves the claim.",
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    decisions = decisions_from_model_response_jsonl(
        response_jsonl=response_jsonl,
        request_jsonl=request_jsonl,
        judge_model="test-judge",
        pass_id="pass-1",
    )
    decision_jsonl = tmp_path / "decisions.jsonl"
    decision_jsonl.write_text(
        "\n".join(json.dumps(decision.to_json_dict(), sort_keys=True) for decision in decisions)
        + "\n",
        encoding="utf-8",
    )
    loaded = read_model_judge_decisions_jsonl(decision_jsonl)
    report = model_judge_agreement_report(
        (
            *loaded,
            ModelJudgeDecision(
                request_id=request.id,
                label=SupportLabel.SUPPORTS,
                confidence=0.82,
                rationale="Second pass agrees.",
                judge_model="test-judge",
                rubric_id=DEFAULT_MODEL_JUDGE_RUBRIC.id,
                rubric_version=DEFAULT_MODEL_JUDGE_RUBRIC.version,
                source_hash=request.source_hash,
                pass_id="pass-2",
            ),
        ),
        min_passes=2,
    )

    assert loaded[0].source_hash == request.source_hash
    assert report.request_count == 1
    assert report.unanimous_count == 1
    assert report.disagreement_count == 0
    assert report.warnings == ()
