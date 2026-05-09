from __future__ import annotations

import pytest

from gia_evidence_finder.training_runs import (
    modal_reranker_probe_by_id,
    modal_reranker_probes,
)


def test_modal_reranker_probes_are_named_and_serializable() -> None:
    probes = modal_reranker_probes()
    probe_ids = {probe.id for probe in probes}

    assert {"minilm_l6_reference", "minilm_l12_larger", "electra_base_larger"} <= probe_ids
    assert len(probe_ids) == len(probes)
    for probe in probes:
        payload = probe.to_json_dict()
        assert payload["id"] == probe.id
        assert payload["base_model"] == probe.base_model
        assert payload["run_name"] == probe.run_name


def test_modal_reranker_probe_lookup_reports_known_ids() -> None:
    probe = modal_reranker_probe_by_id("minilm_l12_larger")

    assert probe.base_model == "cross-encoder/ms-marco-MiniLM-L-12-v2"

    with pytest.raises(ValueError, match="Known probes"):
        modal_reranker_probe_by_id("missing")
