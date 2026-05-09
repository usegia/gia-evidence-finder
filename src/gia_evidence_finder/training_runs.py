from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModalRerankerProbe:
    id: str
    description: str
    base_model: str
    run_name: str
    epochs: int
    batch_size: int
    learning_rate: float
    warmup_ratio: float
    max_length: int
    seed: int
    expected_role: str

    def __post_init__(self) -> None:
        if not self.id.strip():
            raise ValueError("probe id must not be empty")
        if not self.base_model.strip():
            raise ValueError("probe base_model must not be empty")
        if not self.run_name.strip():
            raise ValueError("probe run_name must not be empty")
        if self.epochs < 1:
            raise ValueError("probe epochs must be positive")
        if self.batch_size < 1:
            raise ValueError("probe batch_size must be positive")
        if not 0.0 < self.learning_rate < 1.0:
            raise ValueError("probe learning_rate must be in (0, 1)")
        if not 0.0 <= self.warmup_ratio <= 1.0:
            raise ValueError("probe warmup_ratio must be in [0, 1]")
        if self.max_length < 1:
            raise ValueError("probe max_length must be positive")

    def to_json_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "description": self.description,
            "base_model": self.base_model,
            "run_name": self.run_name,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "warmup_ratio": self.warmup_ratio,
            "max_length": self.max_length,
            "seed": self.seed,
            "expected_role": self.expected_role,
        }


def modal_reranker_probes() -> tuple[ModalRerankerProbe, ...]:
    return (
        ModalRerankerProbe(
            id="minilm_l6_reference",
            description="Compact reference run used to compare against earlier MiniLM probes.",
            base_model="cross-encoder/ms-marco-MiniLM-L-6-v2",
            run_name="reviewed-msmarco-minilm-l6-reference-20260508-e4-seed13",
            epochs=4,
            batch_size=16,
            learning_rate=2e-5,
            warmup_ratio=0.1,
            max_length=512,
            seed=13,
            expected_role="reference",
        ),
        ModalRerankerProbe(
            id="minilm_l12_larger",
            description=(
                "Larger MiniLM cross-encoder probe with the same training contract as the "
                "reference run."
            ),
            base_model="cross-encoder/ms-marco-MiniLM-L-12-v2",
            run_name="reviewed-msmarco-minilm-l12-larger-20260508-e4-seed13",
            epochs=4,
            batch_size=16,
            learning_rate=2e-5,
            warmup_ratio=0.1,
            max_length=512,
            seed=13,
            expected_role="larger_model_probe",
        ),
        ModalRerankerProbe(
            id="electra_base_larger",
            description=(
                "Higher-capacity MS MARCO cross-encoder probe for H200-backed quality checks."
            ),
            base_model="cross-encoder/ms-marco-electra-base",
            run_name="reviewed-msmarco-electra-base-20260508-e3-seed13",
            epochs=3,
            batch_size=12,
            learning_rate=1e-5,
            warmup_ratio=0.1,
            max_length=512,
            seed=13,
            expected_role="larger_model_probe",
        ),
    )


def modal_reranker_probe_by_id(probe_id: str) -> ModalRerankerProbe:
    probes = {probe.id: probe for probe in modal_reranker_probes()}
    try:
        return probes[probe_id]
    except KeyError as exc:
        known = ", ".join(sorted(probes))
        raise ValueError(
            f"unknown Modal reranker probe {probe_id!r}. Known probes: {known}"
        ) from exc
