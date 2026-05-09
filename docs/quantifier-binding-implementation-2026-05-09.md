# Quantifier Binding v2 Implementation

Date: 2026-05-09

## Goal

Quantifier Binding v2 upgrades gia evidence finder from value-only checks to
deterministic role-bound checks. A candidate span must now prove that the right
number or date is attached to the right predicate when the intent is ambiguous
or high risk.

Example:

```text
claim: Project started in 2024 and ended in 2025.
span:  Project started in 2025 and ended in 2024.
```

The span contains both years, but it is a contradiction because the values are
bound to the wrong roles.

## Contract

The public contract now includes `QuantifierBinding`.

- `quantifier`: raw parsed `EvidenceQuantifier`
- `role`: deterministic local role, such as `started`, `ended`, `rent`, or
  `p95_latency`
- `subject_terms`: nearby subject tokens when cheaply available
- `predicate_terms`: local aliases that caused the role assignment
- `local_text`: small text window used for binding
- `binding_confidence`: deterministic confidence for traceability

`QuantifierRequirement` keeps the raw quantifier and adds:

- `role`
- `binding_required`
- `subject_terms`
- `predicate_terms`

Single simple quantifier requirements remain value-level compatible with v1.
Multi-quantifier and high-risk claims require role compatibility.

## Deterministic Binding

`bind_quantifiers(text)` extracts raw quantifiers, takes local windows around
each value, and maps nearby aliases to stable roles. The first implementation
covers:

- date roles: created, started, ended, completed, due, available, founded,
  announced, updated, published, released
- metrics: latency, p50 latency, p95 latency, retry count, cache hit rate,
  speedup
- apartment and business values: rent, deposit, broker fee, application fee,
  bedrooms, bathrooms, lease term
- counts: engineers, designers, seats, projects

Unknown local roles stay unbound instead of guessing.

## Scoring Semantics

When `binding_required=True`:

- same role and matching value supports the requirement
- same role and wrong value is a contradiction
- matching value under the wrong role is a role mismatch and contradiction for
  swapped-role evidence
- missing role-bound values are insufficient context, not support

The ranker exposes role-binding trace features:

- `quantifier_role_mismatch`
- `quantifier_role_missing_count`

## Benchmark

The new suite is `quantifier_binding_v2`.

The new series is `quantifier_binding_evidence_benchmark_v2`.

It contains 24 manually reviewed cases covering:

- date role swaps
- metric and threshold binding
- apartment and money binding
- counts, Python versions, fee swaps, team composition, and date ranges
- insufficient context when only one required bound value appears

`quantifier_numeric_date_v1` remains the value-level regression suite and is not
rewritten into a binding suite.

## Deliberate Non-Goals

This milestone does not add Duckling, SUTime, quantulum3, ADS, LLM calls, or new
mandatory dependencies. Those tools may become optional raw extraction backends
later. Runtime support decisions remain deterministic, typed, and explainable.
