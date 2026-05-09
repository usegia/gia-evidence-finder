# Quantifier and Date Evidence Implementation

Date: 2026-05-09

gia evidence finder now has a deterministic quantifier layer for explicit dates
and numeric constraints. The goal is to prevent semantically similar spans from
being accepted when they contain the wrong year, date, threshold, amount,
percentage, duration, or multiplier.

## Scope

The implementation supports explicit:

- years and dates, including ISO dates and month/day/year phrases;
- simple numbers with adjacent units;
- currency amounts, percentages, multipliers, and durations;
- comparators such as `before`, `after`, `under`, `over`, `at least`,
  `at most`, and `exceeds`.

Unsupported vague phrases such as `recent`, `cheap`, `large`, or `fast` remain
outside the deterministic contract.

## Runtime Contract

Raw claims compiled with `compile_intent` now carry typed
`QuantifierRequirement` values. The default ranker computes quantifier features
for each candidate span:

- `quantifier_coverage`
- `quantifier_mismatch`
- `quantifier_contradiction`
- `quantifier_requirement_count`

The extractor uses those features as a hard safety layer. A candidate with the
right semantic anchors but a contradictory explicit quantity or date is labeled
`contradicts`, not `supports`. A candidate that lacks the required explicit
quantity/date cannot become support for a quantifier-bearing intent.

## Benchmark

The focused benchmark suite is `quantifier_numeric_date_v1`, with 20 reviewed
cases across project-management, people-search, apartment-search, and technical
metric examples. It is also exposed as the train/dev/test series
`quantifier_evidence_benchmark_v1`.

The benchmark covers:

- exact year support and contradiction;
- before/after date thresholds;
- at-least and at-most numeric thresholds;
- latency, rent, lease duration, bedrooms, percentages, and multipliers;
- insufficient-context spans that mention the right subject but omit the needed
  explicit quantity/date.
