from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date

from gia_evidence_finder.contracts import (
    EvidenceQuantifier,
    QuantifierBinding,
    QuantifierKind,
    QuantifierOperator,
    QuantifierRequirement,
)

_MONTHS = {
    "january": 1,
    "jan": 1,
    "february": 2,
    "feb": 2,
    "march": 3,
    "mar": 3,
    "april": 4,
    "apr": 4,
    "may": 5,
    "june": 6,
    "jun": 6,
    "july": 7,
    "jul": 7,
    "august": 8,
    "aug": 8,
    "september": 9,
    "sep": 9,
    "october": 10,
    "oct": 10,
    "november": 11,
    "nov": 11,
    "december": 12,
    "dec": 12,
}
_MONTH_PATTERN = "|".join(sorted(_MONTHS, key=len, reverse=True))
_ISO_DATE_RE = re.compile(r"\b(20\d{2}|19\d{2})-(0[1-9]|1[0-2])-([0-2]\d|3[01])\b")
_MONTH_DATE_RE = re.compile(
    rf"\b({_MONTH_PATTERN})\s+([0-2]?\d|3[01]),?\s+(20\d{{2}}|19\d{{2}})\b",
    re.IGNORECASE,
)
_MONTH_YEAR_RE = re.compile(rf"\b({_MONTH_PATTERN})\s+(20\d{{2}}|19\d{{2}})\b", re.IGNORECASE)
_YEAR_RE = re.compile(r"\b(20\d{2}|19\d{2})\b")
_MONEY_RE = re.compile(r"(?<!\w)\$([0-9][0-9,]*(?:\.[0-9]+)?)")
_PERCENT_RE = re.compile(r"(?<!\w)([0-9]+(?:\.[0-9]+)?)\s*%")
_MULTIPLIER_RE = re.compile(r"(?<!\w)([0-9]+(?:\.[0-9]+)?)\s*x\b", re.IGNORECASE)
_NUMBER_UNIT_RE = re.compile(
    r"(?<![\w$])([0-9]+(?:\.[0-9]+)?)\s*-?\s*"
    r"(retries|retry|times|ms|milliseconds|seconds|second|minutes|minute|hours|hour|days|day|"
    r"months|month|bedrooms|bedroom|beds|bed|br|bathrooms|bathroom|baths|bath|"
    r"engineers|engineer|designers|designer|seats|seat|projects|project|points|"
    r"versions|version)\b",
    re.IGNORECASE,
)
_BARE_NUMBER_RE = re.compile(r"(?<![\w$])([0-9]+(?:\.[0-9]+)?)(?![\w%])")
_COMPARATOR_PATTERNS: tuple[tuple[re.Pattern[str], QuantifierOperator], ...] = (
    (
        re.compile(r"\b(at least|minimum|min|no less than|not less than|>=)\b", re.IGNORECASE),
        QuantifierOperator.GTE,
    ),
    (
        re.compile(r"\b(at most|maximum|max|no more than|not more than|<=)\b", re.IGNORECASE),
        QuantifierOperator.LTE,
    ),
    (
        re.compile(
            r"\b(more than|greater than|over|above|after|since|exceeds|>)\b",
            re.IGNORECASE,
        ),
        QuantifierOperator.GT,
    ),
    (
        re.compile(r"\b(less than|under|below|before|<)\b", re.IGNORECASE),
        QuantifierOperator.LT,
    ),
    (
        re.compile(r"\b(exactly|equal to|equals|=)\b", re.IGNORECASE),
        QuantifierOperator.EQ,
    ),
)
_DATE_CONTEXT_RE = re.compile(
    r"\b(created|founded|announced|available|due|started|ended|published|released|updated|"
    r"before|after|since|until|in|on)\b",
    re.IGNORECASE,
)
_TOKEN_RE = re.compile(r"[a-zA-Z0-9_$%.-]+")
_HIGH_RISK_ROLES = frozenset(
    {
        "application_fee",
        "available",
        "bathrooms",
        "bedrooms",
        "broker_fee",
        "cache_hit_rate",
        "created",
        "deposit",
        "due",
        "ended",
        "lease_term",
        "p50_latency",
        "p95_latency",
        "rent",
        "retry_count",
        "started",
    }
)
_ROLE_ALIASES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("application_fee", ("application fee",)),
    ("broker_fee", ("broker fee",)),
    ("cache_hit_rate", ("cache hit rate", "hit rate")),
    ("p95_latency", ("p95 latency", "latency p95")),
    ("p50_latency", ("p50 latency", "latency p50")),
    ("retry_count", ("retried", "retries", "retry", "attempts")),
    ("lease_term", ("lease term", "minimum lease", "minimum term")),
    ("bedrooms", ("bedrooms", "bedroom", "beds", "bed", "br")),
    ("bathrooms", ("bathrooms", "bathroom", "baths", "bath")),
    ("started", ("started", "start date", "began")),
    ("completed", ("completed", "complete", "finished")),
    ("created", ("created", "opened", "filed", "launched")),
    ("founded", ("founded",)),
    ("announced", ("announced",)),
    ("available", ("available", "move-in", "move in")),
    ("published", ("published",)),
    ("released", ("released",)),
    ("updated", ("updated",)),
    ("ended", ("ended", "closed", "until")),
    ("due", ("due", "deadline", "target date")),
    ("latency", ("latency", "response time")),
    ("speedup", ("faster", "speedup", "improved by")),
    ("deposit", ("deposit",)),
    ("rent", ("monthly rent", "rent", "price")),
    ("engineers", ("engineers", "engineer")),
    ("designers", ("designers", "designer")),
    ("projects", ("projects", "project")),
    ("seats", ("seats", "seat")),
    ("python_version", ("python",)),
)
_ROLE_PRIORITY = {role: index for index, (role, _) in enumerate(_ROLE_ALIASES)}


@dataclass(frozen=True)
class QuantifierScore:
    score: float
    coverage: float
    mismatch: float
    contradiction: float
    requirement_count: int
    matched_count: int
    missing_count: int
    role_mismatch: float = 0.0
    role_missing_count: int = 0


def requirements_from_text(text: str) -> tuple[QuantifierRequirement, ...]:
    bindings = bind_quantifiers(text)
    binding_required = _binding_required(bindings)
    return tuple(
        QuantifierRequirement(
            name=f"quantifier_{index}_{binding.quantifier.kind.value}",
            quantifier=binding.quantifier,
            role=binding.role,
            binding_required=binding_required and bool(binding.role),
            subject_terms=binding.subject_terms,
            predicate_terms=binding.predicate_terms,
        )
        for index, binding in enumerate(bindings, start=1)
    )


def bind_quantifiers(text: str) -> tuple[QuantifierBinding, ...]:
    quantifiers = extract_quantifiers(text)
    if not quantifiers:
        return ()
    occurrences = _quantifier_occurrences(text, quantifiers)
    bindings: list[QuantifierBinding] = []
    used_roles: set[str] = set()
    for quantifier, start, end in occurrences:
        local_text = _local_text(text, start, end)
        role, predicate_terms, confidence = _role_for_quantifier(quantifier, local_text, used_roles)
        if role:
            used_roles.add(role)
        bindings.append(
            QuantifierBinding(
                quantifier=quantifier,
                role=role,
                subject_terms=_subject_terms(local_text, predicate_terms),
                predicate_terms=predicate_terms,
                local_text=local_text,
                binding_confidence=confidence,
            )
        )
    return tuple(bindings)


def extract_quantifiers(text: str) -> tuple[EvidenceQuantifier, ...]:
    occupied: list[range] = []
    quantifiers: list[EvidenceQuantifier] = []

    for match in _ISO_DATE_RE.finditer(text):
        year, month, day = (int(group) for group in match.groups())
        if _valid_date(year, month, day):
            quantifiers.append(
                _quantifier(
                    QuantifierKind.DATE,
                    float(date(year, month, day).toordinal()),
                    match.group(0),
                    text,
                    match.start(),
                    normalized=f"{year:04d}-{month:02d}-{day:02d}",
                )
            )
            occupied.append(range(match.start(), match.end()))

    for match in _MONTH_DATE_RE.finditer(text):
        if _overlaps(match.start(), match.end(), occupied):
            continue
        month = _MONTHS[match.group(1).lower()]
        day = int(match.group(2))
        year = int(match.group(3))
        if _valid_date(year, month, day):
            quantifiers.append(
                _quantifier(
                    QuantifierKind.DATE,
                    float(date(year, month, day).toordinal()),
                    match.group(0),
                    text,
                    match.start(),
                    normalized=f"{year:04d}-{month:02d}-{day:02d}",
                )
            )
            occupied.append(range(match.start(), match.end()))

    for match in _MONTH_YEAR_RE.finditer(text):
        if _overlaps(match.start(), match.end(), occupied):
            continue
        year = int(match.group(2))
        quantifiers.append(
            _quantifier(
                QuantifierKind.YEAR,
                float(year),
                match.group(0),
                text,
                match.start(),
                unit=match.group(1).lower(),
                normalized=f"{year:04d}-{_MONTHS[match.group(1).lower()]:02d}",
            )
        )
        occupied.append(range(match.start(), match.end()))

    for match in _MONEY_RE.finditer(text):
        quantifiers.append(
            _quantifier(
                QuantifierKind.MONEY,
                _number(match.group(1)),
                match.group(0),
                text,
                match.start(),
                unit="usd",
            )
        )
        occupied.append(range(match.start(), match.end()))

    for regex, kind, unit in (
        (_PERCENT_RE, QuantifierKind.PERCENT, "percent"),
        (_MULTIPLIER_RE, QuantifierKind.MULTIPLIER, "x"),
    ):
        for match in regex.finditer(text):
            if _overlaps(match.start(), match.end(), occupied):
                continue
            quantifiers.append(
                _quantifier(
                    kind,
                    _number(match.group(1)),
                    match.group(0),
                    text,
                    match.start(),
                    unit=unit,
                )
            )
            occupied.append(range(match.start(), match.end()))

    for match in _NUMBER_UNIT_RE.finditer(text):
        if _overlaps(match.start(), match.end(), occupied):
            continue
        unit = _normalize_unit(match.group(2))
        kind = (
            QuantifierKind.DURATION
            if unit in {"ms", "second", "minute", "hour", "day", "month"}
            else QuantifierKind.NUMBER
        )
        quantifiers.append(
            _quantifier(
                kind,
                _number(match.group(1)),
                match.group(0),
                text,
                match.start(),
                unit=unit,
            )
        )
        occupied.append(range(match.start(), match.end()))

    for match in _YEAR_RE.finditer(text):
        if _overlaps(match.start(), match.end(), occupied):
            continue
        if _looks_like_date_context(text, match.start()):
            quantifiers.append(
                _quantifier(
                    QuantifierKind.YEAR,
                    float(match.group(1)),
                    match.group(0),
                    text,
                    match.start(),
                    normalized=match.group(1),
                )
            )
            occupied.append(range(match.start(), match.end()))

    for match in _BARE_NUMBER_RE.finditer(text):
        if _overlaps(match.start(), match.end(), occupied):
            continue
        if _looks_like_bare_requirement(text, match.start()):
            quantifiers.append(
                _quantifier(
                    QuantifierKind.NUMBER,
                    _number(match.group(1)),
                    match.group(0),
                    text,
                    match.start(),
                )
            )
            occupied.append(range(match.start(), match.end()))

    return tuple(quantifiers)


def quantifier_score(
    requirements: tuple[QuantifierRequirement, ...],
    text: str,
) -> QuantifierScore:
    required = tuple(requirement for requirement in requirements if requirement.required)
    if not required:
        return QuantifierScore(1.0, 1.0, 0.0, 0.0, 0, 0, 0)
    observed = bind_quantifiers(text)
    matched = 0
    mismatched = 0
    missing = 0
    role_mismatched = 0
    role_missing = 0
    for requirement in required:
        comparable = tuple(
            candidate
            for candidate in observed
            if _comparable(requirement.quantifier, candidate.quantifier)
        )
        if not comparable:
            missing += 1
            continue
        if requirement.binding_required:
            role_matches = tuple(
                candidate
                for candidate in comparable
                if requirement.role and candidate.role == requirement.role
            )
            if not role_matches:
                if any(
                    _satisfies(requirement.quantifier, candidate.quantifier)
                    for candidate in comparable
                ):
                    role_mismatched += 1
                else:
                    role_missing += 1
                continue
            if any(
                _satisfies(requirement.quantifier, candidate.quantifier)
                for candidate in role_matches
            ):
                matched += 1
            else:
                mismatched += 1
            continue
        if any(
            _satisfies(requirement.quantifier, candidate.quantifier)
            for candidate in comparable
        ):
            matched += 1
        else:
            mismatched += 1
    count = len(required)
    coverage = matched / count
    mismatch = mismatched / count
    role_mismatch = role_mismatched / count
    contradiction = min(1.0, mismatch + role_mismatch)
    score = max(0.0, coverage - mismatch - role_mismatch)
    return QuantifierScore(
        score=score,
        coverage=coverage,
        mismatch=mismatch,
        contradiction=contradiction,
        requirement_count=count,
        matched_count=matched,
        missing_count=missing,
        role_mismatch=role_mismatch,
        role_missing_count=role_missing,
    )


def _quantifier(
    kind: QuantifierKind,
    value: float,
    surface: str,
    text: str,
    start: int,
    *,
    unit: str = "",
    normalized: str = "",
) -> EvidenceQuantifier:
    return EvidenceQuantifier(
        kind=kind,
        value=value,
        unit=unit,
        surface=surface,
        operator=_operator_near(text, start),
        normalized=normalized,
    )


def _binding_required(bindings: tuple[QuantifierBinding, ...]) -> bool:
    if len(bindings) <= 1:
        return bool(bindings and bindings[0].role in _HIGH_RISK_ROLES)
    comparable_groups: dict[tuple[QuantifierKind, str], int] = {}
    for binding in bindings:
        key = _comparable_group(binding.quantifier)
        comparable_groups[key] = comparable_groups.get(key, 0) + 1
    return any(count > 1 for count in comparable_groups.values()) or any(
        binding.role in _HIGH_RISK_ROLES for binding in bindings
    )


def _comparable_group(quantifier: EvidenceQuantifier) -> tuple[QuantifierKind, str]:
    if quantifier.kind in {QuantifierKind.DATE, QuantifierKind.YEAR}:
        return (QuantifierKind.DATE, "")
    return (quantifier.kind, quantifier.unit)


def _quantifier_occurrences(
    text: str,
    quantifiers: tuple[EvidenceQuantifier, ...],
) -> tuple[tuple[EvidenceQuantifier, int, int], ...]:
    cursor = 0
    occurrences: list[tuple[EvidenceQuantifier, int, int]] = []
    for quantifier in quantifiers:
        start = text.lower().find(quantifier.surface.lower(), cursor)
        if start < 0:
            start = text.lower().find(quantifier.surface.lower())
        if start < 0:
            start = 0
        end = start + len(quantifier.surface)
        occurrences.append((quantifier, start, end))
        cursor = end
    return tuple(occurrences)


def _local_text(text: str, start: int, end: int) -> str:
    left_boundary = max(
        text.rfind(separator, 0, start)
        for separator in (".", ";", "\n", "-")
    )
    right_candidates = [
        index
        for separator in (".", ";", "\n")
        if (index := text.find(separator, end)) >= 0
    ]
    left = 0 if left_boundary < 0 else left_boundary + 1
    right = min(right_candidates) if right_candidates else len(text)
    return text[left:right].strip(" -")


def _role_for_quantifier(
    quantifier: EvidenceQuantifier,
    local_text: str,
    used_roles: set[str],
) -> tuple[str, tuple[str, ...], float]:
    normalized = local_text.lower()
    candidates: list[tuple[int, int, str, str]] = []
    for role, aliases in _ROLE_ALIASES:
        for alias in aliases:
            index = normalized.find(alias)
            if index >= 0:
                candidates.append(
                    (
                        _role_distance(normalized, alias, quantifier.surface),
                        index,
                        role,
                        alias,
                    )
                )
    if not candidates:
        return "", (), 0.0
    candidates.sort(key=lambda item: (item[0], _ROLE_PRIORITY[item[2]], item[1]))
    for distance, _index, role, alias in candidates:
        if role not in used_roles:
            return role, (alias,), max(0.55, 1.0 - min(distance, 40) / 60)
    distance, _index, role, alias = candidates[0]
    return role, (alias,), max(0.45, 1.0 - min(distance, 40) / 60)


def _role_distance(text: str, alias: str, surface: str) -> int:
    alias_index = text.find(alias)
    surface_index = text.find(surface.lower())
    if alias_index < 0 or surface_index < 0:
        return 1000
    if alias_index <= surface_index:
        return surface_index - alias_index
    return 40 + (alias_index - surface_index)


def _subject_terms(local_text: str, predicate_terms: tuple[str, ...]) -> tuple[str, ...]:
    tokens = tuple(token.lower().strip(".,:;()") for token in _TOKEN_RE.findall(local_text))
    predicate_tokens = {
        token
        for predicate in predicate_terms
        for token in predicate.lower().replace("-", " ").split()
    }
    stop_tokens = {
        "and",
        "after",
        "before",
        "in",
        "is",
        "on",
        "the",
        "to",
        "was",
        "with",
    }
    terms = tuple(
        token
        for token in tokens
        if len(token) > 2
        and token not in predicate_tokens
        and token not in stop_tokens
        and not any(character.isdigit() for character in token)
    )
    return tuple(dict.fromkeys(terms[:4]))


def _operator_near(text: str, start: int) -> QuantifierOperator:
    prefix = text[max(0, start - 32) : start]
    for pattern, operator in _COMPARATOR_PATTERNS:
        if pattern.search(prefix):
            return operator
    return QuantifierOperator.EQ


def _comparable(requirement: EvidenceQuantifier, candidate: EvidenceQuantifier) -> bool:
    if requirement.kind in {QuantifierKind.DATE, QuantifierKind.YEAR}:
        return candidate.kind in {QuantifierKind.DATE, QuantifierKind.YEAR}
    if requirement.kind != candidate.kind:
        return False
    return not requirement.unit or not candidate.unit or requirement.unit == candidate.unit


def _satisfies(requirement: EvidenceQuantifier, candidate: EvidenceQuantifier) -> bool:
    required_value = requirement.value
    candidate_value = candidate.value
    if requirement.kind == QuantifierKind.YEAR and candidate.kind == QuantifierKind.DATE:
        candidate_value = float(date.fromordinal(int(candidate.value)).year)
    if requirement.kind == QuantifierKind.DATE and candidate.kind == QuantifierKind.YEAR:
        required_value = float(date.fromordinal(int(requirement.value)).year)
    if requirement.operator == QuantifierOperator.EQ:
        return candidate_value == required_value
    if requirement.operator == QuantifierOperator.LT:
        return candidate_value < required_value
    if requirement.operator == QuantifierOperator.LTE:
        return candidate_value <= required_value
    if requirement.operator == QuantifierOperator.GT:
        return candidate_value > required_value
    return candidate_value >= required_value


def _normalize_unit(unit: str) -> str:
    normalized = unit.lower()
    return {
        "milliseconds": "ms",
        "seconds": "second",
        "minutes": "minute",
        "hours": "hour",
        "days": "day",
        "months": "month",
        "retries": "retry",
        "times": "retry",
        "bedrooms": "bedroom",
        "bathrooms": "bathroom",
        "baths": "bathroom",
        "bath": "bathroom",
        "beds": "bedroom",
        "bed": "bedroom",
        "br": "bedroom",
        "engineers": "engineer",
        "designers": "designer",
        "seats": "seat",
        "projects": "project",
        "versions": "version",
    }.get(normalized, normalized)


def _looks_like_date_context(text: str, start: int) -> bool:
    return _DATE_CONTEXT_RE.search(text[max(0, start - 32) : start + 16]) is not None


def _looks_like_bare_requirement(text: str, start: int) -> bool:
    context = text[max(0, start - 32) : start + 16]
    if _role_for_quantifier(
        EvidenceQuantifier(QuantifierKind.NUMBER, 0.0, "0"),
        context,
        set(),
    )[0]:
        return True
    if any(
        pattern.search(text[max(0, start - 32) : start])
        for pattern, _ in _COMPARATOR_PATTERNS
    ):
        return True
    return True


def _number(value: str) -> float:
    return float(value.replace(",", ""))


def _valid_date(year: int, month: int, day: int) -> bool:
    try:
        date(year, month, day)
    except ValueError:
        return False
    return True


def _overlaps(start: int, end: int, ranges: list[range]) -> bool:
    return any(start in existing or (end - 1) in existing for existing in ranges)
