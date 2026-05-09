from __future__ import annotations

from gia_evidence_finder import (
    EvidenceExtractor,
    MarkdownSpanParser,
    SupportLabel,
    compile_intent,
    polarity_benchmark_suite,
    popular_readme_benchmark_suite,
    relation_readme_benchmark_suite,
)


def test_compile_intent_extracts_required_and_excluded_facets() -> None:
    intent = compile_intent(
        intent_id="claim",
        claim="Code OSS itself includes Microsoft-specific customizations.",
    )

    required_phrases = {
        phrase
        for facet in intent.required_facets
        for phrase in facet.phrases
    }
    excluded_phrases = {
        phrase
        for facet in intent.excluded_facets
        for phrase in facet.phrases
    }

    assert "Code OSS" in required_phrases
    assert "Microsoft-specific" in required_phrases
    assert "distribution of" in excluded_phrases


def test_compile_intent_accepts_positional_claim() -> None:
    intent = compile_intent("UltraChess is browser-based")

    assert intent.id == "claim"
    assert intent.label == "UltraChess is browser-based"


def test_compiled_hyphenated_claim_supports_direct_sentence() -> None:
    document = MarkdownSpanParser().parse(
        """# README

No evaluation, no search, no opening book.
UltraChess is a browser-based chess variant playground.
""",
        document_id="readme",
    )
    intent = compile_intent("UltraChess is browser-based")

    result = EvidenceExtractor.default().extract(intent, document)

    assert not result.abstained
    assert result.matches[0].label == SupportLabel.SUPPORTS
    assert result.matches[0].span.text == "UltraChess is a browser-based chess variant playground."


def test_compiled_unhyphenated_claim_supports_hyphenated_sentence() -> None:
    document = MarkdownSpanParser().parse(
        """# README

UltraChess is a browser-based chess variant playground.
""",
        document_id="readme",
    )
    intent = compile_intent("UltraChess is browser based")

    result = EvidenceExtractor.default().extract(intent, document)

    assert not result.abstained
    assert result.matches[0].label == SupportLabel.SUPPORTS
    assert result.matches[0].span.text == "UltraChess is a browser-based chess variant playground."


def test_compiled_itself_claim_rejects_indirect_distribution_evidence() -> None:
    suite = popular_readme_benchmark_suite()
    document = next(case.document for case in suite.cases if case.document.id == "vscode")
    intent = compile_intent(
        intent_id="claim",
        claim="Code OSS itself includes Microsoft-specific customizations.",
    )

    result = EvidenceExtractor.default().extract(intent, document)

    assert result.abstained
    assert result.candidates[0].span.id == "vscode:s0003"
    assert result.candidates[0].score < intent.min_support_score


def test_compiled_itself_claim_rejects_dependency_attribute_transfer() -> None:
    suite = relation_readme_benchmark_suite()
    document = next(case.document for case in suite.cases if case.document.id == "fastapi")
    intent = compile_intent(
        intent_id="claim",
        claim="Pydantic itself is on par with NodeJS and Go for performance.",
    )

    result = EvidenceExtractor.default().extract(intent, document)

    assert result.abstained
    assert result.candidates[0].span.id == "fastapi:s0004"
    assert result.candidates[0].score < intent.min_support_score


def test_compiled_positive_claim_rejects_negated_limitation_span() -> None:
    suite = polarity_benchmark_suite()
    document = next(
        case.document for case in suite.cases if case.document.id == "ultrachess_readme"
    )
    intent = compile_intent(
        intent_id="claim",
        claim="ultrachess includes evaluation, search, and an opening book.",
    )

    result = EvidenceExtractor.default().extract(intent, document)

    assert result.abstained
    assert result.candidates[0].label == SupportLabel.CONTRADICTS
    assert result.candidates[0].span.text == "No evaluation, no search, no opening book."


def test_compiled_negative_claim_supports_matching_negated_span() -> None:
    suite = polarity_benchmark_suite()
    document = next(
        case.document for case in suite.cases if case.document.id == "ultrachess_readme"
    )
    intent = compile_intent(
        intent_id="claim",
        claim="ultrachess has no evaluation, no search, and no opening book.",
    )

    result = EvidenceExtractor.default().extract(intent, document)

    assert not result.abstained
    assert result.matches[0].label == SupportLabel.SUPPORTS
    assert result.matches[0].span.text == "No evaluation, no search, no opening book."
