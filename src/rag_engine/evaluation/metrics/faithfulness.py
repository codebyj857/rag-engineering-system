"""
Faithfulness evaluation metric.

Measures how strongly a generated answer is supported by the evidence
retrieved by the RAG system.

This implementation is deterministic and does not require an LLM API.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


class FaithfulnessMetricError(Exception):
    """Base exception for faithfulness metric errors."""


@dataclass(frozen=True, slots=True)
class FaithfulnessScore:
    """
    Faithfulness metric result.

    Attributes:
        score: Normalized score between 0 and 1.
        supported_terms: Terms found in the evidence.
        unsupported_terms: Terms from the answer not found in evidence.
        evidence_count: Number of evidence chunks used.
    """

    score: float
    supported_terms: tuple[str, ...]
    unsupported_terms: tuple[str, ...]
    evidence_count: int


class FaithfulnessMetric:
    """
    Deterministic evidence-support metric.

    The metric extracts meaningful terms from the generated answer and
    measures how many of those terms occur in the supplied evidence.
    """

    DEFAULT_MIN_TERM_LENGTH = 3

    _STOP_WORDS = frozenset(
        {
            "the",
            "and",
            "that",
            "this",
            "with",
            "from",
            "into",
            "have",
            "has",
            "are",
            "was",
            "were",
            "for",
            "not",
            "but",
            "can",
            "will",
            "would",
            "could",
            "should",
            "about",
            "what",
            "when",
            "where",
            "which",
            "while",
            "their",
            "there",
            "they",
            "them",
            "then",
            "than",
            "also",
            "using",
            "used",
            "uses",
            "its",
            "our",
            "your",
            "you",
            "how",
            "why",
            "who",
            "does",
            "did",
            "been",
            "being",
            "more",
            "most",
            "some",
            "such",
            "only",
            "very",
            "over",
            "under",
            "between",
            "through",
            "each",
            "other",
        }
    )

    def __init__(
        self,
        *,
        min_term_length: int = DEFAULT_MIN_TERM_LENGTH,
    ) -> None:
        if min_term_length < 1:
            raise FaithfulnessMetricError(
                "min_term_length must be at least 1."
            )

        self._min_term_length = min_term_length

    def calculate(
        self,
        *,
        answer: str,
        evidence: tuple[str, ...] | list[str],
    ) -> FaithfulnessScore:
        """
        Calculate faithfulness of an answer against evidence.

        Args:
            answer: Generated answer.
            evidence: Retrieved evidence text.

        Returns:
            FaithfulnessScore containing the normalized score and
            supporting term information.
        """

        if not isinstance(answer, str) or not answer.strip():
            raise FaithfulnessMetricError(
                "Answer cannot be empty."
            )

        if not isinstance(evidence, (tuple, list)):
            raise FaithfulnessMetricError(
                "Evidence must be a tuple or list of strings."
            )

        cleaned_evidence = tuple(
            item.strip()
            for item in evidence
            if isinstance(item, str) and item.strip()
        )

        answer_terms = self._extract_terms(answer)

        if not answer_terms:
            return FaithfulnessScore(
                score=1.0,
                supported_terms=(),
                unsupported_terms=(),
                evidence_count=len(cleaned_evidence),
            )

        evidence_text = " ".join(cleaned_evidence).lower()

        supported = tuple(
            term
            for term in answer_terms
            if term in evidence_text
        )

        unsupported = tuple(
            term
            for term in answer_terms
            if term not in evidence_text
        )

        score = len(supported) / len(answer_terms)

        return FaithfulnessScore(
            score=round(score, 4),
            supported_terms=supported,
            unsupported_terms=unsupported,
            evidence_count=len(cleaned_evidence),
        )

    def _extract_terms(self, text: str) -> tuple[str, ...]:
        """Extract meaningful normalized terms from text."""

        tokens = re.findall(
            r"[A-Za-z0-9]+",
            text.lower(),
        )

        terms: list[str] = []

        for token in tokens:
            if len(token) < self._min_term_length:
                continue

            if token in self._STOP_WORDS:
                continue

            if token not in terms:
                terms.append(token)

        return tuple(terms)


def calculate_faithfulness(
    *,
    answer: str,
    evidence: tuple[str, ...] | list[str],
) -> FaithfulnessScore:
    """Convenience function for calculating faithfulness."""

    return FaithfulnessMetric().calculate(
        answer=answer,
        evidence=evidence,
    )