"""
Correctness evaluation metric.

Measures lexical similarity between a generated answer and its
reference answer.

This implementation is deterministic and does not require an LLM API.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


class CorrectnessMetricError(Exception):
    """Base exception for correctness metric errors."""


@dataclass(frozen=True, slots=True)
class CorrectnessScore:
    """
    Correctness metric result.

    Attributes:
        score: Normalized similarity score between 0 and 1.
        matched_terms: Terms shared by generated and reference answers.
        missing_terms: Important reference terms absent from the answer.
        extra_terms: Terms present in the answer but absent from the
            reference.
    """

    score: float
    matched_terms: tuple[str, ...]
    missing_terms: tuple[str, ...]
    extra_terms: tuple[str, ...]


class CorrectnessMetric:
    """
    Deterministic answer-correctness metric.

    The metric compares meaningful normalized terms between the generated
    answer and the reference answer.

    The score uses an F1-style calculation:

        precision = matched / generated_terms
        recall    = matched / reference_terms
        F1        = 2 * precision * recall / (precision + recall)
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
            raise CorrectnessMetricError(
                "min_term_length must be at least 1."
            )

        self._min_term_length = min_term_length

    def calculate(
        self,
        *,
        answer: str,
        reference_answer: str,
    ) -> CorrectnessScore:
        """
        Calculate correctness against a reference answer.

        Args:
            answer: Generated answer.
            reference_answer: Expected/reference answer.

        Returns:
            CorrectnessScore containing the normalized score and
            term-level comparison details.
        """

        if not isinstance(answer, str) or not answer.strip():
            raise CorrectnessMetricError(
                "Generated answer cannot be empty."
            )

        if (
            not isinstance(reference_answer, str)
            or not reference_answer.strip()
        ):
            raise CorrectnessMetricError(
                "Reference answer cannot be empty."
            )

        answer_terms = self._extract_terms(answer)
        reference_terms = self._extract_terms(reference_answer)

        if not reference_terms:
            raise CorrectnessMetricError(
                "Reference answer does not contain enough meaningful terms."
            )

        if not answer_terms:
            return CorrectnessScore(
                score=0.0,
                matched_terms=(),
                missing_terms=reference_terms,
                extra_terms=(),
            )

        reference_set = set(reference_terms)
        answer_set = set(answer_terms)

        matched = tuple(
            term
            for term in answer_terms
            if term in reference_set
        )

        missing = tuple(
            term
            for term in reference_terms
            if term not in answer_set
        )

        extra = tuple(
            term
            for term in answer_terms
            if term not in reference_set
        )

        matched_count = len(set(matched))

        precision = matched_count / len(answer_set)
        recall = matched_count / len(reference_set)

        if precision + recall == 0:
            score = 0.0
        else:
            score = (
                2 * precision * recall
            ) / (precision + recall)

        return CorrectnessScore(
            score=round(score, 4),
            matched_terms=tuple(dict.fromkeys(matched)),
            missing_terms=missing,
            extra_terms=tuple(dict.fromkeys(extra)),
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


def calculate_correctness(
    *,
    answer: str,
    reference_answer: str,
) -> CorrectnessScore:
    """Convenience function for calculating correctness."""

    return CorrectnessMetric().calculate(
        answer=answer,
        reference_answer=reference_answer,
    )