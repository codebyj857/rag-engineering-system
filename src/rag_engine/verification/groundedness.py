"""
Deterministic groundedness verification for generated RAG answers.

This module checks whether important terms from a generated answer are
supported by the retrieved evidence. It is intentionally independent of
any LLM provider so verification can run during development and testing
without consuming API credits.

The verifier is a lightweight signal, not a formal proof of factual
correctness.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Sequence


class GroundednessError(Exception):
    """Base exception for groundedness verification errors."""


class InvalidGroundednessInputError(GroundednessError):
    """Raised when groundedness input is invalid."""


@dataclass(frozen=True, slots=True)
class GroundednessResult:
    """
    Result of deterministic groundedness analysis.

    Attributes:
        grounded: Whether the answer passes the configured threshold.
        score: Support score between 0.0 and 1.0.
        supported_terms: Important answer terms found in the evidence.
        unsupported_terms: Important answer terms absent from the evidence.
        evidence_count: Number of evidence items analyzed.
        threshold: Score required for a grounded result.
    """

    grounded: bool
    score: float
    supported_terms: tuple[str, ...]
    unsupported_terms: tuple[str, ...]
    evidence_count: int
    threshold: float


class GroundednessVerifier:
    """
    Perform deterministic evidence-support analysis.

    The verifier extracts meaningful terms from the generated answer and
    checks how many occur in the supplied evidence. Very common English
    stop words are ignored.

    Args:
        threshold: Minimum support score required to classify an answer
            as grounded.
        min_term_length: Minimum length for a term to be considered.
    """

    _STOP_WORDS = {
        "a",
        "about",
        "after",
        "again",
        "against",
        "all",
        "also",
        "am",
        "an",
        "and",
        "any",
        "are",
        "as",
        "at",
        "be",
        "because",
        "been",
        "before",
        "being",
        "between",
        "both",
        "but",
        "by",
        "can",
        "could",
        "did",
        "do",
        "does",
        "for",
        "from",
        "further",
        "had",
        "has",
        "have",
        "he",
        "her",
        "here",
        "hers",
        "him",
        "his",
        "how",
        "i",
        "if",
        "in",
        "into",
        "is",
        "it",
        "its",
        "itself",
        "just",
        "me",
        "more",
        "most",
        "my",
        "no",
        "nor",
        "not",
        "of",
        "on",
        "once",
        "only",
        "or",
        "other",
        "our",
        "ours",
        "out",
        "over",
        "own",
        "same",
        "she",
        "should",
        "so",
        "some",
        "such",
        "than",
        "that",
        "the",
        "their",
        "theirs",
        "them",
        "then",
        "there",
        "these",
        "they",
        "this",
        "those",
        "through",
        "to",
        "too",
        "under",
        "until",
        "up",
        "very",
        "was",
        "we",
        "were",
        "what",
        "when",
        "where",
        "which",
        "while",
        "who",
        "whom",
        "why",
        "will",
        "with",
        "would",
        "you",
        "your",
        "yours",
    }

    def __init__(
        self,
        threshold: float = 0.60,
        min_term_length: int = 3,
    ) -> None:
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(
                "threshold must be between 0.0 and 1.0."
            )

        if min_term_length < 1:
            raise ValueError(
                "min_term_length must be at least 1."
            )

        self._threshold = threshold
        self._min_term_length = min_term_length

    @property
    def threshold(self) -> float:
        """Return the configured groundedness threshold."""
        return self._threshold

    def verify(
        self,
        answer: str,
        evidence: Sequence[str],
    ) -> GroundednessResult:
        """
        Check whether the answer is supported by the evidence.

        Args:
            answer: Generated answer text.
            evidence: Retrieved evidence texts.

        Returns:
            GroundednessResult containing the support analysis.

        Raises:
            InvalidGroundednessInputError: If answer or evidence is invalid.
        """
        normalized_answer = self._validate_answer(answer)
        normalized_evidence = self._validate_evidence(evidence)

        answer_terms = self._extract_terms(normalized_answer)

        if not answer_terms:
            return GroundednessResult(
                grounded=False,
                score=0.0,
                supported_terms=(),
                unsupported_terms=(),
                evidence_count=len(normalized_evidence),
                threshold=self._threshold,
            )

        evidence_text = " ".join(normalized_evidence).lower()
        evidence_terms = self._extract_terms(evidence_text)

        supported = tuple(
            term
            for term in answer_terms
            if term in evidence_terms
        )

        unsupported = tuple(
            term
            for term in answer_terms
            if term not in evidence_terms
        )

        score = len(supported) / len(answer_terms)

        return GroundednessResult(
            grounded=score >= self._threshold,
            score=round(score, 4),
            supported_terms=supported,
            unsupported_terms=unsupported,
            evidence_count=len(normalized_evidence),
            threshold=self._threshold,
        )

    def _extract_terms(self, text: str) -> tuple[str, ...]:
        """
        Extract unique meaningful terms while preserving order.
        """
        tokens = re.findall(r"[a-zA-Z0-9][a-zA-Z0-9_-]*", text.lower())

        terms: list[str] = []
        seen: set[str] = set()

        for token in tokens:
            cleaned = token.strip("_-")

            if len(cleaned) < self._min_term_length:
                continue

            if cleaned in self._STOP_WORDS:
                continue

            if cleaned in seen:
                continue

            seen.add(cleaned)
            terms.append(cleaned)

        return tuple(terms)

    @staticmethod
    def _validate_answer(answer: str) -> str:
        """Validate generated answer text."""
        if not isinstance(answer, str):
            raise InvalidGroundednessInputError(
                "answer must be a string."
            )

        normalized = answer.strip()

        if not normalized:
            raise InvalidGroundednessInputError(
                "answer cannot be empty."
            )

        return normalized

    @staticmethod
    def _validate_evidence(
        evidence: Sequence[str],
    ) -> list[str]:
        """Validate and normalize evidence texts."""
        if not isinstance(evidence, Sequence) or isinstance(
            evidence,
            (str, bytes),
        ):
            raise InvalidGroundednessInputError(
                "evidence must be a sequence of strings."
            )

        normalized: list[str] = []

        for item in evidence:
            if not isinstance(item, str):
                raise InvalidGroundednessInputError(
                    "Every evidence item must be a string."
                )

            cleaned = item.strip()

            if cleaned:
                normalized.append(cleaned)

        return normalized


def verify_groundedness(
    answer: str,
    evidence: Sequence[str],
    threshold: float = 0.60,
) -> GroundednessResult:
    """
    Convenience function for groundedness verification.
    """
    verifier = GroundednessVerifier(threshold=threshold)

    return verifier.verify(
        answer=answer,
        evidence=evidence,
    )