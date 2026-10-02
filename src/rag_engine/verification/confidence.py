"""
Confidence estimation for RAG responses.

This module combines retrieval and groundedness signals into a single
confidence estimate. It is a heuristic engineering metric rather than
a probability that the answer is factually correct.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence


class ConfidenceError(Exception):
    """Base exception for confidence calculation errors."""


class InvalidConfidenceInputError(ConfidenceError):
    """Raised when confidence inputs are invalid."""


@dataclass(frozen=True, slots=True)
class ConfidenceResult:
    """
    Result of RAG confidence estimation.

    Attributes:
        score: Confidence score between 0.0 and 1.0.
        level: Human-readable confidence level.
        retrieval_score: Aggregated retrieval relevance.
        groundedness_score: Groundedness score.
        evidence_score: Evidence availability score.
    """

    score: float
    level: str
    retrieval_score: float
    groundedness_score: float
    evidence_score: float


class ConfidenceEstimator:
    """
    Estimate response confidence from multiple RAG signals.

    Default weighting:

    - retrieval relevance: 40%
    - groundedness: 40%
    - evidence availability: 20%

    Retrieval scores are accepted in either of these forms:

    - normalized scores already in the range [0.0, 1.0]
    - raw retrieval/reranker scores, which are normalized internally
      using min-max normalization when they fall outside [0.0, 1.0]

    Args:
        retrieval_weight: Weight assigned to retrieval quality.
        groundedness_weight: Weight assigned to groundedness.
        evidence_weight: Weight assigned to evidence availability.
    """

    def __init__(
        self,
        retrieval_weight: float = 0.40,
        groundedness_weight: float = 0.40,
        evidence_weight: float = 0.20,
    ) -> None:
        weights = (
            retrieval_weight,
            groundedness_weight,
            evidence_weight,
        )

        if any(weight < 0.0 or weight > 1.0 for weight in weights):
            raise ValueError(
                "Confidence weights must be between 0.0 and 1.0."
            )

        total = sum(weights)

        if abs(total - 1.0) > 1e-9:
            raise ValueError(
                "Confidence weights must sum to 1.0."
            )

        self._retrieval_weight = retrieval_weight
        self._groundedness_weight = groundedness_weight
        self._evidence_weight = evidence_weight

    def estimate(
        self,
        retrieval_scores: Sequence[float],
        groundedness_score: float,
        evidence_count: int,
    ) -> ConfidenceResult:
        """
        Calculate an overall confidence estimate.

        Args:
            retrieval_scores: Relevance scores from retrieved chunks.
            groundedness_score: Groundedness score between 0 and 1.
            evidence_count: Number of evidence chunks available.

        Returns:
            ConfidenceResult.

        Raises:
            InvalidConfidenceInputError: If an input is invalid.
        """
        normalized_scores = self._validate_retrieval_scores(
            retrieval_scores
        )

        normalized_groundedness = self._validate_score(
            groundedness_score,
            "groundedness_score",
        )

        if not isinstance(evidence_count, int):
            raise InvalidConfidenceInputError(
                "evidence_count must be an integer."
            )

        if evidence_count < 0:
            raise InvalidConfidenceInputError(
                "evidence_count cannot be negative."
            )

        retrieval_score = (
            sum(normalized_scores) / len(normalized_scores)
            if normalized_scores
            else 0.0
        )

        evidence_score = self._calculate_evidence_score(
            evidence_count
        )

        score = (
            retrieval_score * self._retrieval_weight
            + normalized_groundedness * self._groundedness_weight
            + evidence_score * self._evidence_weight
        )

        score = round(
            max(0.0, min(1.0, score)),
            4,
        )

        return ConfidenceResult(
            score=score,
            level=self._classify(score),
            retrieval_score=round(retrieval_score, 4),
            groundedness_score=round(normalized_groundedness, 4),
            evidence_score=round(evidence_score, 4),
        )

    @staticmethod
    def _validate_retrieval_scores(
        scores: Sequence[float],
    ) -> list[float]:
        """
        Validate and normalize retrieval relevance scores.

        Scores already in [0.0, 1.0] are preserved.

        Raw scores outside [0.0, 1.0], such as cross-encoder reranker
        scores, are converted to [0.0, 1.0] using min-max normalization.
        """
        if isinstance(scores, (str, bytes)):
            raise InvalidConfidenceInputError(
                "retrieval_scores must be a sequence of numbers."
            )

        raw_scores: list[float] = []

        for score in scores:
            if not isinstance(score, (int, float)):
                raise InvalidConfidenceInputError(
                    "Every retrieval score must be numeric."
                )

            numeric_score = float(score)

            if not math.isfinite(numeric_score):
                raise InvalidConfidenceInputError(
                    "Retrieval scores must be finite numbers."
                )

            raw_scores.append(numeric_score)

        if not raw_scores:
            return []

        if all(0.0 <= score <= 1.0 for score in raw_scores):
            return raw_scores

        minimum = min(raw_scores)
        maximum = max(raw_scores)

        if maximum == minimum:
            return [1.0 for _ in raw_scores]

        return [
            (score - minimum) / (maximum - minimum)
            for score in raw_scores
        ]

    @staticmethod
    def _validate_score(
        value: float,
        name: str,
    ) -> float:
        """Validate a normalized score."""
        if not isinstance(value, (int, float)):
            raise InvalidConfidenceInputError(
                f"{name} must be numeric."
            )

        normalized = float(value)

        if not math.isfinite(normalized):
            raise InvalidConfidenceInputError(
                f"{name} must be a finite number."
            )

        if not 0.0 <= normalized <= 1.0:
            raise InvalidConfidenceInputError(
                f"{name} must be between 0.0 and 1.0."
            )

        return normalized

    @staticmethod
    def _calculate_evidence_score(
        evidence_count: int,
    ) -> float:
        """
        Convert evidence availability into a normalized score.

        Three or more evidence chunks provide the maximum availability
        score. This is intentionally capped because more chunks do not
        automatically mean better evidence.
        """
        if evidence_count <= 0:
            return 0.0

        return min(evidence_count / 3.0, 1.0)

    @staticmethod
    def _classify(score: float) -> str:
        """Convert a numerical score into a confidence level."""
        if score >= 0.80:
            return "high"

        if score >= 0.60:
            return "medium"

        return "low"


def estimate_confidence(
    retrieval_scores: Sequence[float],
    groundedness_score: float,
    evidence_count: int,
) -> ConfidenceResult:
    """
    Convenience function for confidence estimation.
    """
    estimator = ConfidenceEstimator()

    return estimator.estimate(
        retrieval_scores=retrieval_scores,
        groundedness_score=groundedness_score,
        evidence_count=evidence_count,
    )