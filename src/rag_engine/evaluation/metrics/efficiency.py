"""
Efficiency evaluation metrics.

This module measures computational efficiency indicators such as
latency and token usage during RAG evaluation.

The metric values are normalized to the range [0, 1] so they can be
stored consistently with the project's MetricResult model.

No LLM or external API is required.
"""

from __future__ import annotations

from dataclasses import dataclass


class EfficiencyMetricError(Exception):
    """Base exception for efficiency metric errors."""


@dataclass(frozen=True, slots=True)
class EfficiencyScore:
    """
    Efficiency metric result.

    Attributes:
        efficiency_score: Combined normalized efficiency score.
        latency_score: Normalized latency efficiency.
        token_score: Normalized token efficiency.
        latency_seconds: Raw measured latency.
        total_tokens: Raw total token usage.
    """

    efficiency_score: float
    latency_score: float
    token_score: float
    latency_seconds: float
    total_tokens: int


class EfficiencyMetric:
    """
    Deterministic efficiency evaluator.

    Lower latency and lower token usage produce higher efficiency
    scores.

    The score uses configurable reference limits rather than assuming
    that a specific hardware environment has a fixed performance.
    """

    DEFAULT_LATENCY_REFERENCE_SECONDS = 10.0
    DEFAULT_TOKEN_REFERENCE = 2000

    def __init__(
        self,
        *,
        latency_reference_seconds: float = DEFAULT_LATENCY_REFERENCE_SECONDS,
        token_reference: int = DEFAULT_TOKEN_REFERENCE,
        latency_weight: float = 0.5,
        token_weight: float = 0.5,
    ) -> None:
        if latency_reference_seconds <= 0:
            raise EfficiencyMetricError(
                "latency_reference_seconds must be greater than 0."
            )

        if token_reference <= 0:
            raise EfficiencyMetricError(
                "token_reference must be greater than 0."
            )

        if latency_weight < 0 or token_weight < 0:
            raise EfficiencyMetricError(
                "Efficiency weights cannot be negative."
            )

        weight_sum = latency_weight + token_weight

        if weight_sum == 0:
            raise EfficiencyMetricError(
                "At least one efficiency weight must be greater than 0."
            )

        self._latency_reference_seconds = latency_reference_seconds
        self._token_reference = token_reference

        self._latency_weight = latency_weight / weight_sum
        self._token_weight = token_weight / weight_sum

    def calculate(
        self,
        *,
        latency_seconds: float,
        total_tokens: int,
    ) -> EfficiencyScore:
        """
        Calculate efficiency metrics.

        Args:
            latency_seconds:
                Measured end-to-end latency for the evaluation example.
            total_tokens:
                Total prompt and completion tokens used.

        Returns:
            EfficiencyScore containing normalized and raw measurements.
        """

        if latency_seconds < 0:
            raise EfficiencyMetricError(
                "latency_seconds cannot be negative."
            )

        if total_tokens < 0:
            raise EfficiencyMetricError(
                "total_tokens cannot be negative."
            )

        latency_score = self._inverse_normalized_score(
            latency_seconds,
            self._latency_reference_seconds,
        )

        token_score = self._inverse_normalized_score(
            float(total_tokens),
            float(self._token_reference),
        )

        efficiency_score = (
            latency_score * self._latency_weight
            + token_score * self._token_weight
        )

        return EfficiencyScore(
            efficiency_score=round(efficiency_score, 4),
            latency_score=round(latency_score, 4),
            token_score=round(token_score, 4),
            latency_seconds=latency_seconds,
            total_tokens=total_tokens,
        )

    @staticmethod
    def _inverse_normalized_score(
        value: float,
        reference: float,
    ) -> float:
        """
        Convert a cost-like value into an efficiency score.

        A value of zero receives 1.0.
        A value equal to the reference receives 0.5.
        Values significantly above the reference approach 0.
        """

        if value == 0:
            return 1.0

        score = reference / (reference + value)

        return max(0.0, min(1.0, score))


def calculate_efficiency(
    *,
    latency_seconds: float,
    total_tokens: int,
) -> EfficiencyScore:
    """Convenience function for calculating efficiency."""

    return EfficiencyMetric().calculate(
        latency_seconds=latency_seconds,
        total_tokens=total_tokens,
    )