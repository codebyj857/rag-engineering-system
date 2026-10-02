"""
Evaluation result models.

This module defines the structured result objects used to represent
individual evaluation runs, metric values, and aggregate comparisons.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


class EvaluationResultError(Exception):
    """Base exception for evaluation result errors."""


class InvalidMetricResultError(EvaluationResultError):
    """Raised when a metric result contains invalid values."""


@dataclass(frozen=True, slots=True)
class MetricResult:
    """
    Result produced by a single evaluation metric.

    Attributes:
        name: Metric name.
        value: Numeric metric value.
        metadata: Optional additional metric information.
    """

    name: str
    value: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise InvalidMetricResultError(
                "Metric name cannot be empty."
            )

        if not 0.0 <= self.value <= 1.0:
            raise InvalidMetricResultError(
                f"Metric '{self.name}' must be between 0 and 1. "
                f"Received: {self.value}"
            )


@dataclass(frozen=True, slots=True)
class ExampleEvaluationResult:
    """
    Evaluation result for one question.

    Stores the generated answer, reference answer, retrieved evidence,
    latency, token usage, and calculated metrics.
    """

    question: str
    reference_answer: str
    generated_answer: str
    strategy: str
    metrics: tuple[MetricResult, ...]
    retrieved_chunk_ids: tuple[str, ...] = ()
    relevant_document_ids: tuple[str, ...] = ()
    latency_seconds: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.question.strip():
            raise EvaluationResultError(
                "Question cannot be empty."
            )

        if not self.strategy.strip():
            raise EvaluationResultError(
                "Strategy cannot be empty."
            )

        if self.latency_seconds < 0:
            raise EvaluationResultError(
                "Latency cannot be negative."
            )

        if min(
            self.prompt_tokens,
            self.completion_tokens,
            self.total_tokens,
        ) < 0:
            raise EvaluationResultError(
                "Token counts cannot be negative."
            )


@dataclass(frozen=True, slots=True)
class EvaluationRunResult:
    """
    Aggregate result for one complete evaluation run.

    A run corresponds to one strategy evaluated against the complete
    evaluation dataset.
    """

    strategy: str
    examples: tuple[ExampleEvaluationResult, ...]
    aggregate_metrics: tuple[MetricResult, ...]
    dataset_size: int
    started_at: str
    completed_at: str
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def create(
        cls,
        *,
        strategy: str,
        examples: tuple[ExampleEvaluationResult, ...],
        aggregate_metrics: tuple[MetricResult, ...],
        dataset_size: int,
        metadata: dict[str, Any] | None = None,
    ) -> "EvaluationRunResult":
        """Create a completed evaluation run with UTC timestamps."""

        now = datetime.now(timezone.utc).isoformat()

        return cls(
            strategy=strategy,
            examples=examples,
            aggregate_metrics=aggregate_metrics,
            dataset_size=dataset_size,
            started_at=now,
            completed_at=now,
            metadata=dict(metadata or {}),
        )

    def get_metric(self, name: str) -> MetricResult | None:
        """Return an aggregate metric by name."""

        normalized_name = name.strip().lower()

        for metric in self.aggregate_metrics:
            if metric.name.lower() == normalized_name:
                return metric

        return None

    @property
    def example_count(self) -> int:
        """Return the number of evaluated examples."""

        return len(self.examples)


@dataclass(frozen=True, slots=True)
class EvaluationComparison:
    """
    Comparison of multiple evaluation runs.

    No ranking or winner is implied by this structure. It simply
    stores comparable results for each evaluated strategy.
    """

    runs: tuple[EvaluationRunResult, ...]
    metrics: tuple[str, ...]
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def strategies(self) -> tuple[str, ...]:
        """Return the strategies represented in the comparison."""

        return tuple(run.strategy for run in self.runs)