"""
Evaluation orchestration.

This module runs a RAG pipeline against an evaluation dataset and
calculates faithfulness, correctness, retrieval, and efficiency
metrics for every example.

The evaluator is intentionally dependency-injected so it can be tested
with fake pipelines without making real LLM API calls.
"""

from __future__ import annotations

import time
import uuid
from datetime import datetime, timezone
from typing import Protocol

from rag_engine.evaluation.dataset import EvaluationDataset
from rag_engine.evaluation.metrics.correctness import CorrectnessMetric
from rag_engine.evaluation.metrics.efficiency import EfficiencyMetric
from rag_engine.evaluation.metrics.faithfulness import FaithfulnessMetric
from rag_engine.evaluation.metrics.retrieval import RetrievalMetric
from rag_engine.evaluation.results import (
    EvaluationRunResult,
    ExampleEvaluationResult,
    MetricResult,
)
from rag_engine.pipelines.base import PipelineRequest, PipelineResponse


class EvaluatorError(Exception):
    """Base exception for evaluation errors."""


class EvaluationExecutionError(EvaluatorError):
    """Raised when an evaluation example cannot be executed."""


class EvaluationPipeline(Protocol):
    """Protocol describing the pipeline interface required by the evaluator."""

    def run(self, request: PipelineRequest) -> PipelineResponse:
        """Execute a RAG pipeline request."""


class Evaluator:
    """
    Evaluation orchestrator for a single RAG strategy.

    The evaluator receives an already-configured pipeline and executes
    every example in the supplied dataset.

    A fresh session ID is generated for every evaluation example so that
    conversation history from one dataset question cannot leak into
    another question.
    """

    def __init__(
        self,
        *,
        pipeline: EvaluationPipeline,
        faithfulness_metric: FaithfulnessMetric | None = None,
        correctness_metric: CorrectnessMetric | None = None,
        retrieval_metric: RetrievalMetric | None = None,
        efficiency_metric: EfficiencyMetric | None = None,
    ) -> None:
        if pipeline is None:
            raise EvaluatorError(
                "A configured evaluation pipeline is required."
            )

        self._pipeline = pipeline
        self._faithfulness_metric = (
            faithfulness_metric or FaithfulnessMetric()
        )
        self._correctness_metric = (
            correctness_metric or CorrectnessMetric()
        )
        self._retrieval_metric = (
            retrieval_metric or RetrievalMetric()
        )
        self._efficiency_metric = (
            efficiency_metric or EfficiencyMetric()
        )

    def evaluate(
        self,
        dataset: EvaluationDataset,
        *,
        strategy: str,
        top_k: int = 5,
        verify: bool = True,
    ) -> EvaluationRunResult:
        """
        Evaluate the configured pipeline against a complete dataset.

        Args:
            dataset: Validated evaluation dataset.
            strategy: Retrieval strategy being evaluated.
            top_k: Number of retrieval results to evaluate.
            verify: Whether pipeline groundedness verification is enabled.

        Returns:
            Aggregate EvaluationRunResult.

        Raises:
            EvaluatorError:
                If the dataset or evaluation configuration is invalid.
            EvaluationExecutionError:
                If an individual example fails.
        """

        if not isinstance(dataset, EvaluationDataset):
            raise EvaluatorError(
                "dataset must be an EvaluationDataset."
            )

        normalized_strategy = strategy.strip().lower()

        if not normalized_strategy:
            raise EvaluatorError(
                "Evaluation strategy cannot be empty."
            )

        if top_k < 1:
            raise EvaluatorError(
                "top_k must be at least 1."
            )

        started_at = datetime.now(timezone.utc).isoformat()
        example_results: list[ExampleEvaluationResult] = []

        for example in dataset:
            example_results.append(
                self._evaluate_example(
                    example=example,
                    strategy=normalized_strategy,
                    top_k=top_k,
                    verify=verify,
                )
            )

        aggregate_metrics = self._aggregate_metrics(
            example_results
        )

        completed_at = datetime.now(timezone.utc).isoformat()

        return EvaluationRunResult(
            strategy=normalized_strategy,
            examples=tuple(example_results),
            aggregate_metrics=tuple(aggregate_metrics),
            dataset_size=dataset.size,
            started_at=started_at,
            completed_at=completed_at,
            metadata={
                "top_k": top_k,
                "verification_enabled": verify,
            },
        )

    def _evaluate_example(
        self,
        *,
        example,
        strategy: str,
        top_k: int,
        verify: bool,
    ) -> ExampleEvaluationResult:
        """Evaluate one dataset example."""

        session_id = f"evaluation-{uuid.uuid4()}"

        request = PipelineRequest(
            question=example.question,
            session_id=session_id,
            top_k=top_k,
            verify=verify,
        )

        started = time.perf_counter()

        try:
            response = self._pipeline.run(request)
        except Exception as exc:
            raise EvaluationExecutionError(
                f"Evaluation failed for question: "
                f"{example.question[:100]!r}"
            ) from exc

        latency_seconds = time.perf_counter() - started

        evidence_text = tuple(
            evidence.content
            for evidence in response.evidence
        )

        retrieved_document_ids = self._extract_document_ids(
            response
        )

        retrieved_chunk_ids = tuple(
            evidence.chunk_id
            for evidence in response.evidence
        )

        generation = response.generation

        prompt_tokens = generation.prompt_tokens
        completion_tokens = generation.completion_tokens
        total_tokens = generation.total_tokens

        if total_tokens == 0:
            total_tokens = prompt_tokens + completion_tokens

        faithfulness = self._faithfulness_metric.calculate(
            answer=response.answer,
            evidence=evidence_text,
        )

        correctness = self._correctness_metric.calculate(
            answer=response.answer,
            reference_answer=example.reference_answer,
        )

        retrieval = self._retrieval_metric.calculate(
            retrieved_documents=retrieved_document_ids,
            relevant_documents=example.relevant_documents,
            top_k=top_k,
        )

        efficiency = self._efficiency_metric.calculate(
            latency_seconds=latency_seconds,
            total_tokens=total_tokens,
        )

        metrics = (
            MetricResult(
                name="faithfulness",
                value=faithfulness.score,
                metadata={
                    "supported_terms": faithfulness.supported_terms,
                    "unsupported_terms": faithfulness.unsupported_terms,
                    "evidence_count": faithfulness.evidence_count,
                },
            ),
            MetricResult(
                name="correctness",
                value=correctness.score,
                metadata={
                    "matched_terms": correctness.matched_terms,
                    "missing_terms": correctness.missing_terms,
                    "extra_terms": correctness.extra_terms,
                },
            ),
            MetricResult(
                name="precision_at_k",
                value=retrieval.precision_at_k,
                metadata={
                    "retrieved_count": retrieval.retrieved_count,
                    "relevant_count": retrieval.relevant_count,
                },
            ),
            MetricResult(
                name="recall_at_k",
                value=retrieval.recall_at_k,
                metadata={
                    "matched_documents": retrieval.matched_documents,
                    "missing_documents": retrieval.missing_documents,
                },
            ),
            MetricResult(
                name="f1_at_k",
                value=retrieval.f1_at_k,
            ),
            MetricResult(
                name="hit_at_k",
                value=retrieval.hit_at_k,
            ),
            MetricResult(
                name="efficiency",
                value=efficiency.efficiency_score,
                metadata={
                    "latency_score": efficiency.latency_score,
                    "token_score": efficiency.token_score,
                    "latency_seconds": efficiency.latency_seconds,
                    "total_tokens": efficiency.total_tokens,
                },
            ),
        )

        return ExampleEvaluationResult(
            question=example.question,
            reference_answer=example.reference_answer,
            generated_answer=response.answer,
            strategy=strategy,
            metrics=metrics,
            retrieved_chunk_ids=retrieved_chunk_ids,
            relevant_document_ids=example.relevant_documents,
            latency_seconds=latency_seconds,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            metadata={
                "retrieval_query": response.retrieval_query,
                "session_id": session_id,
                "verification": (
                    response.verification is not None
                ),
            },
        )

    @staticmethod
    def _extract_document_ids(
        response: PipelineResponse,
    ) -> tuple[str, ...]:
        """
        Extract document identifiers from retrieved evidence metadata.

        The ingestion/chunking layer stores document_id in chunk
        metadata, so evaluation uses that identifier when available.
        """

        document_ids: list[str] = []

        for evidence in response.evidence:
            document_id = evidence.metadata.get("document_id")

            if isinstance(document_id, str) and document_id.strip():
                normalized = document_id.strip()

                if normalized not in document_ids:
                    document_ids.append(normalized)

        return tuple(document_ids)

    @staticmethod
    def _aggregate_metrics(
        examples: list[ExampleEvaluationResult],
    ) -> list[MetricResult]:
        """Calculate mean values for every metric across examples."""

        if not examples:
            raise EvaluatorError(
                "Cannot aggregate an empty evaluation result."
            )

        metric_values: dict[str, list[float]] = {}

        for example in examples:
            for metric in example.metrics:
                metric_values.setdefault(
                    metric.name,
                    [],
                ).append(metric.value)

        aggregate_metrics: list[MetricResult] = []

        for metric_name, values in metric_values.items():
            average = sum(values) / len(values)

            aggregate_metrics.append(
                MetricResult(
                    name=metric_name,
                    value=round(average, 4),
                    metadata={
                        "example_count": len(values),
                    },
                )
            )

        return aggregate_metrics


def evaluate_pipeline(
    *,
    pipeline: EvaluationPipeline,
    dataset: EvaluationDataset,
    strategy: str,
    top_k: int = 5,
    verify: bool = True,
) -> EvaluationRunResult:
    """
    Convenience function for evaluating a configured RAG pipeline.
    """

    evaluator = Evaluator(
        pipeline=pipeline,
    )

    return evaluator.evaluate(
        dataset,
        strategy=strategy,
        top_k=top_k,
        verify=verify,
    )