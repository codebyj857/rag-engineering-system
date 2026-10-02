"""
End-to-end tests for the evaluation engine.

These tests use a fake RAG pipeline so no external LLM API is called.
"""

from __future__ import annotations

from rag_engine.evaluation.dataset import (
    EvaluationDataset,
    EvaluationExample,
)
from rag_engine.evaluation.evaluator import Evaluator
from rag_engine.generation.generator import GenerationResult
from rag_engine.pipelines.base import (
    PipelineEvidence,
    PipelineRequest,
    PipelineResponse,
)


class FakePipeline:
    """Deterministic pipeline used for evaluation tests."""

    def run(self, request: PipelineRequest) -> PipelineResponse:
        """Return a deterministic RAG response."""

        evidence = (
            PipelineEvidence(
                chunk_id="chunk-001",
                content=(
                    "RAG combines retrieval with generation. "
                    "The system retrieves relevant documents "
                    "and provides them as context to the model."
                ),
                source="rag_basics.txt",
                filename="rag_basics.txt",
                score=0.95,
                rank=1,
                metadata={
                    "document_id": "doc-rag-basics",
                },
            ),
            PipelineEvidence(
                chunk_id="chunk-002",
                content=(
                    "Retrieval augmented generation can improve "
                    "answers by grounding generation in external "
                    "information."
                ),
                source="rag_basics.txt",
                filename="rag_basics.txt",
                score=0.87,
                rank=2,
                metadata={
                    "document_id": "doc-rag-basics",
                },
            ),
        )

        generation = GenerationResult(
            answer=(
                "RAG combines retrieval with generation by "
                "providing relevant documents as context."
            ),
            model="fake-model",
            prompt_tokens=100,
            completion_tokens=40,
            total_tokens=140,
        )

        return PipelineResponse(
            session_id=request.session_id,
            original_query=request.question,
            retrieval_query=request.question,
            strategy="hybrid",
            answer=generation.answer,
            evidence=evidence,
            verification=None,
            confidence=None,
            generation=generation,
            metadata={},
        )


def test_evaluator_runs_complete_dataset() -> None:
    """The evaluator should process every dataset example."""

    dataset = EvaluationDataset(
        (
            EvaluationExample(
                question="What is RAG?",
                reference_answer=(
                    "RAG combines retrieval with generation "
                    "using relevant documents as context."
                ),
                relevant_documents=("doc-rag-basics",),
            ),
            EvaluationExample(
                question="Why is RAG useful?",
                reference_answer=(
                    "RAG can ground generated answers in "
                    "external information."
                ),
                relevant_documents=("doc-rag-basics",),
            ),
        )
    )

    evaluator = Evaluator(
        pipeline=FakePipeline(),
    )

    result = evaluator.evaluate(
        dataset,
        strategy="hybrid",
        top_k=2,
    )

    assert result.strategy == "hybrid"
    assert result.dataset_size == 2
    assert result.example_count == 2

    metric_names = {
        metric.name
        for metric in result.aggregate_metrics
    }

    assert "faithfulness" in metric_names
    assert "correctness" in metric_names
    assert "precision_at_k" in metric_names
    assert "recall_at_k" in metric_names
    assert "f1_at_k" in metric_names
    assert "hit_at_k" in metric_names
    assert "efficiency" in metric_names


def test_evaluator_creates_independent_sessions() -> None:
    """Each evaluation example should receive a fresh session."""

    dataset = EvaluationDataset(
        (
            EvaluationExample(
                question="What is RAG?",
                reference_answer="RAG combines retrieval and generation.",
                relevant_documents=("doc-rag-basics",),
            ),
            EvaluationExample(
                question="Why is retrieval useful?",
                reference_answer="Retrieval provides relevant context.",
                relevant_documents=("doc-rag-basics",),
            ),
        )
    )

    evaluator = Evaluator(
        pipeline=FakePipeline(),
    )

    result = evaluator.evaluate(
        dataset,
        strategy="naive",
        top_k=2,
    )

    session_ids = {
        example.metadata["session_id"]
        for example in result.examples
    }

    assert len(session_ids) == 2