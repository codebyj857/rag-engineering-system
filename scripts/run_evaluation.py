"""
Run the complete RAG strategy evaluation experiment.

This script evaluates the same evaluation dataset against:
    1. Naive retrieval
    2. Hybrid retrieval
    3. Reranked retrieval

The same vector store, generation configuration, verification
configuration, and evaluation settings are used across all strategies.

Results are saved as one JSON artifact containing the three complete
EvaluationRunResult objects.

The script intentionally does not rank or declare a winner. It records
the measured results so they can be compared later through tables and
visualizations.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from rag_engine.config import get_settings
from rag_engine.evaluation.dataset import load_evaluation_dataset
from rag_engine.evaluation.evaluator import Evaluator
from rag_engine.generation.generator import Generator
from rag_engine.memory.query_contextualizer import QueryContextualizer
from rag_engine.memory.session_store import SessionStore
from rag_engine.pipelines.hybrid import HybridRAGPipeline
from rag_engine.pipelines.naive import NaiveRAGPipeline
from rag_engine.pipelines.reranked import RerankedRAGPipeline
from rag_engine.retrieval.hybrid import HybridRetriever
from rag_engine.retrieval.naive import NaiveRetriever
from rag_engine.retrieval.reranker import RerankedRetriever
from rag_engine.verification.confidence import ConfidenceEstimator
from rag_engine.verification.groundedness import GroundednessVerifier
from rag_engine.indexing.vector_store import VectorStore


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATASET_PATH = (
    PROJECT_ROOT
    / "data"
    / "evaluation"
    / "questions.jsonl"
)

RUNS_DIRECTORY = (
    PROJECT_ROOT
    / "data"
    / "evaluation"
    / "runs"
)

OUTPUT_PATH = RUNS_DIRECTORY / "comparison_run.json"

TOP_K = 5
VERIFY = True


def _serialize_run(run_result) -> dict:
    """Convert an EvaluationRunResult into JSON-compatible data."""

    return asdict(run_result)


def _build_shared_services() -> tuple[
    VectorStore,
    Generator,
    QueryContextualizer,
    GroundednessVerifier,
    ConfidenceEstimator,
]:
    """
    Build services shared by all three strategies.

    Sharing the vector store ensures every strategy evaluates the same
    indexed corpus. The other services are also configured identically.
    """

    vector_store = VectorStore()

    generator = Generator()

    contextualizer = QueryContextualizer()

    groundedness_verifier = GroundednessVerifier()

    confidence_estimator = ConfidenceEstimator()

    return (
        vector_store,
        generator,
        contextualizer,
        groundedness_verifier,
        confidence_estimator,
    )


def _build_pipelines() -> dict[str, object]:
    """Build the three fully configured RAG pipelines."""

    (
        vector_store,
        generator,
        contextualizer,
        groundedness_verifier,
        confidence_estimator,
    ) = _build_shared_services()

    pipelines = {
        "naive": NaiveRAGPipeline(
            retriever=NaiveRetriever(
                vector_store=vector_store,
                top_k=TOP_K,
            ),
            session_store=SessionStore(),
            contextualizer=contextualizer,
            generator=generator,
            groundedness_verifier=groundedness_verifier,
            confidence_estimator=confidence_estimator,
        ),
        "hybrid": HybridRAGPipeline(
            retriever=HybridRetriever(
                vector_store=vector_store,
                top_k=TOP_K,
            ),
            session_store=SessionStore(),
            contextualizer=contextualizer,
            generator=generator,
            groundedness_verifier=groundedness_verifier,
            confidence_estimator=confidence_estimator,
        ),
        "reranked": RerankedRAGPipeline(
            retriever=RerankedRetriever(
                vector_store=vector_store,
                top_k=TOP_K,
            ),
            session_store=SessionStore(),
            contextualizer=contextualizer,
            generator=generator,
            groundedness_verifier=groundedness_verifier,
            confidence_estimator=confidence_estimator,
        ),
    }

    return pipelines


def main() -> None:
    """Run the complete three-strategy evaluation experiment."""

    settings = get_settings()

    print("=" * 72)
    print("RAG ENGINEERING SYSTEM — STRATEGY EVALUATION")
    print("=" * 72)

    print(f"Dataset: {DATASET_PATH}")
    print(f"Top-k: {TOP_K}")
    print(f"Verification enabled: {VERIFY}")
    print()

    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Evaluation dataset not found: {DATASET_PATH}"
        )

    if not settings.groq_api_key:
        raise RuntimeError(
            "GROQ_API_KEY is not configured. "
            "Add the real API key to .env before running the live "
            "evaluation experiment."
        )

    dataset = load_evaluation_dataset(
        DATASET_PATH
    )

    print(f"Loaded evaluation questions: {dataset.size}")
    print()

    pipelines = _build_pipelines()

    evaluator_results = {}

    for strategy, pipeline in pipelines.items():
        print("-" * 72)
        print(f"Running strategy: {strategy}")
        print("-" * 72)

        evaluator = Evaluator(
            pipeline=pipeline,
        )

        result = evaluator.evaluate(
            dataset,
            strategy=strategy,
            top_k=TOP_K,
            verify=VERIFY,
        )

        evaluator_results[strategy] = result

        print(
            f"Completed {result.example_count} questions "
            f"for {strategy}."
        )

        print("Aggregate metrics:")

        for metric in result.aggregate_metrics:
            print(
                f"  {metric.name}: "
                f"{metric.value:.4f}"
            )

        print()

    RUNS_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    output = {
        "experiment": {
            "dataset": str(DATASET_PATH),
            "dataset_size": dataset.size,
            "top_k": TOP_K,
            "verification_enabled": VERIFY,
            "strategies": list(evaluator_results.keys()),
        },
        "runs": {
            strategy: _serialize_run(result)
            for strategy, result in evaluator_results.items()
        },
    }

    with OUTPUT_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            output,
            file,
            indent=2,
            ensure_ascii=False,
        )

    print("=" * 72)
    print("EVALUATION COMPLETE")
    print("=" * 72)
    print(f"Total strategies: {len(evaluator_results)}")
    print(
        f"Total evaluated examples: "
        f"{sum(result.example_count for result in evaluator_results.values())}"
    )
    print(f"Results saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()