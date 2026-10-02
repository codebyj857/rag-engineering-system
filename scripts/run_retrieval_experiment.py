"""
Run a local retrieval-strategy experiment.

This experiment compares:

    1. Naive retrieval
    2. Hybrid retrieval
    3. Reranked retrieval

The experiment performs retrieval only.

It does NOT:
    - call an LLM
    - use the Groq API
    - generate answers
    - consume API quota

It evaluates all strategies against the same evaluation dataset and
records document-level and chunk-level retrieval metrics, MRR,
rank movement, retrieval scores, candidate-pool information, and latency.

Document-level metrics use UNIQUE document IDs in retrieval rank order.
Chunk-level metrics use individual chunk IDs.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from statistics import mean
from typing import Any

from rag_engine.evaluation.dataset import load_evaluation_dataset
from rag_engine.indexing.vector_store import VectorStore
from rag_engine.retrieval.hybrid import HybridRetriever
from rag_engine.retrieval.naive import NaiveRetriever
from rag_engine.retrieval.reranker import RerankedRetriever


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATASET_PATH = PROJECT_ROOT / "data" / "evaluation" / "questions.jsonl"

RUNS_DIRECTORY = PROJECT_ROOT / "data" / "evaluation" / "runs"

OUTPUT_PATH = RUNS_DIRECTORY / "retrieval_experiment.json"

TOP_K = 5

RERANKER_CANDIDATE_MULTIPLIER = 4

RERANKER_CANDIDATE_POOL = max(
    TOP_K * RERANKER_CANDIDATE_MULTIPLIER,
    10,
)


def _unique_preserving_order(values: list[str]) -> list[str]:
    """Return unique values while preserving their first-seen order."""
    seen: set[str] = set()
    unique_values: list[str] = []

    for value in values:
        if value not in seen:
            seen.add(value)
            unique_values.append(value)

    return unique_values


def _document_ids(results: list[Any]) -> list[str]:
    """
    Extract unique document IDs in retrieval rank order.

    Multiple chunks can belong to the same source document. For
    document-level evaluation, each document is counted once.
    """
    document_ids = [
        str(result.metadata["document_id"])
        for result in results
        if "document_id" in result.metadata
    ]

    return _unique_preserving_order(document_ids)


def _retrieved_chunk_ids(results: list[Any]) -> list[str]:
    """Extract retrieved chunk IDs in rank order."""
    return [result.chunk_id for result in results]


def _precision_at_k(
    retrieved_ids: list[str],
    relevant_ids: set[str],
    k: int,
) -> float:
    """
    Calculate precision@k.

    The denominator is the number of retrieved unique items considered,
    capped at k. This is appropriate for the document-level metric because
    duplicate chunks from one document are removed before evaluation.
    """
    if k <= 0:
        return 0.0

    top_results = retrieved_ids[:k]

    if not top_results:
        return 0.0

    relevant_count = sum(
        item_id in relevant_ids
        for item_id in top_results
    )

    return relevant_count / len(top_results)


def _recall_at_k(
    retrieved_ids: list[str],
    relevant_ids: set[str],
    k: int,
) -> float:
    """Calculate recall@k."""
    if not relevant_ids:
        return 0.0

    top_results = retrieved_ids[:k]

    relevant_retrieved = len(
        set(top_results).intersection(relevant_ids)
    )

    return relevant_retrieved / len(relevant_ids)


def _f1(
    precision: float,
    recall: float,
) -> float:
    """Calculate F1 from precision and recall."""
    if precision + recall == 0:
        return 0.0

    return (
        2 * precision * recall
        / (precision + recall)
    )


def _hit_at_k(
    retrieved_ids: list[str],
    relevant_ids: set[str],
    k: int,
) -> float:
    """Return 1 when at least one relevant item appears in top-k."""
    top_results = retrieved_ids[:k]

    return (
        1.0
        if set(top_results).intersection(relevant_ids)
        else 0.0
    )


def _mrr(
    retrieved_ids: list[str],
    relevant_ids: set[str],
    k: int,
) -> float:
    """
    Calculate Mean Reciprocal Rank for one query.

    Returns reciprocal rank of the first relevant result within top-k.
    Returns 0 when no relevant result is found.
    """
    for rank, item_id in enumerate(
        retrieved_ids[:k],
        start=1,
    ):
        if item_id in relevant_ids:
            return 1.0 / rank

    return 0.0


def _first_relevant_rank(
    retrieved_ids: list[str],
    relevant_ids: set[str],
    k: int,
) -> int | None:
    """Return the 1-based rank of the first relevant result."""
    for rank, item_id in enumerate(
        retrieved_ids[:k],
        start=1,
    ):
        if item_id in relevant_ids:
            return rank

    return None


def _relevant_ranks(
    retrieved_ids: list[str],
    relevant_ids: set[str],
) -> dict[str, int]:
    """Return ranks for all relevant IDs found in supplied results."""
    ranks: dict[str, int] = {}

    for rank, item_id in enumerate(
        retrieved_ids,
        start=1,
    ):
        if item_id in relevant_ids and item_id not in ranks:
            ranks[item_id] = rank

    return ranks


def _rank_movement(
    before_rank: int | None,
    after_rank: int | None,
) -> int | None:
    """
    Calculate rank movement.

    Positive:
        moved upward.

    Negative:
        moved downward.

    Zero:
        unchanged.

    None:
        relevant chunk unavailable at one of the measured ranks.
    """
    if before_rank is None or after_rank is None:
        return None

    return before_rank - after_rank


def _result_scores(
    results: list[Any],
) -> list[dict[str, Any]]:
    """Extract retrieval/reranker scores from results."""
    scores: list[dict[str, Any]] = []

    for result in results:
        metadata = result.metadata

        entry: dict[str, Any] = {
            "chunk_id": result.chunk_id,
            "rank": result.rank,
            "score": round(float(result.score), 6),
        }

        if "vector_score" in metadata:
            entry["vector_score"] = round(
                float(metadata["vector_score"]),
                6,
            )

        if "reranker_score" in metadata:
            entry["reranker_score"] = round(
                float(metadata["reranker_score"]),
                6,
            )

        scores.append(entry)

    return scores


def _build_retrievers(
    vector_store: VectorStore,
) -> dict[str, object]:
    """Create all three retrieval strategies."""
    return {
        "naive": NaiveRetriever(
            vector_store=vector_store,
            top_k=TOP_K,
        ),
        "hybrid": HybridRetriever(
            vector_store=vector_store,
            top_k=TOP_K,
        ),
        "reranked": RerankedRetriever(
            vector_store=vector_store,
            top_k=TOP_K,
            candidate_multiplier=RERANKER_CANDIDATE_MULTIPLIER,
        ),
    }


def _get_reranker_candidate_results(
    vector_store: VectorStore,
    question: str,
) -> list[Any]:
    """
    Retrieve the semantic candidate pool used by reranking.

    Used to measure original vector rank before cross-encoder reranking.
    """
    return vector_store.search(
        query=question,
        top_k=RERANKER_CANDIDATE_POOL,
    )


def _calculate_rank_movement(
    vector_store: VectorStore,
    question: str,
    relevant_chunks: set[str],
    reranked_results: list[Any],
) -> dict[str, Any]:
    """
    Measure relevant-chunk movement from vector ranking
    to reranked ranking.
    """
    candidate_results = _get_reranker_candidate_results(
        vector_store=vector_store,
        question=question,
    )

    candidate_chunk_ids = _retrieved_chunk_ids(
        candidate_results
    )

    reranked_chunk_ids = _retrieved_chunk_ids(
        reranked_results
    )

    before_ranks = _relevant_ranks(
        candidate_chunk_ids,
        relevant_chunks,
    )

    after_ranks = _relevant_ranks(
        reranked_chunk_ids,
        relevant_chunks,
    )

    movement: dict[str, Any] = {}

    for chunk_id in sorted(relevant_chunks):
        before_rank = before_ranks.get(chunk_id)
        after_rank = after_ranks.get(chunk_id)

        movement[chunk_id] = {
            "original_vector_rank": before_rank,
            "reranked_rank": after_rank,
            "rank_movement": _rank_movement(
                before_rank,
                after_rank,
            ),
        }

    measured_movements = [
        item["rank_movement"]
        for item in movement.values()
        if item["rank_movement"] is not None
    ]

    return {
        "candidate_pool_size": len(candidate_results),
        "candidate_chunk_ids": candidate_chunk_ids,
        "relevant_chunk_rank_movement": movement,
        "mean_rank_movement": (
            round(mean(measured_movements), 4)
            if measured_movements
            else None
        ),
    }


def _evaluate_strategy(
    strategy: str,
    retriever: object,
    dataset,
    vector_store: VectorStore,
) -> dict[str, Any]:
    """Evaluate one retrieval strategy over the complete dataset."""
    examples: list[dict[str, Any]] = []

    for index, example in enumerate(
        dataset,
        start=1,
    ):
        relevant_documents = set(
            example.relevant_documents
        )

        relevant_chunks = set(
            example.relevant_chunks
        )

        started = time.perf_counter()

        retrieval_response = retriever.retrieve(
            example.question,
            top_k=TOP_K,
        )

        latency = time.perf_counter() - started

        results = retrieval_response.results

        retrieved_document_ids = _document_ids(results)

        retrieved_chunk_ids = _retrieved_chunk_ids(results)

        # --------------------------------------------------------------
        # Document-level metrics
        # --------------------------------------------------------------
        document_precision = _precision_at_k(
            retrieved_document_ids,
            relevant_documents,
            TOP_K,
        )

        document_recall = _recall_at_k(
            retrieved_document_ids,
            relevant_documents,
            TOP_K,
        )

        document_f1 = _f1(
            document_precision,
            document_recall,
        )

        document_hit = _hit_at_k(
            retrieved_document_ids,
            relevant_documents,
            TOP_K,
        )

        # --------------------------------------------------------------
        # Chunk-level metrics
        # --------------------------------------------------------------
        chunk_precision = _precision_at_k(
            retrieved_chunk_ids,
            relevant_chunks,
            TOP_K,
        )

        chunk_recall = _recall_at_k(
            retrieved_chunk_ids,
            relevant_chunks,
            TOP_K,
        )

        chunk_f1 = _f1(
            chunk_precision,
            chunk_recall,
        )

        chunk_hit = _hit_at_k(
            retrieved_chunk_ids,
            relevant_chunks,
            TOP_K,
        )

        mrr = _mrr(
            retrieved_chunk_ids,
            relevant_chunks,
            TOP_K,
        )

        first_relevant_rank = _first_relevant_rank(
            retrieved_chunk_ids,
            relevant_chunks,
            TOP_K,
        )

        # --------------------------------------------------------------
        # Rank movement for reranked retrieval
        # --------------------------------------------------------------
        rank_movement = None

        if strategy == "reranked":
            rank_movement = _calculate_rank_movement(
                vector_store=vector_store,
                question=example.question,
                relevant_chunks=relevant_chunks,
                reranked_results=results,
            )

        retrieval_metadata = dict(
            retrieval_response.metadata
        )

        examples.append(
            {
                "question_index": index,
                "question": example.question,
                "category": (
                    example.metadata.get("category")
                    if example.metadata
                    else None
                ),
                "difficulty": (
                    example.metadata.get("difficulty")
                    if example.metadata
                    else None
                ),
                "relevant_document_ids": sorted(
                    relevant_documents
                ),
                "relevant_chunk_ids": sorted(
                    relevant_chunks
                ),
                "retrieved_document_ids": retrieved_document_ids,
                "retrieved_chunk_ids": retrieved_chunk_ids,
                "unique_document_count": len(
                    retrieved_document_ids
                ),
                "retrieved_chunk_count": len(results),
                "document_metrics": {
                    "precision_at_k": round(
                        document_precision,
                        4,
                    ),
                    "recall_at_k": round(
                        document_recall,
                        4,
                    ),
                    "f1_at_k": round(
                        document_f1,
                        4,
                    ),
                    "hit_at_k": round(
                        document_hit,
                        4,
                    ),
                },
                "chunk_metrics": {
                    "precision_at_k": round(
                        chunk_precision,
                        4,
                    ),
                    "recall_at_k": round(
                        chunk_recall,
                        4,
                    ),
                    "f1_at_k": round(
                        chunk_f1,
                        4,
                    ),
                    "hit_at_k": round(
                        chunk_hit,
                        4,
                    ),
                },
                "mrr": round(
                    mrr,
                    4,
                ),
                "first_relevant_chunk_rank": (
                    first_relevant_rank
                ),
                "rank_movement": rank_movement,
                "result_scores": _result_scores(
                    results
                ),
                "latency_seconds": round(
                    latency,
                    6,
                ),
                "retrieval_metadata": retrieval_metadata,
            }
        )

        print(
            f"  [{index:02d}/{dataset.size}] "
            f"{strategy:<9} "
            f"DocP@{TOP_K}={document_precision:.3f} "
            f"DocR@{TOP_K}={document_recall:.3f} "
            f"ChunkP@{TOP_K}={chunk_precision:.3f} "
            f"ChunkR@{TOP_K}={chunk_recall:.3f} "
            f"MRR={mrr:.3f} "
            f"Latency={latency:.4f}s"
        )

    # --------------------------------------------------------------
    # Aggregate metrics
    # --------------------------------------------------------------
    aggregate = {
        "document_precision_at_k": round(
            mean(
                example["document_metrics"]["precision_at_k"]
                for example in examples
            ),
            4,
        ),
        "document_recall_at_k": round(
            mean(
                example["document_metrics"]["recall_at_k"]
                for example in examples
            ),
            4,
        ),
        "document_f1_at_k": round(
            mean(
                example["document_metrics"]["f1_at_k"]
                for example in examples
            ),
            4,
        ),
        "document_hit_at_k": round(
            mean(
                example["document_metrics"]["hit_at_k"]
                for example in examples
            ),
            4,
        ),
        "chunk_precision_at_k": round(
            mean(
                example["chunk_metrics"]["precision_at_k"]
                for example in examples
            ),
            4,
        ),
        "chunk_recall_at_k": round(
            mean(
                example["chunk_metrics"]["recall_at_k"]
                for example in examples
            ),
            4,
        ),
        "chunk_f1_at_k": round(
            mean(
                example["chunk_metrics"]["f1_at_k"]
                for example in examples
            ),
            4,
        ),
        "chunk_hit_at_k": round(
            mean(
                example["chunk_metrics"]["hit_at_k"]
                for example in examples
            ),
            4,
        ),
        "mrr": round(
            mean(
                example["mrr"]
                for example in examples
            ),
            4,
        ),
        "mean_latency_seconds": round(
            mean(
                example["latency_seconds"]
                for example in examples
            ),
            6,
        ),
    }

    if strategy == "reranked":
        rank_movements = [
            example["rank_movement"]["mean_rank_movement"]
            for example in examples
            if example["rank_movement"] is not None
            and example["rank_movement"]["mean_rank_movement"]
            is not None
        ]

        aggregate["mean_relevant_chunk_rank_movement"] = (
            round(mean(rank_movements), 4)
            if rank_movements
            else None
        )

    return {
        "strategy": strategy,
        "dataset_size": dataset.size,
        "top_k": TOP_K,
        "aggregate": aggregate,
        "examples": examples,
    }


def main() -> None:
    """Run the complete local retrieval experiment."""
    print("=" * 72)
    print(
        "RAG ENGINEERING SYSTEM — "
        "LOCAL RETRIEVAL EXPERIMENT"
    )
    print("=" * 72)

    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Evaluation dataset not found: {DATASET_PATH}"
        )

    dataset = load_evaluation_dataset(
        DATASET_PATH
    )

    vector_store = VectorStore()

    print(f"Dataset: {DATASET_PATH}")
    print(f"Questions: {dataset.size}")
    print(f"Top-k: {TOP_K}")
    print(
        f"Vector store chunks: "
        f"{vector_store.count}"
    )
    print(
        f"Reranker candidate pool: "
        f"{RERANKER_CANDIDATE_POOL}"
    )
    print(
        "Document metrics: unique retrieved documents"
    )
    print(
        "Chunk metrics: individual retrieved chunks"
    )
    print()

    if vector_store.count == 0:
        raise RuntimeError(
            "Vector store is empty. "
            "Run scripts/index_documents.py first."
        )

    retrievers = _build_retrievers(
        vector_store
    )

    results: dict[str, Any] = {}

    for strategy, retriever in retrievers.items():
        print("-" * 72)
        print(
            f"Running strategy: {strategy}"
        )
        print("-" * 72)

        results[strategy] = _evaluate_strategy(
            strategy=strategy,
            retriever=retriever,
            dataset=dataset,
            vector_store=vector_store,
        )

        print()
        print("Aggregate metrics:")

        for metric, value in (
            results[strategy]["aggregate"].items()
        ):
            print(
                f"  {metric}: {value}"
            )

        print()

    RUNS_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    output = {
        "experiment": {
            "type": "retrieval_only",
            "dataset": str(DATASET_PATH),
            "dataset_size": dataset.size,
            "top_k": TOP_K,
            "vector_store_chunks": vector_store.count,
            "document_metric_definition": (
                "Precision, recall, F1, and hit rate are "
                "calculated over unique document IDs "
                "preserving retrieval order."
            ),
            "chunk_metric_definition": (
                "Precision, recall, F1, hit rate, and MRR "
                "are calculated over individual chunk IDs."
            ),
            "reranker_candidate_multiplier": (
                RERANKER_CANDIDATE_MULTIPLIER
            ),
            "reranker_candidate_pool": (
                RERANKER_CANDIDATE_POOL
            ),
            "strategies": list(
                results.keys()
            ),
            "total_evaluations": (
                dataset.size * len(results)
            ),
            "llm_used": False,
            "groq_api_used": False,
        },
        "runs": results,
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
    print(
        "LOCAL RETRIEVAL EXPERIMENT COMPLETE"
    )
    print("=" * 72)
    print(
        f"Total strategies: {len(results)}"
    )
    print(
        f"Total retrieval evaluations: "
        f"{dataset.size * len(results)}"
    )
    print(
        f"Results saved to: {OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()