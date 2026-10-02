"""
Hybrid retrieval for the RAG Engineering System.

Hybrid retrieval combines semantic similarity with lexical keyword
matching. The two signals are normalized and fused into a single
relevance score.
"""

import re
from collections import Counter

from rag_engine.indexing.vector_store import VectorStore
from rag_engine.retrieval.base import (
    RetrievalExecutionError,
    RetrievalResponse,
    RetrievalResult,
    RetrievalStrategy,
)


class HybridRetriever(RetrievalStrategy):
    """
    Retrieve documents using semantic and lexical relevance.

    The vector store provides a broad semantic candidate pool. Lexical
    overlap is then calculated against those candidates and combined with
    semantic relevance.
    """

    name = "hybrid"

    def __init__(
        self,
        vector_store: VectorStore,
        top_k: int = 5,
        semantic_weight: float = 0.7,
        lexical_weight: float = 0.3,
    ) -> None:
        super().__init__(top_k=top_k)

        if semantic_weight < 0 or lexical_weight < 0:
            raise ValueError(
                "Hybrid retrieval weights cannot be negative."
            )

        total_weight = semantic_weight + lexical_weight

        if total_weight == 0:
            raise ValueError(
                "At least one hybrid retrieval weight must be greater than zero."
            )

        self.vector_store = vector_store
        self.semantic_weight = semantic_weight / total_weight
        self.lexical_weight = lexical_weight / total_weight

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
    ) -> RetrievalResponse:
        """Retrieve and rank chunks using hybrid relevance."""
        normalized_query = self._validate_query(query)
        resolved_top_k = self._resolve_top_k(top_k)

        candidate_k = max(resolved_top_k * 3, 10)

        try:
            vector_results = self.vector_store.search(
                query=normalized_query,
                top_k=candidate_k,
            )

            if not vector_results:
                return self._build_response(
                    query=normalized_query,
                    strategy=self.name,
                    results=[],
                    metadata={
                        "retrieval_method": "semantic_lexical_fusion",
                        "top_k": resolved_top_k,
                    },
                )

            query_tokens = self._tokenize(normalized_query)

            scored_results: list[tuple[float, object, float, float]] = []

            for result in vector_results:
                lexical_score = self._lexical_score(
                    query_tokens=query_tokens,
                    content=result.content,
                )

                semantic_score = max(0.0, min(1.0, result.score))

                combined_score = (
                    self.semantic_weight * semantic_score
                    + self.lexical_weight * lexical_score
                )

                scored_results.append(
                    (
                        combined_score,
                        result,
                        semantic_score,
                        lexical_score,
                    )
                )

            scored_results.sort(
                key=lambda item: item[0],
                reverse=True,
            )

            results: list[RetrievalResult] = []

            for rank, (
                combined_score,
                result,
                semantic_score,
                lexical_score,
            ) in enumerate(
                scored_results[:resolved_top_k],
                start=1,
            ):
                metadata = {
                    **result.metadata,
                    "semantic_score": semantic_score,
                    "lexical_score": lexical_score,
                    "hybrid_score": combined_score,
                }

                results.append(
                    RetrievalResult(
                        chunk_id=result.chunk_id,
                        content=result.content,
                        source=result.source,
                        filename=result.filename,
                        file_type=result.file_type,
                        score=combined_score,
                        rank=rank,
                        metadata=metadata,
                    )
                )

            return self._build_response(
                query=normalized_query,
                strategy=self.name,
                results=results,
                metadata={
                    "retrieval_method": "semantic_lexical_fusion",
                    "semantic_weight": self.semantic_weight,
                    "lexical_weight": self.lexical_weight,
                    "candidate_pool": len(vector_results),
                    "top_k": resolved_top_k,
                },
            )

        except Exception as exc:
            if isinstance(exc, RetrievalExecutionError):
                raise

            raise RetrievalExecutionError(
                "Hybrid retrieval failed."
            ) from exc

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """Convert text into normalized alphanumeric tokens."""
        return re.findall(r"\b[a-zA-Z0-9]+\b", text.lower())

    @classmethod
    def _lexical_score(
        cls,
        query_tokens: list[str],
        content: str,
    ) -> float:
        """
        Calculate normalized lexical overlap.

        The score combines query-token coverage with a small frequency
        component and remains bounded between zero and one.
        """
        if not query_tokens:
            return 0.0

        document_tokens = cls._tokenize(content)

        if not document_tokens:
            return 0.0

        query_counts = Counter(query_tokens)
        document_counts = Counter(document_tokens)

        matched = sum(
            min(count, document_counts.get(token, 0))
            for token, count in query_counts.items()
        )

        total_query_terms = sum(query_counts.values())

        return min(1.0, matched / total_query_terms)