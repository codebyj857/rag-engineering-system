"""
Naive semantic retrieval for the RAG Engineering System.

Naive retrieval uses only vector similarity from the persistent vector
store. It serves as the baseline against which hybrid and reranked
strategies can be evaluated.
"""

from rag_engine.indexing.vector_store import VectorStore
from rag_engine.retrieval.base import (
    RetrievalExecutionError,
    RetrievalResponse,
    RetrievalStrategy,
)


class NaiveRetriever(RetrievalStrategy):
    """Retrieve documents using semantic vector similarity only."""

    name = "naive"

    def __init__(
        self,
        vector_store: VectorStore,
        top_k: int = 5,
    ) -> None:
        super().__init__(top_k=top_k)
        self.vector_store = vector_store

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
    ) -> RetrievalResponse:
        """Retrieve the most semantically similar chunks."""
        normalized_query = self._validate_query(query)
        resolved_top_k = self._resolve_top_k(top_k)

        try:
            vector_results = self.vector_store.search(
                query=normalized_query,
                top_k=resolved_top_k,
            )

            results = self._convert_results(vector_results)

            return self._build_response(
                query=normalized_query,
                strategy=self.name,
                results=results,
                metadata={
                    "retrieval_method": "vector_similarity",
                    "top_k": resolved_top_k,
                },
            )

        except Exception as exc:
            if isinstance(exc, RetrievalExecutionError):
                raise

            raise RetrievalExecutionError(
                "Naive semantic retrieval failed."
            ) from exc