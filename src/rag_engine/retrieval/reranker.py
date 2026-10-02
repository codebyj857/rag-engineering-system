"""
Cross-encoder reranking for the RAG Engineering System.

This module performs a two-stage retrieval process:

    1. Retrieve a broad candidate pool using vector similarity.
    2. Rerank those candidates using a cross-encoder.

The cross-encoder evaluates the query and candidate text together,
providing a more precise relevance signal than independent embeddings.
"""

from sentence_transformers import CrossEncoder

from rag_engine.indexing.vector_store import VectorStore
from rag_engine.retrieval.base import (
    RetrievalExecutionError,
    RetrievalResponse,
    RetrievalResult,
    RetrievalStrategy,
)


class RerankedRetriever(RetrievalStrategy):
    """Retrieve semantic candidates and rerank them with a cross-encoder."""

    name = "reranked"

    DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    def __init__(
        self,
        vector_store: VectorStore,
        top_k: int = 5,
        candidate_multiplier: int = 4,
        model_name: str | None = None,
    ) -> None:
        super().__init__(top_k=top_k)

        if candidate_multiplier < 1:
            raise ValueError(
                "candidate_multiplier must be at least 1."
            )

        self.vector_store = vector_store
        self.candidate_multiplier = candidate_multiplier
        self.model_name = model_name or self.DEFAULT_MODEL
        self._model: CrossEncoder | None = None

    @property
    def model(self) -> CrossEncoder:
        """Lazily load the cross-encoder model."""
        if self._model is None:
            try:
                self._model = CrossEncoder(self.model_name)
            except Exception as exc:
                raise RetrievalExecutionError(
                    f"Failed to load reranker model '{self.model_name}'."
                ) from exc

        return self._model

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
    ) -> RetrievalResponse:
        """Retrieve candidates and rerank them using a cross-encoder."""
        normalized_query = self._validate_query(query)
        resolved_top_k = self._resolve_top_k(top_k)

        candidate_k = max(
            resolved_top_k * self.candidate_multiplier,
            10,
        )

        try:
            candidates = self.vector_store.search(
                query=normalized_query,
                top_k=candidate_k,
            )

            if not candidates:
                return self._build_response(
                    query=normalized_query,
                    strategy=self.name,
                    results=[],
                    metadata={
                        "retrieval_method": "vector_then_cross_encoder",
                        "top_k": resolved_top_k,
                        "candidate_pool": 0,
                        "reranker_model": self.model_name,
                    },
                )

            pairs = [
                (normalized_query, candidate.content)
                for candidate in candidates
            ]

            scores = self.model.predict(
                pairs,
                show_progress_bar=False,
            )

            ranked = sorted(
                zip(candidates, scores),
                key=lambda item: float(item[1]),
                reverse=True,
            )

            results: list[RetrievalResult] = []

            for rank, (candidate, score) in enumerate(
                ranked[:resolved_top_k],
                start=1,
            ):
                reranker_score = float(score)

                metadata = {
                    **candidate.metadata,
                    "vector_score": candidate.score,
                    "reranker_score": reranker_score,
                }

                results.append(
                    RetrievalResult(
                        chunk_id=candidate.chunk_id,
                        content=candidate.content,
                        source=candidate.source,
                        filename=candidate.filename,
                        file_type=candidate.file_type,
                        score=reranker_score,
                        rank=rank,
                        metadata=metadata,
                    )
                )

            return self._build_response(
                query=normalized_query,
                strategy=self.name,
                results=results,
                metadata={
                    "retrieval_method": "vector_then_cross_encoder",
                    "top_k": resolved_top_k,
                    "candidate_pool": len(candidates),
                    "reranker_model": self.model_name,
                },
            )

        except Exception as exc:
            if isinstance(exc, RetrievalExecutionError):
                raise

            raise RetrievalExecutionError(
                "Reranked retrieval failed."
            ) from exc