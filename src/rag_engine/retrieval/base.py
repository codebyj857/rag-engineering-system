"""
Core retrieval abstractions for the RAG Engineering System.

This module defines the common interface and data structures shared by all
retrieval strategies.

Concrete implementations such as naive, hybrid, and reranked retrieval
must follow the RetrievalStrategy contract defined here.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Sequence

from rag_engine.indexing.vector_store import VectorSearchResult


class RetrievalError(Exception):
    """Base exception for retrieval-related failures."""


class RetrievalConfigurationError(RetrievalError):
    """Raised when a retrieval strategy is incorrectly configured."""


class RetrievalExecutionError(RetrievalError):
    """Raised when a retrieval operation fails."""


@dataclass(frozen=True, slots=True)
class RetrievalResult:
    """
    Represents one piece of evidence returned by a retrieval strategy.

    Attributes:
        chunk_id: Identifier of the retrieved chunk.
        content: Text content of the chunk.
        source: Original source path or identifier.
        filename: Original filename.
        file_type: Source document type.
        score: Strategy-specific relevance score.
        rank: One-based position in the final ranked results.
        metadata: Additional chunk metadata.
    """

    chunk_id: str
    content: str
    source: str
    filename: str
    file_type: str
    score: float
    rank: int
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_vector_result(
        cls,
        result: VectorSearchResult,
        rank: int,
    ) -> "RetrievalResult":
        """
        Convert a vector-store result into a retrieval result.

        Args:
            result: Result returned by the vector store.
            rank: One-based result rank.

        Returns:
            Normalized RetrievalResult instance.
        """
        return cls(
            chunk_id=result.chunk_id,
            content=result.content,
            source=result.source,
            filename=result.filename,
            file_type=result.file_type,
            score=result.score,
            rank=rank,
            metadata=dict(result.metadata),
        )


@dataclass(frozen=True, slots=True)
class RetrievalResponse:
    """
    Complete response returned by a retrieval strategy.

    Attributes:
        query: Original user query.
        strategy: Name of the retrieval strategy.
        results: Ranked retrieval results.
        total_results: Number of returned results.
        metadata: Strategy-specific execution metadata.
    """

    query: str
    strategy: str
    results: list[RetrievalResult]
    total_results: int
    metadata: dict[str, Any] = field(default_factory=dict)


class RetrievalStrategy(ABC):
    """
    Abstract interface for all retrieval strategies.

    Concrete implementations must provide their own retrieval algorithm
    while returning the same application-level RetrievalResponse structure.
    """

    name: str = "base"

    def __init__(
        self,
        top_k: int = 5,
    ) -> None:
        if top_k < 1:
            raise RetrievalConfigurationError(
                "top_k must be at least 1."
            )

        self.top_k = top_k

    @abstractmethod
    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
    ) -> RetrievalResponse:
        """
        Retrieve relevant chunks for a query.

        Args:
            query: User's search query.
            top_k: Optional override for the number of results.

        Returns:
            Ranked retrieval response.

        Raises:
            RetrievalExecutionError: If retrieval cannot be completed.
        """
        raise NotImplementedError

    def _validate_query(self, query: str) -> str:
        """
        Normalize and validate a retrieval query.

        Args:
            query: Raw user query.

        Returns:
            Stripped query.

        Raises:
            RetrievalExecutionError: If the query is empty.
        """
        normalized_query = query.strip()

        if not normalized_query:
            raise RetrievalExecutionError(
                "Retrieval query cannot be empty."
            )

        return normalized_query

    def _resolve_top_k(self, top_k: int | None) -> int:
        """
        Resolve the effective top-k value.

        Args:
            top_k: Optional per-request override.

        Returns:
            Valid top-k value.

        Raises:
            RetrievalConfigurationError: If top_k is invalid.
        """
        resolved_top_k = self.top_k if top_k is None else top_k

        if resolved_top_k < 1:
            raise RetrievalConfigurationError(
                "top_k must be at least 1."
            )

        return resolved_top_k

    @staticmethod
    def _convert_results(
        results: Sequence[VectorSearchResult],
    ) -> list[RetrievalResult]:
        """
        Convert vector-store results into normalized retrieval results.

        Args:
            results: Vector search results.

        Returns:
            Ranked RetrievalResult objects.
        """
        return [
            RetrievalResult.from_vector_result(
                result=result,
                rank=index + 1,
            )
            for index, result in enumerate(results)
        ]

    @staticmethod
    def _build_response(
        query: str,
        strategy: str,
        results: list[RetrievalResult],
        metadata: dict[str, Any] | None = None,
    ) -> RetrievalResponse:
        """
        Build a standardized retrieval response.
        """
        return RetrievalResponse(
            query=query,
            strategy=strategy,
            results=results,
            total_results=len(results),
            metadata=metadata or {},
        )