"""
Reranked Retrieval-Augmented Generation pipeline.

This pipeline retrieves a larger candidate set using vector similarity
and then applies cross-encoder reranking before generation.
"""

from __future__ import annotations

from rag_engine.generation.generator import Generator
from rag_engine.memory.query_contextualizer import QueryContextualizer
from rag_engine.memory.session_store import SessionStore
from rag_engine.retrieval.base import RetrievalStrategy
from rag_engine.retrieval.reranker import RerankedRetriever
from rag_engine.verification.confidence import ConfidenceEstimator
from rag_engine.verification.groundedness import GroundednessVerifier

from .base import (
    PipelineConfigurationError,
    PipelineRequest,
    PipelineResponse,
    RAGPipeline,
)


class RerankedRAGPipeline(RAGPipeline):
    """
    RAG pipeline using cross-encoder reranking.

    The retriever first obtains a broad candidate set and then
    reranks those candidates using a cross-encoder model.
    """

    strategy_name = "reranked"

    def __init__(
        self,
        *,
        retriever: RetrievalStrategy | None = None,
        session_store: SessionStore,
        contextualizer: QueryContextualizer,
        generator: Generator,
        groundedness_verifier: GroundednessVerifier,
        confidence_estimator: ConfidenceEstimator,
    ) -> None:
        """
        Initialize the reranked RAG pipeline.

        Args:
            retriever: Optional preconfigured reranked retriever.
                Dependency injection keeps the pipeline testable.
            session_store: Conversation/session storage.
            contextualizer: Query contextualization service.
            generator: Answer generation service.
            groundedness_verifier: Groundedness verification service.
            confidence_estimator: Confidence estimation service.
        """

        selected_retriever = (
            retriever
            if retriever is not None
            else RerankedRetriever()
        )

        if selected_retriever.name != self.strategy_name:
            raise PipelineConfigurationError(
                "RerankedRAGPipeline requires a retriever using "
                "the 'reranked' strategy."
            )

        super().__init__(
            strategy_name=self.strategy_name,
            retriever=selected_retriever,
            session_store=session_store,
            contextualizer=contextualizer,
            generator=generator,
            groundedness_verifier=groundedness_verifier,
            confidence_estimator=confidence_estimator,
        )

    def run(self, request: PipelineRequest) -> PipelineResponse:
        """Execute the reranked RAG pipeline."""

        return super().run(request)