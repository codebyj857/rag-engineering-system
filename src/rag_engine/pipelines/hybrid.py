"""
Hybrid Retrieval-Augmented Generation pipeline.

This pipeline combines semantic vector retrieval with lexical
keyword-overlap retrieval through the HybridRetriever.
"""

from __future__ import annotations

from rag_engine.generation.generator import Generator
from rag_engine.memory.query_contextualizer import QueryContextualizer
from rag_engine.memory.session_store import SessionStore
from rag_engine.retrieval.base import RetrievalStrategy
from rag_engine.retrieval.hybrid import HybridRetriever
from rag_engine.verification.confidence import ConfidenceEstimator
from rag_engine.verification.groundedness import GroundednessVerifier

from .base import (
    PipelineConfigurationError,
    PipelineRequest,
    PipelineResponse,
    RAGPipeline,
)


class HybridRAGPipeline(RAGPipeline):
    """
    RAG pipeline using hybrid retrieval.

    Hybrid retrieval combines semantic similarity with lexical
    token-overlap scoring before passing evidence to the shared
    generation and verification workflow.
    """

    strategy_name = "hybrid"

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
        Initialize the hybrid RAG pipeline.

        Args:
            retriever: Optional preconfigured hybrid retriever.
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
            else HybridRetriever()
        )

        if selected_retriever.name != self.strategy_name:
            raise PipelineConfigurationError(
                "HybridRAGPipeline requires a retriever using "
                "the 'hybrid' strategy."
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
        """Execute the hybrid RAG pipeline."""

        return super().run(request)