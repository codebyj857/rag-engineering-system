"""
Naive Retrieval-Augmented Generation pipeline.

This pipeline uses vector-similarity retrieval and delegates the
remaining RAG orchestration to the shared RAGPipeline implementation.
"""

from __future__ import annotations

from rag_engine.generation.generator import Generator
from rag_engine.memory.query_contextualizer import QueryContextualizer
from rag_engine.memory.session_store import SessionStore
from rag_engine.retrieval.base import RetrievalStrategy
from rag_engine.retrieval.naive import NaiveRetriever
from rag_engine.verification.confidence import ConfidenceEstimator
from rag_engine.verification.groundedness import GroundednessVerifier

from .base import (
    PipelineConfigurationError,
    PipelineRequest,
    PipelineResponse,
    RAGPipeline,
)


class NaiveRAGPipeline(RAGPipeline):
    """
    RAG pipeline using naive vector-similarity retrieval.

    All orchestration is inherited from RAGPipeline. This class only
    configures and validates the naive retrieval strategy.
    """

    strategy_name = "naive"

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
        Initialize the naive RAG pipeline.

        Args:
            retriever: Optional preconfigured naive retriever.
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
            else NaiveRetriever()
        )

        if selected_retriever.name != self.strategy_name:
            raise PipelineConfigurationError(
                "NaiveRAGPipeline requires a retriever using "
                "the 'naive' strategy."
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
        """Execute the naive RAG pipeline."""

        return super().run(request)