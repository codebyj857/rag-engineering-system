"""
Base orchestration for Retrieval-Augmented Generation pipelines.

This module provides the shared workflow used by all RAG pipeline
variants:

1. Session history retrieval
2. Query contextualization
3. Retrieval
4. Answer generation
5. Groundedness verification
6. Confidence estimation
7. Conversation persistence

Concrete pipelines only provide their retrieval strategy.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from rag_engine.generation.generator import (
    GenerationEvidence,
    GenerationResult,
    Generator,
)
from rag_engine.memory.query_contextualizer import (
    ContextualizedQuery,
    QueryContextualizer,
)
from rag_engine.memory.session_store import (
    ChatMessage,
    SessionStore,
)
from rag_engine.retrieval.base import (
    RetrievalResponse,
    RetrievalResult,
    RetrievalStrategy,
)
from rag_engine.verification.confidence import (
    ConfidenceEstimator,
    ConfidenceResult,
)
from rag_engine.verification.groundedness import (
    GroundednessResult,
    GroundednessVerifier,
)


logger = logging.getLogger(__name__)


class PipelineError(Exception):
    """Base exception for pipeline-related failures."""


class PipelineConfigurationError(PipelineError):
    """Raised when a pipeline dependency is incorrectly configured."""


class PipelineExecutionError(PipelineError):
    """Raised when pipeline execution fails."""


@dataclass(frozen=True, slots=True)
class PipelineRequest:
    """Input required to execute a RAG pipeline."""

    question: str
    session_id: str
    top_k: int = 5
    verify: bool = True


@dataclass(frozen=True, slots=True)
class PipelineEvidence:
    """Evidence exposed by the pipeline response."""

    chunk_id: str
    content: str
    source: str
    filename: str
    score: float
    rank: int
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_retrieval_result(
        cls,
        result: RetrievalResult,
    ) -> "PipelineEvidence":
        """Create pipeline evidence from a retrieval result."""

        return cls(
            chunk_id=result.chunk_id,
            content=result.content,
            source=result.source,
            filename=result.filename,
            score=result.score,
            rank=result.rank,
            metadata=dict(result.metadata),
        )


@dataclass(frozen=True, slots=True)
class PipelineVerification:
    """Public representation of groundedness verification."""

    grounded: bool
    score: float
    supported_terms: tuple[str, ...]
    unsupported_terms: tuple[str, ...]
    evidence_count: int
    threshold: float

    @classmethod
    def from_result(
        cls,
        result: GroundednessResult,
    ) -> "PipelineVerification":
        """Create pipeline verification from a groundedness result."""

        return cls(
            grounded=result.grounded,
            score=result.score,
            supported_terms=result.supported_terms,
            unsupported_terms=result.unsupported_terms,
            evidence_count=result.evidence_count,
            threshold=result.threshold,
        )


@dataclass(frozen=True, slots=True)
class PipelineResponse:
    """Complete result produced by a RAG pipeline."""

    session_id: str
    original_query: str
    retrieval_query: str
    strategy: str
    answer: str
    evidence: tuple[PipelineEvidence, ...]
    verification: PipelineVerification | None
    confidence: ConfidenceResult
    generation: GenerationResult
    metadata: dict[str, Any] = field(default_factory=dict)


class RAGPipeline:
    """
    Shared RAG pipeline orchestration.

    The retrieval strategy is injected by concrete pipeline
    implementations. All other RAG stages remain identical across
    naive, hybrid, and reranked pipelines.
    """

    def __init__(
        self,
        *,
        strategy_name: str,
        retriever: RetrievalStrategy,
        session_store: SessionStore,
        contextualizer: QueryContextualizer,
        generator: Generator,
        groundedness_verifier: GroundednessVerifier,
        confidence_estimator: ConfidenceEstimator,
    ) -> None:
        if not strategy_name.strip():
            raise PipelineConfigurationError(
                "strategy_name cannot be empty."
            )

        if retriever is None:
            raise PipelineConfigurationError(
                "retriever must be provided."
            )

        if session_store is None:
            raise PipelineConfigurationError(
                "session_store must be provided."
            )

        if contextualizer is None:
            raise PipelineConfigurationError(
                "contextualizer must be provided."
            )

        if generator is None:
            raise PipelineConfigurationError(
                "generator must be provided."
            )

        if groundedness_verifier is None:
            raise PipelineConfigurationError(
                "groundedness_verifier must be provided."
            )

        if confidence_estimator is None:
            raise PipelineConfigurationError(
                "confidence_estimator must be provided."
            )

        self.strategy_name = strategy_name.strip().lower()
        self._retriever = retriever
        self._session_store = session_store
        self._contextualizer = contextualizer
        self._generator = generator
        self._groundedness_verifier = groundedness_verifier
        self._confidence_estimator = confidence_estimator

    def run(self, request: PipelineRequest) -> PipelineResponse:
        """
        Execute the complete RAG workflow.

        This is the single shared orchestration path used by every
        retrieval strategy.
        """

        self._validate_request(request)

        try:
            history = self._session_store.get_history(
                request.session_id
            )

            contextualized = self._contextualize(
                request.question,
                history,
            )

            retrieval_response = self._retrieve(
                contextualized,
                request.top_k,
            )

            generation_result = self._generate(
                request.question,
                history,
                retrieval_response,
            )

            verification_result = self._verify(
                generation_result.answer,
                retrieval_response,
            )

            confidence_result = self._calculate_confidence(
                retrieval_response,
                verification_result,
            )

            self._persist_conversation(
                session_id=request.session_id,
                question=request.question,
                answer=generation_result.answer,
            )

            return self._build_response(
                request=request,
                contextualized=contextualized,
                retrieval_response=retrieval_response,
                generation_result=generation_result,
                verification_result=(
                    verification_result
                    if request.verify
                    else None
                ),
                confidence_result=confidence_result,
            )

        except PipelineError:
            raise
        except Exception as exc:
            logger.exception(
                "Pipeline execution failed: strategy=%s session_id=%s",
                self.strategy_name,
                request.session_id,
            )
            raise PipelineExecutionError(
                f"RAG pipeline execution failed: {exc}"
            ) from exc

    @staticmethod
    def _validate_request(request: PipelineRequest) -> None:
        """Validate pipeline input."""

        if not isinstance(request, PipelineRequest):
            raise PipelineConfigurationError(
                "request must be a PipelineRequest instance."
            )

        if not request.question.strip():
            raise PipelineExecutionError(
                "Question cannot be empty."
            )

        if not request.session_id.strip():
            raise PipelineExecutionError(
                "session_id cannot be empty."
            )

        if request.top_k < 1:
            raise PipelineExecutionError(
                "top_k must be at least 1."
            )

    def _contextualize(
        self,
        query: str,
        history: list[ChatMessage],
    ) -> ContextualizedQuery:
        """Contextualize the current question using conversation history."""

        try:
            return self._contextualizer.contextualize(
                query=query,
                history=history,
            )
        except Exception as exc:
            logger.exception("Query contextualization failed.")
            raise PipelineExecutionError(
                "Query contextualization failed."
            ) from exc

    def _retrieve(
        self,
        contextualized: ContextualizedQuery,
        top_k: int,
    ) -> RetrievalResponse:
        """Retrieve evidence using the configured strategy."""

        try:
            return self._retriever.retrieve(
                contextualized.standalone_query,
                top_k=top_k,
            )
        except Exception as exc:
            logger.exception(
                "Retrieval failed: strategy=%s",
                self.strategy_name,
            )
            raise PipelineExecutionError(
                f"Retrieval failed for strategy '{self.strategy_name}'."
            ) from exc

    def _generate(
        self,
        question: str,
        history: list[ChatMessage],
        retrieval_response: RetrievalResponse,
    ) -> GenerationResult:
        """Generate an answer using retrieved evidence and history."""

        evidence = tuple(
            GenerationEvidence(
                content=result.content,
                source=result.source,
                filename=result.filename,
                score=result.score,
                rank=result.rank,
            )
            for result in retrieval_response.results
        )

        conversation_context = self._format_history(history)

        try:
            return self._generator.generate(
                question=question,
                evidence=evidence,
                conversation_context=conversation_context,
            )
        except Exception as exc:
            logger.exception("Answer generation failed.")
            raise PipelineExecutionError(
                "Answer generation failed."
            ) from exc

    def _verify(
        self,
        answer: str,
        retrieval_response: RetrievalResponse,
    ) -> GroundednessResult:
        """Verify the generated answer against retrieved evidence."""

        evidence = tuple(
            result.content
            for result in retrieval_response.results
            if result.content.strip()
        )

        try:
            return self._groundedness_verifier.verify(
                answer=answer,
                evidence=evidence,
            )
        except Exception as exc:
            logger.exception("Groundedness verification failed.")
            raise PipelineExecutionError(
                "Groundedness verification failed."
            ) from exc

    def _calculate_confidence(
        self,
        retrieval_response: RetrievalResponse,
        verification_result: GroundednessResult,
    ) -> ConfidenceResult:
        """Calculate final confidence from retrieval and verification."""

        retrieval_scores = tuple(
            result.score
            for result in retrieval_response.results
        )

        try:
            return self._confidence_estimator.estimate(
                retrieval_scores=retrieval_scores,
                groundedness_score=verification_result.score,
                evidence_count=len(retrieval_response.results),
            )
        except Exception as exc:
            logger.exception("Confidence estimation failed.")
            raise PipelineExecutionError(
                "Confidence estimation failed."
            ) from exc

    def _persist_conversation(
        self,
        *,
        session_id: str,
        question: str,
        answer: str,
    ) -> None:
        """Persist the completed user/assistant exchange."""

        try:
            self._session_store.add_user_message(
                session_id,
                question,
            )
            self._session_store.add_assistant_message(
                session_id,
                answer,
            )
        except Exception as exc:
            logger.exception(
                "Conversation persistence failed: session_id=%s",
                session_id,
            )
            raise PipelineExecutionError(
                "Failed to persist conversation."
            ) from exc

    def _build_response(
        self,
        *,
        request: PipelineRequest,
        contextualized: ContextualizedQuery,
        retrieval_response: RetrievalResponse,
        generation_result: GenerationResult,
        verification_result: GroundednessResult | None,
        confidence_result: ConfidenceResult,
    ) -> PipelineResponse:
        """Build the final pipeline response."""

        evidence = tuple(
            PipelineEvidence.from_retrieval_result(result)
            for result in retrieval_response.results
        )

        verification = (
            PipelineVerification.from_result(
                verification_result
            )
            if verification_result is not None
            else None
        )

        metadata = {
            "retrieval_total_results": retrieval_response.total_results,
            "context_used_history": contextualized.used_history,
            "history_messages_used": (
                contextualized.history_messages_used
            ),
            "verification_enabled": request.verify,
        }

        return PipelineResponse(
            session_id=request.session_id,
            original_query=contextualized.original_query,
            retrieval_query=contextualized.standalone_query,
            strategy=self.strategy_name,
            answer=generation_result.answer,
            evidence=evidence,
            verification=verification,
            confidence=confidence_result,
            generation=generation_result,
            metadata=metadata,
        )

    @staticmethod
    def _format_history(
        history: list[ChatMessage],
    ) -> str:
        """Format recent conversation history for the generator."""

        if not history:
            return ""

        return "\n".join(
            f"{message.role.capitalize()}: {message.content}"
            for message in history[-10:]
        )