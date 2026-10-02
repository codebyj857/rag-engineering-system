from __future__ import annotations

from functools import lru_cache

from rag_engine.config import get_settings
from rag_engine.generation.generator import Generator
from rag_engine.memory.query_contextualizer import QueryContextualizer
from rag_engine.memory.session_store import get_session_store
from rag_engine.pipelines.hybrid import HybridRAGPipeline
from rag_engine.pipelines.naive import NaiveRAGPipeline
from rag_engine.pipelines.reranked import RerankedRAGPipeline
from rag_engine.pipelines.base import RAGPipeline
from rag_engine.retrieval.hybrid import HybridRetriever
from rag_engine.retrieval.naive import NaiveRetriever
from rag_engine.retrieval.reranker import RerankedRetriever
from rag_engine.verification.confidence import ConfidenceEstimator
from rag_engine.verification.groundedness import GroundednessVerifier
from rag_engine.indexing.vector_store import VectorStore


@lru_cache(maxsize=1)
def get_vector_store() -> VectorStore:
    return VectorStore()


@lru_cache(maxsize=1)
def get_common_dependencies() -> tuple[
    object,
    QueryContextualizer,
    Generator,
    GroundednessVerifier,
    ConfidenceEstimator,
]:
    return (
        get_session_store(),
        QueryContextualizer(),
        Generator(),
        GroundednessVerifier(),
        ConfidenceEstimator(),
    )


@lru_cache(maxsize=1)
def get_pipeline(strategy: str) -> RAGPipeline:
    settings = get_settings()
    normalized = strategy.strip().lower()

    session_store, contextualizer, generator, verifier, confidence = (
        get_common_dependencies()
    )
    vector_store = get_vector_store()

    if normalized == "naive":
        retriever = NaiveRetriever(
            vector_store=vector_store,
            top_k=settings.default_top_k,
        )
        return NaiveRAGPipeline(
            retriever=retriever,
            session_store=session_store,
            contextualizer=contextualizer,
            generator=generator,
            groundedness_verifier=verifier,
            confidence_estimator=confidence,
        )

    if normalized == "hybrid":
        retriever = HybridRetriever(
            vector_store=vector_store,
            top_k=settings.default_top_k,
        )
        return HybridRAGPipeline(
            retriever=retriever,
            session_store=session_store,
            contextualizer=contextualizer,
            generator=generator,
            groundedness_verifier=verifier,
            confidence_estimator=confidence,
        )

    if normalized == "reranked":
        retriever = RerankedRetriever(
            vector_store=vector_store,
            top_k=settings.default_top_k,
        )
        return RerankedRAGPipeline(
            retriever=retriever,
            session_store=session_store,
            contextualizer=contextualizer,
            generator=generator,
            groundedness_verifier=verifier,
            confidence_estimator=confidence,
        )

    raise ValueError(f"Unsupported retrieval strategy: {strategy}")


def get_default_pipeline() -> RAGPipeline:
    return get_pipeline(get_settings().default_retrieval_strategy)