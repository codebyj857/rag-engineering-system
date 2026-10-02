from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

from rag_engine.indexing.embeddings import EmbeddingService
from rag_engine.indexing.vector_store import VectorStore
from rag_engine.ingestion.chunker import DocumentChunker
from rag_engine.ingestion.loader import load_document
from rag_engine.retrieval.base import RetrievalExecutionError, RetrievalResponse
from rag_engine.retrieval.hybrid import HybridRetriever
from rag_engine.retrieval.naive import NaiveRetriever
from rag_engine.retrieval.reranker import RerankedRetriever


class FakeEmbeddingModel:
    """Deterministic SentenceTransformer-compatible embedding model."""

    def get_embedding_dimension(self) -> int:
        return 5

    def encode(
        self,
        texts,
        batch_size: int = 32,
        convert_to_numpy: bool = True,
        normalize_embeddings: bool = True,
        show_progress_bar: bool = False,
    ) -> np.ndarray:
        def make_vector(text: str) -> list[float]:
            normalized = text.lower()

            return [
                float("python" in normalized),
                float("database" in normalized),
                float("retrieval" in normalized),
                float("machine learning" in normalized),
                float(len(normalized)) / 100.0,
            ]

        if isinstance(texts, str):
            return np.asarray(
                make_vector(texts),
                dtype=float,
            )

        return np.asarray(
            [make_vector(text) for text in texts],
            dtype=float,
        )


class FakeCrossEncoder:
    """Deterministic CrossEncoder-compatible reranker."""

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name

    def predict(
        self,
        pairs,
        show_progress_bar: bool = False,
    ) -> np.ndarray:
        scores = []

        for query, content in pairs:
            query_tokens = set(query.lower().split())
            content_tokens = set(content.lower().split())

            overlap = len(
                query_tokens & content_tokens
            )

            scores.append(float(overlap))

        return np.asarray(scores, dtype=float)


def _build_vector_store(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    collection_name: str,
) -> VectorStore:
    """Build a local vector store using the deterministic fake model."""

    fake_model = FakeEmbeddingModel()

    monkeypatch.setattr(
        "rag_engine.indexing.embeddings.SentenceTransformer",
        lambda *args, **kwargs: fake_model,
    )

    embedding_service = EmbeddingService(
        model_name="test-retrieval-model",
    )

    return VectorStore(
        persist_directory=tmp_path / "vector_store",
        collection_name=collection_name,
        embedding_service=embedding_service,
    )


def _index_test_documents(
    tmp_path: Path,
    vector_store: VectorStore,
) -> None:
    """Create and index a small deterministic retrieval corpus."""

    documents = {
        "python.txt": (
            "Python is a programming language widely used "
            "for machine learning and artificial intelligence."
        ),
        "database.txt": (
            "Vector databases store embeddings and support "
            "semantic retrieval over documents."
        ),
        "retrieval.txt": (
            "Retrieval systems search a knowledge base and "
            "return relevant documents for a user query."
        ),
    }

    chunker = DocumentChunker(
        chunk_size=200,
        chunk_overlap=20,
    )

    all_chunks = []

    for filename, content in documents.items():
        path = tmp_path / filename

        path.write_text(
            content,
            encoding="utf-8",
        )

        document = load_document(path)
        chunks = chunker.chunk(document)

        assert chunks

        all_chunks.extend(chunks)

    stored_count = vector_store.add_chunks(all_chunks)

    assert stored_count == len(all_chunks)
    assert vector_store.count == len(all_chunks)


def _assert_valid_retrieval_response(
    response: RetrievalResponse,
    expected_strategy: str,
    expected_query: str,
    max_results: int,
) -> None:
    """Validate the common retrieval response contract."""

    assert isinstance(response, RetrievalResponse)
    assert response.query == expected_query
    assert response.strategy == expected_strategy
    assert response.total_results == len(response.results)
    assert response.total_results <= max_results

    ranks = [
        result.rank
        for result in response.results
    ]

    assert ranks == list(
        range(1, len(response.results) + 1)
    )

    for result in response.results:
        assert result.chunk_id
        assert result.content
        assert result.source
        assert result.filename
        assert result.file_type
        assert isinstance(result.score, float)
        assert result.rank >= 1
        assert isinstance(result.metadata, dict)


def test_naive_retrieval(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify semantic vector retrieval."""

    vector_store = _build_vector_store(
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
        collection_name="naive_retrieval_test",
    )

    _index_test_documents(
        tmp_path=tmp_path,
        vector_store=vector_store,
    )

    retriever = NaiveRetriever(
        vector_store=vector_store,
        top_k=2,
    )

    response = retriever.retrieve(
        query="Python machine learning",
    )

    _assert_valid_retrieval_response(
        response=response,
        expected_strategy="naive",
        expected_query="Python machine learning",
        max_results=2,
    )

    assert response.metadata["retrieval_method"] == "vector_similarity"
    assert response.metadata["top_k"] == 2

    assert response.results

    assert response.results[0].filename == "python.txt"

    # Verify per-request top_k override.
    override_response = retriever.retrieve(
        query="Python machine learning",
        top_k=1,
    )

    assert override_response.total_results <= 1
    assert override_response.metadata["top_k"] == 1


def test_hybrid_retrieval(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify semantic + lexical hybrid retrieval."""

    vector_store = _build_vector_store(
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
        collection_name="hybrid_retrieval_test",
    )

    _index_test_documents(
        tmp_path=tmp_path,
        vector_store=vector_store,
    )

    retriever = HybridRetriever(
        vector_store=vector_store,
        top_k=2,
        semantic_weight=0.7,
        lexical_weight=0.3,
    )

    response = retriever.retrieve(
        query="vector database retrieval",
    )

    _assert_valid_retrieval_response(
        response=response,
        expected_strategy="hybrid",
        expected_query="vector database retrieval",
        max_results=2,
    )

    assert response.metadata["retrieval_method"] == (
        "semantic_lexical_fusion"
    )

    assert response.metadata["top_k"] == 2

    assert response.metadata["semantic_weight"] == pytest.approx(
        0.7
    )

    assert response.metadata["lexical_weight"] == pytest.approx(
        0.3
    )

    assert response.metadata["candidate_pool"] >= len(
        response.results
    )

    assert response.results

    for result in response.results:
        assert "semantic_score" in result.metadata
        assert "lexical_score" in result.metadata
        assert "hybrid_score" in result.metadata

        assert 0.0 <= result.metadata["semantic_score"] <= 1.0
        assert 0.0 <= result.metadata["lexical_score"] <= 1.0
        assert 0.0 <= result.metadata["hybrid_score"] <= 1.0


def test_reranked_retrieval(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify vector retrieval followed by cross-encoder reranking."""

    vector_store = _build_vector_store(
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
        collection_name="reranked_retrieval_test",
    )

    _index_test_documents(
        tmp_path=tmp_path,
        vector_store=vector_store,
    )

    fake_cross_encoder = FakeCrossEncoder(
        model_name="test-cross-encoder",
    )

    monkeypatch.setattr(
        "rag_engine.retrieval.reranker.CrossEncoder",
        lambda model_name: fake_cross_encoder,
    )

    retriever = RerankedRetriever(
        vector_store=vector_store,
        top_k=2,
        candidate_multiplier=4,
        model_name="test-cross-encoder",
    )

    response = retriever.retrieve(
        query="vector database retrieval",
    )

    _assert_valid_retrieval_response(
        response=response,
        expected_strategy="reranked",
        expected_query="vector database retrieval",
        max_results=2,
    )

    assert response.metadata["retrieval_method"] == (
        "vector_then_cross_encoder"
    )

    assert response.metadata["top_k"] == 2
    assert response.metadata["reranker_model"] == (
        "test-cross-encoder"
    )

    assert response.metadata["candidate_pool"] >= len(
        response.results
    )

    assert response.results

    for result in response.results:
        assert "vector_score" in result.metadata
        assert "reranker_score" in result.metadata

        assert isinstance(
            result.metadata["vector_score"],
            float,
        )

        assert isinstance(
            result.metadata["reranker_score"],
            float,
        )

        assert result.score == pytest.approx(
            result.metadata["reranker_score"]
        )

    # Reranked results must be ordered from highest to lowest
    # cross-encoder score.
    reranker_scores = [
        result.metadata["reranker_score"]
        for result in response.results
    ]

    assert reranker_scores == sorted(
        reranker_scores,
        reverse=True,
    )


def test_retrieval_query_validation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify retrieval rejects empty queries."""

    vector_store = _build_vector_store(
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
        collection_name="retrieval_validation_test",
    )

    naive = NaiveRetriever(
        vector_store=vector_store,
        top_k=5,
    )

    with pytest.raises(RetrievalExecutionError):
        naive.retrieve("   ")


def test_retrieval_top_k_validation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify invalid top_k values are rejected."""

    vector_store = _build_vector_store(
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
        collection_name="retrieval_top_k_test",
    )

    retriever = NaiveRetriever(
        vector_store=vector_store,
        top_k=5,
    )

    with pytest.raises(Exception):
        retriever.retrieve(
            query="Python",
            top_k=0,
        )

