from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from rag_engine.indexing.embeddings import EmbeddingService
from rag_engine.indexing.vector_store import VectorSearchResult, VectorStore
from rag_engine.ingestion.chunker import DocumentChunker
from rag_engine.ingestion.loader import LoadedDocument, load_document


class FakeEmbeddingModel:
    """Lightweight SentenceTransformer-compatible model for tests."""

    def get_embedding_dimension(self) -> int:
        return 4

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
                float(len(normalized)) / 100.0,
            ]

        # SentenceTransformer returns a single 1D vector for a
        # single string and a 2D array for a sequence of strings.
        if isinstance(texts, str):
            return np.asarray(
                make_vector(texts),
                dtype=float,
            )

        return np.asarray(
            [make_vector(text) for text in texts],
            dtype=float,
        )


def test_document_ingestion_and_indexing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test the complete local ingestion and indexing flow."""

    document_path = tmp_path / "rag_test.txt"

    document_path.write_text(
        (
            "Python is a programming language.\n"
            "Vector databases store embeddings.\n"
            "Retrieval systems search relevant documents."
        ),
        encoding="utf-8",
    )

    # ---------------------------------------------------------------
    # 1. Document loading
    # ---------------------------------------------------------------

    document = load_document(document_path)

    assert isinstance(document, LoadedDocument)
    assert document.filename == "rag_test.txt"
    assert document.file_type == ".txt"
    assert document.source == str(document_path)
    assert "Python" in document.content
    assert document.metadata == {}

    # ---------------------------------------------------------------
    # 2. Document chunking
    # ---------------------------------------------------------------

    chunker = DocumentChunker(
        chunk_size=100,
        chunk_overlap=10,
    )

    chunks = chunker.chunk(document)

    assert chunks
    assert all(chunk.content.strip() for chunk in chunks)
    assert all(chunk.document_id for chunk in chunks)
    assert all(chunk.chunk_id for chunk in chunks)

    document_ids = {
        chunk.document_id
        for chunk in chunks
    }

    assert len(document_ids) == 1

    assert [
        chunk.chunk_index
        for chunk in chunks
    ] == list(range(len(chunks)))

    # ---------------------------------------------------------------
    # 3. Local embedding generation
    # ---------------------------------------------------------------

    fake_model = FakeEmbeddingModel()

    monkeypatch.setattr(
        "rag_engine.indexing.embeddings.SentenceTransformer",
        lambda *args, **kwargs: fake_model,
    )

    embedding_service = EmbeddingService(
        model_name="test-model",
    )

    embedding_result = embedding_service.embed_texts(
        [chunk.content for chunk in chunks]
    )

    assert len(embedding_result.vectors) == len(chunks)
    assert embedding_result.model_name == "test-model"
    assert embedding_result.dimension == 4

    assert all(
        len(vector) == 4
        for vector in embedding_result.vectors
    )

    query_vector = embedding_service.embed_text(
        "Python programming"
    )

    assert isinstance(query_vector, list)
    assert len(query_vector) == 4

    # ---------------------------------------------------------------
    # 4. Persistent vector store
    # ---------------------------------------------------------------

    vector_store = VectorStore(
        persist_directory=tmp_path / "vector_store",
        collection_name="integration_test",
        embedding_service=embedding_service,
    )

    assert vector_store.count == 0

    stored_count = vector_store.add_chunks(chunks)

    assert stored_count == len(chunks)
    assert vector_store.count == len(chunks)

    # ---------------------------------------------------------------
    # 5. Semantic search
    # ---------------------------------------------------------------

    results = vector_store.search(
        query="Python programming language",
        top_k=2,
    )

    assert results
    assert len(results) <= 2

    assert all(
        isinstance(result, VectorSearchResult)
        for result in results
    )

    stored_chunk_ids = {
        chunk.chunk_id
        for chunk in chunks
    }

    result_chunk_ids = {
        result.chunk_id
        for result in results
    }

    assert result_chunk_ids.issubset(stored_chunk_ids)

    for result in results:
        assert result.metadata["document_id"] == chunks[0].document_id
        assert result.filename == "rag_test.txt"
        assert result.file_type == ".txt"
        assert isinstance(result.score, float)


def test_multiple_documents_are_indexed_independently(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test independent indexing and deletion of multiple documents."""

    first_path = tmp_path / "python.txt"
    second_path = tmp_path / "database.txt"

    first_path.write_text(
        "Python is widely used for machine learning.",
        encoding="utf-8",
    )

    second_path.write_text(
        "Vector databases support semantic retrieval.",
        encoding="utf-8",
    )

    # ---------------------------------------------------------------
    # 1. Load documents
    # ---------------------------------------------------------------

    first_document = load_document(first_path)
    second_document = load_document(second_path)

    assert first_document.filename == "python.txt"
    assert second_document.filename == "database.txt"

    # ---------------------------------------------------------------
    # 2. Chunk documents
    # ---------------------------------------------------------------

    chunker = DocumentChunker(
        chunk_size=100,
        chunk_overlap=10,
    )

    first_chunks = chunker.chunk(first_document)
    second_chunks = chunker.chunk(second_document)

    all_chunks = first_chunks + second_chunks

    assert first_chunks
    assert second_chunks

    first_document_ids = {
        chunk.document_id
        for chunk in first_chunks
    }

    second_document_ids = {
        chunk.document_id
        for chunk in second_chunks
    }

    assert len(first_document_ids) == 1
    assert len(second_document_ids) == 1
    assert first_document_ids.isdisjoint(second_document_ids)

    # ---------------------------------------------------------------
    # 3. Local embedding model
    # ---------------------------------------------------------------

    fake_model = FakeEmbeddingModel()

    monkeypatch.setattr(
        "rag_engine.indexing.embeddings.SentenceTransformer",
        lambda *args, **kwargs: fake_model,
    )

    embedding_service = EmbeddingService(
        model_name="test-model",
    )

    # ---------------------------------------------------------------
    # 4. Store both documents
    # ---------------------------------------------------------------

    vector_store = VectorStore(
        persist_directory=tmp_path / "vector_store",
        collection_name="multi_document_test",
        embedding_service=embedding_service,
    )

    stored_count = vector_store.add_chunks(all_chunks)

    assert stored_count == len(all_chunks)
    assert vector_store.count == len(all_chunks)

    # ---------------------------------------------------------------
    # 5. Search
    # ---------------------------------------------------------------

    results = vector_store.search(
        query="semantic retrieval",
        top_k=5,
    )

    assert results
    assert len(results) <= 5

    result_document_ids = {
        result.metadata["document_id"]
        for result in results
    }

    all_document_ids = (
        first_document_ids | second_document_ids
    )

    assert result_document_ids.issubset(all_document_ids)

    assert second_document_ids & result_document_ids

    # ---------------------------------------------------------------
    # 6. Delete one document
    # ---------------------------------------------------------------

    second_document_id = next(iter(second_document_ids))

    deleted_count = vector_store.delete_document(
        second_document_id
    )

    assert deleted_count == len(second_chunks)
    assert vector_store.count == len(first_chunks)

    remaining_results = vector_store.search(
        query="semantic retrieval",
        top_k=5,
    )

    remaining_document_ids = {
        result.metadata["document_id"]
        for result in remaining_results
    }

    assert second_document_id not in remaining_document_ids
    assert remaining_document_ids.issubset(first_document_ids)

