from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from rag_engine.indexing.embeddings import EmbeddingResult
from rag_engine.indexing.vector_store import (
    VectorSearchResult,
    VectorStore,
    VectorStoreConfigurationError,
    VectorStoreOperationError,
)
from rag_engine.ingestion.chunker import DocumentChunk


def make_embedding_service(dimension: int = 3) -> MagicMock:
    service = MagicMock()

    def embed_text(text: str) -> list[float]:
        if "rag" in text.lower():
            return [1.0, 0.0, 0.0]
        if "database" in text.lower():
            return [0.0, 1.0, 0.0]
        return [0.0, 0.0, 1.0]

    def embed_texts(texts: list[str]) -> EmbeddingResult:
        return EmbeddingResult(
            vectors=[embed_text(text) for text in texts],
            model_name="test-model",
            dimension=dimension,
        )

    service.embed_text.side_effect = embed_text
    service.embed_texts.side_effect = embed_texts

    return service


def make_chunk(
    chunk_id: str,
    document_id: str,
    content: str,
    chunk_index: int = 0,
) -> DocumentChunk:
    return DocumentChunk(
        chunk_id=chunk_id,
        document_id=document_id,
        content=content,
        source="data/documents/test.txt",
        filename="test.txt",
        file_type=".txt",
        chunk_index=chunk_index,
        metadata={
            "category": "test",
            "nested": {"key": "value"},
        },
    )


@pytest.fixture
def vector_store(tmp_path):
    return VectorStore(
        persist_directory=tmp_path / "vector_store",
        collection_name="test_collection",
        embedding_service=make_embedding_service(),
    )


def test_vector_store_initializes(vector_store) -> None:
    assert vector_store.count == 0
    assert vector_store.collection_name == "test_collection"


def test_empty_collection_search_returns_empty(vector_store) -> None:
    assert vector_store.search("RAG", top_k=5) == []


def test_add_chunks(vector_store) -> None:
    chunks = [
        make_chunk(
            "chunk-1",
            "doc-1",
            "RAG retrieval uses embeddings.",
        ),
        make_chunk(
            "chunk-2",
            "doc-1",
            "Vector databases store embeddings.",
            chunk_index=1,
        ),
    ]

    added = vector_store.add_chunks(chunks)

    assert added == 2
    assert vector_store.count == 2


def test_add_empty_chunks_returns_zero(vector_store) -> None:
    assert vector_store.add_chunks([]) == 0
    assert vector_store.count == 0


def test_search_returns_structured_results(vector_store) -> None:
    chunks = [
        make_chunk(
            "chunk-1",
            "doc-1",
            "RAG retrieval uses embeddings.",
        ),
        make_chunk(
            "chunk-2",
            "doc-1",
            "Vector databases store embeddings.",
            chunk_index=1,
        ),
    ]

    vector_store.add_chunks(chunks)

    results = vector_store.search("RAG retrieval", top_k=2)

    assert len(results) == 2
    assert all(isinstance(result, VectorSearchResult) for result in results)

    assert results[0].chunk_id == "chunk-1"
    assert results[0].content == "RAG retrieval uses embeddings."
    assert results[0].filename == "test.txt"
    assert results[0].file_type == ".txt"
    assert results[0].source == "data/documents/test.txt"


def test_search_results_are_sorted_by_similarity(vector_store) -> None:
    chunks = [
        make_chunk(
            "chunk-rag",
            "doc-rag",
            "RAG retrieval uses embeddings.",
        ),
        make_chunk(
            "chunk-db",
            "doc-db",
            "Database storage information.",
            chunk_index=0,
        ),
    ]

    vector_store.add_chunks(chunks)

    results = vector_store.search("RAG retrieval", top_k=2)

    assert results[0].chunk_id == "chunk-rag"
    assert results[0].score >= results[1].score


def test_top_k_limits_results(vector_store) -> None:
    chunks = [
        make_chunk("chunk-1", "doc-1", "RAG information."),
        make_chunk("chunk-2", "doc-1", "Database information.", 1),
        make_chunk("chunk-3", "doc-1", "Other information.", 2),
    ]

    vector_store.add_chunks(chunks)

    results = vector_store.search("RAG", top_k=1)

    assert len(results) == 1


def test_search_query_is_stripped(vector_store) -> None:
    chunk = make_chunk(
        "chunk-1",
        "doc-1",
        "RAG retrieval uses embeddings.",
    )

    vector_store.add_chunks([chunk])

    results = vector_store.search("   RAG retrieval   ", top_k=1)

    assert len(results) == 1
    assert results[0].chunk_id == "chunk-1"


def test_empty_search_query_is_rejected(vector_store) -> None:
    with pytest.raises(VectorStoreOperationError):
        vector_store.search("   ")


def test_invalid_top_k_is_rejected(vector_store) -> None:
    with pytest.raises(VectorStoreOperationError):
        vector_store.search("RAG", top_k=0)


def test_metadata_is_preserved_and_normalized(vector_store) -> None:
    chunk = make_chunk(
        "chunk-1",
        "doc-1",
        "RAG retrieval uses embeddings.",
    )

    vector_store.add_chunks([chunk])

    results = vector_store.search("RAG", top_k=1)
    metadata = results[0].metadata

    assert metadata["source"] == "data/documents/test.txt"
    assert metadata["filename"] == "test.txt"
    assert metadata["file_type"] == ".txt"
    assert metadata["document_id"] == "doc-1"
    assert metadata["chunk_index"] == 0
    assert metadata["category"] == "test"

    # Nested metadata is converted to a string for Chroma compatibility.
    assert isinstance(metadata["nested"], str)


def test_delete_document(vector_store) -> None:
    chunks = [
        make_chunk("chunk-1", "doc-1", "RAG information."),
        make_chunk("chunk-2", "doc-1", "More RAG information.", 1),
        make_chunk("chunk-3", "doc-2", "Database information."),
    ]

    vector_store.add_chunks(chunks)

    assert vector_store.count == 3

    deleted = vector_store.delete_document("doc-1")

    assert deleted == 2
    assert vector_store.count == 1


def test_delete_nonexistent_document_returns_zero(vector_store) -> None:
    chunk = make_chunk(
        "chunk-1",
        "doc-1",
        "RAG information.",
    )

    vector_store.add_chunks([chunk])

    assert vector_store.delete_document("does-not-exist") == 0
    assert vector_store.count == 1


def test_empty_document_id_is_rejected(vector_store) -> None:
    with pytest.raises(VectorStoreOperationError):
        vector_store.delete_document("   ")


def test_clear_removes_all_chunks(vector_store) -> None:
    chunks = [
        make_chunk("chunk-1", "doc-1", "RAG information."),
        make_chunk("chunk-2", "doc-2", "Database information."),
    ]

    vector_store.add_chunks(chunks)

    assert vector_store.count == 2

    vector_store.clear()

    assert vector_store.count == 0
    assert vector_store.search("RAG") == []


def test_empty_collection_name_is_rejected(tmp_path) -> None:
    with pytest.raises(VectorStoreConfigurationError):
        VectorStore(
            persist_directory=tmp_path / "vector_store",
            collection_name="   ",
            embedding_service=make_embedding_service(),
        )


def test_prepare_metadata() -> None:
    chunk = make_chunk(
        "chunk-1",
        "doc-1",
        "RAG information.",
    )

    metadata = VectorStore._prepare_metadata(chunk)

    assert metadata["source"] == chunk.source
    assert metadata["filename"] == chunk.filename
    assert metadata["file_type"] == chunk.file_type
    assert metadata["document_id"] == chunk.document_id
    assert metadata["chunk_index"] == chunk.chunk_index
    assert metadata["category"] == "test"
    assert isinstance(metadata["nested"], str)


def test_parse_search_results() -> None:
    raw_results = {
        "ids": [["chunk-1", "chunk-2"]],
        "documents": [["RAG content", "Database content"]],
        "metadatas": [
            [
                {
                    "source": "rag.txt",
                    "filename": "rag.txt",
                    "file_type": ".txt",
                    "document_id": "doc-1",
                },
                {
                    "source": "db.txt",
                    "filename": "db.txt",
                    "file_type": ".txt",
                    "document_id": "doc-2",
                },
            ]
        ],
        "distances": [[0.1, 0.4]],
    }

    results = VectorStore._parse_search_results(raw_results)

    assert len(results) == 2
    assert results[0].chunk_id == "chunk-1"
    assert results[0].score == pytest.approx(0.9)
    assert results[1].score == pytest.approx(0.6)
    assert results[0].metadata["document_id"] == "doc-1"


def test_persistence_across_instances(tmp_path) -> None:
    persist_directory = tmp_path / "persistent_store"

    first_store = VectorStore(
        persist_directory=persist_directory,
        collection_name="persistent_test",
        embedding_service=make_embedding_service(),
    )

    chunk = make_chunk(
        "persistent-chunk",
        "persistent-doc",
        "RAG persistence test.",
    )

    first_store.add_chunks([chunk])

    second_store = VectorStore(
        persist_directory=persist_directory,
        collection_name="persistent_test",
        embedding_service=make_embedding_service(),
    )

    assert second_store.count == 1

    results = second_store.search("RAG persistence", top_k=1)

    assert len(results) == 1
    assert results[0].chunk_id == "persistent-chunk"