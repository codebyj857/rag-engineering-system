"""
Unit tests for the document chunker.
"""

from pathlib import Path

import pytest

from rag_engine.ingestion.chunker import (
    ChunkingError,
    DocumentChunker,
    EmptyDocumentError,
    InvalidChunkConfigurationError,
    chunk_document,
)
from rag_engine.ingestion.loader import LoadedDocument


def make_document(
    content: str,
    *,
    source: str = "documents/example.txt",
    filename: str = "example.txt",
    file_type: str = ".txt",
    metadata: dict | None = None,
) -> LoadedDocument:
    """Create a LoadedDocument for testing."""
    return LoadedDocument(
        source=source,
        filename=filename,
        file_type=file_type,
        content=content,
        metadata=metadata or {},
    )


def test_chunk_document_returns_chunks() -> None:
    """A valid document should produce at least one chunk."""
    document = make_document("A" * 250)

    chunker = DocumentChunker(
        chunk_size=100,
        chunk_overlap=20,
    )

    chunks = chunker.chunk(document)

    assert chunks
    assert all(chunk.content for chunk in chunks)
    assert all(chunk.document_id for chunk in chunks)
    assert all(chunk.chunk_id for chunk in chunks)


def test_chunk_size_is_respected() -> None:
    """Chunks should not exceed the configured character window."""
    document = make_document("A" * 250)

    chunker = DocumentChunker(
        chunk_size=100,
        chunk_overlap=20,
    )

    chunks = chunker.chunk(document)

    assert all(len(chunk.content) <= 100 for chunk in chunks)


def test_chunk_overlap_creates_overlapping_content() -> None:
    """Configured overlap should appear between adjacent chunks."""
    content = "".join(str(index % 10) for index in range(250))
    document = make_document(content)

    chunker = DocumentChunker(
        chunk_size=100,
        chunk_overlap=20,
    )

    chunks = chunker.chunk(document)

    assert len(chunks) >= 2

    for previous, current in zip(chunks, chunks[1:]):
        assert previous.content[-20:] == current.content[:20]


def test_chunk_indices_are_sequential() -> None:
    """Chunk indices should start at zero and increase sequentially."""
    document = make_document("A" * 300)

    chunker = DocumentChunker(
        chunk_size=100,
        chunk_overlap=20,
    )

    chunks = chunker.chunk(document)

    assert [chunk.chunk_index for chunk in chunks] == list(range(len(chunks)))


def test_document_id_is_stable_for_same_document() -> None:
    """Identical document content should produce the same document ID."""
    document_one = make_document(
        "This is a stable document.",
        source="documents/example.txt",
    )
    document_two = make_document(
        "This is a stable document.",
        source="documents/example.txt",
    )

    chunker = DocumentChunker()

    chunks_one = chunker.chunk(document_one)
    chunks_two = chunker.chunk(document_two)

    assert chunks_one[0].document_id == chunks_two[0].document_id


def test_document_id_is_independent_of_source_path() -> None:
    """
    The same document loaded through different filesystem paths should
    receive the same document ID.
    """
    document_relative = make_document(
        "This document should have one stable identity.",
        source="data/documents/example.txt",
    )

    document_absolute = make_document(
        "This document should have one stable identity.",
        source=str(Path.cwd() / "data" / "documents" / "example.txt"),
    )

    chunker = DocumentChunker()

    relative_chunks = chunker.chunk(document_relative)
    absolute_chunks = chunker.chunk(document_absolute)

    assert relative_chunks[0].document_id == absolute_chunks[0].document_id


def test_chunk_ids_are_stable_for_same_document() -> None:
    """Identical documents should produce identical chunk IDs."""
    content = "A" * 250

    document_one = make_document(content)
    document_two = make_document(content)

    chunker = DocumentChunker(
        chunk_size=100,
        chunk_overlap=20,
    )

    chunks_one = chunker.chunk(document_one)
    chunks_two = chunker.chunk(document_two)

    assert [chunk.chunk_id for chunk in chunks_one] == [
        chunk.chunk_id for chunk in chunks_two
    ]


def test_chunk_ids_change_when_content_changes() -> None:
    """Changing document content should change its chunk identity."""
    document_one = make_document("A" * 250)
    document_two = make_document("B" * 250)

    chunker = DocumentChunker(
        chunk_size=100,
        chunk_overlap=20,
    )

    chunks_one = chunker.chunk(document_one)
    chunks_two = chunker.chunk(document_two)

    assert chunks_one[0].document_id != chunks_two[0].document_id
    assert chunks_one[0].chunk_id != chunks_two[0].chunk_id


def test_metadata_contains_chunk_information() -> None:
    """Generated chunks should contain useful positional metadata."""
    document = make_document(
        "A" * 250,
        metadata={"category": "test"},
    )

    chunker = DocumentChunker(
        chunk_size=100,
        chunk_overlap=20,
    )

    chunks = chunker.chunk(document)

    first = chunks[0]

    assert first.metadata["category"] == "test"
    assert first.metadata["document_id"] == first.document_id
    assert first.metadata["chunk_index"] == 0
    assert "chunk_start" in first.metadata
    assert "chunk_end" in first.metadata
    assert "chunk_size" in first.metadata


def test_empty_document_raises_error() -> None:
    """Empty documents should be rejected."""
    document = make_document("   \n\t ")

    chunker = DocumentChunker()

    with pytest.raises(EmptyDocumentError):
        chunker.chunk(document)


def test_chunk_size_below_minimum_raises_error() -> None:
    """Chunk sizes below the minimum should be rejected."""
    with pytest.raises(InvalidChunkConfigurationError):
        DocumentChunker(chunk_size=99)


def test_negative_overlap_raises_error() -> None:
    """Negative overlap should be rejected."""
    with pytest.raises(InvalidChunkConfigurationError):
        DocumentChunker(
            chunk_size=100,
            chunk_overlap=-1,
        )


def test_overlap_equal_to_chunk_size_raises_error() -> None:
    """Overlap must be smaller than the chunk size."""
    with pytest.raises(InvalidChunkConfigurationError):
        DocumentChunker(
            chunk_size=100,
            chunk_overlap=100,
        )


def test_overlap_greater_than_chunk_size_raises_error() -> None:
    """Overlap greater than the chunk size should be rejected."""
    with pytest.raises(InvalidChunkConfigurationError):
        DocumentChunker(
            chunk_size=100,
            chunk_overlap=101,
        )


def test_chunk_document_convenience_function() -> None:
    """The module-level convenience function should work."""
    document = make_document("A" * 250)

    chunks = chunk_document(
        document,
        chunk_size=100,
        chunk_overlap=20,
    )

    assert chunks
    assert all(chunk.document_id for chunk in chunks)


def test_source_path_is_preserved_in_chunk() -> None:
    """The original source path should remain available as metadata."""
    source = "some/path/document.txt"
    document = make_document(
        "A" * 150,
        source=source,
    )

    chunker = DocumentChunker(
        chunk_size=100,
        chunk_overlap=20,
    )

    chunks = chunker.chunk(document)

    assert chunks[0].source == source


def test_filename_and_file_type_are_preserved() -> None:
    """Source filename and type should be preserved."""
    document = make_document(
        "A" * 150,
        filename="knowledge.txt",
        file_type=".txt",
    )

    chunks = DocumentChunker(
        chunk_size=100,
        chunk_overlap=20,
    ).chunk(document)

    assert chunks[0].filename == "knowledge.txt"
    assert chunks[0].file_type == ".txt"


def test_chunks_preserve_document_identity() -> None:
    """All chunks from one document should share one document ID."""
    document = make_document("A" * 400)

    chunks = DocumentChunker(
        chunk_size=100,
        chunk_overlap=20,
    ).chunk(document)

    document_ids = {chunk.document_id for chunk in chunks}

    assert len(document_ids) == 1


def test_chunk_ids_are_unique_within_document() -> None:
    """Each generated chunk should have a unique chunk ID."""
    document = make_document("A" * 500)

    chunks = DocumentChunker(
        chunk_size=100,
        chunk_overlap=20,
    ).chunk(document)

    chunk_ids = [chunk.chunk_id for chunk in chunks]

    assert len(chunk_ids) == len(set(chunk_ids))


def test_chunk_content_is_trimmed() -> None:
    """Leading and trailing whitespace should be removed from chunks."""
    document = make_document(
        "   " + "A" * 150 + "   ",
    )

    chunks = DocumentChunker(
        chunk_size=100,
        chunk_overlap=20,
    ).chunk(document)

    assert chunks[0].content == chunks[0].content.strip()