"""
Document chunking for the RAG Engineering System.

This module converts loaded documents into smaller, overlapping chunks
that can later be embedded and indexed for retrieval.

The chunker is deliberately independent of the vector store and embedding
implementation so that ingestion remains modular and testable.
"""

from dataclasses import dataclass
from hashlib import sha256
from typing import Any

from rag_engine.config import get_settings
from rag_engine.ingestion.loader import LoadedDocument


class ChunkingError(Exception):
    """Base exception for document chunking failures."""


class InvalidChunkConfigurationError(ChunkingError):
    """Raised when chunking configuration is invalid."""


class EmptyDocumentError(ChunkingError):
    """Raised when a document contains no usable text."""


@dataclass(frozen=True, slots=True)
class DocumentChunk:
    """
    Represents a single chunk produced from a loaded document.

    Attributes:
        chunk_id: Stable identifier for the chunk.
        document_id: Stable identifier for the source document.
        content: Text contained in the chunk.
        source: Original source path or identifier.
        filename: Original filename.
        file_type: Source document type.
        chunk_index: Zero-based position of the chunk.
        metadata: Additional document and chunk metadata.
    """

    chunk_id: str
    document_id: str
    content: str
    source: str
    filename: str
    file_type: str
    chunk_index: int
    metadata: dict[str, Any]


class DocumentChunker:
    """
    Split loaded documents into overlapping text chunks.

    Chunk size and overlap are measured in characters rather than tokens.
    This keeps the ingestion layer independent of any particular embedding
    or language model tokenizer.
    """

    def __init__(
        self,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
    ) -> None:
        settings = get_settings()

        self.chunk_size = (
            chunk_size if chunk_size is not None else settings.chunk_size
        )
        self.chunk_overlap = (
            chunk_overlap
            if chunk_overlap is not None
            else settings.chunk_overlap
        )

        self._validate_configuration()

    def chunk(self, document: LoadedDocument) -> list[DocumentChunk]:
        """
        Split a loaded document into overlapping chunks.

        Args:
            document: Loaded document produced by DocumentLoader.

        Returns:
            A list of DocumentChunk objects in document order.

        Raises:
            EmptyDocumentError: If the document contains no usable content.
            ChunkingError: If chunk creation fails.
        """
        content = document.content.strip()

        if not content:
            raise EmptyDocumentError(
                f"Document '{document.filename}' contains no usable text."
            )

        try:
            document_id = self._create_document_id(document)
            chunks: list[DocumentChunk] = []

            start = 0
            chunk_index = 0

            while start < len(content):
                end = min(start + self.chunk_size, len(content))
                chunk_content = content[start:end].strip()

                if chunk_content:
                    chunk_id = self._create_chunk_id(
                        document_id=document_id,
                        chunk_index=chunk_index,
                        content=chunk_content,
                    )

                    chunk_metadata = {
                        **document.metadata,
                        "document_id": document_id,
                        "chunk_index": chunk_index,
                        "chunk_start": start,
                        "chunk_end": end,
                        "chunk_size": len(chunk_content),
                    }

                    chunks.append(
                        DocumentChunk(
                            chunk_id=chunk_id,
                            document_id=document_id,
                            content=chunk_content,
                            source=document.source,
                            filename=document.filename,
                            file_type=document.file_type,
                            chunk_index=chunk_index,
                            metadata=chunk_metadata,
                        )
                    )

                    chunk_index += 1

                if end >= len(content):
                    break

                next_start = end - self.chunk_overlap

                if next_start <= start:
                    raise ChunkingError(
                        "Chunking failed to make forward progress. "
                        "Check chunk_size and chunk_overlap configuration."
                    )

                start = next_start

            if not chunks:
                raise EmptyDocumentError(
                    f"Document '{document.filename}' produced no usable chunks."
                )

            return chunks

        except ChunkingError:
            raise
        except Exception as exc:
            raise ChunkingError(
                f"Failed to chunk document '{document.filename}'."
            ) from exc

    def _validate_configuration(self) -> None:
        """Validate chunk size and overlap settings."""
        if self.chunk_size < 100:
            raise InvalidChunkConfigurationError(
                "chunk_size must be at least 100 characters."
            )

        if self.chunk_overlap < 0:
            raise InvalidChunkConfigurationError(
                "chunk_overlap cannot be negative."
            )

        if self.chunk_overlap >= self.chunk_size:
            raise InvalidChunkConfigurationError(
                "chunk_overlap must be smaller than chunk_size."
            )

    @staticmethod
    def _create_document_id(document: LoadedDocument) -> str:
        """
            Create a stable content-based identifier for a source document.
        
            The local source path is intentionally excluded so the same document
            receives the same ID regardless of whether it is loaded using a
            relative or absolute filesystem path.
            """
        fingerprint = (
                 f"{document.filename}|"
                 f"{document.file_type}|"
                 f"{document.content}"
                    )

        return sha256(fingerprint.encode("utf-8")).hexdigest()[:16]

    @staticmethod
    def _create_chunk_id(
        document_id: str,
        chunk_index: int,
        content: str,
    ) -> str:
        """Create a stable identifier for an individual chunk."""
        fingerprint = (
            f"{document_id}|"
            f"{chunk_index}|"
            f"{content}"
        )

        return sha256(fingerprint.encode("utf-8")).hexdigest()[:16]


def chunk_document(
    document: LoadedDocument,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> list[DocumentChunk]:
    """
    Convenience function for chunking a single loaded document.

    Args:
        document: Loaded document to chunk.
        chunk_size: Optional character-based chunk size.
        chunk_overlap: Optional character overlap.

    Returns:
        List of generated document chunks.
    """
    chunker = DocumentChunker(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    return chunker.chunk(document)