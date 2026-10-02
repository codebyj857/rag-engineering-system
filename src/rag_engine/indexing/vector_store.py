"""
Persistent vector storage for the RAG Engineering System.

This module provides a small abstraction around ChromaDB so the rest of the
application does not need to know about the underlying vector database.

Responsibilities:
    - Persist document chunks and embeddings.
    - Perform similarity searches.
    - Store chunk metadata.
    - Manage the application collection.
    - Return structured retrieval results.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import chromadb

from rag_engine.config import get_settings
from rag_engine.indexing.embeddings import EmbeddingService
from rag_engine.ingestion.chunker import DocumentChunk


class VectorStoreError(Exception):
    """Base exception for vector-store operations."""


class VectorStoreConfigurationError(VectorStoreError):
    """Raised when vector-store configuration is invalid."""


class VectorStoreOperationError(VectorStoreError):
    """Raised when a vector-store operation fails."""


@dataclass(frozen=True, slots=True)
class VectorSearchResult:
    """
    Represents a single vector similarity-search result.

    Attributes:
        chunk_id: Identifier of the matched chunk.
        content: Text contained in the matched chunk.
        source: Original document source.
        filename: Original document filename.
        file_type: Original document type.
        score: Similarity score where higher values indicate closer matches.
        metadata: Additional stored metadata.
    """

    chunk_id: str
    content: str
    source: str
    filename: str
    file_type: str
    score: float
    metadata: dict[str, Any]


class VectorStore:
    """
    Persistent ChromaDB-backed vector store.

    The vector store receives already-created embeddings. This keeps
    embedding generation separate from persistence and allows the embedding
    implementation to be replaced independently.
    """

    DEFAULT_COLLECTION_NAME = "rag_documents"

    def __init__(
        self,
        persist_directory: str | Path | None = None,
        collection_name: str | None = None,
        embedding_service: EmbeddingService | None = None,
    ) -> None:
        settings = get_settings()

        self.persist_directory = Path(
            persist_directory
            if persist_directory is not None
            else settings.documents_dir.parent / "vector_store"
        )

        self.collection_name = (
            collection_name or self.DEFAULT_COLLECTION_NAME
        ).strip()

        if not self.collection_name:
            raise VectorStoreConfigurationError(
                "Collection name cannot be empty."
            )

        self.embedding_service = (
            embedding_service or EmbeddingService()
        )

        self.persist_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        try:
            self._client = chromadb.PersistentClient(
                path=str(self.persist_directory)
            )

            self._collection = self._client.get_or_create_collection(
                name=self.collection_name,
                metadata={
                    "description": (
                        "Persistent document collection for "
                        "the RAG Engineering System."
                    )
                },
            )

        except Exception as exc:
            raise VectorStoreError(
                "Failed to initialize the persistent vector store."
            ) from exc

    @property
    def count(self) -> int:
        """Return the number of stored chunks."""
        try:
            return self._collection.count()
        except Exception as exc:
            raise VectorStoreOperationError(
                "Failed to retrieve vector-store document count."
            ) from exc

    def add_chunks(
        self,
        chunks: Sequence[DocumentChunk],
    ) -> int:
        """
        Add document chunks to the vector store.

        Embeddings are generated locally using the configured
        EmbeddingService.

        Args:
            chunks: Chunks to add to the vector store.

        Returns:
            Number of chunks submitted to the store.

        Raises:
            VectorStoreOperationError: If chunks cannot be stored.
        """
        if not chunks:
            return 0

        try:
            texts = [chunk.content for chunk in chunks]

            embedding_result = self.embedding_service.embed_texts(texts)

            self._collection.upsert(
                ids=[chunk.chunk_id for chunk in chunks],
                embeddings=embedding_result.vectors,
                documents=texts,
                metadatas=[
                    self._prepare_metadata(chunk)
                    for chunk in chunks
                ],
            )

            return len(chunks)

        except Exception as exc:
            if isinstance(exc, VectorStoreError):
                raise

            raise VectorStoreOperationError(
                "Failed to add document chunks to the vector store."
            ) from exc

    def search(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[VectorSearchResult]:
        """
        Search the vector store using semantic similarity.

        Args:
            query: User query to search for.
            top_k: Maximum number of results to return.

        Returns:
            Ranked vector search results.

        Raises:
            VectorStoreOperationError: If the search fails.
        """
        normalized_query = query.strip()

        if not normalized_query:
            raise VectorStoreOperationError(
                "Search query cannot be empty."
            )

        if top_k < 1:
            raise VectorStoreOperationError(
                "top_k must be at least 1."
            )

        if self.count == 0:
            return []

        try:
            query_vector = self.embedding_service.embed_text(
                normalized_query
            )

            results = self._collection.query(
                query_embeddings=[query_vector],
                n_results=min(top_k, self.count),
                include=[
                    "documents",
                    "metadatas",
                    "distances",
                ],
            )

            return self._parse_search_results(results)

        except Exception as exc:
            if isinstance(exc, VectorStoreError):
                raise

            raise VectorStoreOperationError(
                "Failed to search the vector store."
            ) from exc

    def delete_document(self, document_id: str) -> int:
        """
        Delete all chunks belonging to a document.

        Args:
            document_id: Stable document identifier.

        Returns:
            Number of matching chunks found before deletion.

        Raises:
            VectorStoreOperationError: If deletion fails.
        """
        normalized_document_id = document_id.strip()

        if not normalized_document_id:
            raise VectorStoreOperationError(
                "document_id cannot be empty."
            )

        try:
            existing = self._collection.get(
                where={"document_id": normalized_document_id},
                include=[],
            )

            ids = existing.get("ids", [])

            if not ids:
                return 0

            self._collection.delete(ids=ids)

            return len(ids)

        except Exception as exc:
            raise VectorStoreOperationError(
                f"Failed to delete document '{document_id}'."
            ) from exc

    def clear(self) -> None:
        """Delete all chunks from the current collection."""
        try:
            ids = self._collection.get(include=[])["ids"]

            if ids:
                self._collection.delete(ids=ids)

        except Exception as exc:
            raise VectorStoreOperationError(
                "Failed to clear the vector store."
            ) from exc

    @staticmethod
    def _prepare_metadata(
        chunk: DocumentChunk,
    ) -> dict[str, Any]:
        """
        Convert chunk metadata into Chroma-compatible metadata.

        Chroma metadata values must be primitive scalar values, so nested
        structures are converted to strings where necessary.
        """
        metadata: dict[str, Any] = {
            "source": chunk.source,
            "filename": chunk.filename,
            "file_type": chunk.file_type,
            "document_id": chunk.document_id,
            "chunk_index": chunk.chunk_index,
        }

        for key, value in chunk.metadata.items():
            if key in metadata:
                continue

            if isinstance(value, (str, int, float, bool)):
                metadata[key] = value
            elif value is not None:
                metadata[key] = str(value)

        return metadata

    @staticmethod
    def _parse_search_results(
        results: dict[str, Any],
    ) -> list[VectorSearchResult]:
        """Convert Chroma's response into application-level results."""
        ids = results.get("ids", [[]])[0]
        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        parsed_results: list[VectorSearchResult] = []

        for index, chunk_id in enumerate(ids):
            metadata = metadatas[index] or {}
            document = documents[index] or ""
            distance = distances[index] if distances else 0.0

            # Chroma returns distances rather than similarity scores.
            # With cosine distance, smaller distance means greater
            # similarity. Convert it into a simple higher-is-better score.
            score = 1.0 - float(distance)

            parsed_results.append(
                VectorSearchResult(
                    chunk_id=chunk_id,
                    content=document,
                    source=str(metadata.get("source", "")),
                    filename=str(metadata.get("filename", "")),
                    file_type=str(metadata.get("file_type", "")),
                    score=score,
                    metadata=dict(metadata),
                )
            )

        return parsed_results