"""
Index all source documents into the persistent Chroma vector store.

Pipeline:
    data/documents/
        -> DocumentLoader
        -> DocumentChunker
        -> VectorStore

This script does not use the LLM/Groq API.
Embeddings are generated locally by EmbeddingService.
"""

from __future__ import annotations

from pathlib import Path

from rag_engine.indexing.vector_store import VectorStore
from rag_engine.ingestion.chunker import DocumentChunker
from rag_engine.ingestion.loader import load_document


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DOCUMENTS_DIRECTORY = PROJECT_ROOT / "data" / "documents"

SUPPORTED_EXTENSIONS = {
    ".pdf",
    ".txt",
    ".docx",
    ".csv",
    ".json",
}


def main() -> None:
    """Load, chunk, and index all supported documents."""

    print("=" * 72)
    print("RAG ENGINEERING SYSTEM — DOCUMENT INDEXING")
    print("=" * 72)

    if not DOCUMENTS_DIRECTORY.exists():
        raise FileNotFoundError(
            f"Documents directory not found: {DOCUMENTS_DIRECTORY}"
        )

    document_paths = sorted(
        path
        for path in DOCUMENTS_DIRECTORY.iterdir()
        if path.is_file()
        and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )

    if not document_paths:
        raise RuntimeError(
            f"No supported documents found in {DOCUMENTS_DIRECTORY}"
        )

    print(f"Documents directory: {DOCUMENTS_DIRECTORY}")
    print(f"Documents found: {len(document_paths)}")
    print()

    chunker = DocumentChunker()
    vector_store = VectorStore()

    all_chunks = []

    for document_path in document_paths:
        print(f"Loading: {document_path.name}")

        document = load_document(document_path)
        chunks = chunker.chunk(document)

        print(
            f"  document_id: {chunks[0].document_id if chunks else 'N/A'}"
        )
        print(f"  chunks: {len(chunks)}")

        all_chunks.extend(chunks)

    print()
    print(f"Total chunks prepared: {len(all_chunks)}")

    if not all_chunks:
        raise RuntimeError("No chunks were generated.")

    print()
    print("Indexing chunks into persistent vector store...")

    indexed_count = vector_store.add_chunks(all_chunks)

    print(f"Indexed chunks: {indexed_count}")
    print(f"Vector store total chunks: {vector_store.count}")

    print()
    print("=" * 72)
    print("INDEXING COMPLETE")
    print("=" * 72)


if __name__ == "__main__":
    main()