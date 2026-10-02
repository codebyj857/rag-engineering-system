from pathlib import Path

from rag_engine.ingestion.chunker import DocumentChunker
from rag_engine.ingestion.loader import load_document


def main() -> None:
    documents_dir = Path("data/documents")
    chunker = DocumentChunker()

    for path in sorted(documents_dir.glob("*.txt")):
        document = load_document(path)
        chunks = chunker.chunk(document)

        print("=" * 100)
        print(f"DOCUMENT: {path.name}")
        print(f"DOCUMENT ID: {chunks[0].document_id}")
        print(f"CHUNKS: {len(chunks)}")
        print("=" * 100)

        for chunk in chunks:
            print()
            print(
                f"CHUNK INDEX: {chunk.chunk_index} | "
                f"CHUNK ID: {chunk.chunk_id}"
            )
            print("-" * 100)
            print(chunk.content)
            print("-" * 100)


if __name__ == "__main__":
    main()