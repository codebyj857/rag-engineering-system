"""
Document ingestion API routes.

Uploads are loaded, chunked, embedded, and indexed into the persistent
vector store before a successful response is returned.
"""

from pathlib import Path
from uuid import uuid4

from fastapi import APIRouter, File, HTTPException, UploadFile, status

from rag_engine.config import get_settings
from rag_engine.indexing.vector_store import VectorStore
from rag_engine.ingestion.chunker import DocumentChunker
from rag_engine.ingestion.loader import load_document


router = APIRouter(
    prefix="/ingest",
    tags=["Ingestion"],
)


SUPPORTED_EXTENSIONS: frozenset[str] = frozenset(
    {
        ".pdf",
        ".txt",
        ".docx",
        ".csv",
        ".json",
    }
)


def _validate_file_extension(filename: str) -> str:
    extension = Path(filename).suffix.lower()

    if extension not in SUPPORTED_EXTENSIONS:
        supported = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"Unsupported file type '{extension or 'unknown'}'. "
                f"Supported types: {supported}."
            ),
        )

    return extension


def _safe_filename(filename: str) -> str:
    original_name = Path(filename).name
    return f"{uuid4().hex}_{original_name}"


@router.post(
    "",
    status_code=status.HTTP_201_CREATED,
)
async def ingest_document(
    file: UploadFile = File(...),
) -> dict[str, object]:
    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file must have a filename.",
        )

    extension = _validate_file_extension(file.filename)
    settings = get_settings()

    settings.documents_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    safe_filename = _safe_filename(file.filename)
    destination = settings.documents_dir / safe_filename

    try:
        file_content = await file.read()

        if not file_content:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Uploaded file is empty.",
            )

        destination.write_bytes(file_content)

        document = load_document(destination)

        chunker = DocumentChunker(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        )
        chunks = chunker.chunk(document)

        if not chunks:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Document produced no usable chunks.",
            )

        vector_store = VectorStore()
        indexed_count = vector_store.add_chunks(chunks)

        return {
            "status": "indexed",
            "filename": file.filename,
            "stored_filename": safe_filename,
            "extension": extension,
            "document_id": chunks[0].metadata["document_id"],
            "chunk_count": len(chunks),
            "indexed_count": indexed_count,
            "message": "Document uploaded and indexed successfully.",
        }

    except HTTPException:
        raise

    except Exception as exc:
        if destination.exists():
            try:
                destination.unlink()
            except OSError:
                pass

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Document ingestion and indexing failed.",
        ) from exc

    finally:
        await file.close()