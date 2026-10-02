"""
Document loading and text extraction.

This module converts supported document formats into a common internal
representation that can be consumed by the chunking and indexing layers.

Supported formats:
    - PDF
    - TXT
    - DOCX
    - CSV
    - JSON

The loader is intentionally responsible only for reading documents and
extracting their content. Chunking, embedding, and indexing belong to
separate layers.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from io import StringIO
from pathlib import Path
from typing import Any

import pymupdf
from docx import Document as DocxDocument


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class DocumentLoaderError(Exception):
    """
    Base exception for document loading failures.
    """


class UnsupportedDocumentTypeError(DocumentLoaderError):
    """
    Raised when a document format is not supported.
    """


class DocumentExtractionError(DocumentLoaderError):
    """
    Raised when content cannot be extracted from a supported document.
    """


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class LoadedDocument:
    """
    Normalized representation of a loaded document.

    Attributes:
        source: Original document path.
        filename: Original filename.
        file_type: Normalized file extension.
        content: Extracted textual content.
        metadata: Additional document metadata.
    """

    source: str
    filename: str
    file_type: str
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Supported formats
# ---------------------------------------------------------------------------


SUPPORTED_EXTENSIONS: frozenset[str] = frozenset(
    {
        ".pdf",
        ".txt",
        ".docx",
        ".csv",
        ".json",
    }
)


# ---------------------------------------------------------------------------
# Public loader
# ---------------------------------------------------------------------------


class DocumentLoader:
    """
    Load supported documents into a normalized representation.

    The loader uses the file extension to select the appropriate extraction
    strategy.
    """

    def load(self, path: Path) -> LoadedDocument:
        """
        Load a document from disk.

        Args:
            path: Path to the document.

        Returns:
            LoadedDocument: Normalized document representation.

        Raises:
            FileNotFoundError: If the document does not exist.
            UnsupportedDocumentTypeError: If the extension is unsupported.
            DocumentExtractionError: If extraction fails.
        """

        if not path.exists():
            raise FileNotFoundError(
                f"Document does not exist: {path}"
            )

        if not path.is_file():
            raise DocumentLoaderError(
                f"Document path is not a file: {path}"
            )

        extension = path.suffix.lower()

        if extension not in SUPPORTED_EXTENSIONS:
            supported = ", ".join(sorted(SUPPORTED_EXTENSIONS))

            raise UnsupportedDocumentTypeError(
                f"Unsupported document type '{extension or 'unknown'}'. "
                f"Supported types: {supported}."
            )

        try:
            content, metadata = self._extract_content(
                path,
                extension,
            )

        except DocumentLoaderError:
            raise

        except (OSError, UnicodeError, ValueError) as exc:
            raise DocumentExtractionError(
                f"Failed to extract content from '{path.name}'."
            ) from exc

        normalized_content = self._normalize_text(content)

        if not normalized_content:
            raise DocumentExtractionError(
                f"Document '{path.name}' contains no readable text."
            )

        return LoadedDocument(
            source=str(path),
            filename=path.name,
            file_type=extension,
            content=normalized_content,
            metadata=metadata,
        )

    # -----------------------------------------------------------------------
    # Extraction dispatch
    # -----------------------------------------------------------------------

    def _extract_content(
        self,
        path: Path,
        extension: str,
    ) -> tuple[str, dict[str, Any]]:
        """
        Dispatch extraction to the appropriate format-specific loader.

        Args:
            path: Document path.
            extension: Normalized file extension.

        Returns:
            tuple[str, dict[str, Any]]: Extracted text and metadata.
        """

        loaders = {
            ".pdf": self._load_pdf,
            ".txt": self._load_text,
            ".docx": self._load_docx,
            ".csv": self._load_csv,
            ".json": self._load_json,
        }

        loader = loaders[extension]
        return loader(path)

    # -----------------------------------------------------------------------
    # PDF
    # -----------------------------------------------------------------------

    def _load_pdf(
        self,
        path: Path,
    ) -> tuple[str, dict[str, Any]]:
        """
        Extract text from a PDF document.

        Args:
            path: PDF path.

        Returns:
            tuple[str, dict[str, Any]]: Extracted text and metadata.
        """

        pages: list[str] = []

        try:
            with pymupdf.open(path) as document:
                metadata = {
                    "page_count": len(document),
                    "pdf_metadata": document.metadata or {},
                }

                for page_number, page in enumerate(document, start=1):
                    text = page.get_text("text").strip()

                    if text:
                        pages.append(
                            f"[Page {page_number}]\n{text}"
                        )

        except Exception as exc:
            raise DocumentExtractionError(
                f"Failed to read PDF '{path.name}'."
            ) from exc

        return "\n\n".join(pages), metadata

    # -----------------------------------------------------------------------
    # TXT
    # -----------------------------------------------------------------------

    def _load_text(
        self,
        path: Path,
    ) -> tuple[str, dict[str, Any]]:
        """
        Extract text from a plain-text document.

        Args:
            path: Text file path.

        Returns:
            tuple[str, dict[str, Any]]: Text and metadata.
        """

        content = path.read_text(
            encoding="utf-8-sig",
            errors="replace",
        )

        return content, {}

    # -----------------------------------------------------------------------
    # DOCX
    # -----------------------------------------------------------------------

    def _load_docx(
        self,
        path: Path,
    ) -> tuple[str, dict[str, Any]]:
        """
        Extract paragraphs and table content from a DOCX document.

        Args:
            path: DOCX file path.

        Returns:
            tuple[str, dict[str, Any]]: Extracted text and metadata.
        """

        try:
            document = DocxDocument(path)

            sections: list[str] = []

            for paragraph in document.paragraphs:
                text = paragraph.text.strip()

                if text:
                    sections.append(text)

            for table_index, table in enumerate(
                document.tables,
                start=1,
            ):
                rows: list[str] = []

                for row in table.rows:
                    cells = [
                        cell.text.strip()
                        for cell in row.cells
                    ]

                    rows.append(" | ".join(cells))

                if rows:
                    sections.append(
                        f"[Table {table_index}]\n"
                        + "\n".join(rows)
                    )

        except Exception as exc:
            raise DocumentExtractionError(
                f"Failed to read DOCX '{path.name}'."
            ) from exc

        metadata = {
            "paragraph_count": len(document.paragraphs),
            "table_count": len(document.tables),
        }

        return "\n\n".join(sections), metadata

    # -----------------------------------------------------------------------
    # CSV
    # -----------------------------------------------------------------------

    def _load_csv(
        self,
        path: Path,
    ) -> tuple[str, dict[str, Any]]:
        """
        Convert CSV rows into readable textual records.

        Args:
            path: CSV file path.

        Returns:
            tuple[str, dict[str, Any]]: Text representation and metadata.
        """

        content = path.read_text(
            encoding="utf-8-sig",
            errors="replace",
        )

        reader = csv.DictReader(StringIO(content))

        if reader.fieldnames is None:
            raise DocumentExtractionError(
                f"CSV '{path.name}' does not contain a header row."
            )

        rows: list[str] = []

        for row_number, row in enumerate(reader, start=1):
            fields = [
                f"{key}: {value}"
                for key, value in row.items()
                if key is not None and value is not None
            ]

            if fields:
                rows.append(
                    f"[Row {row_number}]\n"
                    + "\n".join(fields)
                )

        metadata = {
            "columns": list(reader.fieldnames),
            "row_count": len(rows),
        }

        return "\n\n".join(rows), metadata

    # -----------------------------------------------------------------------
    # JSON
    # -----------------------------------------------------------------------

    def _load_json(
        self,
        path: Path,
    ) -> tuple[str, dict[str, Any]]:
        """
        Convert JSON data into readable text.

        Args:
            path: JSON file path.

        Returns:
            tuple[str, dict[str, Any]]: Formatted JSON text and metadata.
        """

        content = path.read_text(
            encoding="utf-8-sig",
            errors="replace",
        )

        try:
            data = json.loads(content)

        except json.JSONDecodeError as exc:
            raise DocumentExtractionError(
                f"Invalid JSON in '{path.name}'."
            ) from exc

        formatted = json.dumps(
            data,
            ensure_ascii=False,
            indent=2,
        )

        metadata = {
            "json_type": type(data).__name__,
        }

        return formatted, metadata

    # -----------------------------------------------------------------------
    # Text normalization
    # -----------------------------------------------------------------------

    @staticmethod
    def _normalize_text(text: str) -> str:
        """
        Normalize extracted text for downstream processing.

        Consecutive whitespace on each line is reduced while preserving
        meaningful paragraph and line boundaries.

        Args:
            text: Raw extracted text.

        Returns:
            str: Normalized text.
        """

        lines = [
            " ".join(line.split())
            for line in text.splitlines()
        ]

        return "\n".join(
            line for line in lines if line
        ).strip()


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------


def load_document(path: str | Path) -> LoadedDocument:
    """
    Load a document using the default document loader.

    Args:
        path: Document path.

    Returns:
        LoadedDocument: Normalized loaded document.
    """

    return DocumentLoader().load(Path(path))
