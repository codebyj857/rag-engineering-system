"""
Embedding generation for the RAG Engineering System.

This module converts text into dense vector representations used by the
retrieval layer. Embeddings are generated locally so retrieval does not
depend on the LLM provider or consume generation API calls.
"""

from dataclasses import dataclass
from typing import Sequence

from sentence_transformers import SentenceTransformer


class EmbeddingError(Exception):
    """Base exception for embedding-related failures."""


class EmbeddingConfigurationError(EmbeddingError):
    """Raised when embedding configuration is invalid."""


class EmbeddingGenerationError(EmbeddingError):
    """Raised when embeddings cannot be generated."""


@dataclass(frozen=True, slots=True)
class EmbeddingResult:
    """
    Result of an embedding operation.

    Attributes:
        vectors: Dense embedding vectors.
        model_name: Name of the embedding model used.
        dimension: Number of dimensions in each vector.
    """

    vectors: list[list[float]]
    model_name: str
    dimension: int


class EmbeddingService:
    """
    Generate vector embeddings using a local Sentence Transformer model.

    The service loads the model lazily so importing the application does not
    immediately download or initialize a potentially large model.
    """

    DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

    def __init__(self, model_name: str | None = None) -> None:
        self.model_name = model_name or self.DEFAULT_MODEL

        if not self.model_name.strip():
            raise EmbeddingConfigurationError(
                "Embedding model name cannot be empty."
            )

        self._model: SentenceTransformer | None = None

    @property
    def model(self) -> SentenceTransformer:
        """
        Return the loaded embedding model.

        The model is initialized only when first accessed.
        """
        if self._model is None:
            try:
                self._model = SentenceTransformer(self.model_name)
            except Exception as exc:
                raise EmbeddingError(
                    f"Failed to load embedding model '{self.model_name}'."
                ) from exc

        return self._model

    @property
    def dimension(self) -> int:
        """Return the dimensionality of the configured embedding model."""
        try:
            return self.model.get_embedding_dimension()
        except Exception as exc:
            raise EmbeddingError(
                "Failed to determine embedding dimension."
            ) from exc

    def embed_text(self, text: str) -> list[float]:
        """
        Generate an embedding for a single text string.

        Args:
            text: Input text to embed.

        Returns:
            A dense vector represented as a list of floats.

        Raises:
            EmbeddingGenerationError: If the input is invalid or encoding
                fails.
        """
        normalized_text = text.strip()

        if not normalized_text:
            raise EmbeddingGenerationError(
                "Cannot generate an embedding for empty text."
            )

        try:
            vector = self.model.encode(
                normalized_text,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )

            return vector.tolist()

        except Exception as exc:
            raise EmbeddingGenerationError(
                "Failed to generate embedding for the provided text."
            ) from exc

    def embed_texts(
        self,
        texts: Sequence[str],
        batch_size: int = 32,
    ) -> EmbeddingResult:
        """
        Generate embeddings for multiple texts.

        Args:
            texts: Sequence of text strings to embed.
            batch_size: Number of texts processed in each model batch.

        Returns:
            EmbeddingResult containing all generated vectors.

        Raises:
            EmbeddingGenerationError: If inputs are invalid or encoding fails.
        """
        if not texts:
            raise EmbeddingGenerationError(
                "Cannot generate embeddings for an empty collection."
            )

        if batch_size < 1:
            raise EmbeddingGenerationError(
                "batch_size must be at least 1."
            )

        normalized_texts = [text.strip() for text in texts]

        if any(not text for text in normalized_texts):
            raise EmbeddingGenerationError(
                "Embedding input contains empty text."
            )

        try:
            vectors = self.model.encode(
                normalized_texts,
                batch_size=batch_size,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )

            vector_list = vectors.tolist()

            return EmbeddingResult(
                vectors=vector_list,
                model_name=self.model_name,
                dimension=self.dimension,
            )

        except Exception as exc:
            raise EmbeddingGenerationError(
                "Failed to generate embeddings for the provided texts."
            ) from exc


def get_embedding_service(
    model_name: str | None = None,
) -> EmbeddingService:
    """
    Create an embedding service.

    Args:
        model_name: Optional Sentence Transformer model name.

    Returns:
        Configured EmbeddingService instance.
    """
    return EmbeddingService(model_name=model_name)