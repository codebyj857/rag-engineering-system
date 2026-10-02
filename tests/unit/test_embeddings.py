from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from rag_engine.indexing.embeddings import (
    EmbeddingConfigurationError,
    EmbeddingError,
    EmbeddingGenerationError,
    EmbeddingResult,
    EmbeddingService,
    get_embedding_service,
)


def create_mock_model(
    dimension: int = 3,
) -> MagicMock:
    model = MagicMock()
    model.get_embedding_dimension.return_value = dimension

    def encode(
        texts,
        batch_size=None,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    ):
        if isinstance(texts, str):
            return MagicMock(
                tolist=lambda: [0.1, 0.2, 0.3][:dimension]
            )

        return MagicMock(
            tolist=lambda: [
                [0.1, 0.2, 0.3][:dimension]
                for _ in texts
            ]
        )

    model.encode.side_effect = encode
    return model


def test_default_model_name() -> None:
    service = EmbeddingService()

    assert service.model_name == "sentence-transformers/all-MiniLM-L6-v2"


def test_custom_model_name() -> None:
    service = EmbeddingService("custom/test-model")

    assert service.model_name == "custom/test-model"


def test_empty_model_name_is_rejected() -> None:
    with pytest.raises(EmbeddingConfigurationError):
        EmbeddingService("   ")


def test_model_is_loaded_lazily() -> None:
    service = EmbeddingService()

    assert service._model is None


@patch("rag_engine.indexing.embeddings.SentenceTransformer")
def test_model_is_initialized_when_accessed(
    mock_sentence_transformer,
) -> None:
    mock_model = create_mock_model()
    mock_sentence_transformer.return_value = mock_model

    service = EmbeddingService("test-model")

    assert service._model is None

    model = service.model

    mock_sentence_transformer.assert_called_once_with("test-model")
    assert model is mock_model
    assert service._model is mock_model


@patch("rag_engine.indexing.embeddings.SentenceTransformer")
def test_model_is_cached(
    mock_sentence_transformer,
) -> None:
    mock_model = create_mock_model()
    mock_sentence_transformer.return_value = mock_model

    service = EmbeddingService("test-model")

    first = service.model
    second = service.model

    assert first is second
    mock_sentence_transformer.assert_called_once_with("test-model")


@patch("rag_engine.indexing.embeddings.SentenceTransformer")
def test_model_loading_failure_is_wrapped(
    mock_sentence_transformer,
) -> None:
    mock_sentence_transformer.side_effect = RuntimeError("load failed")

    service = EmbeddingService("test-model")

    with pytest.raises(EmbeddingError):
        _ = service.model


@patch("rag_engine.indexing.embeddings.SentenceTransformer")
def test_dimension(
    mock_sentence_transformer,
) -> None:
    mock_model = create_mock_model(dimension=384)
    mock_sentence_transformer.return_value = mock_model

    service = EmbeddingService()

    assert service.dimension == 384
    mock_model.get_embedding_dimension.assert_called_once()


@patch("rag_engine.indexing.embeddings.SentenceTransformer")
def test_embed_text(
    mock_sentence_transformer,
) -> None:
    mock_model = create_mock_model()
    mock_sentence_transformer.return_value = mock_model

    service = EmbeddingService()

    vector = service.embed_text("  hello world  ")

    assert vector == [0.1, 0.2, 0.3]

    mock_model.encode.assert_called_once_with(
        "hello world",
        convert_to_numpy=True,
        normalize_embeddings=True,
    )


def test_embed_empty_text_is_rejected() -> None:
    service = EmbeddingService()

    with pytest.raises(EmbeddingGenerationError):
        service.embed_text("   ")


@patch("rag_engine.indexing.embeddings.SentenceTransformer")
def test_embed_texts(
    mock_sentence_transformer,
) -> None:
    mock_model = create_mock_model()
    mock_sentence_transformer.return_value = mock_model

    service = EmbeddingService()

    result = service.embed_texts(
        ["  first text  ", "second text"],
        batch_size=2,
    )

    assert isinstance(result, EmbeddingResult)
    assert result.vectors == [
        [0.1, 0.2, 0.3],
        [0.1, 0.2, 0.3],
    ]
    assert result.model_name == service.model_name
    assert result.dimension == 3

    mock_model.encode.assert_called_once_with(
        ["first text", "second text"],
        batch_size=2,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )


def test_embed_texts_empty_collection_is_rejected() -> None:
    service = EmbeddingService()

    with pytest.raises(EmbeddingGenerationError):
        service.embed_texts([])


def test_embed_texts_invalid_batch_size_is_rejected() -> None:
    service = EmbeddingService()

    with pytest.raises(EmbeddingGenerationError):
        service.embed_texts(["hello"], batch_size=0)


def test_embed_texts_with_empty_item_is_rejected() -> None:
    service = EmbeddingService()

    with pytest.raises(EmbeddingGenerationError):
        service.embed_texts(["hello", "   "])


@patch("rag_engine.indexing.embeddings.SentenceTransformer")
def test_embed_text_failure_is_wrapped(
    mock_sentence_transformer,
) -> None:
    mock_model = MagicMock()
    mock_model.encode.side_effect = RuntimeError("encoding failed")
    mock_sentence_transformer.return_value = mock_model

    service = EmbeddingService()

    with pytest.raises(EmbeddingGenerationError):
        service.embed_text("hello")


@patch("rag_engine.indexing.embeddings.SentenceTransformer")
def test_embed_texts_failure_is_wrapped(
    mock_sentence_transformer,
) -> None:
    mock_model = MagicMock()
    mock_model.encode.side_effect = RuntimeError("encoding failed")
    mock_sentence_transformer.return_value = mock_model

    service = EmbeddingService()

    with pytest.raises(EmbeddingGenerationError):
        service.embed_texts(["hello", "world"])


def test_get_embedding_service() -> None:
    service = get_embedding_service("custom/test-model")

    assert isinstance(service, EmbeddingService)
    assert service.model_name == "custom/test-model"