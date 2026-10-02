from __future__ import annotations

import pytest
from pydantic import ValidationError

from rag_engine.config import Settings


def test_settings_defaults() -> None:
    settings = Settings()

    assert settings.app_name == "RAG Engineering System"
    assert settings.app_version == "1.0.0"
    assert settings.environment == "development"
    assert settings.debug is True
    assert settings.api_host == "127.0.0.1"
    assert settings.api_port == 8000

    assert settings.llm_model == "openai/gpt-oss-120b"
    assert settings.llm_temperature == 0.2

    assert settings.default_retrieval_strategy == "reranked"
    assert settings.default_top_k == 5

    assert settings.chunk_size == 800
    assert settings.chunk_overlap == 120


@pytest.mark.parametrize(
    "strategy",
    ["naive", "hybrid", "reranked"],
)
def test_valid_retrieval_strategies(strategy: str) -> None:
    settings = Settings(default_retrieval_strategy=strategy)

    assert settings.default_retrieval_strategy == strategy


def test_invalid_retrieval_strategy() -> None:
    with pytest.raises((ValidationError, ValueError)):
        Settings(default_retrieval_strategy="invalid")


@pytest.mark.parametrize(
    "environment",
    ["development", "testing", "production"],
)
def test_valid_environments(environment: str) -> None:
    settings = Settings(environment=environment)

    assert settings.environment == environment


def test_invalid_environment() -> None:
    with pytest.raises((ValidationError, ValueError)):
        Settings(environment="invalid")


def test_chunk_overlap_must_be_less_than_chunk_size() -> None:
    with pytest.raises((ValidationError, ValueError)):
        Settings(
            chunk_size=500,
            chunk_overlap=500,
        )

    with pytest.raises((ValidationError, ValueError)):
        Settings(
            chunk_size=500,
            chunk_overlap=600,
        )


def test_chunk_overlap_cannot_be_negative() -> None:
    with pytest.raises((ValidationError, ValueError)):
        Settings(chunk_overlap=-1)


def test_temperature_default() -> None:
    settings = Settings()

    assert settings.llm_temperature == 0.2


def test_top_k_boundaries() -> None:
    assert Settings(default_top_k=1).default_top_k == 1
    assert Settings(default_top_k=50).default_top_k == 50


def test_top_k_out_of_range() -> None:
    with pytest.raises((ValidationError, ValueError)):
        Settings(default_top_k=0)

    with pytest.raises((ValidationError, ValueError)):
        Settings(default_top_k=51)