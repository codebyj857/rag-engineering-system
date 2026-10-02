"""
Central configuration management for the RAG Engineering System.

This module provides one validated configuration object for the entire
application. Configuration values can come from environment variables
or the project's .env file.

Secrets such as API keys must never be hardcoded in application code.
"""

from functools import lru_cache
from pathlib import Path

from pydantic import Field, ValidationError, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------

# config.py:
#   <project>/src/rag_engine/config.py
#
# parents[0] -> rag_engine
# parents[1] -> src
# parents[2] -> project root

PROJECT_ROOT = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------


class Settings(BaseSettings):
    """
    Validated application configuration.

    Environment variables take precedence over values defined in .env.
    """

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # -----------------------------------------------------------------------
    # Application
    # -----------------------------------------------------------------------

    app_name: str = "RAG Engineering System"

    app_version: str = "1.0.0"

    environment: str = "development"

    debug: bool = True

    # -----------------------------------------------------------------------
    # API server
    # -----------------------------------------------------------------------

    api_host: str = "127.0.0.1"

    api_port: int = Field(
        default=8000,
        ge=1,
        le=65535,
    )

    # -----------------------------------------------------------------------
    # Frontend
    # -----------------------------------------------------------------------

    frontend_url: str = "http://127.0.0.1:5500"

    # -----------------------------------------------------------------------
    # LLM
    # -----------------------------------------------------------------------

    groq_api_key: str = ""

    llm_model: str = "openai/gpt-oss-120b"

    llm_temperature: float = Field(
        default=0.2,
        ge=0.0,
        le=2.0,
    )

    # -----------------------------------------------------------------------
    # Retrieval
    # -----------------------------------------------------------------------

    default_retrieval_strategy: str = "reranked"

    default_top_k: int = Field(
        default=5,
        ge=1,
        le=50,
    )

    # -----------------------------------------------------------------------
    # Document processing
    # -----------------------------------------------------------------------

    chunk_size: int = Field(
        default=800,
        ge=100,
    )

    chunk_overlap: int = Field(
        default=120,
        ge=0,
    )

    # -----------------------------------------------------------------------
    # Storage
    # -----------------------------------------------------------------------

    documents_dir: Path = PROJECT_ROOT / "data" / "documents"

    evaluation_dir: Path = PROJECT_ROOT / "data" / "evaluation"

    evaluation_runs_dir: Path = (
        PROJECT_ROOT / "data" / "evaluation" / "runs"
    )

    # -----------------------------------------------------------------------
    # Validators
    # -----------------------------------------------------------------------

    @field_validator("environment")
    @classmethod
    def validate_environment(cls, value: str) -> str:
        """
        Validate the application environment.

        Args:
            value: Environment name.

        Returns:
            Normalized environment name.

        Raises:
            ValueError: If an unsupported environment is provided.
        """

        normalized = value.strip().lower()

        allowed = {
            "development",
            "testing",
            "production",
        }

        if normalized not in allowed:
            raise ValueError(
                f"Invalid environment '{value}'. "
                f"Expected one of: {', '.join(sorted(allowed))}."
            )

        return normalized

    @field_validator("default_retrieval_strategy")
    @classmethod
    def validate_retrieval_strategy(cls, value: str) -> str:
        """
        Validate the default retrieval strategy.

        Args:
            value: Retrieval strategy name.

        Returns:
            Normalized retrieval strategy.

        Raises:
            ValueError: If the strategy is unsupported.
        """

        normalized = value.strip().lower()

        allowed = {
            "naive",
            "hybrid",
            "reranked",
        }

        if normalized not in allowed:
            raise ValueError(
                f"Invalid retrieval strategy '{value}'. "
                f"Expected one of: {', '.join(sorted(allowed))}."
            )

        return normalized

    @model_validator(mode="after")
    def validate_chunk_configuration(self) -> "Settings":
        """
        Validate relationships between document chunking settings.

        Returns:
            Validated Settings instance.

        Raises:
            ValueError: If chunk overlap is greater than or equal to
                the chunk size.
        """

        if self.chunk_overlap >= self.chunk_size:
            raise ValueError(
                "chunk_overlap must be smaller than chunk_size."
            )

        return self


# ---------------------------------------------------------------------------
# Cached settings accessor
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Load and cache application settings.

    Configuration errors are intentionally allowed to propagate so that
    invalid application configuration causes a clear startup failure
    instead of silently running with incorrect values.

    Returns:
        Settings: Validated application settings.

    Raises:
        ValidationError: If configuration values are invalid.
        OSError: If configuration cannot be read.
    """

    try:
        return Settings()

    except ValidationError:
        # Preserve Pydantic's detailed validation information.
        raise

    except OSError:
        # Configuration access failures should stop application startup.
        raise