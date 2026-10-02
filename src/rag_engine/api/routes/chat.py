"""
Chat API route.

Connects the HTTP layer to the shared RAG pipeline orchestration.
"""

from typing import Literal

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field, field_validator

from rag_engine.api.dependencies import get_pipeline
from rag_engine.config import get_settings
from rag_engine.pipelines.base import PipelineRequest
from rag_engine.memory.session_store import SessionNotFoundError


router = APIRouter(
    prefix="/chat",
    tags=["Chat"],
)


RetrievalStrategy = Literal[
    "naive",
    "hybrid",
    "reranked",
]


class ChatRequest(BaseModel):
    message: str = Field(
        ...,
        min_length=1,
        max_length=10000,
    )
    session_id: str | None = Field(
        default=None,
        min_length=1,
        max_length=128,
    )
    strategy: RetrievalStrategy | None = None
    top_k: int | None = Field(
        default=None,
        ge=1,
        le=50,
    )
    verify: bool = True

    @field_validator("message")
    @classmethod
    def validate_message(cls, value: str) -> str:
        normalized = value.strip()

        if not normalized:
            raise ValueError("Message cannot be empty.")

        return normalized

    @field_validator("session_id")
    @classmethod
    def validate_session_id(
        cls,
        value: str | None,
    ) -> str | None:
        if value is None:
            return None

        normalized = value.strip()
        return normalized or None


class ChatEvidence(BaseModel):
    chunk_id: str
    content: str
    score: float
    source: str


class VerificationResult(BaseModel):
    enabled: bool
    grounded: bool | None = None
    score: float | None = None
    explanation: str | None = None


class ChatResponse(BaseModel):
    answer: str
    session_id: str
    strategy: RetrievalStrategy
    confidence: float | None = None
    evidence: list[ChatEvidence]
    verification: VerificationResult


@router.post(
    "",
    response_model=ChatResponse,
    status_code=status.HTTP_200_OK,
)
async def chat(
    request: ChatRequest,
) -> ChatResponse:
    settings = get_settings()

    strategy = request.strategy or settings.default_retrieval_strategy
    top_k = request.top_k or settings.default_top_k

    if request.session_id is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="A valid session_id is required.",
        )

    try:
        pipeline = get_pipeline(strategy)

        result = pipeline.run(
            PipelineRequest(
                question=request.message,
                session_id=request.session_id,
                top_k=top_k,
                verify=request.verify,
            )
        )

        evidence = [
            ChatEvidence(
                chunk_id=item.chunk_id,
                content=item.content,
                score=item.score,
                source=item.source,
            )
            for item in result.evidence
        ]

        verification = VerificationResult(
            enabled=request.verify,
            grounded=(
                result.verification.grounded
                if result.verification is not None
                else None
            ),
            score=(
                result.verification.score
                if result.verification is not None
                else None
            ),
        )

        return ChatResponse(
            answer=result.answer,
            session_id=result.session_id,
            strategy=result.strategy,
            confidence=result.confidence.score,
            evidence=evidence,
            verification=verification,
        )

    except SessionNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Session not found.",
        ) from exc

    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="RAG chat processing failed.",
        ) from exc