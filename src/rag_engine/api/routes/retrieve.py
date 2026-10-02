"""
Knowledge retrieval API route.
"""

from typing import Literal

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field, field_validator

from rag_engine.api.dependencies import get_pipeline, get_vector_store
from rag_engine.config import get_settings


router = APIRouter(
    prefix="/retrieve",
    tags=["Retrieval"],
)


RetrievalStrategy = Literal[
    "naive",
    "hybrid",
    "reranked",
]


class RetrievalRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=1,
        max_length=5000,
    )
    strategy: RetrievalStrategy | None = None
    top_k: int | None = Field(
        default=None,
        ge=1,
        le=50,
    )

    @field_validator("question")
    @classmethod
    def validate_question(cls, value: str) -> str:
        normalized = value.strip()

        if not normalized:
            raise ValueError("Question cannot be empty.")

        return normalized


class RetrievedEvidence(BaseModel):
    chunk_id: str
    content: str
    score: float
    source: str


class RetrievalResponse(BaseModel):
    question: str
    strategy: RetrievalStrategy
    top_k: int
    results: list[RetrievedEvidence]


@router.post(
    "",
    response_model=RetrievalResponse,
    status_code=status.HTTP_200_OK,
)
async def retrieve_knowledge(
    request: RetrievalRequest,
) -> RetrievalResponse:
    settings = get_settings()

    strategy = request.strategy or settings.default_retrieval_strategy
    top_k = request.top_k or settings.default_top_k

    try:
        pipeline = get_pipeline(strategy)

        retrieval_response = pipeline._retriever.retrieve(
            request.question,
            top_k=top_k,
        )

        results = [
            RetrievedEvidence(
                chunk_id=result.chunk_id,
                content=result.content,
                score=result.score,
                source=result.source,
            )
            for result in retrieval_response.results
        ]

        return RetrievalResponse(
            question=request.question,
            strategy=strategy,
            top_k=top_k,
            results=results,
        )

    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Knowledge retrieval failed.",
        ) from exc