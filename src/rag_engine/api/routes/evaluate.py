"""
Evaluation API route.

This module exposes the endpoint used by the frontend to execute and
inspect RAG evaluation runs.

The route defines the HTTP contract only. Dataset loading, retrieval
evaluation, generation evaluation, and metric calculation belong to the
evaluation layer.
"""

from typing import Literal

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from rag_engine.config import get_settings


router = APIRouter(
    prefix="/evaluate",
    tags=["Evaluation"],
)


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------


RetrievalStrategy = Literal[
    "naive",
    "hybrid",
    "reranked",
]


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------


class EvaluationRequest(BaseModel):
    """
    Request payload for an evaluation run.
    """

    strategy: RetrievalStrategy | None = Field(
        default=None,
        description="Retrieval strategy to evaluate.",
    )

    top_k: int | None = Field(
        default=None,
        ge=1,
        le=50,
        description="Number of retrieved chunks used during evaluation.",
    )

    full_rag: bool = Field(
        default=True,
        description=(
            "Whether to evaluate the complete RAG pipeline including "
            "generation and verification."
        ),
    )


class EvaluationMetrics(BaseModel):
    """
    Standardized evaluation metrics returned by the API.

    Metric values remain optional because different evaluation modes may
    not calculate every metric.
    """

    retrieval_recall: float | None = None
    retrieval_precision: float | None = None
    faithfulness: float | None = None
    correctness: float | None = None
    groundedness: float | None = None
    average_latency_ms: float | None = None


class EvaluationResponse(BaseModel):
    """
    Structured evaluation result returned to the frontend.
    """

    strategy: RetrievalStrategy
    top_k: int
    full_rag: bool
    total_questions: int
    metrics: EvaluationMetrics


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@router.post(
    "",
    response_model=EvaluationResponse,
    status_code=status.HTTP_200_OK,
)
async def evaluate(
    request: EvaluationRequest,
) -> EvaluationResponse:
    """
    Run an evaluation against the configured evaluation dataset.

    The actual evaluator will eventually:

    1. Load evaluation questions and expected evidence.
    2. Execute the selected retrieval strategy.
    3. Optionally run the complete RAG pipeline.
    4. Calculate retrieval metrics.
    5. Calculate generation and groundedness metrics.
    6. Measure latency.
    7. Persist the evaluation run.

    Args:
        request: Validated evaluation configuration.

    Returns:
        EvaluationResponse: Evaluation metrics and run metadata.

    Raises:
        HTTPException: If the evaluation cannot be executed.
    """

    settings = get_settings()

    strategy = request.strategy or settings.default_retrieval_strategy
    top_k = request.top_k or settings.default_top_k

    try:
        # The evaluator will be connected once the evaluation layer and
        # evaluation dataset are implemented. We intentionally do not
        # fabricate metric values.

        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail=(
                "Evaluation is not connected yet. "
                "The evaluation dataset and evaluator must be "
                "initialized first."
            ),
        )

    except HTTPException:
        raise

    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="RAG evaluation failed.",
        ) from exc

