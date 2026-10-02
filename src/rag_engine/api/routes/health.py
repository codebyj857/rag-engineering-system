"""
Health-check API route.

This endpoint provides a lightweight way for the frontend, monitoring
systems, and deployment infrastructure to verify that the API is running.
"""

from datetime import datetime, timezone

from fastapi import APIRouter

router = APIRouter(
    prefix="/health",
    tags=["Health"],
)


@router.get("")
async def health_check() -> dict[str, str]:
    """
    Return the current API health status.

    Returns:
        dict[str, str]: Basic service status and UTC timestamp.
    """

    return {
        "status": "healthy",
        "service": "rag-engineering-system",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }