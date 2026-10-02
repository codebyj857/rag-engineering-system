"""
Conversation session API routes.
"""

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field, field_validator

from rag_engine.memory.session_store import (
    SessionNotFoundError,
    get_session_store,
)


router = APIRouter(
    prefix="/sessions",
    tags=["Sessions"],
)


class SessionMessage(BaseModel):
    role: str
    content: str
    timestamp: str


class SessionResponse(BaseModel):
    session_id: str
    created_at: str
    updated_at: str
    messages: list[SessionMessage] = Field(default_factory=list)


class CreateSessionResponse(BaseModel):
    session_id: str
    created_at: str


class ResetSessionRequest(BaseModel):
    session_id: str = Field(
        ...,
        min_length=1,
        max_length=128,
    )

    @field_validator("session_id")
    @classmethod
    def validate_session_id(cls, value: str) -> str:
        normalized = value.strip()

        if not normalized:
            raise ValueError("session_id cannot be empty.")

        return normalized


class ResetSessionResponse(BaseModel):
    session_id: str
    status: str


@router.post(
    "/new",
    response_model=CreateSessionResponse,
    status_code=status.HTTP_201_CREATED,
)
async def create_session() -> CreateSessionResponse:
    store = get_session_store()
    session = store.create_session()

    return CreateSessionResponse(
        session_id=session.session_id,
        created_at=session.created_at.isoformat(),
    )


@router.get(
    "/{session_id}",
    response_model=SessionResponse,
    status_code=status.HTTP_200_OK,
)
async def get_session(
    session_id: str,
) -> SessionResponse:
    normalized_session_id = session_id.strip()

    if not normalized_session_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="session_id cannot be empty.",
        )

    try:
        session = get_session_store().get_session(
            normalized_session_id
        )

        return SessionResponse(
            session_id=session.session_id,
            created_at=session.created_at.isoformat(),
            updated_at=session.updated_at.isoformat(),
            messages=[
                SessionMessage(
                    role=message.role,
                    content=message.content,
                    timestamp=message.timestamp.isoformat(),
                )
                for message in session.messages
            ],
        )

    except SessionNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Session not found.",
        ) from exc


@router.post(
    "/reset",
    response_model=ResetSessionResponse,
    status_code=status.HTTP_200_OK,
)
async def reset_session(
    request: ResetSessionRequest,
) -> ResetSessionResponse:
    try:
        get_session_store().reset_session(
            request.session_id
        )

        return ResetSessionResponse(
            session_id=request.session_id,
            status="reset",
        )

    except SessionNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Session not found.",
        ) from exc