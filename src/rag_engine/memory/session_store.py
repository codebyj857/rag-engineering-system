"""
In-memory session storage for multi-turn RAG conversations.

This module manages conversation sessions and their message history.
The store is intentionally in-memory for the current application
architecture. A persistent store such as Redis can be introduced later
without changing the public interface significantly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from threading import RLock
from typing import Literal
from uuid import uuid4


MessageRole = Literal["user", "assistant"]


class SessionStoreError(Exception):
    """Base exception for session-store errors."""


class SessionNotFoundError(SessionStoreError):
    """Raised when a requested session does not exist."""


class InvalidMessageError(SessionStoreError):
    """Raised when a message is invalid."""


class SessionLimitError(SessionStoreError):
    """Raised when a session exceeds its configured message limit."""


@dataclass(frozen=True, slots=True)
class ChatMessage:
    """
    Represents one message in a conversation.

    Attributes:
        role: Message author, either ``user`` or ``assistant``.
        content: Message text.
        timestamp: UTC timestamp when the message was created.
    """

    role: MessageRole
    content: str
    timestamp: datetime


@dataclass(slots=True)
class Session:
    """
    Represents a multi-turn conversation session.

    Attributes:
        session_id: Unique identifier for the session.
        created_at: UTC timestamp when the session was created.
        updated_at: UTC timestamp of the most recent modification.
        messages: Ordered conversation history.
    """

    session_id: str
    created_at: datetime
    updated_at: datetime
    messages: list[ChatMessage] = field(default_factory=list)


class SessionStore:
    """
    Thread-safe in-memory session store.

    The store provides operations required by the RAG application:

    - create sessions
    - retrieve sessions
    - append user/assistant messages
    - retrieve recent history
    - reset sessions
    - delete sessions

    Args:
        max_messages_per_session: Maximum number of messages retained
            for a single session.
    """

    def __init__(self, max_messages_per_session: int = 20) -> None:
        if max_messages_per_session < 2:
            raise ValueError(
                "max_messages_per_session must be at least 2."
            )

        self._max_messages_per_session = max_messages_per_session
        self._sessions: dict[str, Session] = {}
        self._lock = RLock()

    @property
    def max_messages_per_session(self) -> int:
        """Return the configured maximum messages per session."""
        return self._max_messages_per_session

    def create_session(self, session_id: str | None = None) -> Session:
        """
        Create and store a new conversation session.

        Args:
            session_id: Optional caller-provided session ID.

        Returns:
            The newly created session.

        Raises:
            ValueError: If the provided session ID is empty.
        """
        if session_id is not None:
            session_id = session_id.strip()

            if not session_id:
                raise ValueError("session_id cannot be empty.")

        resolved_session_id = session_id or str(uuid4())
        now = self._utc_now()

        session = Session(
            session_id=resolved_session_id,
            created_at=now,
            updated_at=now,
        )

        with self._lock:
            if resolved_session_id in self._sessions:
                raise SessionStoreError(
                    f"Session '{resolved_session_id}' already exists."
                )

            self._sessions[resolved_session_id] = session

        return self._copy_session(session)

    def get_session(self, session_id: str) -> Session:
        """
        Retrieve an existing session.

        Args:
            session_id: Session identifier.

        Returns:
            A copy of the requested session.

        Raises:
            SessionNotFoundError: If the session does not exist.
        """
        normalized_id = self._validate_session_id(session_id)

        with self._lock:
            session = self._sessions.get(normalized_id)

            if session is None:
                raise SessionNotFoundError(
                    f"Session '{normalized_id}' was not found."
                )

            return self._copy_session(session)

    def get_or_create_session(self, session_id: str | None = None) -> Session:
        """
        Retrieve a session or create it when no session ID is provided.

        Args:
            session_id: Optional existing session identifier.

        Returns:
            Existing or newly created session.
        """
        if session_id is None:
            return self.create_session()

        normalized_id = self._validate_session_id(session_id)

        with self._lock:
            if normalized_id in self._sessions:
                return self._copy_session(self._sessions[normalized_id])

        return self.create_session(normalized_id)

    def add_message(
        self,
        session_id: str,
        role: MessageRole,
        content: str,
    ) -> ChatMessage:
        """
        Append a message to an existing session.

        Args:
            session_id: Session identifier.
            role: Either ``user`` or ``assistant``.
            content: Message content.

        Returns:
            The stored message.

        Raises:
            SessionNotFoundError: If the session does not exist.
            InvalidMessageError: If role or content is invalid.
            SessionLimitError: If the session is already full.
        """
        normalized_id = self._validate_session_id(session_id)

        if role not in {"user", "assistant"}:
            raise InvalidMessageError(
                "role must be either 'user' or 'assistant'."
            )

        normalized_content = content.strip()

        if not normalized_content:
            raise InvalidMessageError(
                "Message content cannot be empty."
            )

        now = self._utc_now()
        message = ChatMessage(
            role=role,
            content=normalized_content,
            timestamp=now,
        )

        with self._lock:
            session = self._sessions.get(normalized_id)

            if session is None:
                raise SessionNotFoundError(
                    f"Session '{normalized_id}' was not found."
                )

            if len(session.messages) >= self._max_messages_per_session:
                raise SessionLimitError(
                    f"Session '{normalized_id}' has reached its maximum "
                    f"of {self._max_messages_per_session} messages."
                )

            session.messages.append(message)
            session.updated_at = now

        return message

    def add_user_message(
        self,
        session_id: str,
        content: str,
    ) -> ChatMessage:
        """Append a user message to a session."""
        return self.add_message(session_id, "user", content)

    def add_assistant_message(
        self,
        session_id: str,
        content: str,
    ) -> ChatMessage:
        """Append an assistant message to a session."""
        return self.add_message(session_id, "assistant", content)

    def get_history(
        self,
        session_id: str,
        limit: int | None = None,
    ) -> list[ChatMessage]:
        """
        Retrieve conversation history in chronological order.

        Args:
            session_id: Session identifier.
            limit: Optional number of most recent messages to return.

        Returns:
            List of conversation messages.

        Raises:
            SessionNotFoundError: If the session does not exist.
            ValueError: If limit is invalid.
        """
        if limit is not None and limit < 1:
            raise ValueError("limit must be at least 1.")

        session = self.get_session(session_id)

        if limit is None:
            return list(session.messages)

        return list(session.messages[-limit:])

    def reset_session(self, session_id: str) -> Session:
        """
        Remove all messages from an existing session.

        The session ID remains the same.

        Args:
            session_id: Session identifier.

        Returns:
            Reset session.

        Raises:
            SessionNotFoundError: If the session does not exist.
        """
        normalized_id = self._validate_session_id(session_id)
        now = self._utc_now()

        with self._lock:
            session = self._sessions.get(normalized_id)

            if session is None:
                raise SessionNotFoundError(
                    f"Session '{normalized_id}' was not found."
                )

            session.messages.clear()
            session.updated_at = now

            return self._copy_session(session)

    def delete_session(self, session_id: str) -> None:
        """
        Permanently remove a session.

        Args:
            session_id: Session identifier.

        Raises:
            SessionNotFoundError: If the session does not exist.
        """
        normalized_id = self._validate_session_id(session_id)

        with self._lock:
            if normalized_id not in self._sessions:
                raise SessionNotFoundError(
                    f"Session '{normalized_id}' was not found."
                )

            del self._sessions[normalized_id]

    def session_exists(self, session_id: str) -> bool:
        """Return whether a session exists."""
        normalized_id = self._validate_session_id(session_id)

        with self._lock:
            return normalized_id in self._sessions

    def session_count(self) -> int:
        """Return the number of active sessions."""
        with self._lock:
            return len(self._sessions)

    @staticmethod
    def _validate_session_id(session_id: str) -> str:
        """Validate and normalize a session ID."""
        if not isinstance(session_id, str):
            raise ValueError("session_id must be a string.")

        normalized_id = session_id.strip()

        if not normalized_id:
            raise ValueError("session_id cannot be empty.")

        return normalized_id

    @staticmethod
    def _utc_now() -> datetime:
        """Return the current timezone-aware UTC timestamp."""
        return datetime.now(timezone.utc)

    @staticmethod
    def _copy_session(session: Session) -> Session:
        """Create a defensive copy of a session."""
        return Session(
            session_id=session.session_id,
            created_at=session.created_at,
            updated_at=session.updated_at,
            messages=list(session.messages),
        )


_session_store = SessionStore()


def get_session_store() -> SessionStore:
    """
    Return the application's shared session store.

    Returns:
        Singleton in-memory session store.
    """
    return _session_store