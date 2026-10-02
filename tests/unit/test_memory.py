from datetime import datetime, timezone

import pytest

from rag_engine.memory.query_contextualizer import (
    ContextualizedQuery,
    InvalidQueryError,
    QueryContextualizer,
    contextualize_query,
)

from rag_engine.memory.session_store import (
    ChatMessage,
    InvalidMessageError,
    SessionLimitError,
    SessionNotFoundError,
    SessionStore,
    SessionStoreError,
)


# ============================================================================
# SessionStore
# ============================================================================


def test_session_store_initialization():
    """SessionStore should initialize with the configured message limit."""
    store = SessionStore(max_messages_per_session=10)

    assert store.max_messages_per_session == 10
    assert store.session_count() == 0


def test_session_store_rejects_invalid_message_limit():
    """The session message limit must be at least two."""
    with pytest.raises(ValueError):
        SessionStore(max_messages_per_session=1)

    with pytest.raises(ValueError):
        SessionStore(max_messages_per_session=0)


def test_create_session_with_generated_id():
    """A session without an explicit ID should receive a UUID."""
    store = SessionStore()

    session = store.create_session()

    assert session.session_id
    assert isinstance(session.session_id, str)
    assert session.messages == []