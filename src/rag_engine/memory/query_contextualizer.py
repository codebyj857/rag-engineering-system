"""
Query contextualization for multi-turn RAG conversations.

This module converts follow-up user questions into standalone queries
using conversation history.

The contextualizer is intentionally deterministic at this layer. It does
not depend directly on an LLM or generation provider. This keeps the memory
package modular and allows an LLM-powered contextualizer to be introduced
later without changing the session-store contract.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Sequence

from rag_engine.memory.session_store import ChatMessage


class QueryContextualizationError(Exception):
    """Base exception for query contextualization errors."""


class InvalidQueryError(QueryContextualizationError):
    """Raised when the input query is invalid."""


@dataclass(frozen=True, slots=True)
class ContextualizedQuery:
    """
    Result of contextualizing a user query.

    Attributes:
        original_query: The exact query supplied by the user.
        standalone_query: Query suitable for retrieval.
        used_history: Whether conversation history influenced the result.
        history_messages_used: Number of history messages considered.
    """

    original_query: str
    standalone_query: str
    used_history: bool
    history_messages_used: int


class QueryContextualizer:
    """
    Converts conversational follow-up questions into standalone queries.

    The current implementation uses deterministic heuristics rather than
    an LLM. This makes contextualization predictable, inexpensive, and
    available even when no LLM API key is configured.

    Args:
        max_history_messages: Maximum number of previous messages examined.
    """

    _FOLLOW_UP_PATTERNS = (
        r"^\s*what about\b",
        r"^\s*how about\b",
        r"^\s*what\s+is\s+its\b",
        r"^\s*what\s+are\s+its\b",
        r"^\s*what\s+does\s+it\b",
        r"^\s*how\s+does\s+it\b",
        r"^\s*why\s+does\s+it\b",
        r"^\s*why\s+is\s+it\b",
        r"^\s*can\s+it\b",
        r"^\s*does\s+it\b",
        r"^\s*is\s+it\b",
        r"^\s*are\s+they\b",
        r"^\s*do\s+they\b",
        r"^\s*and\s+what\b",
        r"^\s*and\s+how\b",
        r"^\s*and\s+why\b",
        r"^\s*then\s+what\b",
        r"^\s*what\s+about\s+that\b",
    )

    _REFERENCE_TERMS = {
        "it",
        "its",
        "they",
        "them",
        "their",
        "this",
        "that",
        "these",
        "those",
        "he",
        "him",
        "his",
        "she",
        "her",
    }

    def __init__(self, max_history_messages: int = 6) -> None:
        if max_history_messages < 1:
            raise ValueError(
                "max_history_messages must be at least 1."
            )

        self._max_history_messages = max_history_messages

    @property
    def max_history_messages(self) -> int:
        """Return the maximum number of history messages considered."""
        return self._max_history_messages

    def contextualize(
        self,
        query: str,
        history: Sequence[ChatMessage],
    ) -> ContextualizedQuery:
        """
        Convert a conversational query into a standalone query.

        Args:
            query: Current user query.
            history: Previous conversation messages.

        Returns:
            ContextualizedQuery containing the retrieval-ready query.

        Raises:
            InvalidQueryError: If the query is empty or invalid.
        """
        normalized_query = self._validate_query(query)

        relevant_history = list(history[-self._max_history_messages :])

        if not relevant_history:
            return ContextualizedQuery(
                original_query=normalized_query,
                standalone_query=normalized_query,
                used_history=False,
                history_messages_used=0,
            )

        if not self._needs_context(normalized_query):
            return ContextualizedQuery(
                original_query=normalized_query,
                standalone_query=normalized_query,
                used_history=False,
                history_messages_used=0,
            )

        context = self._build_context(relevant_history)

        if not context:
            return ContextualizedQuery(
                original_query=normalized_query,
                standalone_query=normalized_query,
                used_history=False,
                history_messages_used=0,
            )

        standalone_query = self._combine_context(
            normalized_query,
            context,
        )

        return ContextualizedQuery(
            original_query=normalized_query,
            standalone_query=standalone_query,
            used_history=True,
            history_messages_used=len(relevant_history),
        )

    def _needs_context(self, query: str) -> bool:
        """
        Determine whether a query likely depends on previous conversation.
        """
        normalized = query.lower().strip()

        for pattern in self._FOLLOW_UP_PATTERNS:
            if re.search(pattern, normalized):
                return True

        tokens = self._tokenize(normalized)

        return bool(tokens.intersection(self._REFERENCE_TERMS))

    @staticmethod
    def _build_context(history: Sequence[ChatMessage]) -> str:
        """
        Build compact textual context from conversation history.

        Only non-empty messages are included. The context is deliberately
        compact so it can later be passed to an LLM if desired.
        """
        lines: list[str] = []

        for message in history:
            content = message.content.strip()

            if not content:
                continue

            role = "User" if message.role == "user" else "Assistant"
            lines.append(f"{role}: {content}")

        return "\n".join(lines)

    @staticmethod
    def _combine_context(query: str, context: str) -> str:
        """
        Produce a retrieval-friendly query containing conversation context.

        The format keeps the original question intact while providing the
        previous conversation needed to resolve references.
        """
        return (
            "Conversation context:\n"
            f"{context}\n\n"
            "Current question:\n"
            f"{query}"
        )

    @staticmethod
    def _validate_query(query: str) -> str:
        """Validate and normalize a user query."""
        if not isinstance(query, str):
            raise InvalidQueryError("Query must be a string.")

        normalized = query.strip()

        if not normalized:
            raise InvalidQueryError("Query cannot be empty.")

        return normalized

    @staticmethod
    def _tokenize(text: str) -> set[str]:
        """Extract normalized word tokens from text."""
        return set(re.findall(r"\b[a-zA-Z]+\b", text.lower()))


def contextualize_query(
    query: str,
    history: Sequence[ChatMessage],
    max_history_messages: int = 6,
) -> ContextualizedQuery:
    """
    Convenience function for query contextualization.

    Args:
        query: Current user query.
        history: Previous conversation messages.
        max_history_messages: Maximum history messages to inspect.

    Returns:
        ContextualizedQuery result.
    """
    contextualizer = QueryContextualizer(
        max_history_messages=max_history_messages
    )

    return contextualizer.contextualize(query, history)