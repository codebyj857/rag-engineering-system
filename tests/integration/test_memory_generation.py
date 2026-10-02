from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from rag_engine.generation.generator import (
    GenerationEvidence,
    GenerationResult,
    Generator,
)
from rag_engine.memory.query_contextualizer import (
    ContextualizedQuery,
    InvalidQueryError,
    QueryContextualizer,
)
from rag_engine.memory.session_store import (
    SessionStore,
)


@dataclass
class FakeCompletions:
    """Fake chat-completions interface for offline generation tests."""

    answer: str = (
        "Python is a programming language used for machine learning."
    )
    model: str = "test-model"
    prompt_tokens: int = 120
    completion_tokens: int = 20
    total_tokens: int = 140

    def create(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        temperature: float,
    ) -> SimpleNamespace:
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content=self.answer,
                    )
                )
            ],
            usage=SimpleNamespace(
                prompt_tokens=self.prompt_tokens,
                completion_tokens=self.completion_tokens,
                total_tokens=self.total_tokens,
            ),
            model=self.model,
        )


class FakeChat:
    """Fake chat interface matching the Groq client structure."""

    def __init__(self, completions: FakeCompletions) -> None:
        self.completions = completions


class FakeGroqClient:
    """Minimal Groq-compatible client used by Generator."""

    def __init__(
        self,
        answer: str = (
            "Python is a programming language used for machine learning."
        ),
    ) -> None:
        completions = FakeCompletions(
            answer=answer,
        )

        self.chat = FakeChat(
            completions=completions,
        )


def test_session_store_to_contextualizer_flow() -> None:
    """
    Verify the complete conversation-memory flow:

    session -> user/assistant messages -> history ->
    follow-up query -> contextualized retrieval query
    """

    store = SessionStore(
        max_messages_per_session=10,
    )

    session = store.create_session(
        session_id="integration-memory-session",
    )

    assert session.session_id == "integration-memory-session"
    assert session.messages == []

    store.add_user_message(
        session.session_id,
        "What is retrieval augmented generation?",
    )

    store.add_assistant_message(
        session.session_id,
        "RAG combines retrieval with language model generation.",
    )

    history = store.get_history(
        session.session_id,
    )

    assert len(history) == 2
    assert history[0].role == "user"
    assert history[1].role == "assistant"

    contextualizer = QueryContextualizer(
        max_history_messages=6,
    )

    result = contextualizer.contextualize(
        query="What about its benefits?",
        history=history,
    )

    assert isinstance(
        result,
        ContextualizedQuery,
    )

    assert result.original_query == "What about its benefits?"
    assert result.used_history is True
    assert result.history_messages_used == 2

    assert (
        "Conversation context:"
        in result.standalone_query
    )

    assert (
        "What is retrieval augmented generation?"
        in result.standalone_query
    )

    assert (
        "RAG combines retrieval with language model generation."
        in result.standalone_query
    )

    assert (
        "Current question:"
        in result.standalone_query
    )

    assert (
        "What about its benefits?"
        in result.standalone_query
    )


def test_contextualizer_leaves_independent_questions_unchanged() -> None:
    """Verify that standalone questions do not unnecessarily use history."""

    store = SessionStore()

    session = store.create_session()

    store.add_user_message(
        session.session_id,
        "What is Python?",
    )

    history = store.get_history(
        session.session_id,
    )

    contextualizer = QueryContextualizer()

    result = contextualizer.contextualize(
        query="Explain vector databases.",
        history=history,
    )

    assert result.original_query == (
        "Explain vector databases."
    )

    assert result.standalone_query == (
        "Explain vector databases."
    )

    assert result.used_history is False
    assert result.history_messages_used == 0


def test_contextualizer_without_history() -> None:
    """Verify contextualization with no previous conversation."""

    contextualizer = QueryContextualizer()

    result = contextualizer.contextualize(
        query="What is machine learning?",
        history=[],
    )

    assert result.original_query == (
        "What is machine learning?"
    )

    assert result.standalone_query == (
        "What is machine learning?"
    )

    assert result.used_history is False
    assert result.history_messages_used == 0


def test_generator_uses_injected_client_without_api_call() -> None:
    """
    Verify generation through an injected fake client.

    No Groq API key or network request is required.
    """

    client = FakeGroqClient(
        answer=(
            "Python is a programming language "
            "commonly used for machine learning."
        ),
    )

    generator = Generator(
        client=client,
    )

    evidence = [
        GenerationEvidence(
            content=(
                "Python is a programming language commonly "
                "used for machine learning."
            ),
            source="integration-test",
            filename="python.txt",
            score=0.95,
            rank=1,
        )
    ]

    result = generator.generate(
        question="What is Python used for?",
        evidence=evidence,
    )

    assert isinstance(
        result,
        GenerationResult,
    )

    assert result.answer == (
        "Python is a programming language "
        "commonly used for machine learning."
    )

    assert result.model == "test-model"
    assert result.prompt_tokens == 120
    assert result.completion_tokens == 20
    assert result.total_tokens == 140


def test_generator_includes_evidence_and_conversation_context() -> None:
    """Verify the generator builds the expected grounded prompt."""

    client = FakeGroqClient()

    generator = Generator(
        client=client,
    )

    evidence = [
        GenerationEvidence(
            content="RAG retrieves relevant documents before generation.",
            source="test-source",
            filename="rag.txt",
            score=0.91,
            rank=1,
        ),
        GenerationEvidence(
            content="Retrieved evidence is supplied to the generator.",
            source="test-source",
            filename="evidence.txt",
            score=0.82,
            rank=2,
        ),
    ]

    messages = generator._build_messages(
        question="How does RAG work?",
        evidence=evidence,
        conversation_context=(
            "User: What is retrieval augmented generation?\n"
            "Assistant: RAG retrieves relevant information first."
        ),
    )

    assert len(messages) == 2

    assert messages[0]["role"] == "system"
    assert messages[0]["content"] == generator.SYSTEM_PROMPT

    assert messages[1]["role"] == "user"

    prompt = messages[1]["content"]

    assert "CONVERSATION CONTEXT:" in prompt
    assert (
        "What is retrieval augmented generation?"
        in prompt
    )

    assert "RETRIEVED EVIDENCE:" in prompt
    assert (
        "RAG retrieves relevant documents before generation."
        in prompt
    )
    assert (
        "Retrieved evidence is supplied to the generator."
        in prompt
    )

    assert "USER QUESTION:" in prompt
    assert "How does RAG work?" in prompt
    assert "ANSWER USING THE EVIDENCE ABOVE:" in prompt


def test_memory_and_generation_work_together() -> None:
    """
    Verify the complete local multi-turn generation preparation flow:

    session history -> contextualization -> evidence ->
    generation with conversation context.
    """

    store = SessionStore()

    session = store.create_session(
        session_id="memory-generation-session",
    )

    store.add_user_message(
        session.session_id,
        "What is a vector database?",
    )

    store.add_assistant_message(
        session.session_id,
        "A vector database stores embeddings for similarity search.",
    )

    history = store.get_history(
        session.session_id,
    )

    contextualizer = QueryContextualizer()

    contextualized = contextualizer.contextualize(
        query="How does it help retrieval?",
        history=history,
    )

    assert contextualized.used_history is True
    assert contextualized.history_messages_used == 2

    client = FakeGroqClient(
        answer=(
            "A vector database helps retrieval by storing "
            "embeddings that can be searched by similarity."
        ),
    )

    generator = Generator(
        client=client,
    )

    evidence = [
        GenerationEvidence(
            content=(
                "Vector databases store embeddings and allow "
                "similarity search over those embeddings."
            ),
            source="integration-test",
            filename="vector_databases.txt",
            score=0.94,
            rank=1,
        )
    ]

    result = generator.generate(
        question=contextualized.standalone_query,
        evidence=evidence,
        conversation_context=contextualized.standalone_query,
    )

    assert isinstance(
        result,
        GenerationResult,
    )

    assert result.answer

    assert "vector database" in result.answer.lower()
    assert result.total_tokens == (
        result.prompt_tokens
        + result.completion_tokens
    )


def test_invalid_contextualizer_query_is_rejected() -> None:
    """Verify invalid contextualizer input is rejected."""

    contextualizer = QueryContextualizer()

    with pytest.raises(InvalidQueryError):
        contextualizer.contextualize(
            query="   ",
            history=[],
        )


def test_invalid_generator_question_is_rejected() -> None:
    """Verify invalid generator input is rejected."""

    generator = Generator(
        client=FakeGroqClient(),
    )

    with pytest.raises(Exception):
        generator.generate(
            question="   ",
            evidence=[],
        )

