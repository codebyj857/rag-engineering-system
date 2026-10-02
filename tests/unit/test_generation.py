from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from rag_engine.config import Settings
from rag_engine.generation.generator import (
    GenerationConfigurationError,
    GenerationError,
    GenerationEvidence,
    GenerationResponseError,
    Generator,
)


# ---------------------------------------------------------------------------
# Fake Groq-compatible client
# ---------------------------------------------------------------------------


@dataclass
class FakeCompletionCall:
    model: str
    messages: list[dict[str, str]]
    temperature: float


class FakeCompletions:
    def __init__(
        self,
        response: object | None = None,
        error: Exception | None = None,
    ) -> None:
        self.response = response
        self.error = error
        self.calls: list[FakeCompletionCall] = []

    def create(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        temperature: float,
    ) -> object:
        self.calls.append(
            FakeCompletionCall(
                model=model,
                messages=messages,
                temperature=temperature,
            )
        )

        if self.error is not None:
            raise self.error

        if self.response is None:
            raise AssertionError("Fake response was not configured.")

        return self.response


class FakeChat:
    def __init__(
        self,
        response: object | None = None,
        error: Exception | None = None,
    ) -> None:
        self.completions = FakeCompletions(
            response=response,
            error=error,
        )


class FakeClient:
    def __init__(
        self,
        response: object | None = None,
        error: Exception | None = None,
    ) -> None:
        self.chat = FakeChat(
            response=response,
            error=error,
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_settings(**overrides: object) -> Settings:
    values: dict[str, object] = {
        "groq_api_key": "",
        "llm_model": "openai/gpt-oss-120b",
        "llm_temperature": 0.2,
    }
    values.update(overrides)
    return Settings(**values)


def make_response(
    *,
    answer: str = "Python is a programming language.",
    model: str = "openai/gpt-oss-120b",
    prompt_tokens: int = 100,
    completion_tokens: int = 20,
    total_tokens: int = 120,
) -> object:
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content=answer,
                )
            )
        ],
        usage=SimpleNamespace(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
        ),
        model=model,
    )


def make_evidence(
    *,
    content: str = "Python is a high-level programming language.",
    source: str = "document-1",
    filename: str = "python.txt",
    score: float = 0.91,
    rank: int = 1,
) -> GenerationEvidence:
    return GenerationEvidence(
        content=content,
        source=source,
        filename=filename,
        score=score,
        rank=rank,
    )


# ---------------------------------------------------------------------------
# Initialization / configuration
# ---------------------------------------------------------------------------


def test_generator_uses_settings() -> None:
    settings = make_settings(
        llm_model="test-model",
        llm_temperature=0.7,
    )

    generator = Generator(settings=settings)

    assert generator.model_name == "test-model"
    assert generator.temperature == 0.7


def test_injected_client_is_used() -> None:
    response = make_response()
    client = FakeClient(response=response)

    generator = Generator(
        settings=make_settings(),
        client=client,
    )

    result = generator.generate(
        question="What is Python?",
        evidence=[make_evidence()],
    )

    assert result.answer == "Python is a programming language."
    assert len(client.chat.completions.calls) == 1


def test_missing_api_key_raises_configuration_error() -> None:
    generator = Generator(
        settings=make_settings(groq_api_key=""),
    )

    with pytest.raises(GenerationConfigurationError):
        generator.generate(
            question="What is Python?",
            evidence=[make_evidence()],
        )


def test_injected_client_takes_priority_over_missing_api_key() -> None:
    response = make_response()
    client = FakeClient(response=response)

    generator = Generator(
        settings=make_settings(groq_api_key=""),
        client=client,
    )

    result = generator.generate(
        question="What is Python?",
        evidence=[make_evidence()],
    )

    assert result.answer == "Python is a programming language."


# ---------------------------------------------------------------------------
# Question validation
# ---------------------------------------------------------------------------


def test_question_is_stripped_before_generation() -> None:
    response = make_response()
    client = FakeClient(response=response)

    generator = Generator(
        settings=make_settings(),
        client=client,
    )

    generator.generate(
        question="   What is Python?   ",
        evidence=[],
    )

    call = client.chat.completions.calls[0]
    user_message = call.messages[1]["content"]

    assert "USER QUESTION:\nWhat is Python?" in user_message


def test_non_string_question_raises_generation_error() -> None:
    generator = Generator(
        settings=make_settings(),
        client=FakeClient(response=make_response()),
    )

    with pytest.raises(GenerationError, match="Question must be a string"):
        generator.generate(
            question=123,  # type: ignore[arg-type]
            evidence=[],
        )


@pytest.mark.parametrize(
    "question",
    [
        "",
        "   ",
        "\n\t",
    ],
)
def test_empty_question_raises_generation_error(question: str) -> None:
    generator = Generator(
        settings=make_settings(),
        client=FakeClient(response=make_response()),
    )

    with pytest.raises(GenerationError, match="Question cannot be empty"):
        generator.generate(
            question=question,
            evidence=[],
        )


# ---------------------------------------------------------------------------
# Evidence formatting
# ---------------------------------------------------------------------------


def test_format_evidence_returns_no_evidence_message_when_empty() -> None:
    result = Generator._format_evidence([])

    assert result == "No retrieved evidence is available."


def test_format_evidence_contains_all_required_fields() -> None:
    evidence = [
        make_evidence(
            content="First document content.",
            source="source-a",
            filename="first.txt",
            score=0.87654,
            rank=1,
        ),
        make_evidence(
            content="Second document content.",
            source="source-b",
            filename="second.txt",
            score=0.65432,
            rank=2,
        ),
    ]

    result = Generator._format_evidence(evidence)

    assert "[Evidence 1]" in result
    assert "Source: source-a" in result
    assert "Filename: first.txt" in result
    assert "Relevance score: 0.8765" in result
    assert "First document content." in result

    assert "[Evidence 2]" in result
    assert "Source: source-b" in result
    assert "Filename: second.txt" in result
    assert "Relevance score: 0.6543" in result
    assert "Second document content." in result


def test_format_evidence_preserves_evidence_order() -> None:
    evidence = [
        make_evidence(
            content="FIRST",
            rank=1,
        ),
        make_evidence(
            content="SECOND",
            rank=2,
        ),
        make_evidence(
            content="THIRD",
            rank=3,
        ),
    ]

    result = Generator._format_evidence(evidence)

    assert result.index("FIRST") < result.index("SECOND")
    assert result.index("SECOND") < result.index("THIRD")


# ---------------------------------------------------------------------------
# Message construction
# ---------------------------------------------------------------------------


def test_build_messages_contains_system_and_user_messages() -> None:
    generator = Generator(settings=make_settings())

    messages = generator._build_messages(
        question="What is Python?",
        evidence=[make_evidence()],
        conversation_context=None,
    )

    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"

    assert generator.SYSTEM_PROMPT == messages[0]["content"]


def test_build_messages_contains_evidence_and_question() -> None:
    generator = Generator(settings=make_settings())

    messages = generator._build_messages(
        question="What is Python?",
        evidence=[
            make_evidence(
                content="Python is a programming language.",
            )
        ],
        conversation_context=None,
    )

    user_content = messages[1]["content"]

    assert "RETRIEVED EVIDENCE:" in user_content
    assert "Python is a programming language." in user_content
    assert "USER QUESTION:" in user_content
    assert "What is Python?" in user_content
    assert "ANSWER USING THE EVIDENCE ABOVE:" in user_content


def test_build_messages_includes_conversation_context() -> None:
    generator = Generator(settings=make_settings())

    messages = generator._build_messages(
        question="What about its creator?",
        evidence=[make_evidence()],
        conversation_context="User previously asked about Python.",
    )

    user_content = messages[1]["content"]

    assert "CONVERSATION CONTEXT:" in user_content
    assert "User previously asked about Python." in user_content


def test_blank_conversation_context_is_not_included() -> None:
    generator = Generator(settings=make_settings())

    messages = generator._build_messages(
        question="What is Python?",
        evidence=[make_evidence()],
        conversation_context="   ",
    )

    user_content = messages[1]["content"]

    assert "CONVERSATION CONTEXT:" not in user_content


# ---------------------------------------------------------------------------
# Successful generation
# ---------------------------------------------------------------------------


def test_generate_returns_structured_result() -> None:
    response = make_response(
        answer="  Python is a programming language.  ",
        model="test-model",
        prompt_tokens=120,
        completion_tokens=30,
        total_tokens=150,
    )
    client = FakeClient(response=response)

    generator = Generator(
        settings=make_settings(
            llm_model="test-model",
            llm_temperature=0.4,
        ),
        client=client,
    )

    result = generator.generate(
        question="What is Python?",
        evidence=[make_evidence()],
    )

    assert result.answer == "Python is a programming language."
    assert result.model == "test-model"
    assert result.prompt_tokens == 120
    assert result.completion_tokens == 30
    assert result.total_tokens == 150


def test_generate_passes_model_temperature_and_messages_to_client() -> None:
    response = make_response()
    client = FakeClient(response=response)

    generator = Generator(
        settings=make_settings(
            llm_model="test-model",
            llm_temperature=0.65,
        ),
        client=client,
    )

    generator.generate(
        question="What is Python?",
        evidence=[make_evidence()],
    )

    call = client.chat.completions.calls[0]

    assert call.model == "test-model"
    assert call.temperature == 0.65
    assert len(call.messages) == 2
    assert call.messages[0]["role"] == "system"
    assert call.messages[1]["role"] == "user"


def test_generate_uses_conversation_context() -> None:
    response = make_response()
    client = FakeClient(response=response)

    generator = Generator(
        settings=make_settings(),
        client=client,
    )

    generator.generate(
        question="What about its creator?",
        evidence=[make_evidence()],
        conversation_context="The previous topic was Python.",
    )

    user_content = client.chat.completions.calls[0].messages[1]["content"]

    assert "CONVERSATION CONTEXT:" in user_content
    assert "The previous topic was Python." in user_content
    assert "What about its creator?" in user_content


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------


def test_parse_response_handles_missing_usage_fields() -> None:
    response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content="A valid answer.",
                )
            )
        ],
        usage=SimpleNamespace(),
        model="test-model",
    )

    result = Generator._parse_response(response)

    assert result.answer == "A valid answer."
    assert result.model == "test-model"
    assert result.prompt_tokens == 0
    assert result.completion_tokens == 0
    assert result.total_tokens == 0


@pytest.mark.parametrize(
    "response",
    [
        SimpleNamespace(
            choices=[],
            usage=SimpleNamespace(),
            model="test-model",
        ),
        SimpleNamespace(
            usage=SimpleNamespace(),
            model="test-model",
        ),
        SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(),
                )
            ],
            usage=SimpleNamespace(),
            model="test-model",
        ),
    ],
)
def test_parse_response_rejects_invalid_structure(response: object) -> None:
    with pytest.raises(GenerationResponseError):
        Generator._parse_response(response)


@pytest.mark.parametrize(
    "answer",
    [
        "",
        "   ",
        "\n\t",
        None,
    ],
)
def test_parse_response_rejects_empty_answer(
    answer: object,
) -> None:
    response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content=answer,
                )
            )
        ],
        usage=SimpleNamespace(
            prompt_tokens=1,
            completion_tokens=1,
            total_tokens=2,
        ),
        model="test-model",
    )

    with pytest.raises(GenerationResponseError, match="empty answer"):
        Generator._parse_response(response)


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


def test_unexpected_client_error_is_wrapped_as_generation_error() -> None:
    client = FakeClient(
        error=RuntimeError("unexpected failure"),
    )

    generator = Generator(
        settings=make_settings(),
        client=client,
    )

    with pytest.raises(
        GenerationError,
        match="unexpected error occurred during generation",
    ):
        generator.generate(
            question="What is Python?",
            evidence=[make_evidence()],
        )


def test_get_generator_returns_generator() -> None:
    from rag_engine.generation.generator import get_generator

    generator = get_generator()

    assert isinstance(generator, Generator)