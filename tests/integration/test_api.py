from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from fastapi.testclient import TestClient

from rag_engine.api.routes.chat import ChatRequest
from rag_engine.api.routes.evaluate import EvaluationRequest
from rag_engine.api.routes.sessions import ResetSessionRequest
from rag_engine.config import get_settings
from rag_engine.main import create_app
from rag_engine.memory.session_store import get_session_store


def test_application_starts_and_registers_expected_routes() -> None:
    """Verify the FastAPI application starts with all expected routes."""

    app = create_app()
    client = TestClient(app)

    response = client.get("/health")

    assert response.status_code == 200

    payload = response.json()

    assert payload["status"] == "healthy"
    assert payload["service"] == "rag-engineering-system"
    assert "timestamp" in payload

    paths = set(app.openapi()["paths"].keys())

    assert "/health" in paths
    assert "/ingest" in paths
    assert "/chat" in paths
    assert "/retrieve" in paths
    assert "/sessions/new" in paths
    assert "/sessions/{session_id}" in paths
    assert "/sessions/reset" in paths
    assert "/evaluate" in paths


def test_health_endpoint_returns_expected_contract() -> None:
    """Verify the health endpoint response structure."""

    client = TestClient(create_app())

    response = client.get("/health")

    assert response.status_code == 200

    payload = response.json()

    assert set(payload.keys()) == {
        "status",
        "service",
        "timestamp",
    }

    assert payload["status"] == "healthy"
    assert payload["service"] == "rag-engineering-system"
    assert isinstance(payload["timestamp"], str)
    assert payload["timestamp"].endswith("+00:00")


def test_ingest_indexes_supported_text_file(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Verify a supported document is stored, chunked, and indexed."""

    settings = get_settings()

    monkeypatch.setattr(
        settings,
        "documents_dir",
        tmp_path,
    )

    client = TestClient(create_app())

    response = client.post(
        "/ingest",
        files={
            "file": (
                "knowledge.txt",
                b"RAG systems retrieve relevant evidence before generation.",
                "text/plain",
            )
        },
    )

    assert response.status_code == 201

    payload = response.json()

    assert payload["status"] == "indexed"
    assert payload["filename"] == "knowledge.txt"
    assert payload["extension"] == ".txt"
    assert payload["message"] == (
        "Document uploaded and indexed successfully."
    )
    assert isinstance(payload["document_id"], str)
    assert payload["document_id"]
    assert payload["chunk_count"] >= 1
    assert payload["indexed_count"] == payload["chunk_count"]

    stored_filename = payload["stored_filename"]

    assert stored_filename.endswith("_knowledge.txt")
    assert (tmp_path / stored_filename).exists()

    assert (
        (tmp_path / stored_filename).read_bytes()
        == b"RAG systems retrieve relevant evidence before generation."
    )


def test_ingest_rejects_unsupported_file_type() -> None:
    """Verify unsupported document extensions are rejected."""

    client = TestClient(create_app())

    response = client.post(
        "/ingest",
        files={
            "file": (
                "malware.exe",
                b"not a supported document",
                "application/octet-stream",
            )
        },
    )

    assert response.status_code == 400

    payload = response.json()

    assert "Unsupported file type" in payload["detail"]
    assert ".exe" in payload["detail"]


def test_ingest_rejects_empty_file(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Verify empty uploads are rejected."""

    settings = get_settings()

    monkeypatch.setattr(
        settings,
        "documents_dir",
        tmp_path,
    )

    client = TestClient(create_app())

    response = client.post(
        "/ingest",
        files={
            "file": (
                "empty.txt",
                b"",
                "text/plain",
            )
        },
    )

    assert response.status_code == 400
    assert response.json()["detail"] == (
        "Uploaded file is empty."
    )

    assert list(tmp_path.iterdir()) == []


def test_retrieve_returns_results_with_explicit_configuration() -> None:
    """Verify connected retrieval returns evidence."""

    client = TestClient(create_app())

    response = client.post(
        "/retrieve",
        json={
            "question": "What is retrieval augmented generation?",
            "strategy": "hybrid",
            "top_k": 7,
        },
    )

    assert response.status_code == 200

    payload = response.json()

    assert payload["question"] == (
        "What is retrieval augmented generation?"
    )
    assert payload["strategy"] == "hybrid"
    assert payload["top_k"] == 7
    assert isinstance(payload["results"], list)
    assert len(payload["results"]) <= 7

    if payload["results"]:
        result = payload["results"][0]

        assert set(result.keys()) == {
            "chunk_id",
            "content",
            "score",
            "source",
        }
        assert isinstance(result["chunk_id"], str)
        assert isinstance(result["content"], str)
        assert isinstance(result["score"], float)
        assert isinstance(result["source"], str)


def test_retrieve_normalizes_question() -> None:
    """Verify retrieval question whitespace is normalized."""

    client = TestClient(create_app())

    response = client.post(
        "/retrieve",
        json={
            "question": "   Explain vector databases.   ",
            "strategy": "naive",
            "top_k": 3,
        },
    )

    assert response.status_code == 200

    payload = response.json()

    assert payload["question"] == "Explain vector databases."
    assert payload["strategy"] == "naive"
    assert payload["top_k"] == 3
    assert len(payload["results"]) <= 3


def test_retrieve_rejects_invalid_strategy() -> None:
    """Verify unsupported retrieval strategies are rejected."""

    client = TestClient(create_app())

    response = client.post(
        "/retrieve",
        json={
            "question": "What is RAG?",
            "strategy": "invalid",
            "top_k": 5,
        },
    )

    assert response.status_code == 422


def test_retrieve_rejects_invalid_top_k() -> None:
    """Verify retrieval top_k validation."""

    client = TestClient(create_app())

    response = client.post(
        "/retrieve",
        json={
            "question": "What is RAG?",
            "strategy": "naive",
            "top_k": 0,
        },
    )

    assert response.status_code == 422


def test_chat_requires_session_id() -> None:
    """Verify chat rejects requests without a session ID."""

    client = TestClient(create_app())

    response = client.post(
        "/chat",
        json={
            "message": "What is RAG?",
            "strategy": "naive",
            "top_k": 5,
        },
    )

    assert response.status_code == 400

    assert response.json()["detail"] == (
        "A valid session_id is required."
    )


def test_chat_connected_pipeline_contract_without_groq(
    monkeypatch,
) -> None:
    """
    Verify the connected chat route contract without making a Groq call.

    A small fake pipeline is injected so this test validates API mapping
    while preserving the real API quota for final browser testing.
    """

    client = TestClient(create_app())

    session = get_session_store().create_session()

    fake_verification = SimpleNamespace(
        grounded=True,
        score=0.95,
    )

    fake_evidence = (
        SimpleNamespace(
            chunk_id="test-chunk",
            content="RAG retrieves relevant evidence before generation.",
            score=0.91,
            source="test-source",
        ),
    )

    fake_confidence = SimpleNamespace(
        score=0.92,
    )

    fake_result = SimpleNamespace(
        session_id=session.session_id,
        strategy="reranked",
        answer="RAG retrieves relevant evidence before generation.",
        evidence=fake_evidence,
        verification=fake_verification,
        confidence=fake_confidence,
    )

    class FakePipeline:
        def run(self, request):
            assert request.question == "What is RAG?"
            assert request.session_id == session.session_id
            assert request.top_k == 5
            assert request.verify is True
            return fake_result

    monkeypatch.setattr(
        "rag_engine.api.routes.chat.get_pipeline",
        lambda strategy: FakePipeline(),
    )

    response = client.post(
        "/chat",
        json={
            "message": "What is RAG?",
            "session_id": session.session_id,
            "strategy": "reranked",
            "top_k": 5,
            "verify": True,
        },
    )

    assert response.status_code == 200

    payload = response.json()

    assert payload["answer"] == (
        "RAG retrieves relevant evidence before generation."
    )
    assert payload["session_id"] == session.session_id
    assert payload["strategy"] == "reranked"
    assert payload["confidence"] == 0.92

    assert payload["evidence"] == [
        {
            "chunk_id": "test-chunk",
            "content": (
                "RAG retrieves relevant evidence before generation."
            ),
            "score": 0.91,
            "source": "test-source",
        }
    ]

    assert payload["verification"] == {
        "enabled": True,
        "grounded": True,
        "score": 0.95,
        "explanation": None,
    }


def test_chat_normalizes_message() -> None:
    """Verify ChatRequest normalization through the HTTP layer."""

    request = ChatRequest(
        message="   What is RAG?   ",
        session_id="   session-123   ",
    )

    assert request.message == "What is RAG?"
    assert request.session_id == "session-123"


def test_chat_rejects_empty_message() -> None:
    """Verify empty chat messages are rejected by Pydantic validation."""

    client = TestClient(create_app())

    response = client.post(
        "/chat",
        json={
            "message": "   ",
            "session_id": "session-123",
        },
    )

    assert response.status_code == 422


def test_create_session_returns_session_id() -> None:
    """Verify a new conversation session can be created."""

    client = TestClient(create_app())

    response = client.post("/sessions/new")

    assert response.status_code == 201

    payload = response.json()

    assert set(payload.keys()) == {
        "session_id",
        "created_at",
    }

    assert isinstance(payload["session_id"], str)
    assert payload["session_id"]
    assert isinstance(payload["created_at"], str)
    assert payload["created_at"].endswith("+00:00")


def test_get_session_returns_created_session() -> None:
    """Verify an existing session can be retrieved."""

    client = TestClient(create_app())

    create_response = client.post("/sessions/new")

    assert create_response.status_code == 201

    session_id = create_response.json()["session_id"]

    response = client.get(
        f"/sessions/{session_id}",
    )

    assert response.status_code == 200

    payload = response.json()

    assert payload["session_id"] == session_id
    assert isinstance(payload["created_at"], str)
    assert isinstance(payload["updated_at"], str)
    assert payload["messages"] == []


def test_get_session_returns_not_found_for_unknown_session() -> None:
    """Verify unknown sessions return HTTP 404."""

    client = TestClient(create_app())

    response = client.get(
        "/sessions/does-not-exist",
    )

    assert response.status_code == 404
    assert response.json()["detail"] == "Session not found."


def test_get_session_rejects_whitespace_only_id() -> None:
    """
    Verify invalid session identifiers are rejected.

    FastAPI path parameters cannot normally represent an empty path segment,
    so a whitespace-only path is used to exercise the route validation.
    """

    client = TestClient(create_app())

    response = client.get(
        "/sessions/%20%20%20",
    )

    assert response.status_code == 400
    assert response.json()["detail"] == (
        "session_id cannot be empty."
    )


def test_reset_session_resets_existing_session() -> None:
    """Verify an existing conversation session can be reset."""

    client = TestClient(create_app())

    create_response = client.post("/sessions/new")

    assert create_response.status_code == 201

    session_id = create_response.json()["session_id"]

    response = client.post(
        "/sessions/reset",
        json={
            "session_id": session_id,
        },
    )

    assert response.status_code == 200

    payload = response.json()

    assert payload == {
        "session_id": session_id,
        "status": "reset",
    }

    get_response = client.get(
        f"/sessions/{session_id}",
    )

    assert get_response.status_code == 200

    session_payload = get_response.json()

    assert session_payload["session_id"] == session_id
    assert session_payload["messages"] == []


def test_reset_session_returns_not_found_for_unknown_session() -> None:
    """Verify resetting an unknown session returns HTTP 404."""

    client = TestClient(create_app())

    response = client.post(
        "/sessions/reset",
        json={
            "session_id": "integration-session",
        },
    )

    assert response.status_code == 404
    assert response.json()["detail"] == "Session not found."


def test_reset_session_normalizes_session_id() -> None:
    """Verify ResetSessionRequest strips surrounding whitespace."""

    request = ResetSessionRequest(
        session_id="   session-123   ",
    )

    assert request.session_id == "session-123"


def test_evaluate_returns_not_implemented() -> None:
    """Verify the evaluation API contract remains intentionally disconnected."""

    client = TestClient(create_app())

    response = client.post(
        "/evaluate",
        json={
            "strategy": "hybrid",
            "top_k": 10,
            "full_rag": True,
        },
    )

    assert response.status_code == 501

    assert response.json()["detail"] == (
        "Evaluation is not connected yet. "
        "The evaluation dataset and evaluator must be "
        "initialized first."
    )


def test_evaluate_accepts_retrieval_only_mode() -> None:
    """Verify evaluation request validation for retrieval-only mode."""

    request = EvaluationRequest(
        strategy="naive",
        top_k=3,
        full_rag=False,
    )

    assert request.strategy == "naive"
    assert request.top_k == 3
    assert request.full_rag is False


def test_cors_allows_configured_frontend_origin() -> None:
    """Verify application-level CORS middleware uses the configured origin."""

    settings = get_settings()

    client = TestClient(create_app())

    response = client.get(
        "/health",
        headers={
            "Origin": settings.frontend_url,
        },
    )

    assert response.status_code == 200

    assert response.headers.get(
        "access-control-allow-origin"
    ) == settings.frontend_url

    assert response.headers.get(
        "access-control-allow-credentials"
    ) == "true"