import pytest

from rag_engine.indexing.vector_store import VectorSearchResult
from rag_engine.retrieval.base import (
    RetrievalConfigurationError,
    RetrievalExecutionError,
    RetrievalResult,
    RetrievalStrategy,
)
from rag_engine.retrieval.hybrid import HybridRetriever
from rag_engine.retrieval.naive import NaiveRetriever
from rag_engine.retrieval.reranker import RerankedRetriever


class FakeVectorStore:
    """Deterministic fake vector store for retrieval unit tests."""

    def __init__(self, results=None, error=None):
        self.results = results or []
        self.error = error
        self.calls = []

    def search(self, query: str, top_k: int = 5):
        self.calls.append(
            {
                "query": query,
                "top_k": top_k,
            }
        )

        if self.error is not None:
            raise self.error

        return self.results[:top_k]


def make_vector_result(
    chunk_id: str,
    content: str,
    score: float,
    document_id: str = "doc-1",
) -> VectorSearchResult:
    """Create a deterministic vector search result."""
    return VectorSearchResult(
        chunk_id=chunk_id,
        content=content,
        source=f"data/{document_id}.txt",
        filename=f"{document_id}.txt",
        file_type=".txt",
        score=score,
        metadata={
            "document_id": document_id,
            "chunk_index": 0,
        },
    )


@pytest.fixture
def vector_results():
    return [
        make_vector_result(
            chunk_id="chunk-1",
            content="Python is a programming language used for machine learning.",
            score=0.95,
            document_id="doc-python",
        ),
        make_vector_result(
            chunk_id="chunk-2",
            content="Machine learning uses algorithms to learn patterns from data.",
            score=0.85,
            document_id="doc-ml",
        ),
        make_vector_result(
            chunk_id="chunk-3",
            content="Vector databases store embeddings for semantic search.",
            score=0.75,
            document_id="doc-vector",
        ),
        make_vector_result(
            chunk_id="chunk-4",
            content="Transformers are neural network architectures used in NLP.",
            score=0.65,
            document_id="doc-transformers",
        ),
    ]


# ---------------------------------------------------------------------------
# Base retrieval abstractions
# ---------------------------------------------------------------------------


def test_retrieval_strategy_requires_implementation():
    """The base retrieval strategy must remain abstract."""

    with pytest.raises(TypeError):
        RetrievalStrategy()


def test_retrieval_strategy_rejects_invalid_top_k():
    """A retrieval strategy cannot be configured with top_k < 1."""

    class DummyRetriever(RetrievalStrategy):
        name = "dummy"

        def retrieve(self, query, top_k=None):
            return self._build_response(
                query=query,
                strategy=self.name,
                results=[],
            )

    with pytest.raises(RetrievalConfigurationError):
        DummyRetriever(top_k=0)


def test_retrieval_strategy_resolves_top_k():
    """Per-request top_k overrides the configured default."""

    class DummyRetriever(RetrievalStrategy):
        name = "dummy"

        def retrieve(self, query, top_k=None):
            resolved = self._resolve_top_k(top_k)
            return self._build_response(
                query=query,
                strategy=self.name,
                results=[],
                metadata={"resolved_top_k": resolved},
            )

    retriever = DummyRetriever(top_k=5)

    assert retriever._resolve_top_k(None) == 5
    assert retriever._resolve_top_k(3) == 3

    with pytest.raises(RetrievalConfigurationError):
        retriever._resolve_top_k(0)


def test_retrieval_strategy_validates_query():
    """Whitespace-only queries must be rejected."""

    class DummyRetriever(RetrievalStrategy):
        name = "dummy"

        def retrieve(self, query, top_k=None):
            return self._build_response(
                query=self._validate_query(query),
                strategy=self.name,
                results=[],
            )

    retriever = DummyRetriever()

    with pytest.raises(RetrievalExecutionError):
        retriever._validate_query("   ")

    assert retriever._validate_query("  machine learning  ") == (
        "machine learning"
    )


def test_retrieval_result_from_vector_result(vector_results):
    """Vector results must convert into normalized retrieval results."""

    result = RetrievalResult.from_vector_result(
        vector_results[0],
        rank=1,
    )

    assert result.chunk_id == "chunk-1"
    assert result.content.startswith("Python")
    assert result.source == "data/doc-python.txt"
    assert result.filename == "doc-python.txt"
    assert result.file_type == ".txt"
    assert result.score == 0.95
    assert result.rank == 1
    assert result.metadata["document_id"] == "doc-python"


def test_retrieval_result_conversion_assigns_sequential_ranks(
    vector_results,
):
    """Converted results must receive one-based sequential ranks."""

    class DummyRetriever(RetrievalStrategy):
        name = "dummy"

        def retrieve(self, query, top_k=None):
            raise NotImplementedError

    results = DummyRetriever._convert_results(vector_results)

    assert [result.rank for result in results] == [1, 2, 3, 4]
    assert [result.chunk_id for result in results] == [
        "chunk-1",
        "chunk-2",
        "chunk-3",
        "chunk-4",
    ]


def test_build_response_sets_total_results():
    """Response total_results must match the number of results."""

    class DummyRetriever(RetrievalStrategy):
        name = "dummy"

        def retrieve(self, query, top_k=None):
            raise NotImplementedError

    results = [
        RetrievalResult(
            chunk_id="chunk-1",
            content="content",
            source="source.txt",
            filename="source.txt",
            file_type=".txt",
            score=0.9,
            rank=1,
        )
    ]

    response = DummyRetriever._build_response(
        query="test",
        strategy="dummy",
        results=results,
        metadata={"test": True},
    )

    assert response.query == "test"
    assert response.strategy == "dummy"
    assert response.total_results == 1
    assert response.metadata["test"] is True


# ---------------------------------------------------------------------------
# Naive retrieval
# ---------------------------------------------------------------------------


def test_naive_retriever_returns_vector_results(vector_results):
    """Naive retrieval should directly use vector similarity results."""

    store = FakeVectorStore(vector_results)
    retriever = NaiveRetriever(store, top_k=3)

    response = retriever.retrieve("machine learning")

    assert response.query == "machine learning"
    assert response.strategy == "naive"
    assert response.total_results == 3
    assert len(response.results) == 3

    assert response.results[0].chunk_id == "chunk-1"
    assert response.results[0].score == 0.95
    assert response.results[0].rank == 1

    assert response.metadata["retrieval_method"] == "vector_similarity"
    assert response.metadata["top_k"] == 3


def test_naive_retriever_strips_query(vector_results):
    """Naive retrieval should normalize whitespace around queries."""

    store = FakeVectorStore(vector_results)
    retriever = NaiveRetriever(store)

    response = retriever.retrieve("   Python   ")

    assert response.query == "Python"
    assert store.calls[0]["query"] == "Python"


def test_naive_retriever_honors_top_k_override(vector_results):
    """A request-level top_k should be forwarded to the vector store."""

    store = FakeVectorStore(vector_results)
    retriever = NaiveRetriever(store, top_k=5)

    response = retriever.retrieve(
        "machine learning",
        top_k=2,
    )

    assert response.metadata["top_k"] == 2
    assert store.calls[0]["top_k"] == 2
    assert len(response.results) == 2


def test_naive_retriever_rejects_empty_query(vector_results):
    """Naive retrieval should reject empty queries."""

    store = FakeVectorStore(vector_results)
    retriever = NaiveRetriever(store)

    with pytest.raises(RetrievalExecutionError):
        retriever.retrieve("   ")

    assert store.calls == []


def test_naive_retriever_rejects_invalid_top_k(vector_results):
    """Naive retrieval should reject invalid request-level top_k."""

    store = FakeVectorStore(vector_results)
    retriever = NaiveRetriever(store)

    with pytest.raises(RetrievalConfigurationError):
        retriever.retrieve("python", top_k=0)

    assert store.calls == []


def test_naive_retriever_wraps_vector_store_errors():
    """Vector-store failures should become RetrievalExecutionError."""

    store = FakeVectorStore(
        error=RuntimeError("vector store unavailable")
    )
    retriever = NaiveRetriever(store)

    with pytest.raises(
        RetrievalExecutionError,
        match="Naive semantic retrieval failed",
    ):
        retriever.retrieve("python")


def test_naive_retriever_handles_empty_results():
    """Naive retrieval should return a valid empty response."""

    store = FakeVectorStore([])
    retriever = NaiveRetriever(store)

    response = retriever.retrieve("python")

    assert response.results == []
    assert response.total_results == 0
    assert response.strategy == "naive"


# ---------------------------------------------------------------------------
# Hybrid retrieval
# ---------------------------------------------------------------------------


def test_hybrid_retriever_initializes_weights():
    """Hybrid weights should be normalized to sum to one."""

    store = FakeVectorStore()

    retriever = HybridRetriever(
        store,
        semantic_weight=7,
        lexical_weight=3,
    )

    assert retriever.semantic_weight == pytest.approx(0.7)
    assert retriever.lexical_weight == pytest.approx(0.3)
    assert (
        retriever.semantic_weight + retriever.lexical_weight
        == pytest.approx(1.0)
    )


def test_hybrid_retriever_rejects_negative_weights():
    """Negative hybrid weights are invalid."""

    store = FakeVectorStore()

    with pytest.raises(ValueError):
        HybridRetriever(
            store,
            semantic_weight=-0.1,
            lexical_weight=1.0,
        )


def test_hybrid_retriever_rejects_zero_total_weight():
    """Both weights cannot be zero."""

    store = FakeVectorStore()

    with pytest.raises(ValueError):
        HybridRetriever(
            store,
            semantic_weight=0,
            lexical_weight=0,
        )


def test_hybrid_retriever_combines_semantic_and_lexical_scores(
    vector_results,
):
    """Hybrid retrieval should produce fused relevance scores."""

    store = FakeVectorStore(vector_results)
    retriever = HybridRetriever(
        store,
        top_k=2,
        semantic_weight=0.7,
        lexical_weight=0.3,
    )

    response = retriever.retrieve("Python machine learning")

    assert response.strategy == "hybrid"
    assert len(response.results) == 2
    assert response.total_results == 2

    for result in response.results:
        assert 0.0 <= result.score <= 1.0
        assert "semantic_score" in result.metadata
        assert "lexical_score" in result.metadata
        assert "hybrid_score" in result.metadata

    assert response.metadata["retrieval_method"] == (
        "semantic_lexical_fusion"
    )
    assert response.metadata["candidate_pool"] == 4
    assert response.metadata["top_k"] == 2


def test_hybrid_retriever_requests_expanded_candidate_pool(
    vector_results,
):
    """Hybrid retrieval should request at least top_k * 3 candidates."""

    store = FakeVectorStore(vector_results)
    retriever = HybridRetriever(store, top_k=4)

    retriever.retrieve("machine learning")

    assert store.calls[0]["top_k"] == 12


def test_hybrid_retriever_candidate_pool_has_minimum_ten(
    vector_results,
):
    """Hybrid retrieval should request at least ten candidates."""

    store = FakeVectorStore(vector_results)
    retriever = HybridRetriever(store, top_k=2)

    retriever.retrieve("machine learning")

    assert store.calls[0]["top_k"] == 10


def test_hybrid_retriever_lexical_score():
    """Lexical scoring should correctly measure query-token coverage."""

    score = HybridRetriever._lexical_score(
        query_tokens=["python", "machine", "learning"],
        content="Python is widely used for machine learning.",
    )

    assert score == pytest.approx(1.0)


def test_hybrid_retriever_lexical_score_partial_match():
    """Partial lexical overlap should produce a score between zero and one."""

    score = HybridRetriever._lexical_score(
        query_tokens=["python", "machine", "learning"],
        content="Python is a programming language.",
    )

    assert score == pytest.approx(1 / 3)


def test_hybrid_retriever_lexical_score_no_match():
    """No lexical overlap should produce zero."""

    score = HybridRetriever._lexical_score(
        query_tokens=["python", "machine"],
        content="Vector databases store embeddings.",
    )

    assert score == 0.0


def test_hybrid_retriever_lexical_score_empty_query_tokens():
    """Empty query tokens should produce zero."""

    score = HybridRetriever._lexical_score(
        query_tokens=[],
        content="Python machine learning",
    )

    assert score == 0.0


def test_hybrid_retriever_lexical_score_empty_document():
    """Empty document content should produce zero."""

    score = HybridRetriever._lexical_score(
        query_tokens=["python"],
        content="",
    )

    assert score == 0.0


def test_hybrid_retriever_tokenization():
    """Tokenization should normalize case and ignore punctuation."""

    tokens = HybridRetriever._tokenize(
        "Python, Machine-Learning! 123."
    )

    assert tokens == [
        "python",
        "machine",
        "learning",
        "123",
    ]


def test_hybrid_retriever_empty_results():
    """Hybrid retrieval should handle an empty candidate pool."""

    store = FakeVectorStore([])
    retriever = HybridRetriever(store)

    response = retriever.retrieve("python")

    assert response.results == []
    assert response.total_results == 0
    assert response.strategy == "hybrid"
    assert "candidate_pool" not in response.metadata


def test_hybrid_retriever_wraps_vector_store_errors():
    """Hybrid vector-store failures should become RetrievalExecutionError."""

    store = FakeVectorStore(
        error=RuntimeError("vector store unavailable")
    )
    retriever = HybridRetriever(store)

    with pytest.raises(
        RetrievalExecutionError,
        match="Hybrid retrieval failed",
    ):
        retriever.retrieve("python")


# ---------------------------------------------------------------------------
# Reranked retrieval
# ---------------------------------------------------------------------------


class FakeCrossEncoder:
    """Fake cross-encoder that returns deterministic scores."""

    def __init__(self, scores):
        self.scores = scores
        self.calls = []

    def predict(self, pairs, show_progress_bar=False):
        self.calls.append(
            {
                "pairs": pairs,
                "show_progress_bar": show_progress_bar,
            }
        )
        return self.scores


def test_reranked_retriever_configuration(vector_results):
    """Reranked retriever should preserve configuration."""

    store = FakeVectorStore(vector_results)

    retriever = RerankedRetriever(
        store,
        top_k=3,
        candidate_multiplier=5,
        model_name="fake-reranker",
    )

    assert retriever.top_k == 3
    assert retriever.candidate_multiplier == 5
    assert retriever.model_name == "fake-reranker"
    assert retriever._model is None


def test_reranked_retriever_rejects_invalid_candidate_multiplier():
    """Candidate multiplier must be at least one."""

    store = FakeVectorStore()

    with pytest.raises(ValueError):
        RerankedRetriever(
            store,
            candidate_multiplier=0,
        )


def test_reranked_retriever_lazy_loads_model(monkeypatch):
    """CrossEncoder should be loaded only when the model is accessed."""

    created = []

    class FakeModel:
        pass

    def fake_cross_encoder(model_name):
        created.append(model_name)
        return FakeModel()

    monkeypatch.setattr(
        "rag_engine.retrieval.reranker.CrossEncoder",
        fake_cross_encoder,
    )

    store = FakeVectorStore()

    retriever = RerankedRetriever(
        store,
        model_name="fake-model",
    )

    assert created == []
    assert retriever._model is None

    model = retriever.model

    assert isinstance(model, FakeModel)
    assert created == ["fake-model"]
    assert retriever.model is model
    assert created == ["fake-model"]


def test_reranked_retriever_model_load_error(monkeypatch):
    """CrossEncoder loading failures should become RetrievalExecutionError."""

    def fake_cross_encoder(model_name):
        raise RuntimeError("model download failed")

    monkeypatch.setattr(
        "rag_engine.retrieval.reranker.CrossEncoder",
        fake_cross_encoder,
    )

    retriever = RerankedRetriever(
        FakeVectorStore(),
        model_name="broken-model",
    )

    with pytest.raises(
        RetrievalExecutionError,
        match="Failed to load reranker model",
    ):
        _ = retriever.model


def test_reranked_retriever_reranks_candidates(vector_results):
    """Reranked retrieval should sort candidates by cross-encoder score."""

    store = FakeVectorStore(vector_results)

    retriever = RerankedRetriever(
        store,
        top_k=2,
        candidate_multiplier=2,
        model_name="fake-reranker",
    )

    fake_model = FakeCrossEncoder(
        scores=[0.10, 0.95, 0.40, 0.80]
    )
    retriever._model = fake_model

    response = retriever.retrieve("machine learning")

    assert response.strategy == "reranked"
    assert response.total_results == 2
    assert len(response.results) == 2

    assert response.results[0].chunk_id == "chunk-2"
    assert response.results[0].score == pytest.approx(0.95)
    assert response.results[0].rank == 1

    assert response.results[1].chunk_id == "chunk-4"
    assert response.results[1].score == pytest.approx(0.80)
    assert response.results[1].rank == 2

    assert response.results[0].metadata["vector_score"] == 0.85
    assert response.results[0].metadata["reranker_score"] == pytest.approx(
        0.95
    )

    assert response.metadata["retrieval_method"] == (
        "vector_then_cross_encoder"
    )
    assert response.metadata["candidate_pool"] == 4
    assert response.metadata["reranker_model"] == "fake-reranker"


def test_reranked_retriever_requests_expanded_candidate_pool(
    vector_results,
):
    """Reranked retrieval should request top_k * candidate_multiplier."""

    store = FakeVectorStore(vector_results)

    retriever = RerankedRetriever(
        store,
        top_k=2,
        candidate_multiplier=2,
    )

    retriever._model = FakeCrossEncoder(
        scores=[0.1, 0.2, 0.3, 0.4]
    )

    retriever.retrieve("machine learning")

    assert store.calls[0]["top_k"] == 10


def test_reranked_retriever_candidate_pool_respects_multiplier(
    vector_results,
):
    """A large top_k should scale the candidate pool correctly."""

    store = FakeVectorStore(vector_results)

    retriever = RerankedRetriever(
        store,
        top_k=4,
        candidate_multiplier=3,
    )

    retriever._model = FakeCrossEncoder(
        scores=[0.1, 0.2, 0.3, 0.4]
    )

    retriever.retrieve("machine learning")

    assert store.calls[0]["top_k"] == 12


def test_reranked_retriever_passes_correct_pairs(vector_results):
    """The reranker should receive query/content pairs."""

    store = FakeVectorStore(vector_results)

    retriever = RerankedRetriever(
        store,
        top_k=2,
        model_name="fake-reranker",
    )

    fake_model = FakeCrossEncoder(
        scores=[0.1, 0.2, 0.3, 0.4]
    )
    retriever._model = fake_model

    retriever.retrieve("  machine learning  ")

    assert fake_model.calls[0]["show_progress_bar"] is False

    pairs = fake_model.calls[0]["pairs"]

    assert pairs[0] == (
        "machine learning",
        vector_results[0].content,
    )
    assert len(pairs) == 4


def test_reranked_retriever_empty_results_does_not_load_model():
    """No candidates means the reranker model should not be loaded."""

    store = FakeVectorStore([])

    retriever = RerankedRetriever(
        store,
        model_name="fake-reranker",
    )

    response = retriever.retrieve("python")

    assert response.results == []
    assert response.total_results == 0
    assert response.strategy == "reranked"
    assert response.metadata["candidate_pool"] == 0
    assert retriever._model is None


def test_reranked_retriever_rejects_empty_query(vector_results):
    """Reranked retrieval should reject empty queries."""

    store = FakeVectorStore(vector_results)
    retriever = RerankedRetriever(store)

    with pytest.raises(RetrievalExecutionError):
        retriever.retrieve("   ")

    assert store.calls == []


def test_reranked_retriever_rejects_invalid_top_k(vector_results):
    """Reranked retrieval should reject invalid request-level top_k."""

    store = FakeVectorStore(vector_results)
    retriever = RerankedRetriever(store)

    with pytest.raises(RetrievalConfigurationError):
        retriever.retrieve("python", top_k=0)

    assert store.calls == []


def test_reranked_retriever_wraps_model_errors(vector_results):
    """Reranker prediction failures should become RetrievalExecutionError."""

    class BrokenModel:
        def predict(self, pairs, show_progress_bar=False):
            raise RuntimeError("prediction failed")

    store = FakeVectorStore(vector_results)

    retriever = RerankedRetriever(store)

    retriever._model = BrokenModel()

    with pytest.raises(
        RetrievalExecutionError,
        match="Reranked retrieval failed",
    ):
        retriever.retrieve("machine learning")


def test_reranked_retriever_wraps_vector_store_errors():
    """Reranked vector-store failures should become RetrievalExecutionError."""

    store = FakeVectorStore(
        error=RuntimeError("vector store unavailable")
    )

    retriever = RerankedRetriever(store)

    with pytest.raises(
        RetrievalExecutionError,
        match="Reranked retrieval failed",
    ):
        retriever.retrieve("python")