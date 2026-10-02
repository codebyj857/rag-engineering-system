from __future__ import annotations

import pytest

from rag_engine.verification.confidence import (
    ConfidenceEstimator,
    ConfidenceResult,
    InvalidConfidenceInputError,
    estimate_confidence,
)
from rag_engine.verification.groundedness import (
    GroundednessResult,
    GroundednessVerifier,
    InvalidGroundednessInputError,
    verify_groundedness,
)


# ============================================================================
# Groundedness
# ============================================================================


def test_groundedness_initialization_defaults() -> None:
    verifier = GroundednessVerifier()
    assert verifier.threshold == 0.60


@pytest.mark.parametrize(
    "threshold",
    [-0.1, 1.1],
)
def test_groundedness_rejects_invalid_threshold(
    threshold: float,
) -> None:
    with pytest.raises(ValueError, match="threshold"):
        GroundednessVerifier(threshold=threshold)


@pytest.mark.parametrize(
    "min_term_length",
    [0, -1],
)
def test_groundedness_rejects_invalid_min_term_length(
    min_term_length: int,
) -> None:
    with pytest.raises(ValueError, match="min_term_length"):
        GroundednessVerifier(min_term_length=min_term_length)


def test_groundedness_fully_supported_answer() -> None:
    verifier = GroundednessVerifier()
    result = verifier.verify(
        answer="Python is a programming language.",
        evidence=[
            "Python is a high-level programming language used for software development."
        ],
    )

    assert isinstance(result, GroundednessResult)
    assert result.grounded is True
    assert result.score == 1.0
    assert result.evidence_count == 1
    assert result.threshold == 0.60
    assert result.unsupported_terms == ()


def test_groundedness_partially_supported_answer() -> None:
    verifier = GroundednessVerifier()
    result = verifier.verify(
        answer="Python is a programming language created in Amsterdam.",
        evidence=[
            "Python is a high-level programming language."
        ],
    )

    assert result.grounded is True
    assert result.score == 0.6
    assert "python" in result.supported_terms
    assert "programming" in result.supported_terms
    assert "language" in result.supported_terms
    assert "amsterdam" in result.unsupported_terms


def test_groundedness_unsupported_answer() -> None:
    verifier = GroundednessVerifier()
    result = verifier.verify(
        answer="Quantum computers use magical crystals.",
        evidence=[
            "Python is a programming language."
        ],
    )

    assert result.grounded is False
    assert result.score == 0.0
    assert len(result.supported_terms) == 0
    assert len(result.unsupported_terms) > 0


def test_groundedness_is_case_insensitive() -> None:
    verifier = GroundednessVerifier()
    result = verifier.verify(
        answer="PYTHON PROGRAMMING LANGUAGE",
        evidence=[
            "python is a programming language."
        ],
    )

    assert result.grounded is True
    assert result.score == 1.0


def test_groundedness_ignores_stop_words() -> None:
    verifier = GroundednessVerifier()
    result = verifier.verify(
        answer="Python is a language.",
        evidence=[
            "Python is a programming language."
        ],
    )

    assert result.grounded is True
    assert result.score == 1.0
    assert result.supported_terms == ("python", "language")


def test_groundedness_preserves_term_order() -> None:
    verifier = GroundednessVerifier()
    result = verifier.verify(
        answer="Python programming language",
        evidence=[
            "Python is a programming language."
        ],
    )

    assert result.supported_terms == (
        "python",
        "programming",
        "language",
    )


def test_groundedness_deduplicates_terms() -> None:
    verifier = GroundednessVerifier()
    result = verifier.verify(
        answer="Python Python programming programming",
        evidence=[
            "Python is a programming language."
        ],
    )

    assert result.supported_terms == (
        "python",
        "programming",
    )
    assert result.score == 1.0


def test_groundedness_counts_empty_evidence() -> None:
    verifier = GroundednessVerifier()
    result = verifier.verify(
        answer="Python programming language",
        evidence=[],
    )

    assert result.evidence_count == 0
    assert result.score == 0.0
    assert result.grounded is False


def test_groundedness_ignores_empty_evidence_items() -> None:
    verifier = GroundednessVerifier()
    result = verifier.verify(
        answer="Python programming language",
        evidence=[
            "",
            "   ",
            "Python is a programming language.",
        ],
    )

    assert result.evidence_count == 1
    assert result.grounded is True


def test_groundedness_no_meaningful_answer_terms() -> None:
    verifier = GroundednessVerifier()
    result = verifier.verify(
        answer="the and is",
        evidence=["Python is a programming language."],
    )

    assert result.grounded is False
    assert result.score == 0.0
    assert result.supported_terms == ()
    assert result.unsupported_terms == ()
    assert result.evidence_count == 1


@pytest.mark.parametrize(
    "answer",
    [
        "",
        "   ",
        "\n\t",
    ],
)
def test_groundedness_rejects_empty_answer(answer: str) -> None:
    verifier = GroundednessVerifier()

    with pytest.raises(
        InvalidGroundednessInputError,
        match="answer cannot be empty",
    ):
        verifier.verify(
            answer=answer,
            evidence=["Python is a language."],
        )


def test_groundedness_rejects_non_string_answer() -> None:
    verifier = GroundednessVerifier()

    with pytest.raises(
        InvalidGroundednessInputError,
        match="answer must be a string",
    ):
        verifier.verify(
            answer=123,  # type: ignore[arg-type]
            evidence=["Python is a language."],
        )


@pytest.mark.parametrize(
    "evidence",
    [
        "Python is a language.",
        b"Python is a language.",
        123,
        None,
    ],
)
def test_groundedness_rejects_invalid_evidence_container(
    evidence: object,
) -> None:
    verifier = GroundednessVerifier()

    with pytest.raises(InvalidGroundednessInputError):
        verifier.verify(
            answer="Python language",
            evidence=evidence,  # type: ignore[arg-type]
        )


def test_groundedness_rejects_non_string_evidence_item() -> None:
    verifier = GroundednessVerifier()

    with pytest.raises(
        InvalidGroundednessInputError,
        match="Every evidence item must be a string",
    ):
        verifier.verify(
            answer="Python language",
            evidence=[
                "Python is a language.",
                123,  # type: ignore[list-item]
            ],
        )


def test_groundedness_custom_threshold_changes_classification() -> None:
    verifier = GroundednessVerifier(threshold=1.0)

    result = verifier.verify(
        answer="Python programming language database",
        evidence=[
            "Python is a programming language."
        ],
    )

    assert result.grounded is False
    assert result.threshold == 1.0
    assert 0.0 < result.score < 1.0


def test_groundedness_custom_min_term_length() -> None:
    verifier = GroundednessVerifier(min_term_length=5)

    result = verifier.verify(
        answer="Python code model",
        evidence=[
            "Python code model"
        ],
    )

    assert result.supported_terms == (
        "python",
        "model",
    )


def test_groundedness_extract_terms_handles_hyphens_and_underscores() -> None:
    verifier = GroundednessVerifier()

    terms = verifier._extract_terms(
        "RAG-system uses vector_database retrieval."
    )

    assert "rag-system" in terms
    assert "vector_database" in terms
    assert "retrieval" in terms


def test_verify_groundedness_convenience_function() -> None:
    result = verify_groundedness(
        answer="Python is a programming language.",
        evidence=[
            "Python is a programming language."
        ],
    )

    assert isinstance(result, GroundednessResult)
    assert result.grounded is True
    assert result.score == 1.0


# ============================================================================
# Confidence
# ============================================================================


def test_confidence_initialization_defaults() -> None:
    estimator = ConfidenceEstimator()

    result = estimator.estimate(
        retrieval_scores=[0.8],
        groundedness_score=0.8,
        evidence_count=3,
    )

    assert isinstance(result, ConfidenceResult)
    assert result.score == 0.84
    assert result.level == "high"


def test_confidence_weights_are_applied_correctly() -> None:
    estimator = ConfidenceEstimator(
        retrieval_weight=0.40,
        groundedness_weight=0.40,
        evidence_weight=0.20,
    )

    result = estimator.estimate(
        retrieval_scores=[0.8, 0.6],
        groundedness_score=0.9,
        evidence_count=2,
    )

    # retrieval = 0.7
    # groundedness = 0.9
    # evidence = 2 / 3
    # score = 0.7*0.4 + 0.9*0.4 + (2/3)*0.2
    expected = round(
        0.7 * 0.4
        + 0.9 * 0.4
        + (2 / 3) * 0.2,
        4,
    )

    assert result.score == expected
    assert result.retrieval_score == 0.7
    assert result.groundedness_score == 0.9
    assert result.evidence_score == round(2 / 3, 4)


@pytest.mark.parametrize(
    "weights",
    [
        (-0.1, 0.8, 0.3),
        (1.1, 0.0, -0.1),
        (0.5, 0.5, 0.5),
        (0.2, 0.2, 0.2),
    ],
)
def test_confidence_rejects_invalid_weights(
    weights: tuple[float, float, float],
) -> None:
    with pytest.raises(ValueError):
        ConfidenceEstimator(
            retrieval_weight=weights[0],
            groundedness_weight=weights[1],
            evidence_weight=weights[2],
        )


def test_confidence_empty_retrieval_scores() -> None:
    estimator = ConfidenceEstimator()

    result = estimator.estimate(
        retrieval_scores=[],
        groundedness_score=0.8,
        evidence_count=3,
    )

    assert result.retrieval_score == 0.0
    assert result.evidence_score == 1.0
    assert result.score == 0.52
    assert result.level == "low"


def test_confidence_evidence_score_is_capped_at_one() -> None:
    estimator = ConfidenceEstimator()

    result = estimator.estimate(
        retrieval_scores=[1.0],
        groundedness_score=1.0,
        evidence_count=10,
    )

    assert result.evidence_score == 1.0
    assert result.score == 1.0
    assert result.level == "high"


@pytest.mark.parametrize(
    ("evidence_count", "expected"),
    [
        (0, 0.0),
        (1, 1 / 3),
        (2, 2 / 3),
        (3, 1.0),
        (10, 1.0),
    ],
)
def test_confidence_evidence_score(
    evidence_count: int,
    expected: float,
) -> None:
    assert (
        ConfidenceEstimator._calculate_evidence_score(
            evidence_count
        )
        == expected
    )


@pytest.mark.parametrize(
    ("score", "expected_level"),
    [
        (1.0, "high"),
        (0.80, "high"),
        (0.79, "medium"),
        (0.60, "medium"),
        (0.59, "low"),
        (0.0, "low"),
    ],
)
def test_confidence_classification(
    score: float,
    expected_level: str,
) -> None:
    assert ConfidenceEstimator._classify(score) == expected_level


@pytest.mark.parametrize(
    "retrieval_scores",
    [
        "0.8",
        b"0.8",
        [0.5, "0.8"],
        [0.5, None],
        [-0.1],
        [1.1],
    ],
)
def test_confidence_rejects_invalid_retrieval_scores(
    retrieval_scores: object,
) -> None:
    estimator = ConfidenceEstimator()

    with pytest.raises(InvalidConfidenceInputError):
        estimator.estimate(
            retrieval_scores=retrieval_scores,  # type: ignore[arg-type]
            groundedness_score=0.8,
            evidence_count=2,
        )


@pytest.mark.parametrize(
    "groundedness_score",
    [
        -0.1,
        1.1,
        "0.8",
        None,
    ],
)
def test_confidence_rejects_invalid_groundedness_score(
    groundedness_score: object,
) -> None:
    estimator = ConfidenceEstimator()

    with pytest.raises(InvalidConfidenceInputError):
        estimator.estimate(
            retrieval_scores=[0.8],
            groundedness_score=groundedness_score,  # type: ignore[arg-type]
            evidence_count=2,
        )


@pytest.mark.parametrize(
    "evidence_count",
    [
        -1,
        -10,
        1.5,
        "2",
        None,
    ],
)
def test_confidence_rejects_invalid_evidence_count(
    evidence_count: object,
) -> None:
    estimator = ConfidenceEstimator()

    with pytest.raises(InvalidConfidenceInputError):
        estimator.estimate(
            retrieval_scores=[0.8],
            groundedness_score=0.8,
            evidence_count=evidence_count,  # type: ignore[arg-type]
        )


def test_confidence_result_values_are_rounded() -> None:
    estimator = ConfidenceEstimator()

    result = estimator.estimate(
        retrieval_scores=[0.123456, 0.654321],
        groundedness_score=0.987654,
        evidence_count=1,
    )

    assert result.retrieval_score == round(
        (0.123456 + 0.654321) / 2,
        4,
    )
    assert result.groundedness_score == 0.9877
    assert result.evidence_score == round(1 / 3, 4)


def test_estimate_confidence_convenience_function() -> None:
    result = estimate_confidence(
        retrieval_scores=[0.9, 0.8],
        groundedness_score=0.9,
        evidence_count=3,
    )

    assert isinstance(result, ConfidenceResult)
    assert result.score == 0.9
    assert result.level == "high"