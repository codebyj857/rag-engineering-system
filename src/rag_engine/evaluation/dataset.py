"""
Evaluation dataset management.

This module loads and validates JSONL evaluation datasets used to
compare different RAG retrieval and generation strategies.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator


class EvaluationDatasetError(Exception):
    """Base exception for evaluation dataset errors."""


class EvaluationDatasetFileError(EvaluationDatasetError):
    """Raised when an evaluation dataset cannot be read."""


class InvalidEvaluationRecordError(EvaluationDatasetError):
    """Raised when an evaluation record has an invalid structure."""


@dataclass(frozen=True, slots=True)
class EvaluationExample:
    """
    Single evaluation example.

    Attributes:
        question: User question presented to the RAG system.
        reference_answer: Expected/reference answer used for correctness
            evaluation.
        relevant_documents: Document identifiers expected to contain
            relevant evidence.
        relevant_chunks: Exact chunk identifiers expected to contain
            relevant evidence.
        metadata: Additional evaluation metadata.
    """

    question: str
    reference_answer: str
    relevant_documents: tuple[str, ...] = ()
    relevant_chunks: tuple[str, ...] = ()
    metadata: dict[str, object] | None = None

    def __post_init__(self) -> None:
        """Validate the evaluation example."""

        if not self.question.strip():
            raise InvalidEvaluationRecordError(
                "Evaluation question cannot be empty."
            )

        if not self.reference_answer.strip():
            raise InvalidEvaluationRecordError(
                "Reference answer cannot be empty."
            )

        if self.metadata is None:
            object.__setattr__(self, "metadata", {})

    @classmethod
    def from_dict(
        cls,
        record: dict[str, object],
    ) -> "EvaluationExample":
        """
        Create an evaluation example from a dictionary.

        Required fields:
            question
            reference_answer

        Optional fields:
            relevant_documents
            relevant_chunks
            metadata
        """

        if not isinstance(record, dict):
            raise InvalidEvaluationRecordError(
                "Each evaluation record must be a JSON object."
            )

        question = record.get("question")
        reference_answer = record.get("reference_answer")

        if not isinstance(question, str):
            raise InvalidEvaluationRecordError(
                "Field 'question' must be a string."
            )

        if not isinstance(reference_answer, str):
            raise InvalidEvaluationRecordError(
                "Field 'reference_answer' must be a string."
            )

        relevant_documents = record.get(
            "relevant_documents",
            (),
        )

        if relevant_documents is None:
            relevant_documents = ()

        if not isinstance(relevant_documents, list):
            raise InvalidEvaluationRecordError(
                "Field 'relevant_documents' must be a list."
            )

        if not all(
            isinstance(document, str)
            for document in relevant_documents
        ):
            raise InvalidEvaluationRecordError(
                "Every relevant document identifier must be a string."
            )

        relevant_chunks = record.get(
            "relevant_chunks",
            (),
        )

        if relevant_chunks is None:
            relevant_chunks = ()

        if not isinstance(relevant_chunks, list):
            raise InvalidEvaluationRecordError(
                "Field 'relevant_chunks' must be a list."
            )

        if not all(
            isinstance(chunk, str)
            for chunk in relevant_chunks
        ):
            raise InvalidEvaluationRecordError(
                "Every relevant chunk identifier must be a string."
            )

        metadata = record.get("metadata", {})

        if metadata is None:
            metadata = {}

        if not isinstance(metadata, dict):
            raise InvalidEvaluationRecordError(
                "Field 'metadata' must be an object."
            )

        return cls(
            question=question,
            reference_answer=reference_answer,
            relevant_documents=tuple(relevant_documents),
            relevant_chunks=tuple(relevant_chunks),
            metadata=dict(metadata),
        )


class EvaluationDataset:
    """
    Collection of validated evaluation examples.

    The dataset is intentionally independent of the RAG pipeline so it
    can be reused across naive, hybrid, and reranked evaluations.
    """

    def __init__(
        self,
        examples: tuple[EvaluationExample, ...],
    ) -> None:
        if not examples:
            raise EvaluationDatasetError(
                "Evaluation dataset cannot be empty."
            )

        self._examples = examples

    @classmethod
    def from_jsonl(
        cls,
        path: Path | str,
    ) -> "EvaluationDataset":
        """
        Load an evaluation dataset from a JSONL file.

        Each non-empty line must contain exactly one JSON object.
        """

        dataset_path = Path(path)

        if not dataset_path.exists():
            raise EvaluationDatasetFileError(
                f"Evaluation dataset not found: {dataset_path}"
            )

        if not dataset_path.is_file():
            raise EvaluationDatasetFileError(
                f"Evaluation dataset path is not a file: {dataset_path}"
            )

        examples: list[EvaluationExample] = []

        try:
            with dataset_path.open(
                "r",
                encoding="utf-8",
            ) as file:
                for line_number, raw_line in enumerate(
                    file,
                    start=1,
                ):
                    line = raw_line.strip()

                    if not line:
                        continue

                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError as exc:
                        raise EvaluationDatasetFileError(
                            f"Invalid JSON on line {line_number} "
                            f"of {dataset_path}: {exc}"
                        ) from exc

                    try:
                        examples.append(
                            EvaluationExample.from_dict(record)
                        )
                    except InvalidEvaluationRecordError as exc:
                        raise InvalidEvaluationRecordError(
                            f"Invalid evaluation record on line "
                            f"{line_number}: {exc}"
                        ) from exc

        except OSError as exc:
            raise EvaluationDatasetFileError(
                f"Unable to read evaluation dataset: {dataset_path}"
            ) from exc

        if not examples:
            raise EvaluationDatasetError(
                f"Evaluation dataset contains no records: {dataset_path}"
            )

        return cls(tuple(examples))

    @property
    def examples(self) -> tuple[EvaluationExample, ...]:
        """Return all evaluation examples."""

        return self._examples

    @property
    def size(self) -> int:
        """Return the number of evaluation examples."""

        return len(self._examples)

    def __len__(self) -> int:
        """Return the number of examples."""

        return self.size

    def __iter__(self) -> Iterator[EvaluationExample]:
        """Iterate over evaluation examples."""

        return iter(self._examples)

    def __getitem__(self, index: int) -> EvaluationExample:
        """Access an evaluation example by index."""

        return self._examples[index]


def load_evaluation_dataset(
    path: Path | str,
) -> EvaluationDataset:
    """Convenience function for loading an evaluation dataset."""

    return EvaluationDataset.from_jsonl(path)