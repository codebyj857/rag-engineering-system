"""
Retrieval evaluation metrics.

This module provides deterministic metrics for evaluating whether a
retrieval strategy returns relevant documents within its top-k results.

No LLM or external API is required.
"""

from __future__ import annotations

from dataclasses import dataclass


class RetrievalMetricError(Exception):
    """Base exception for retrieval metric errors."""


@dataclass(frozen=True, slots=True)
class RetrievalScore:
    """
    Retrieval metric result.

    Attributes:
        precision_at_k: Fraction of retrieved results that are relevant.
        recall_at_k: Fraction of relevant documents retrieved.
        f1_at_k: Harmonic mean of precision and recall.
        hit_at_k: Whether at least one relevant document was retrieved.
        retrieved_count: Number of retrieved results considered.
        relevant_count: Number of relevant documents expected.
        matched_documents: Relevant documents that were retrieved.
        missing_documents: Relevant documents that were not retrieved.
    """

    precision_at_k: float
    recall_at_k: float
    f1_at_k: float
    hit_at_k: float
    retrieved_count: int
    relevant_count: int
    matched_documents: tuple[str, ...]
    missing_documents: tuple[str, ...]


class RetrievalMetric:
    """
    Deterministic retrieval evaluator.

    Document identifiers are normalized before comparison so that
    differences in case or surrounding whitespace do not affect results.
    """

    def calculate(
        self,
        *,
        retrieved_documents: tuple[str, ...] | list[str],
        relevant_documents: tuple[str, ...] | list[str],
        top_k: int | None = None,
    ) -> RetrievalScore:
        """
        Calculate retrieval metrics.

        Args:
            retrieved_documents:
                Document identifiers returned by the retriever, ordered
                by retrieval rank.
            relevant_documents:
                Document identifiers known to be relevant.
            top_k:
                Maximum number of retrieved results to evaluate. If
                omitted, all supplied retrieved documents are considered.

        Returns:
            RetrievalScore containing precision, recall, F1, and hit rate.
        """

        if not isinstance(retrieved_documents, (tuple, list)):
            raise RetrievalMetricError(
                "retrieved_documents must be a tuple or list."
            )

        if not isinstance(relevant_documents, (tuple, list)):
            raise RetrievalMetricError(
                "relevant_documents must be a tuple or list."
            )

        if top_k is not None and top_k < 1:
            raise RetrievalMetricError(
                "top_k must be at least 1."
            )

        retrieved = self._normalize_identifiers(
            retrieved_documents
        )
        relevant = self._normalize_identifiers(
            relevant_documents
        )

        if not relevant:
            raise RetrievalMetricError(
                "At least one relevant document is required."
            )

        if top_k is not None:
            retrieved = retrieved[:top_k]

        retrieved_set = set(retrieved)
        relevant_set = set(relevant)

        matched = tuple(
            document
            for document in relevant
            if document in retrieved_set
        )

        missing = tuple(
            document
            for document in relevant
            if document not in retrieved_set
        )

        retrieved_count = len(retrieved)
        relevant_count = len(relevant_set)
        matched_count = len(set(matched))

        precision = (
            matched_count / retrieved_count
            if retrieved_count
            else 0.0
        )

        recall = (
            matched_count / relevant_count
            if relevant_count
            else 0.0
        )

        if precision + recall == 0:
            f1 = 0.0
        else:
            f1 = (
                2 * precision * recall
            ) / (precision + recall)

        hit = 1.0 if matched_count > 0 else 0.0

        return RetrievalScore(
            precision_at_k=round(precision, 4),
            recall_at_k=round(recall, 4),
            f1_at_k=round(f1, 4),
            hit_at_k=hit,
            retrieved_count=retrieved_count,
            relevant_count=relevant_count,
            matched_documents=matched,
            missing_documents=missing,
        )

    @staticmethod
    def _normalize_identifiers(
        identifiers: tuple[str, ...] | list[str],
    ) -> tuple[str, ...]:
        """Normalize and deduplicate document identifiers."""

        normalized: list[str] = []

        for identifier in identifiers:
            if not isinstance(identifier, str):
                raise RetrievalMetricError(
                    "Every document identifier must be a string."
                )

            value = identifier.strip().lower()

            if not value:
                continue

            if value not in normalized:
                normalized.append(value)

        return tuple(normalized)


def calculate_retrieval_metrics(
    *,
    retrieved_documents: tuple[str, ...] | list[str],
    relevant_documents: tuple[str, ...] | list[str],
    top_k: int | None = None,
) -> RetrievalScore:
    """Convenience function for calculating retrieval metrics."""

    return RetrievalMetric().calculate(
        retrieved_documents=retrieved_documents,
        relevant_documents=relevant_documents,
        top_k=top_k,
    )