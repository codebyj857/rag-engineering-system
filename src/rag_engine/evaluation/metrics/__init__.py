"""
Evaluation metrics package.

Provides deterministic metrics for evaluating RAG answers, retrieval
quality, and execution efficiency.
"""

from rag_engine.evaluation.metrics.correctness import (
    CorrectnessMetric,
    CorrectnessScore,
    calculate_correctness,
)
from rag_engine.evaluation.metrics.efficiency import (
    EfficiencyMetric,
    EfficiencyScore,
    calculate_efficiency,
)
from rag_engine.evaluation.metrics.faithfulness import (
    FaithfulnessMetric,
    FaithfulnessScore,
    calculate_faithfulness,
)
from rag_engine.evaluation.metrics.retrieval import (
    RetrievalMetric,
    RetrievalScore,
    calculate_retrieval_metrics,
)

__all__ = [
    "CorrectnessMetric",
    "CorrectnessScore",
    "calculate_correctness",
    "EfficiencyMetric",
    "EfficiencyScore",
    "calculate_efficiency",
    "FaithfulnessMetric",
    "FaithfulnessScore",
    "calculate_faithfulness",
    "RetrievalMetric",
    "RetrievalScore",
    "calculate_retrieval_metrics",
]