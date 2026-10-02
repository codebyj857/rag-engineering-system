from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

RESULTS_FILE = (
    ROOT
    / "data"
    / "evaluation"
    / "runs"
    / "retrieval_experiment.json"
)

OUTPUT_DIR = ROOT / "docs" / "evaluation"

CSV_FILE = OUTPUT_DIR / "retrieval_results.csv"
MARKDOWN_FILE = OUTPUT_DIR / "retrieval_results.md"


METRICS = [
    ("Document Precision", "document_precision_at_k", "percent"),
    ("Document Recall", "document_recall_at_k", "percent"),
    ("Document F1", "document_f1_at_k", "percent"),
    ("Document Hit", "document_hit_at_k", "percent"),
    ("Chunk Precision", "chunk_precision_at_k", "percent"),
    ("Chunk Recall", "chunk_recall_at_k", "percent"),
    ("Chunk F1", "chunk_f1_at_k", "percent"),
    ("Chunk Hit", "chunk_hit_at_k", "percent"),
    ("MRR", "mrr", "decimal"),
    ("Mean Latency", "mean_latency_seconds", "seconds"),
]


def load_results() -> dict:
    with RESULTS_FILE.open("r", encoding="utf-8") as file:
        return json.load(file)


def get_metric(run: dict, metric_name: str) -> float:
    value = run["aggregate"][metric_name]

    if isinstance(value, dict):
        value = value["value"]

    return float(value)


def format_value(value: float, value_type: str) -> str:
    if value_type == "percent":
        return f"{value * 100:.2f}%"

    if value_type == "seconds":
        return f"{value:.4f}s"

    return f"{value:.4f}"


def main() -> None:
    if not RESULTS_FILE.exists():
        raise FileNotFoundError(
            f"Results file not found: {RESULTS_FILE}"
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    data = load_results()
    runs = data["runs"]

    strategies = ["naive", "hybrid", "reranked"]

    rows = []

    for label, metric_name, value_type in METRICS:
        row = {
            "Metric @5": label,
        }

        for strategy in strategies:
            value = get_metric(runs[strategy], metric_name)
            row[strategy.capitalize()] = format_value(
                value,
                value_type,
            )

        rows.append(row)

    # CSV
    fieldnames = [
        "Metric @5",
        "Naive",
        "Hybrid",
        "Reranked",
    ]

    with CSV_FILE.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(rows)

    # Markdown
    lines = [
        "# Retrieval Evaluation Results",
        "",
        (
            "Final local retrieval benchmark comparing "
            "Naive, Hybrid, and Reranked retrieval at top-k=5."
        ),
        "",
        "| Metric @5 | Naive | Hybrid | Reranked |",
        "|---|---:|---:|---:|",
    ]

    for row in rows:
        lines.append(
            f"| {row['Metric @5']} | "
            f"{row['Naive']} | "
            f"{row['Hybrid']} | "
            f"{row['Reranked']} |"
        )

    lines.extend(
        [
            "",
            "## Methodology",
            "",
            "- Dataset: 12 evaluation questions",
            "- Indexed collection: 18 document chunks",
            "- Retrieval strategies: Naive, Hybrid, Reranked",
            "- Retrieval depth: top-k=5",
            "- LLM/Groq calls: none",
            "- Metrics were generated from the final retrieval experiment JSON.",
            "",
        ]
    )

    MARKDOWN_FILE.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )

    print("Evaluation table generated successfully.")
    print(f"CSV:      {CSV_FILE}")
    print(f"Markdown: {MARKDOWN_FILE}")


if __name__ == "__main__":
    main()