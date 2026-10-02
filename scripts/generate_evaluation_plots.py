from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
RESULTS_FILE = ROOT / "data" / "evaluation" / "runs" / "retrieval_experiment.json"
OUTPUT_DIR = ROOT / "docs" / "figures"


def load_results() -> dict:
    with RESULTS_FILE.open("r", encoding="utf-8") as file:
        return json.load(file)


def get_aggregate(run: dict, metric: str) -> float:
    value = run["aggregate"][metric]

    if isinstance(value, dict):
        if "value" in value:
            return float(value["value"])

    return float(value)


def save_plot(filename: str) -> None:
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename, dpi=200, bbox_inches="tight")
    plt.close()


def plot_retrieval_quality(runs: dict) -> None:
    strategies = list(runs.keys())

    precision = [
        get_aggregate(runs[strategy], "chunk_precision_at_k") * 100
        for strategy in strategies
    ]
    recall = [
        get_aggregate(runs[strategy], "chunk_recall_at_k") * 100
        for strategy in strategies
    ]
    f1 = [
        get_aggregate(runs[strategy], "chunk_f1_at_k") * 100
        for strategy in strategies
    ]

    x = range(len(strategies))
    width = 0.25

    plt.figure(figsize=(9, 5))
    plt.bar([i - width for i in x], precision, width, label="Precision")
    plt.bar(x, recall, width, label="Recall")
    plt.bar([i + width for i in x], f1, width, label="F1")

    plt.xticks(list(x), [s.capitalize() for s in strategies])
    plt.ylabel("Score (%)")
    plt.title("Retrieval Quality by Strategy")
    plt.ylim(0, 100)
    plt.legend()
    plt.grid(axis="y", alpha=0.25)

    save_plot("retrieval_quality.png")


def plot_mrr(runs: dict) -> None:
    strategies = list(runs.keys())

    mrr = [
        get_aggregate(runs[strategy], "mrr")
        for strategy in strategies
    ]

    plt.figure(figsize=(8, 5))
    bars = plt.bar(
        [s.capitalize() for s in strategies],
        mrr,
    )

    plt.ylabel("Mean Reciprocal Rank")
    plt.title("Ranking Quality — MRR")
    plt.ylim(0, 1)

    for bar, value in zip(bars, mrr):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.02,
            f"{value:.4f}",
            ha="center",
            va="bottom",
        )

    plt.grid(axis="y", alpha=0.25)

    save_plot("mrr_comparison.png")


def plot_quality_vs_latency(runs: dict) -> None:
    strategies = list(runs.keys())

    latency = [
        get_aggregate(runs[strategy], "mean_latency_seconds")
        for strategy in strategies
    ]

    f1 = [
        get_aggregate(runs[strategy], "chunk_f1_at_k") * 100
        for strategy in strategies
    ]

    plt.figure(figsize=(8, 5))

    plt.scatter(latency, f1, s=100)

    for strategy, x, y in zip(strategies, latency, f1):
        plt.annotate(
            strategy.capitalize(),
            (x, y),
            xytext=(8, 8),
            textcoords="offset points",
        )

    plt.xlabel("Mean Latency (seconds)")
    plt.ylabel("Chunk F1 (%)")
    plt.title("Retrieval Quality vs Latency")
    plt.grid(alpha=0.25)

    save_plot("quality_vs_latency.png")


def main() -> None:
    if not RESULTS_FILE.exists():
        raise FileNotFoundError(
            f"Evaluation results not found: {RESULTS_FILE}"
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    data = load_results()
    runs = data["runs"]

    required_strategies = {"naive", "hybrid", "reranked"}

    missing = required_strategies - set(runs)

    if missing:
        raise ValueError(
            f"Missing expected strategies: {sorted(missing)}"
        )

    plot_retrieval_quality(runs)
    plot_mrr(runs)
    plot_quality_vs_latency(runs)

    print("Evaluation plots generated successfully.")
    print(f"Output directory: {OUTPUT_DIR}")

    for filename in (
        "retrieval_quality.png",
        "mrr_comparison.png",
        "quality_vs_latency.png",
    ):
        print(f"  - {OUTPUT_DIR / filename}")


if __name__ == "__main__":
    main()