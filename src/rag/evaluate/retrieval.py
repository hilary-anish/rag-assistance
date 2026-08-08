from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

from rag.retrieve.retriever import HybridRetriever, Result


def hit_rate(retrieved_sources: list[list[str]], relevant_sources: list[str]) -> float:
    hits = 0
    for retrieved, relevant in zip(retrieved_sources, relevant_sources):
        if relevant in retrieved:
            hits += 1
    return hits / len(relevant_sources) if relevant_sources else 0.0


def mrr(retrieved_sources: list[list[str]], relevant_sources: list[str]) -> float:
    total = 0.0
    for retrieved, relevant in zip(retrieved_sources, relevant_sources):
        for rank, source in enumerate(retrieved, 1):
            if source == relevant:
                total += 1.0 / rank
                break
    return total / len(relevant_sources) if relevant_sources else 0.0


def run_retrieval_eval(
    retriever: HybridRetriever,
    gold_data: list[dict],
    k: int = 5,
    retrieve_fn: Callable[[str, int], list[Result]] | None = None,
) -> dict[str, float]:
    if retrieve_fn is None:
        retrieve_fn = retriever.retrieve

    all_retrieved: list[list[str]] = []
    all_relevant: list[str] = []

    for item in gold_data:
        results = retrieve_fn(item["question"], k=k)
        sources = [r.metadata.get("source", "") for r in results]
        all_retrieved.append(sources)
        all_relevant.append(item["relevant_source"])

    return {
        "hit_rate": hit_rate(all_retrieved, all_relevant),
        "mrr": mrr(all_retrieved, all_relevant),
    }


if __name__ == "__main__":
    gold_path = Path("data/eval/gold_qa.json")
    with open(gold_path) as f:
        gold_data = json.load(f)
    print(f"Loaded {len(gold_data)} gold QA pairs\n")

    retriever = HybridRetriever()

    modes = [
        ("Dense only", retriever.dense_only),
        ("Sparse only", retriever.sparse_only),
        ("Hybrid (RRF + rerank)", retriever.retrieve),
    ]

    print(f"{'Mode':<25} {'Hit Rate':>10} {'MRR':>10}")
    print("-" * 47)

    for name, fn in modes:
        scores = run_retrieval_eval(retriever, gold_data, k=5, retrieve_fn=fn)
        print(f"{name:<25} {scores['hit_rate']:>10.3f} {scores['mrr']:>10.3f}")
