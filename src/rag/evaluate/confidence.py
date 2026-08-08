from __future__ import annotations

import json
import math
from pathlib import Path

from rag.retrieve.retriever import HybridRetriever


def confidence(
    reranker_scores: list[float],
    faithfulness: float | None = None,
    threshold: float = 0.0,
) -> dict:
    if not reranker_scores:
        return {
            "confidence": 0.0,
            "flag": "low-confidence / possible hallucination",
            "top_score": 0.0,
            "margin": 0.0,
        }

    sorted_scores = sorted(reranker_scores, reverse=True)
    top_score = sorted_scores[0]
    margin = top_score - sorted_scores[1] if len(sorted_scores) > 1 else top_score

    retrieval_conf = 1.0 / (1.0 + math.exp(-top_score))

    if faithfulness is not None:
        blended = 0.6 * retrieval_conf + 0.4 * faithfulness
    else:
        blended = retrieval_conf

    grounded = top_score > threshold and (faithfulness is None or faithfulness > 0.7)
    flag = "grounded" if grounded else "low-confidence / possible hallucination"

    return {
        "confidence": round(blended, 4),
        "flag": flag,
        "top_score": round(top_score, 4),
        "margin": round(margin, 4),
    }


def assess_gold_set(
    retriever: HybridRetriever,
    gold_data: list[dict],
    k: int = 5,
) -> list[dict]:
    rows: list[dict] = []

    for item in gold_data:
        results = retriever.retrieve(item["question"], k=k)
        scores = [r.score for r in results]
        retrieved_sources = [r.metadata.get("source", "") for r in results]
        correct = item["relevant_source"] in retrieved_sources

        conf = confidence(scores)

        rows.append(
            {
                "question": item["question"],
                "correct": correct,
                **conf,
            }
        )

    header = f"{'#':<3} {'Question':<60} {'Conf':>6} {'Flag':<40} {'Correct':>7}"
    print(header)
    print("-" * len(header))

    for i, row in enumerate(rows, 1):
        q = row["question"][:57] + "..." if len(row["question"]) > 57 else row["question"]
        print(
            f"{i:<3} {q:<60} {row['confidence']:>6.3f} {row['flag']:<40} {'yes' if row['correct'] else 'NO':>7}"
        )

    return rows


if __name__ == "__main__":
    gold_path = Path("data/eval/gold_qa_hard.json")
    with open(gold_path) as f:
        gold_data = json.load(f)
    print(f"Loaded {len(gold_data)} gold QA pairs\n")

    retriever = HybridRetriever()
    rows = assess_gold_set(retriever, gold_data)

    correct_rows = [r for r in rows if r["correct"]]
    incorrect_rows = [r for r in rows if not r["correct"]]
    grounded_correct = sum(1 for r in correct_rows if r["flag"] == "grounded")
    flagged_incorrect = sum(
        1 for r in incorrect_rows if r["flag"] != "grounded"
    )

    print("\n--- Summary ---")
    print(f"Total questions:        {len(rows)}")
    print(f"Retrieval correct:      {len(correct_rows)}/{len(rows)}")
    print(f"Retrieval incorrect:    {len(incorrect_rows)}/{len(rows)}")
    print(f"Correct & grounded:     {grounded_correct}/{len(correct_rows)}")
    print(f"Incorrect & flagged:    {flagged_incorrect}/{len(incorrect_rows)}")
    if correct_rows:
        avg_conf_correct = sum(r["confidence"] for r in correct_rows) / len(correct_rows)
        print(f"Avg confidence (correct):   {avg_conf_correct:.3f}")
    if incorrect_rows:
        avg_conf_incorrect = sum(r["confidence"] for r in incorrect_rows) / len(incorrect_rows)
        print(f"Avg confidence (incorrect): {avg_conf_incorrect:.3f}")
