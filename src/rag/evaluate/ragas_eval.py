from __future__ import annotations

import json
from pathlib import Path

from datasets import Dataset
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from ragas import evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import (
    answer_relevancy,
    context_precision,
    context_recall,
    faithfulness,
)
from ragas.run_config import RunConfig

from rag.generate.answer import generate
from rag.ingest.index import MODEL_NAME
from rag.retrieve.retriever import HybridRetriever

METRICS = [faithfulness, answer_relevancy, context_precision, context_recall]


def run_ragas_eval(
    gold_data: list[dict],
    retriever: HybridRetriever,
    model: str = "llama3.2:3b",
) -> dict[str, float]:
    questions = []
    answers = []
    contexts = []
    ground_truths = []

    print(f"Generating answers for {len(gold_data)} questions...")
    for i, item in enumerate(gold_data, 1):
        q = item["question"]
        results = retriever.retrieve(q, k=5)
        answer = generate(q, results, model=model)

        questions.append(q)
        answers.append(answer)
        contexts.append([r.document for r in results])
        ground_truths.append(item["ground_truth"])
        print(f"  [{i}/{len(gold_data)}] done")

    ds = Dataset.from_dict(
        {
            "question": questions,
            "answer": answers,
            "contexts": contexts,
            "ground_truth": ground_truths,
        }
    )

    llm = LangchainLLMWrapper(ChatOllama(model=model, temperature=0))
    embeddings = LangchainEmbeddingsWrapper(
        HuggingFaceEmbeddings(model_name=MODEL_NAME)
    )

    run_config = RunConfig(timeout=300, max_retries=3, max_workers=4)

    print("\nRunning RAGAS metrics...")
    result = evaluate(
        dataset=ds,
        metrics=METRICS,
        llm=llm,
        embeddings=embeddings,
        raise_exceptions=False,
        show_progress=True,
        run_config=run_config,
    )

    return dict(result._repr_dict)


if __name__ == "__main__":
    gold_path = Path("data/eval/gold_qa_hard.json")
    with open(gold_path) as f:
        gold_data = json.load(f)
    print(f"Loaded {len(gold_data)} gold QA pairs from {gold_path}\n")

    retriever = HybridRetriever()
    print("Running RAGAS evaluation (this may take a while)...\n")

    scores = run_ragas_eval(gold_data, retriever)

    print(f"\n{'Metric':<25} {'Score':>10}")
    print("-" * 37)
    for metric, score in scores.items():
        print(f"{metric:<25} {score:>10.4f}")
