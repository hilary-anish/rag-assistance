from __future__ import annotations

import ollama

from rag.retrieve.retriever import Result

SYSTEM_PROMPT = (
    "You are a precise technical assistant. Follow these rules strictly:\n"
    "1. Answer ONLY using the provided context. Do not use outside knowledge.\n"
    "2. Cite sources inline as [1], [2], etc., matching the context chunk numbers.\n"
    "3. If the context does not contain enough information to answer, say "
    '"I don\'t have enough information to answer this question."\n'
    "4. Be concise and precise. Do not repeat the question or add filler."
)


def build_prompt(question: str, results: list[Result]) -> str:
    context_parts = []
    for i, r in enumerate(results, 1):
        source = r.metadata.get("source", "unknown")
        page = r.metadata.get("page", 0)
        context_parts.append(f"[{i}] (source: {source}, page: {page})\n{r.document}")

    context_block = "\n\n".join(context_parts)
    return f"Context:\n{context_block}\n\nQuestion: {question}"


def generate(
    question: str,
    results: list[Result],
    model: str = "llama3.2:3b",
) -> str:
    user_prompt = build_prompt(question, results)
    response = ollama.chat(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
    )
    return response["message"]["content"]


if __name__ == "__main__":
    from rag.retrieve.retriever import HybridRetriever

    retriever = HybridRetriever()

    question = "What is the difference between bi-encoder and cross-encoder for retrieval?"
    print(f"Question: {question}\n")

    results = retriever.retrieve(question, k=5)

    print("Sources:")
    for i, r in enumerate(results, 1):
        source = r.metadata.get("source", "?")
        print(f"  [{i}] {source} (score={r.score:.4f})")
    print()

    answer = generate(question, results)
    print(f"Answer:\n{answer}")
