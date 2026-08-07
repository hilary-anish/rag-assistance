from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from pydantic import BaseModel, Field

from rag.evaluate.confidence import confidence
from rag.generate.answer import generate
from rag.retrieve.retriever import HybridRetriever


class Query(BaseModel):
    question: str
    k: int = Field(default=5, ge=1, le=20)


class Source(BaseModel):
    text: str
    source: str
    page: int


class Answer(BaseModel):
    answer: str
    sources: list[Source]
    confidence: float
    flag: str


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.retriever = HybridRetriever()
    yield


app = FastAPI(title="Document-Intelligence RAG", version="1.0", lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ask", response_model=Answer)
def ask(query: Query):
    retriever: HybridRetriever = app.state.retriever

    results = retriever.retrieve(query.question, k=query.k)
    answer_text = generate(query.question, results)

    scores = [r.score for r in results]
    conf = confidence(scores)

    sources = [
        Source(
            text=r.document[:200],
            source=r.metadata.get("source", ""),
            page=r.metadata.get("page", 0),
        )
        for r in results
    ]

    return Answer(
        answer=answer_text,
        sources=sources,
        confidence=conf["confidence"],
        flag=conf["flag"],
    )
