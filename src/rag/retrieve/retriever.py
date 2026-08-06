from __future__ import annotations

import re
from dataclasses import dataclass, field

import chromadb
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer

from rag.ingest.index import CHROMA_DIR, COLLECTION_NAME, MODEL_NAME

CROSS_ENCODER_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

_TOKENIZE_RE = re.compile(r"\w+")


def _tokenize(text: str) -> list[str]:
    return _TOKENIZE_RE.findall(text.lower())


@dataclass
class Result:
    chunk_id: str
    document: str
    metadata: dict = field(default_factory=dict)
    score: float = 0.0


class HybridRetriever:
    def __init__(
        self,
        chroma_dir: str = CHROMA_DIR,
        collection_name: str = COLLECTION_NAME,
        embedding_model: str = MODEL_NAME,
        cross_encoder_model: str = CROSS_ENCODER_NAME,
    ):
        client = chromadb.PersistentClient(path=chroma_dir)
        self.collection = client.get_collection(collection_name)

        self.embedder = SentenceTransformer(embedding_model)
        self.cross_encoder = CrossEncoder(cross_encoder_model)

        store = self.collection.get(include=["documents", "metadatas"])
        self._ids: list[str] = store["ids"]
        self._documents: list[str] = store["documents"]
        self._metadatas: list[dict] = store["metadatas"]
        self._id_to_idx: dict[str, int] = {id_: i for i, id_ in enumerate(self._ids)}

        corpus_tokens = [_tokenize(doc) for doc in self._documents]
        self.bm25 = BM25Okapi(corpus_tokens)

    def _dense(self, query: str, k: int) -> list[Result]:
        embedding = self.embedder.encode(query, normalize_embeddings=True).tolist()
        hits = self.collection.query(query_embeddings=[embedding], n_results=k)

        results = []
        for id_, doc, meta in zip(hits["ids"][0], hits["documents"][0], hits["metadatas"][0]):
            results.append(Result(chunk_id=id_, document=doc, metadata=meta))
        return results

    def _sparse(self, query: str, k: int) -> list[Result]:
        tokens = _tokenize(query)
        scores = self.bm25.get_scores(tokens)

        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]

        results = []
        for idx in top_indices:
            if scores[idx] > 0:
                results.append(
                    Result(
                        chunk_id=self._ids[idx],
                        document=self._documents[idx],
                        metadata=self._metadatas[idx],
                        score=float(scores[idx]),
                    )
                )
        return results

    @staticmethod
    def _rrf(rankings: list[list[Result]], k: int = 60) -> list[Result]:
        scores: dict[str, float] = {}
        best: dict[str, Result] = {}

        for ranking in rankings:
            for rank, result in enumerate(ranking):
                scores[result.chunk_id] = scores.get(result.chunk_id, 0.0) + 1.0 / (k + rank + 1)
                if result.chunk_id not in best:
                    best[result.chunk_id] = result

        for chunk_id, score in scores.items():
            best[chunk_id].score = score

        return sorted(best.values(), key=lambda r: r.score, reverse=True)

    def _rerank(self, query: str, results: list[Result], k: int) -> list[Result]:
        if not results:
            return []

        pairs = [[query, r.document] for r in results]
        ce_scores = self.cross_encoder.predict(pairs).tolist()

        for result, score in zip(results, ce_scores):
            result.score = float(score)

        results.sort(key=lambda r: r.score, reverse=True)
        return results[:k]

    def retrieve(self, query: str, k: int = 5, pool: int = 20) -> list[Result]:
        dense_results = self._dense(query, pool)
        sparse_results = self._sparse(query, pool)
        fused = self._rrf([dense_results, sparse_results])
        candidates = fused[:pool]
        return self._rerank(query, candidates, k)

    def dense_only(self, query: str, k: int = 5) -> list[Result]:
        return self._dense(query, k)

    def sparse_only(self, query: str, k: int = 5) -> list[Result]:
        return self._sparse(query, k)


if __name__ == "__main__":
    retriever = HybridRetriever()
    print(f"Loaded {len(retriever._ids)} chunks from ChromaDB\n")

    query = "How does cosine similarity work in vector search?"
    print(f"Query: {query}\n")

    for label, results in [
        ("Dense only", retriever.dense_only(query)),
        ("Sparse only", retriever.sparse_only(query)),
        ("Hybrid (RRF + rerank)", retriever.retrieve(query)),
    ]:
        print(f"--- {label} ---")
        for i, r in enumerate(results, 1):
            source = r.metadata.get("source", "?")
            print(f"  {i}. [{source}] score={r.score:.4f}  {r.document[:80]}...")
        print()
