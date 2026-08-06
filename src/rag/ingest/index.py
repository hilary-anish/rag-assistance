from __future__ import annotations

from pathlib import Path

import chromadb
from langchain_core.documents import Document
from sentence_transformers import SentenceTransformer

MODEL_NAME = "BAAI/bge-small-en-v1.5"
COLLECTION_NAME = "docs"
CHROMA_DIR = "chroma"
CHROMA_BATCH_LIMIT = 41_666


def build_index(
    chunks: list[Document],
    model_name: str = MODEL_NAME,
    chroma_dir: str | Path = CHROMA_DIR,
    collection_name: str = COLLECTION_NAME,
) -> chromadb.Collection:
    chroma_dir = str(Path(chroma_dir).resolve())
    client = chromadb.PersistentClient(path=chroma_dir)

    if collection_name in [c.name for c in client.list_collections()]:
        client.delete_collection(collection_name)

    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )

    model = SentenceTransformer(model_name)
    texts = [c.page_content for c in chunks]
    embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=True).tolist()

    ids = [c.metadata["chunk_id"] for c in chunks]
    metadatas = [
        {
            "source": c.metadata.get("source", ""),
            "page": c.metadata.get("page", 0),
            "chunk_id": c.metadata["chunk_id"],
            "chunk_index": c.metadata.get("chunk_index", 0),
        }
        for c in chunks
    ]

    for start in range(0, len(ids), CHROMA_BATCH_LIMIT):
        end = start + CHROMA_BATCH_LIMIT
        collection.add(
            ids=ids[start:end],
            embeddings=embeddings[start:end],
            documents=texts[start:end],
            metadatas=metadatas[start:end],
        )

    return collection


if __name__ == "__main__":
    from rag.ingest.chunk import chunk, load_documents

    corpus_dir = Path("data/corpus")
    print(f"Loading documents from {corpus_dir.resolve()} ...")

    docs = load_documents(corpus_dir)
    print(f"Loaded {len(docs)} raw document pages/files")

    chunks = chunk(docs)
    print(f"Created {len(chunks)} chunks")

    print(f"\nBuilding index with {MODEL_NAME} ...")
    collection = build_index(chunks)
    print(f"Indexed {collection.count()} chunks into ChromaDB at {CHROMA_DIR}/")
