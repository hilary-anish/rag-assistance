from __future__ import annotations

import hashlib
import os
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

SUPPORTED_TEXT_EXTENSIONS = {".md", ".txt"}
SUPPORTED_PDF_EXTENSIONS = {".pdf"}


def load_documents(folder: str | Path) -> list[Document]:
    folder = Path(folder)
    docs: list[Document] = []

    for path in sorted(folder.rglob("*")):
        if not path.is_file():
            continue

        suffix = path.suffix.lower()
        rel = str(path.relative_to(folder))

        if suffix in SUPPORTED_PDF_EXTENSIONS:
            loader = PyPDFLoader(str(path))
            for doc in loader.load():
                doc.metadata["source"] = rel
                docs.append(doc)

        elif suffix in SUPPORTED_TEXT_EXTENSIONS:
            loader = TextLoader(str(path), encoding="utf-8")
            for doc in loader.load():
                doc.metadata["source"] = rel
                doc.metadata["page"] = 0
                docs.append(doc)

    return docs


def _make_chunk_id(source: str, page: int, index: int) -> str:
    raw = f"{source}::page{page}::chunk{index}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def chunk(
    docs: list[Document],
    size: int = 800,
    overlap: int = 120,
) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=size,
        chunk_overlap=overlap,
        length_function=len,
        separators=["\n\n", "\n", " ", ""],
    )

    chunks: list[Document] = []
    counter: dict[tuple[str, int], int] = {}

    for doc in docs:
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", 0)
        splits = splitter.split_documents([doc])

        for split in splits:
            key = (source, page)
            idx = counter.get(key, 0)
            counter[key] = idx + 1

            split.metadata["chunk_id"] = _make_chunk_id(source, page, idx)
            split.metadata["chunk_index"] = idx
            chunks.append(split)

    return chunks


if __name__ == "__main__":
    corpus_dir = Path(os.environ.get("CORPUS_DIR", "data/corpus"))
    print(f"Loading documents from {corpus_dir.resolve()} ...")

    docs = load_documents(corpus_dir)
    print(f"Loaded {len(docs)} raw document pages/files")

    chunks = chunk(docs)
    print(f"Created {len(chunks)} chunks (size=800, overlap=120)")

    if chunks:
        sample = chunks[0]
        print(f"\nSample chunk metadata: {sample.metadata}")
        print(f"Sample chunk text ({len(sample.page_content)} chars): {sample.page_content[:120]}...")
