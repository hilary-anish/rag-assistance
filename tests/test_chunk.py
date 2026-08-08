from __future__ import annotations

from pathlib import Path

import pytest
from langchain_core.documents import Document

from rag.ingest.chunk import chunk, load_documents


@pytest.fixture()
def corpus_dir(tmp_path: Path) -> Path:
    doc = tmp_path / "sample.txt"
    doc.write_text("Line one.\n\nLine two.\n\nLine three.", encoding="utf-8")
    return tmp_path


@pytest.fixture()
def large_corpus_dir(tmp_path: Path) -> Path:
    doc = tmp_path / "large.txt"
    doc.write_text("word " * 500, encoding="utf-8")
    return tmp_path


class TestLoadDocuments:
    def test_returns_documents_from_folder(self, corpus_dir: Path):
        docs = load_documents(corpus_dir)
        assert len(docs) >= 1
        assert all(isinstance(d, Document) for d in docs)

    def test_sets_source_metadata(self, corpus_dir: Path):
        docs = load_documents(corpus_dir)
        assert docs[0].metadata["source"] == "sample.txt"

    def test_ignores_unsupported_extensions(self, tmp_path: Path):
        (tmp_path / "image.png").write_bytes(b"\x89PNG")
        docs = load_documents(tmp_path)
        assert docs == []


class TestChunk:
    def test_produces_chunks_with_metadata(self, corpus_dir: Path):
        docs = load_documents(corpus_dir)
        chunks = chunk(docs)
        assert len(chunks) >= 1
        for c in chunks:
            assert "chunk_id" in c.metadata
            assert "chunk_index" in c.metadata
            assert "source" in c.metadata

    def test_chunk_index_starts_at_zero(self, corpus_dir: Path):
        docs = load_documents(corpus_dir)
        chunks = chunk(docs)
        assert chunks[0].metadata["chunk_index"] == 0

    def test_chunk_size_respected(self, large_corpus_dir: Path):
        docs = load_documents(large_corpus_dir)
        chunks = chunk(docs, size=200, overlap=20)
        assert len(chunks) > 1
        for c in chunks:
            assert len(c.page_content) <= 200

    def test_overlap_produces_more_chunks(self, large_corpus_dir: Path):
        docs = load_documents(large_corpus_dir)
        no_overlap = chunk(docs, size=200, overlap=0)
        with_overlap = chunk(docs, size=200, overlap=50)
        assert len(with_overlap) >= len(no_overlap)

    def test_chunk_ids_are_unique(self, large_corpus_dir: Path):
        docs = load_documents(large_corpus_dir)
        chunks = chunk(docs, size=200, overlap=20)
        ids = [c.metadata["chunk_id"] for c in chunks]
        assert len(ids) == len(set(ids))
