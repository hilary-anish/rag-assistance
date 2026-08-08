from __future__ import annotations

from dataclasses import dataclass, field
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient


@dataclass
class FakeResult:
    chunk_id: str = "abc123"
    document: str = "Some retrieved text for testing."
    metadata: dict = field(default_factory=lambda: {"source": "test.pdf", "page": 1})
    score: float = 2.5


@pytest.fixture()
def client():
    with patch("rag.serving.api.HybridRetriever") as MockRetriever:
        mock_instance = MagicMock()
        mock_instance.retrieve.return_value = [FakeResult()]
        MockRetriever.return_value = mock_instance

        from rag.serving.api import app

        with TestClient(app) as c:
            yield c


class TestHealth:
    def test_health_returns_ok(self, client: TestClient):
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}


class TestAsk:
    @patch("rag.serving.api.generate", return_value="Mocked answer text.")
    def test_ask_returns_answer(self, mock_generate, client: TestClient):
        resp = client.post("/ask", json={"question": "What is RAG?"})
        assert resp.status_code == 200

        body = resp.json()
        assert body["answer"] == "Mocked answer text."
        assert body["flag"] == "grounded"
        assert body["confidence"] > 0
        assert len(body["sources"]) == 1
        assert body["sources"][0]["source"] == "test.pdf"

    @patch("rag.serving.api.generate", return_value="Answer.")
    def test_ask_passes_k_parameter(self, mock_generate, client: TestClient):
        resp = client.post("/ask", json={"question": "test", "k": 3})
        assert resp.status_code == 200

        from rag.serving.api import app
        app.state.retriever.retrieve.assert_called_with("test", k=3)

    def test_ask_rejects_missing_question(self, client: TestClient):
        resp = client.post("/ask", json={})
        assert resp.status_code == 422
