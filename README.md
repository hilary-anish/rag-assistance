# Document-Intelligence RAG

Production-grade Retrieval-Augmented Generation system with hybrid search, cross-encoder reranking, confidence-calibrated answers, and rigorous evaluation — running fully local on Ollama.

---

## Architecture

```
                         ┌──────────────────────────────────────────────┐
                         │              Document Ingestion              │
                         │                                              │
                         │  PDFs / Markdown                             │
                         │       │                                      │
                         │       ▼                                      │
                         │  Recursive Chunking (800 tok, 120 overlap)   │
                         │       │                                      │
                         │       ▼                                      │
                         │  BGE-small-en-v1.5 Embeddings                │
                         │       │                                      │
                         │       ▼                                      │
                         │  ChromaDB (cosine, persistent)               │
                         └──────────────────────────────────────────────┘

                         ┌──────────────────────────────────────────────┐
                         │            Hybrid Retrieval Pipeline          │
                         │                                              │
  User Query ───────────►│  ┌─────────────┐    ┌──────────────────┐     │
                         │  │ Dense Search │    │ Sparse BM25      │     │
                         │  │ (BGE embed)  │    │ (tokenized corp) │     │
                         │  └──────┬──────┘    └────────┬─────────┘     │
                         │         │                    │               │
                         │         └────────┬───────────┘               │
                         │                  ▼                           │
                         │        Reciprocal Rank Fusion                │
                         │                  │                           │
                         │                  ▼                           │
                         │     Cross-Encoder Reranking                  │
                         │     (ms-marco-MiniLM-L-6-v2)                │
                         │                  │                           │
                         └──────────────────┼───────────────────────────┘
                                            │
                         ┌──────────────────┼───────────────────────────┐
                         │                  ▼            Generation      │
                         │        Ollama (llama3.2:3b)                  │
                         │        Grounded system prompt                │
                         │        with inline citations                 │
                         │                  │                           │
                         │                  ▼                           │
                         │        Confidence Scoring                    │
                         │        (sigmoid + faithfulness blend)        │
                         │                  │                           │
                         │                  ▼                           │
                         │        Flagged Answer + Sources              │
                         └──────────────────────────────────────────────┘
```

## Key Features

**Hybrid Retrieval with Reranking.** Dense semantic search (BGE embeddings) and sparse lexical search (BM25) are fused via Reciprocal Rank Fusion, then reranked by a cross-encoder. This three-stage pipeline catches both semantic and keyword matches while promoting the most relevant chunks to the top. The ablation results below quantify the gain at each stage.

**Uncertainty-Aware Responses.** Every answer carries a calibrated confidence score and a binary flag (`grounded` / `low-confidence`). The score blends the cross-encoder's reranker signal with an optional RAGAS faithfulness score (0.6 retrieval + 0.4 faithfulness). Weak retrievals are flagged before the user ever sees the answer — not after.

**RAGAS Evaluation Harness.** End-to-end answer quality is measured with faithfulness, answer relevancy, context precision, and context recall using a local Ollama judge. No API keys, no cost, fully reproducible.

**Rigorous Retrieval Evaluation.** Two hand-curated gold QA sets (30 standard, 30 adversarial) with ablation across dense-only, sparse-only, and hybrid configurations. Metrics: Hit Rate@5 and Mean Reciprocal Rank.

**Fully Local, Zero Cost.** The entire stack — embeddings, reranking, generation, and evaluation — runs on local hardware via Ollama and HuggingFace models. No API keys, no cloud dependencies, no per-query cost.

---

## Retrieval Ablation Results

Evaluated on 30 gold QA pairs across EU AI Act regulations, RAG research papers, computer vision papers, and technical documentation.

| Configuration            | Hit Rate@5 | MRR   |
| :----------------------- | ---------: | ----: |
| Dense only               |      0.967 | 0.917 |
| Sparse (BM25) only       |      1.000 | 0.928 |
| Hybrid (RRF + rerank)    |      1.000 | 0.983 |

The hybrid pipeline achieves perfect recall and near-perfect ranking. Dense search alone misses one query where the answer relies on exact terminology (BM25 catches it). Cross-encoder reranking lifts MRR from 0.928 to 0.983 by promoting the most relevant chunk to rank 1.

---

## Confidence Scoring

Confidence scoring bridges the gap between retrieval quality and answer trustworthiness.

| Metric                              | Value  |
| :---------------------------------- | -----: |
| Grounded (high confidence)          | 26/30  |
| Flagged (low confidence)            |  4/30  |
| Average confidence                  |  0.807 |

The confidence scorer applies a sigmoid transform to the top cross-encoder score, optionally blended with RAGAS faithfulness (0.6 retrieval + 0.4 faithfulness). The binary flag uses two conditions: the top reranker score must exceed the threshold, and faithfulness (when available) must exceed 0.7.

The 4 flagged queries have negative or near-zero reranker scores — meaning the cross-encoder found weak semantic overlap between the query and the retrieved chunks, even when the correct source document appeared in the result set. This is the intended behavior: the flag catches cases where retrieval technically "hit" but the match quality is too low to trust.

---

## Tech Stack

| Layer          | Technology                                              |
| :------------- | :------------------------------------------------------ |
| Language       | Python 3.11                                             |
| Orchestration  | LangChain (document loading, text splitting)            |
| Vector Store   | ChromaDB (persistent, cosine similarity)                |
| Embeddings     | sentence-transformers (`BAAI/bge-small-en-v1.5`)       |
| Reranking      | sentence-transformers (`cross-encoder/ms-marco-MiniLM-L-6-v2`) |
| Sparse Search  | rank-bm25                                               |
| LLM            | Ollama (`llama3.2:3b`)                                  |
| API            | FastAPI + Uvicorn                                       |
| UI             | Streamlit                                               |
| Evaluation     | RAGAS, custom retrieval harness                         |
| Infrastructure | Docker, Docker Compose, GitHub Actions CI               |
| Testing        | pytest, pytest-cov, ruff                                |

---

## Quick Start

### Docker (recommended)

```bash
git clone https://github.com/hilary-anish/rag-assistance.git
cd rag-assistance

docker compose up --build
```

This starts the Ollama server and the RAG API. Once running:

```bash
# Pull the LLM (first time only)
docker compose exec ollama ollama pull llama3.2:3b

# Build the index
docker compose exec api python -m rag.ingest.index

# Test
curl http://localhost:8000/health
```

### Manual Setup

```bash
# Install uv (if not installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create environment and install
uv venv
uv pip install -e ".[dev]"

# Start Ollama and pull the model
ollama serve &
ollama pull llama3.2:3b

# Build the document index
uv run python -m rag.ingest.index

# Start the API
uv run uvicorn rag.serving.api:app --host 0.0.0.0 --port 8000

# Start the Streamlit UI (separate terminal)
uv run streamlit run app/streamlit_app.py
```

---

## API Documentation

### `GET /health`

Health check endpoint.

```bash
curl http://localhost:8000/health
```

```json
{"status": "ok"}
```

### `POST /ask`

Submit a question and receive a grounded answer with sources and confidence.

**Request:**

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What are the penalties under the EU AI Act?", "k": 5}'
```

| Field      | Type   | Default | Description                     |
| :--------- | :----- | ------: | :------------------------------ |
| `question` | string |       — | The question to answer          |
| `k`        | int    |       5 | Number of chunks to retrieve (1–20) |

**Response:**

```json
{
  "answer": "Under the EU AI Act, penalties vary by violation severity: up to €35 million or 7% of global annual turnover for prohibited AI practices [1], up to €15 million or 3% for other violations [1], and up to €7.5 million or 1% for supplying incorrect information [1].",
  "sources": [
    {
      "text": "Penalties for infringements of this Regulation are set...",
      "source": "eu-regulations/eu-ai-act.pdf",
      "page": 142
    }
  ],
  "confidence": 0.9241,
  "flag": "grounded"
}
```

| Field        | Type   | Description                                        |
| :----------- | :----- | :------------------------------------------------- |
| `answer`     | string | Generated answer with inline citations `[1]`, `[2]` |
| `sources`    | array  | Retrieved chunks with source file and page number  |
| `confidence` | float  | Calibrated confidence score (0–1)                  |
| `flag`       | string | `"grounded"` or `"low-confidence / possible hallucination"` |

---

## Project Structure

```
rag-assistance/
├── src/rag/
│   ├── ingest/
│   │   ├── chunk.py            # Document loading + recursive chunking
│   │   └── index.py            # ChromaDB index construction
│   ├── retrieve/
│   │   └── retriever.py        # HybridRetriever: dense + BM25 + RRF + rerank
│   ├── generate/
│   │   └── answer.py           # Ollama generation with grounded system prompt
│   ├── evaluate/
│   │   ├── confidence.py       # Confidence scoring + hallucination flags
│   │   ├── retrieval.py        # Hit rate, MRR evaluation harness
│   │   └── ragas_eval.py       # RAGAS end-to-end answer evaluation
│   └── serving/
│       └── api.py              # FastAPI endpoints (/health, /ask)
├── app/
│   └── streamlit_app.py        # Chat UI with confidence badges
├── data/
│   ├── corpus/                 # Source documents (PDFs, Markdown)
│   │   ├── arxiv-papers/       # RAG research papers (4 PDFs)
│   │   ├── computer-vision/    # Vision model papers (10 PDFs)
│   │   ├── eu-regulations/     # EU AI Act (1 PDF)
│   │   └── technical-docs/     # Technical explainers (10 Markdown files)
│   └── eval/
│       ├── gold_qa.json        # 30 standard gold QA pairs
│       └── gold_qa_hard.json   # 30 adversarial gold QA pairs
├── tests/
│   ├── test_chunk.py           # Chunking + document loading tests
│   ├── test_confidence.py      # Confidence scoring tests
│   ├── test_retrieval.py       # Hit rate + MRR metric tests
│   └── test_api.py             # API endpoint tests (fully mocked)
├── Dockerfile
├── docker-compose.yml
├── pyproject.toml
└── .github/workflows/ci.yml    # Lint + test on push/PR
```

---

## Evaluation Methodology

### Gold Set Design

Two gold QA sets were hand-curated, each with 30 question–answer pairs spanning all four corpus domains:

- **Standard set** (`gold_qa.json`): Questions answerable from a single clearly relevant chunk. Used for retrieval ablation and baseline metrics.
- **Adversarial set** (`gold_qa_hard.json`): Questions requiring specific numerical details, multi-hop reasoning across chunks, or distinguishing between similar concepts. Used for confidence calibration and stress testing.

Each entry contains:
- `question`: The natural-language query
- `ground_truth`: The expected answer (for RAGAS evaluation)
- `relevant_source`: The filename of the authoritative source document (for retrieval evaluation)

### Retrieval Metrics

- **Hit Rate@k**: Fraction of queries where the relevant source appears anywhere in the top-k results. Measures recall — does the pipeline find the right document?
- **Mean Reciprocal Rank (MRR)**: Average of 1/rank for the first relevant result. Measures ranking quality — is the right document at the top?

Both metrics are computed per-query and averaged. The ablation table above reports results at k=5 across three configurations: dense-only, sparse-only, and the full hybrid pipeline.

### RAGAS Metrics

End-to-end answer quality is evaluated with four RAGAS metrics, using a local Ollama model as the LLM judge:

- **Faithfulness**: Does the answer only contain claims supported by the retrieved context?
- **Answer Relevancy**: Does the answer address the question asked?
- **Context Precision**: Are the relevant chunks ranked higher than irrelevant ones?
- **Context Recall**: Does the retrieved context cover all aspects of the ground truth?

### Confidence Scoring

The confidence pipeline operates post-generation and provides a trust signal independent of the LLM's own certainty:

1. **Sigmoid transform** on the top cross-encoder score maps raw reranker logits to [0, 1]
2. **Faithfulness blending** (when available): `0.6 * retrieval_confidence + 0.4 * faithfulness`
3. **Binary flag**: `grounded` requires top score > threshold AND faithfulness > 0.7; otherwise `low-confidence / possible hallucination`
4. **Margin**: Gap between the top two reranker scores — a large margin indicates the retriever is decisive, a small margin suggests ambiguity

---

## Future Improvements

- **Streaming generation** — Token-by-token streaming in the API and Streamlit UI for better UX on longer answers
- **Multi-hop retrieval** — Iterative retrieval for questions that require synthesizing information across multiple documents
- **Chunk-level citation grounding** — Map each sentence in the generated answer to the specific chunk that supports it, enabling per-claim verification
- **Adaptive chunking** — Use document structure (headings, sections, tables) instead of fixed token windows to produce more semantically coherent chunks
- **User feedback loop** — Log queries, retrieved chunks, and user ratings to identify retrieval gaps and fine-tune the reranker
- **Embedding model fine-tuning** — Fine-tune BGE on the domain corpus with contrastive learning to improve dense retrieval on domain-specific terminology
- **Query expansion** — Use the LLM to generate multiple query reformulations before retrieval, improving recall on ambiguous or under-specified queries
