# Cross-Encoder vs. Bi-Encoder for Retrieval and Reranking

## Overview

Bi-encoders and cross-encoders are two architectures for computing the semantic similarity between text pairs. They represent a fundamental trade-off between computational efficiency and accuracy. In modern retrieval systems, they are typically used together in a two-stage retrieve-then-rerank pipeline: the bi-encoder performs fast initial retrieval over millions of documents, and the cross-encoder reranks the top results for higher precision.

## Bi-Encoder Architecture

A bi-encoder (also called a dual encoder or two-tower model) encodes the query and the document independently using the same or separate encoder networks. The similarity between the query and document is computed from their individual embeddings, typically using cosine similarity or dot product.

### How It Works

1. The query $q$ is encoded to a vector: $v_q = \text{Encoder}(q)$.
2. The document $d$ is encoded to a vector: $v_d = \text{Encoder}(d)$.
3. Similarity is computed: $\text{score}(q, d) = \text{cosine}(v_q, v_d)$.

The key property is that the query and document encodings are completely independent—the document embedding does not depend on the query. This allows all document embeddings to be pre-computed and indexed offline.

### Advantages

- **Scalable retrieval**: Document embeddings are computed once and stored. At query time, only the query needs to be encoded. Approximate nearest neighbor (ANN) search over millions or billions of pre-computed embeddings takes milliseconds.
- **Low latency**: A single forward pass for the query embedding (typically 5–20ms on GPU) plus a vector search (typically 1–10ms) yields total retrieval latency under 50ms.
- **Asymmetric encoding**: The query and document can use different encoders optimized for their respective roles. Some models use a lightweight query encoder and a heavier document encoder, since documents are encoded offline.

### Limitations

- **No cross-attention**: Because the query and document are encoded independently, the model cannot perform fine-grained token-level interaction between them. It must compress all information about a text into a single fixed-size vector, losing nuance.
- **Lower accuracy**: On retrieval benchmarks, bi-encoders consistently underperform cross-encoders, particularly for queries that require understanding subtle word interactions or negation.
- **Embedding quality ceiling**: The quality of retrieval is bounded by how well the embedding captures the text's meaning. Ambiguous queries or complex documents may not be well-represented by a single vector.

### Training

Bi-encoders are trained with contrastive learning objectives:

- **In-batch negatives**: Given a batch of (query, positive_document) pairs, all other documents in the batch serve as negatives. Efficient but may include false negatives.
- **Hard negatives**: Mining difficult negatives (documents that are similar to the query but not relevant) significantly improves training quality. Common sources include BM25 top results or previous bi-encoder retrieval results.
- **Distillation from cross-encoder**: A cross-encoder scores a large set of (query, document) pairs, and the bi-encoder is trained to reproduce these scores. This transfers some of the cross-encoder's accuracy to the bi-encoder.

### Notable Bi-Encoder Models

- **DPR (Dense Passage Retrieval)**: Separate BERT encoders for queries and passages. Trained on Natural Questions with in-batch negatives.
- **E5 (EmbEddings from bidirEctional Encoder rEpresentations)**: Trained with weakly supervised contrastive pre-training on large text pair datasets.
- **BGE (BAAI General Embedding)**: Includes instruction-tuned variants where a task-specific prefix improves retrieval quality.
- **GTE (General Text Embeddings)**: Multi-stage training with contrastive learning on diverse text pairs.
- **Nomic Embed**: Open-source, long-context (8192 tokens) bi-encoder with strong benchmark performance.

## Cross-Encoder Architecture

A cross-encoder processes the query and document jointly as a single input sequence. Both texts are concatenated (with a separator token) and fed through a single Transformer, allowing full cross-attention between all query and document tokens. The output is a relevance score rather than separate embeddings.

### How It Works

1. Concatenate query and document: $\text{input} = [\text{CLS}] \; q \; [\text{SEP}] \; d \; [\text{SEP}]$.
2. Process through a Transformer encoder.
3. Take the [CLS] token output and pass it through a classification head to produce a relevance score: $\text{score}(q, d) = \sigma(W \cdot h_{\text{CLS}} + b)$.

### Advantages

- **Higher accuracy**: Cross-attention between query and document tokens allows the model to capture fine-grained interactions. The model can recognize that "Python is not a snake" is not relevant to "types of snakes," whereas a bi-encoder might match on the shared "snake" token.
- **Handles complex matching**: Negation, coreference, paraphrase, and compositional meaning are better captured because the model sees both texts simultaneously.
- **Strong zero-shot performance**: Cross-encoders fine-tuned on diverse relevance datasets (like MS MARCO) generalize well to new domains without additional training.

### Limitations

- **Not scalable for retrieval**: Because the score depends on both the query and the document, every (query, document) pair requires a full forward pass. Scoring 1 million documents requires 1 million forward passes per query, which is prohibitively slow.
- **No pre-computation**: Document representations cannot be pre-computed because they depend on the query. There is no embedding to index.
- **Latency**: Reranking 100 documents with a cross-encoder takes approximately 200–500ms on a GPU, depending on document length and model size. This is acceptable as a reranking step but not for initial retrieval.

### Training

Cross-encoders are trained as binary classifiers or regression models on (query, document, relevance_label) triples:

- **Binary classification**: Relevant pairs labeled 1, irrelevant labeled 0. Trained with binary cross-entropy loss.
- **Regression**: Relevance is a continuous score (e.g., 0–3 from human annotations). Trained with MSE loss.
- **Pairwise ranking**: Given a query and two documents (one more relevant), trained to rank them correctly using margin-based or cross-entropy loss.

### Notable Cross-Encoder Models

- **MS MARCO cross-encoders**: Fine-tuned on the MS MARCO passage ranking dataset with approximately 500K training queries.
- **MonoT5**: A T5-based cross-encoder that frames relevance as a text generation task—the model generates "true" or "false" and the probability of "true" is the relevance score.
- **Cohere Rerank**: A commercial cross-encoder API optimized for reranking retrieved documents. Supports long documents (up to 4096 tokens).
- **BGE Reranker**: Open-source cross-encoder from BAAI with competitive performance.
- **RankLLaMA**: Uses a decoder-only LLM as a cross-encoder, achieving strong reranking performance by leveraging the LLM's pre-trained knowledge.

## The Two-Stage Retrieve-and-Rerank Pipeline

The standard architecture combines both models to get the best of both worlds:

### Stage 1: Retrieval (Bi-Encoder)

- Retrieve the top-k candidates (typically k = 100–1000) from the full document corpus using the bi-encoder and ANN search.
- Latency: ~10–50ms.
- Purpose: Efficiently narrow down the candidate set from millions to hundreds.

### Stage 2: Reranking (Cross-Encoder)

- Score each of the top-k candidates with the cross-encoder.
- Re-sort by cross-encoder score.
- Return the top-n results (typically n = 5–20).
- Latency: ~200–500ms for 100 candidates.
- Purpose: Improve precision by applying fine-grained relevance scoring.

### Why Two Stages?

The bi-encoder alone provides good recall (the relevant document is usually in the top-100) but imperfect precision (it may not be ranked #1). The cross-encoder has excellent precision but cannot be applied to the full corpus. Together, the bi-encoder ensures the relevant documents are in the candidate set, and the cross-encoder puts the best ones at the top.

### Practical Impact

On the MS MARCO passage ranking benchmark, a typical bi-encoder achieves MRR@10 of ~0.33–0.38. Adding a cross-encoder reranker improves this to ~0.38–0.42. The improvement is consistent across domains and query types, making reranking a standard component in production retrieval systems.

## Late Interaction Models

Late interaction models (ColBERT and variants) represent a middle ground between bi-encoders and cross-encoders:

- Each token in the query and document gets its own embedding (not pooled into a single vector).
- Similarity is computed as the sum of maximum similarities between each query token and all document tokens (MaxSim operation).
- Document token embeddings can be pre-computed and indexed, enabling retrieval-time efficiency closer to bi-encoders.
- Accuracy is closer to cross-encoders because token-level interactions are preserved.

ColBERTv2 reduces the storage overhead through residual compression, bringing per-document storage down from ~1.5 KB (128 tokens × 128 dimensions × float16 / 2) to ~25 bytes per token with aggressive quantization. This makes it practical for large-scale deployment, though still significantly more expensive than single-vector bi-encoders.
