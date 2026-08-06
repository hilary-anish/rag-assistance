# Vector Embeddings

## Overview

Vector embeddings are dense numerical representations of data—text, images, audio, or other modalities—mapped into a continuous vector space. Each embedding is a fixed-length array of floating-point numbers (typically 128 to 4096 dimensions) where semantic similarity between inputs is preserved as geometric proximity in the vector space. Two sentences with similar meanings will have embeddings that are close together, while unrelated sentences will be far apart.

Embeddings are the foundational building block of modern information retrieval, recommendation systems, and retrieval-augmented generation (RAG) pipelines.

## How Embeddings Work

An embedding model is a neural network trained to map inputs to vectors such that semantically similar inputs cluster together. The training process typically involves a contrastive objective: the model learns to pull positive pairs (semantically similar inputs) closer together and push negative pairs apart.

Given an input text $x$, an embedding model $f$ produces a vector:

```
v = f(x) ∈ ℝ^d
```

where $d$ is the embedding dimensionality. The model parameters are learned during training so that for similar inputs $x_1$ and $x_2$, the distance $\|f(x_1) - f(x_2)\|$ is small.

### Training Objectives

- **Contrastive loss**: Minimizes distance between positive pairs and maximizes distance between negative pairs. Common formulations include triplet loss and InfoNCE.
- **Cosine similarity loss**: Directly optimizes the cosine similarity between embeddings of matched pairs.
- **Multiple Negatives Ranking Loss (MNRL)**: Given a batch of (query, positive) pairs, treats all other positives in the batch as negatives. Efficient because it requires no explicit negative sampling.

## Types of Embedding Models

### Word2Vec (2013)

Word2Vec, developed by Mikolov et al. at Google, produces static word-level embeddings. It offers two architectures:

- **CBOW (Continuous Bag of Words)**: Predicts a target word from its surrounding context words. Faster to train and works well with frequent words.
- **Skip-gram**: Predicts surrounding context words from a target word. Performs better on rare words and smaller datasets.

Word2Vec embeddings are typically 100–300 dimensions. A key limitation is that each word receives a single embedding regardless of context—"bank" in "river bank" and "bank account" share the same vector.

### GloVe (2014)

Global Vectors for Word Representation (GloVe), developed at Stanford, constructs embeddings from a global word co-occurrence matrix. Unlike Word2Vec's local context window approach, GloVe captures corpus-wide statistical information. The resulting embeddings exhibit linear substructures that encode analogies (e.g., king - man + woman ≈ queen).

### BERT Embeddings (2018)

BERT (Bidirectional Encoder Representations from Transformers) produces contextualized embeddings—the same word receives different vectors depending on its surrounding context. BERT uses a Transformer encoder with 12 or 24 layers and is pre-trained on masked language modeling and next-sentence prediction.

To extract embeddings from BERT, common approaches include:

- **[CLS] token embedding**: Using the output vector of the special classification token.
- **Mean pooling**: Averaging the output vectors across all tokens.
- **Max pooling**: Taking the element-wise maximum across token vectors.

BERT embeddings are 768-dimensional (base) or 1024-dimensional (large). While powerful for downstream tasks, raw BERT embeddings without fine-tuning perform poorly for semantic similarity because the [CLS] token is not trained for that purpose.

### Sentence-Transformers (2019)

Sentence-Transformers, introduced by Reimers and Gurevych, fine-tune BERT-like models specifically for producing high-quality sentence and paragraph embeddings. The key innovation is the Siamese/triplet network architecture that processes sentence pairs through shared weights and trains with a similarity objective.

Popular models include:

- **all-MiniLM-L6-v2**: 384 dimensions, fast inference, good general-purpose quality. Roughly 80M parameters.
- **all-mpnet-base-v2**: 768 dimensions, higher quality, slower inference. Based on MPNet architecture.
- **e5-large-v2**: 1024 dimensions, strong performance on retrieval benchmarks. Trained with weakly supervised contrastive pre-training.
- **BGE (BAAI General Embedding)**: Family of models from Beijing Academy of AI, competitive with proprietary embeddings on MTEB benchmarks.

### Commercial Embedding APIs

- **OpenAI text-embedding-3-small/large**: 1536 or 3072 dimensions, supports Matryoshka representation learning for flexible dimensionality reduction.
- **Cohere embed-v3**: 1024 dimensions, trained with compression-aware objectives. Supports int8 and binary quantization natively.
- **Voyage AI**: Domain-specific models (code, legal, finance) with strong benchmark performance.

## Properties of Good Embeddings

- **Semantic fidelity**: Similar meanings map to nearby vectors.
- **Isotropy**: Embeddings are roughly uniformly distributed in the vector space rather than clustered in a narrow cone (anisotropy degrades retrieval quality).
- **Dimensionality efficiency**: The information is distributed across dimensions without excessive redundancy.
- **Domain alignment**: The embedding space reflects the similarity structure relevant to the target domain.

## Practical Considerations

### Dimensionality

Higher dimensions capture more information but increase storage and computation costs. For most applications, 384–1024 dimensions provide a good trade-off. Matryoshka Representation Learning (MRL) trains models so that truncating the vector to fewer dimensions preserves most of the quality, enabling flexible dimension-accuracy trade-offs at query time.

### Normalization

Most embedding models output L2-normalized vectors (unit vectors on a hypersphere). With normalized vectors, cosine similarity reduces to a dot product, simplifying computation. If an embedding model does not normalize by default, explicit normalization is recommended before indexing.

### Batching and Throughput

Embedding generation benefits significantly from batching. Processing 32–512 texts per batch on a GPU can increase throughput by 10–50x compared to single-text inference. For large-scale indexing, this is critical for practical wall-clock times.

### Storage

A single float32 embedding of 1536 dimensions requires 6,144 bytes. For 1 million documents, this amounts to roughly 6 GB of raw embedding storage. Quantization techniques (float16, int8, binary) can reduce this by 2–8x with minimal quality loss.
