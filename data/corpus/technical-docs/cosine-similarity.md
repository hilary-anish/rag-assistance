# Cosine Similarity and Distance Metrics in Vector Search

## Overview

Vector search relies on mathematical measures of similarity or distance between vectors to find the most relevant documents for a given query. The choice of metric affects both the quality of retrieval results and the types of index structures that can be used for efficient search. The three most common metrics are cosine similarity, dot product, and Euclidean distance.

## Cosine Similarity

Cosine similarity measures the cosine of the angle between two vectors, independent of their magnitudes. It ranges from -1 (opposite directions) to +1 (same direction), with 0 indicating orthogonality.

```
cosine_similarity(A, B) = (A · B) / (‖A‖ × ‖B‖)
```

Where `A · B` is the dot product and `‖A‖` is the L2 norm (Euclidean length) of vector A.

### Properties

- **Magnitude invariant**: Cosine similarity is unaffected by vector scaling. A vector [1, 2, 3] and [2, 4, 6] have cosine similarity of 1.0. This is desirable when the direction of the vector (representing semantic meaning) matters more than its magnitude.
- **Bounded range**: The output is always in [-1, 1] for real-valued vectors, making it interpretable and easy to threshold.
- **Equivalent to dot product for normalized vectors**: When vectors are L2-normalized (unit vectors), cosine similarity equals the dot product. Most modern embedding models output normalized vectors, so in practice the two metrics are interchangeable.

### Cosine Distance

Cosine distance is derived from cosine similarity:

```
cosine_distance(A, B) = 1 - cosine_similarity(A, B)
```

This transforms the similarity into a distance metric in the range [0, 2], where 0 means identical direction. Cosine distance is not a true metric in the mathematical sense because it does not satisfy the triangle inequality in all cases, though it does for non-negative vectors.

## Dot Product (Inner Product)

The dot product is the sum of element-wise products of two vectors:

```
dot_product(A, B) = Σ(a_i × b_i) for i = 1 to d
```

### Properties

- **Magnitude sensitive**: Unlike cosine similarity, the dot product scales with vector magnitudes. Longer vectors produce larger dot products, which can be either an advantage or a disadvantage depending on the application.
- **Faster computation**: The dot product requires no normalization step, making it slightly faster than cosine similarity for unnormalized vectors.
- **Maximum inner product search (MIPS)**: When using models that encode relevance in vector magnitude (e.g., some recommendation systems where magnitude indicates confidence), dot product is the appropriate metric. MIPS is a well-studied problem with dedicated index structures.

### When to Use Dot Product Over Cosine

- When the embedding model produces normalized vectors (the metrics are equivalent).
- When vector magnitude carries meaningful information, such as term importance or confidence scores.
- When performance is critical and you want to avoid the normalization overhead.

## Euclidean Distance (L2 Distance)

Euclidean distance is the straight-line distance between two points in the vector space:

```
euclidean_distance(A, B) = √(Σ(a_i - b_i)² for i = 1 to d)
```

In practice, the squared Euclidean distance is often used to avoid the square root computation, as it preserves the ordering of distances.

### Properties

- **True metric**: Euclidean distance satisfies all metric space axioms (non-negativity, identity of indiscernibles, symmetry, and triangle inequality), enabling the use of metric tree index structures like ball trees and VP-trees.
- **Magnitude sensitive**: Like dot product, Euclidean distance is affected by vector magnitudes.
- **Relationship to cosine similarity**: For L2-normalized vectors, squared Euclidean distance and cosine similarity are monotonically related: `‖A - B‖² = 2(1 - cos(A, B))`. This means ranking by one is equivalent to ranking by the other for normalized vectors.

## Other Distance Metrics

### Manhattan Distance (L1 Distance)

```
manhattan_distance(A, B) = Σ|a_i - b_i| for i = 1 to d
```

Manhattan distance sums the absolute differences across dimensions. It is more robust to outliers than Euclidean distance because it does not square the differences. In high-dimensional spaces, the relative contrast between nearest and farthest neighbors diminishes for all Lp norms, but L1 tends to maintain slightly better discrimination than L2.

### Hamming Distance

Hamming distance counts the number of positions where corresponding elements differ. It is primarily used with binary embeddings (vectors of 0s and 1s) and can be computed extremely efficiently using XOR and popcount CPU instructions. Binary embeddings sacrifice accuracy for massive gains in storage and computation speed—a binary vector of 768 bits requires only 96 bytes compared to 3,072 bytes for a float32 vector of the same dimensionality.

### Jaccard Similarity

Jaccard similarity measures the overlap between two sets:

```
jaccard(A, B) = |A ∩ B| / |A ∪ B|
```

It is used primarily with sparse representations like bag-of-words or binary feature vectors rather than dense embeddings.

## Choosing a Metric for Vector Search

| Scenario | Recommended Metric |
|---|---|
| Normalized embeddings (most embedding models) | Dot product (equivalent to cosine similarity, fastest) |
| Unnormalized embeddings where direction matters | Cosine similarity |
| Unnormalized embeddings where magnitude matters | Dot product |
| Need for exact metric space properties | Euclidean distance |
| Binary embeddings | Hamming distance |
| Sparse, set-based representations | Jaccard similarity |

## Impact on Approximate Nearest Neighbor (ANN) Indexes

The choice of distance metric affects which ANN index structures are available and how they perform:

- **HNSW (Hierarchical Navigable Small World)**: Supports cosine, dot product, and Euclidean. The graph construction uses the chosen metric for neighbor selection, so switching metrics requires rebuilding the index.
- **IVF (Inverted File Index)**: Clusters are formed using k-means with the selected metric. Cosine similarity–based IVF typically normalizes vectors first and uses L2 for clustering.
- **Product Quantization (PQ)**: Works with Euclidean distance natively. For cosine similarity, vectors are normalized before quantization.
- **ScaNN (Scalable Nearest Neighbors)**: Uses a learned quantization approach optimized for maximum inner product search.

## Practical Recommendations

1. **Normalize your vectors** if the embedding model does not do so by default. This makes cosine similarity and dot product equivalent, giving you the fastest metric (dot product) without sacrificing correctness.
2. **Use the metric the embedding model was trained with**. If the model was trained with cosine similarity loss, use cosine similarity (or dot product after normalization). Mixing metrics can degrade retrieval quality.
3. **Benchmark on your data**. The theoretical differences between metrics are often smaller than the practical differences caused by data distribution, embedding quality, and index configuration.
