# Evaluation Metrics for Information Retrieval

## Overview

Evaluating information retrieval (IR) systems requires quantifying how well a system returns relevant documents in response to a query. Different metrics capture different aspects of retrieval quality: whether relevant documents are retrieved at all (recall), whether the retrieved documents are relevant (precision), and whether relevant documents appear near the top of the results (ranking quality). Choosing the right metric depends on the application—a web search engine, a RAG pipeline, and a legal discovery system have fundamentally different evaluation priorities.

## Binary Relevance Metrics

These metrics treat relevance as binary: each document is either relevant or not relevant for a given query.

### Precision

Precision measures the fraction of retrieved documents that are relevant:

```
Precision = |relevant ∩ retrieved| / |retrieved|
```

**Precision@k (P@k)** evaluates precision at a fixed cutoff of k results. P@10 measures the fraction of the top 10 results that are relevant. For example, if 7 of the top 10 results are relevant, P@10 = 0.7.

Precision is important when the cost of showing irrelevant results is high—for example, in a RAG pipeline where irrelevant retrieved passages can cause the LLM to hallucinate or generate confused responses.

### Recall

Recall measures the fraction of all relevant documents that are retrieved:

```
Recall = |relevant ∩ retrieved| / |relevant|
```

**Recall@k (R@k)** evaluates how many of the total relevant documents appear in the top k results. If there are 20 relevant documents in the corpus and 15 appear in the top 100 results, R@100 = 0.75.

Recall is critical when missing a relevant document has serious consequences—in legal e-discovery, medical literature review, or patent search, failing to find a relevant document can be more costly than returning some irrelevant ones.

### F1 Score

The F1 score is the harmonic mean of precision and recall:

```
F1 = 2 × (Precision × Recall) / (Precision + Recall)
```

It balances the trade-off between precision and recall. The harmonic mean penalizes extreme imbalances: a system with P=0.9 and R=0.1 gets F1=0.18, not 0.5.

## Ranking Metrics

Ranking metrics evaluate the quality of the ordered list of results, rewarding systems that place relevant documents higher in the ranking.

### Mean Reciprocal Rank (MRR)

MRR measures how early the first relevant document appears in the ranking:

```
MRR = (1/|Q|) × Σ (1 / rank_i) for each query i
```

where $rank_i$ is the position of the first relevant document for query $i$.

**Example**: For three queries where the first relevant document appears at positions 1, 3, and 2:

```
MRR = (1/3) × (1/1 + 1/3 + 1/2) = (1/3) × 1.833 = 0.611
```

MRR is appropriate when the user typically needs only one correct answer—for example, factoid question answering or navigational search. It ignores the positions of all relevant documents after the first one.

### Mean Average Precision (MAP)

MAP is the mean of Average Precision (AP) scores across all queries. AP for a single query is:

```
AP = (1/|relevant|) × Σ (P@k × rel(k)) for k = 1 to n
```

where $rel(k)$ is 1 if the document at rank $k$ is relevant and 0 otherwise. In other words, AP is the average of precision values computed at each rank position where a relevant document is found.

**Example**: Suppose there are 4 relevant documents and the system returns them at positions 1, 3, 6, and 10:

```
AP = (1/4) × (1/1 + 2/3 + 3/6 + 4/10) = (1/4) × (1.0 + 0.667 + 0.5 + 0.4) = 0.642
```

MAP rewards systems that rank all relevant documents highly, not just the first one. It is the standard metric for benchmarks like MS MARCO and TREC.

### MAP vs. MRR

- **MRR**: Only cares about the first relevant result. Use for single-answer tasks.
- **MAP**: Cares about all relevant results and their positions. Use when multiple relevant documents matter—e.g., comprehensive literature search, multi-passage RAG.

## Graded Relevance Metrics

These metrics use graded relevance judgments (e.g., 0 = not relevant, 1 = marginally relevant, 2 = relevant, 3 = highly relevant) rather than binary relevance.

### Normalized Discounted Cumulative Gain (NDCG)

NDCG is the standard metric for evaluating ranked results with graded relevance.

**Cumulative Gain (CG)** at rank $k$:

```
CG@k = Σ rel_i for i = 1 to k
```

This simply sums the relevance scores of the top $k$ documents. It does not account for position.

**Discounted Cumulative Gain (DCG)** introduces a position-based discount:

```
DCG@k = Σ (rel_i / log2(i + 1)) for i = 1 to k
```

Documents at lower positions contribute less to the score. The logarithmic discount models the diminishing probability that a user will examine results further down the list.

**Normalized DCG** divides DCG by the ideal DCG (IDCG)—the DCG of the perfect ranking where all documents are sorted by relevance:

```
NDCG@k = DCG@k / IDCG@k
```

NDCG ranges from 0 to 1, where 1 indicates a perfect ranking.

**Example**: Consider four documents with relevance scores [3, 1, 2, 0]:

```
DCG@4 = 3/log2(2) + 1/log2(3) + 2/log2(4) + 0/log2(5)
       = 3/1 + 1/1.585 + 2/2 + 0
       = 3.0 + 0.631 + 1.0 + 0
       = 4.631

IDCG@4 (ideal order [3, 2, 1, 0]):
       = 3/1 + 2/1.585 + 1/2 + 0
       = 3.0 + 1.262 + 0.5 + 0
       = 4.762

NDCG@4 = 4.631 / 4.762 = 0.972
```

### When to Use NDCG

NDCG is the preferred metric when:
- Relevance is not binary (some documents are more relevant than others).
- Ranking quality matters more than simple retrieval.
- The evaluation data has graded annotations (common in TREC and industry datasets).

NDCG@10 is the primary metric on the MTEB (Massive Text Embedding Benchmark) leaderboard for retrieval tasks.

## Retrieval-Specific Metrics

### Hit Rate (Hit@k)

Hit@k measures the fraction of queries for which at least one relevant document appears in the top $k$ results:

```
Hit@k = (1/|Q|) × Σ 1(at least one relevant in top k) for each query
```

Hit@k is simpler than MRR—it only asks "did we find anything relevant?" without caring about exact position. It is commonly used to evaluate the first-stage retrieval in a retrieve-and-rerank pipeline, where the goal is to ensure the relevant document is in the candidate set.

### R-Precision

R-Precision evaluates precision at the cutoff equal to the number of relevant documents:

```
R-Precision = |relevant ∩ top-R| / R
```

where $R$ is the total number of relevant documents for the query. This normalizes for the varying number of relevant documents across queries.

## Metrics for RAG Systems

RAG pipelines introduce additional evaluation dimensions beyond traditional IR:

### Retrieval Metrics (Context Quality)

- **Context Precision**: Are the retrieved passages relevant to the question?
- **Context Recall**: Do the retrieved passages contain all the information needed to answer the question?
- **Context Relevance**: What fraction of the retrieved content is actually useful (penalizes retrieving large passages when only a sentence is relevant)?

### Generation Metrics (Answer Quality)

- **Faithfulness**: Is the generated answer supported by the retrieved context? Measures whether the LLM hallucinated beyond the provided evidence.
- **Answer Relevance**: Does the generated answer actually address the question?
- **Answer Correctness**: Is the answer factually correct (compared to a ground truth)?

Frameworks like RAGAS, TruLens, and DeepEval provide automated evaluation of these metrics, often using an LLM as a judge.

## Evaluation Pitfalls

### Incomplete Relevance Judgments

Most evaluation datasets have incomplete relevance judgments—only a subset of documents are labeled for each query. Unjudged documents are typically assumed irrelevant, which penalizes systems that retrieve truly relevant but unjudged documents. Metrics like bpref (binary preference) are designed to handle incomplete judgments more fairly.

### Position Bias in Click Data

When using click logs for evaluation, users tend to click on higher-ranked results regardless of relevance (position bias). Metrics derived from click data must account for this using techniques like inverse propensity weighting.

### Metric Sensitivity to Cutoff

P@5 and P@20 can tell very different stories about the same system. Always report the cutoff value and choose it based on the application. A chatbot that shows 3 results cares about P@3; a search engine showing 10 results per page cares about P@10 and NDCG@10.

### Statistical Significance

Differences in IR metrics between systems are often small. Always report confidence intervals or conduct significance tests (paired t-test, bootstrap test) before claiming one system is better than another. A 0.01 improvement in MRR may or may not be meaningful depending on the variance.
