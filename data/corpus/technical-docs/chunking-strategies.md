# Document Chunking Strategies

## Overview

Chunking is the process of splitting documents into smaller segments before embedding and indexing them in a vector database. It is a critical preprocessing step in any retrieval-augmented generation (RAG) pipeline because embedding models and LLM context windows have limited input sizes, and because retrieval quality depends heavily on chunk granularity. A chunk that is too large dilutes the relevant information with noise; a chunk that is too small loses the context needed to understand the content.

## Why Chunking Matters

Embedding models produce a single vector representation for an entire input. When a 10-page document is embedded as one chunk, the resulting vector is a compressed average of all topics covered. A query about a specific detail will match poorly because the embedding is diluted by unrelated content. Conversely, chunking the document into focused segments means each embedding captures a specific topic, enabling precise retrieval.

The ideal chunk contains exactly the information needed to answer a class of questions—complete enough to be self-contained, focused enough to be relevant.

## Fixed-Size Chunking

The simplest approach splits text into chunks of a fixed number of tokens or characters with optional overlap.

### How It Works

1. Tokenize the document (or count characters).
2. Split into chunks of $n$ tokens.
3. Optionally overlap consecutive chunks by $m$ tokens.

### Parameters

- **Chunk size**: Typically 256–1024 tokens. Smaller chunks (128–256) improve retrieval precision for specific facts. Larger chunks (512–1024) provide more context for complex topics.
- **Overlap**: Typically 10–20% of chunk size. Overlap ensures that information near chunk boundaries is not lost. A sentence split between two chunks will appear in full in at least one of them if the overlap is sufficient.

### Advantages

- Dead simple to implement. No dependencies on document structure.
- Predictable chunk sizes, which simplifies batching for embedding generation and ensures consistent token usage in LLM prompts.
- Works for any text format.

### Limitations

- Ignores document structure. A chunk boundary may fall in the middle of a sentence, paragraph, or logical section.
- No guarantee that a chunk is semantically coherent—it may contain the end of one topic and the beginning of another.
- Overlap increases storage and computation by 10–20% without proportional quality gains.

## Recursive Character Text Splitting

Recursive splitting, popularized by LangChain's `RecursiveCharacterTextSplitter`, attempts to split text at natural boundaries while respecting a maximum chunk size.

### How It Works

1. Define a hierarchy of separators, from most preferred to least: `["\n\n", "\n", " ", ""]`.
2. Try to split the text using the first separator (double newline = paragraph boundary).
3. If any resulting chunk exceeds the maximum size, recursively split it using the next separator.
4. Continue until all chunks are within the size limit.

### Advantages

- Respects natural text boundaries (paragraphs, sentences, words) when possible.
- Falls back gracefully to smaller boundaries when a single paragraph is too long.
- Minimal configuration—just set the chunk size and overlap.

### Limitations

- Relies on whitespace conventions that may not hold for all document types (e.g., dense technical text without paragraph breaks, HTML, or code).
- Does not understand document semantics—it splits on formatting, not meaning.
- Different separator hierarchies may be needed for different document types (code, markdown, HTML).

### Separator Hierarchies for Different Formats

- **Markdown**: `["\n## ", "\n### ", "\n#### ", "\n\n", "\n", " "]` — split on headers first.
- **Python code**: `["\nclass ", "\ndef ", "\n\n", "\n", " "]` — split on class/function boundaries.
- **HTML**: `["</div>", "</p>", "</section>", "\n\n", "\n", " "]` — split on element boundaries.
- **LaTeX**: `["\n\\section", "\n\\subsection", "\n\n", "\n", " "]` — split on section commands.

## Semantic Chunking

Semantic chunking uses embedding similarity to determine where to split text, grouping semantically related content together regardless of formatting.

### How It Works

1. Split the text into base units (sentences or small paragraphs).
2. Embed each base unit.
3. Compute the cosine similarity between consecutive units.
4. Identify breakpoints where similarity drops below a threshold (or drops significantly relative to neighbors).
5. Group consecutive units between breakpoints into chunks.

### Advantages

- Chunks are semantically coherent—each chunk covers a single topic or subtopic.
- Adapts to the content structure rather than imposing arbitrary boundaries.
- Works well for documents with implicit topic shifts that are not marked by formatting.

### Limitations

- Requires an embedding model for preprocessing, adding computation cost and latency to the ingestion pipeline.
- The similarity threshold or breakpoint detection method requires tuning per corpus.
- Variable chunk sizes may require additional handling for context window management.
- Slower ingestion compared to rule-based methods.

### Breakpoint Detection Methods

- **Percentile threshold**: A breakpoint occurs where the inter-sentence similarity falls below the $p$-th percentile of all inter-sentence similarities in the document. Typical values are p = 25–40.
- **Standard deviation**: A breakpoint occurs where similarity drops more than $k$ standard deviations below the mean. Typical values are k = 1–2.
- **Gradient-based**: Compute the gradient of the similarity curve and place breakpoints at local minima.

## Structure-Aware Chunking

Structure-aware chunking leverages document structure (headers, sections, tables, lists) to create chunks that align with the document's organization.

### Markdown/HTML Heading-Based Splitting

Split documents at heading boundaries (H1, H2, H3), creating one chunk per section. Subsections can be kept with their parent section or split into separate chunks depending on size.

### Advantages

- Chunks align with the author's intended content organization.
- Headers provide natural metadata (section titles) that can be used for filtering or hybrid search.
- Preserves the hierarchical context (a subsection chunk can inherit its parent heading as metadata).

### Limitations

- Requires structured documents. Unstructured text (emails, chat logs, transcripts) has no headings to split on.
- Section sizes vary widely. Some sections may be too large (requiring further splitting) or too small (benefiting from merging with adjacent sections).
- Tables, code blocks, and lists within sections may need special handling.

## Specialized Chunking Approaches

### Proposition-Based Chunking

Dense-X Retrieval proposes chunking documents into self-contained propositions—atomic factual statements that can stand alone. A paragraph like "Einstein was born in Ulm in 1879. He later moved to Switzerland." becomes two propositions: "Albert Einstein was born in Ulm, Germany in 1879" and "Albert Einstein moved to Switzerland."

An LLM is used to decompose text into propositions, each rephrased to be self-contained (resolving pronouns, adding context). This maximizes retrieval precision but significantly increases the number of chunks and ingestion cost.

### Parent-Child Chunking (Small-to-Big)

Embed small chunks for precise retrieval but return their parent (larger) chunks to the LLM for generation:

1. Split documents into large parent chunks (e.g., 2048 tokens).
2. Split each parent into smaller child chunks (e.g., 256 tokens).
3. Embed and index only the child chunks.
4. At retrieval time, match on child chunks but return the parent chunk to the LLM.

This combines the retrieval precision of small chunks with the contextual completeness of large chunks.

### Sliding Window with Stride

A variant of fixed-size chunking where the window slides by a stride smaller than the chunk size, creating heavily overlapping chunks. With a chunk size of 512 tokens and a stride of 128 tokens, each token appears in approximately 4 chunks. This maximizes the chance that any relevant passage is fully contained in at least one chunk, at the cost of 4× storage and embedding computation.

## Chunk Size Selection

### Empirical Guidelines

| Use Case | Recommended Chunk Size | Rationale |
|---|---|---|
| Factoid QA | 128–256 tokens | Small chunks for precise fact retrieval |
| Complex QA / Summarization | 512–1024 tokens | Larger context for multi-sentence reasoning |
| Code retrieval | Function or class level | Logical code units; splitting mid-function is rarely useful |
| Legal/medical documents | Section level with fallback to 512 tokens | Domain documents have strong section structure |
| Conversational data | Per-message or per-turn | Natural unit of conversation |

### Evaluation-Driven Selection

The only reliable way to choose chunk size is to evaluate retrieval quality on representative queries:

1. Create a test set of (query, relevant_passage) pairs.
2. Chunk the corpus with multiple chunk sizes (e.g., 128, 256, 512, 1024).
3. Measure Recall@k and Hit@k for each configuration.
4. Select the chunk size that maximizes the target metric.

A 2× change in chunk size can produce a 5–15% change in Recall@10, making this a high-impact tuning parameter.

## Metadata Enrichment

Regardless of chunking strategy, enriching chunks with metadata improves retrieval:

- **Source document title and URL**: Enables citation and deduplication.
- **Section headers**: Parent section titles provide hierarchical context.
- **Page numbers**: For PDF documents, enabling users to locate the source.
- **Document type and date**: Enables filtering by recency or type.
- **Summary or contextual header**: Prepending a brief summary of the document or section to each chunk (contextual retrieval) helps the embedding model produce more accurate vectors.
