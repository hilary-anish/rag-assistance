# Fine-Tuning vs. RAG: When to Use Each

## Overview

Fine-tuning and Retrieval-Augmented Generation (RAG) are two approaches for adapting large language models (LLMs) to specific domains or tasks. Fine-tuning modifies the model's weights by training on domain-specific data. RAG augments the model's input by retrieving relevant documents at inference time and including them in the prompt context. The two approaches are not mutually exclusive and can be combined, but understanding their trade-offs is essential for choosing the right approach for a given problem.

## Fine-Tuning

Fine-tuning continues the training process on a domain-specific dataset, adjusting some or all of the model's parameters to specialize its behavior.

### Types of Fine-Tuning

- **Full fine-tuning**: Updates all model parameters. Requires significant compute (comparable to pre-training for large updates) and risks catastrophic forgetting of general capabilities. Rarely practical for models above 7B parameters without substantial infrastructure.
- **LoRA (Low-Rank Adaptation)**: Freezes original weights and trains low-rank decomposition matrices ($A \in \mathbb{R}^{d \times r}$, $B \in \mathbb{R}^{r \times d}$, where $r \ll d$) that are added to the original weight matrices. Typically trains only 0.1–1% of parameters. A 7B model might add only 4–20M trainable parameters.
- **QLoRA**: Combines LoRA with 4-bit quantization of the base model, enabling fine-tuning of 65B+ models on a single GPU with 48 GB of VRAM.
- **Instruction tuning**: Fine-tuning on (instruction, response) pairs to improve the model's ability to follow directions. This is how base models become chat models.
- **RLHF/DPO**: Alignment training using human preferences. Reinforcement Learning from Human Feedback (RLHF) uses a reward model and PPO; Direct Preference Optimization (DPO) optimizes preferences directly without a separate reward model.

### When to Fine-Tune

- **Style and format adaptation**: When the model needs to consistently produce output in a specific format, tone, or style that is difficult to achieve through prompting alone. Medical report generation, legal document drafting, or matching an organization's communication style.
- **Domain-specific reasoning**: When the model needs to learn reasoning patterns unique to a domain. Financial analysis, scientific reasoning, or code generation in a niche programming language.
- **Latency-critical applications**: Fine-tuning bakes knowledge into the model weights, eliminating the retrieval step and reducing inference latency. The model generates answers directly without needing to process long retrieved contexts.
- **Consistent behavioral changes**: When you need to modify how the model responds across all inputs, not just specific queries—for example, always responding in a certain persona or always structuring output as JSON.
- **Small, stable knowledge bases**: When the knowledge is relatively static and small enough to be learned during training.

### Limitations of Fine-Tuning

- **Knowledge cutoff**: The fine-tuned model only knows what was in its training data. It cannot access information that has changed since training.
- **Hallucination risk**: Fine-tuning can increase confident hallucination if the training data does not cover a query's topic. The model may generate plausible-sounding but incorrect information.
- **Cost and iteration speed**: Fine-tuning requires a curated dataset, compute resources, and time. Iterating on the training data and retraining is slower than updating a document store.
- **Catastrophic forgetting**: Aggressive fine-tuning can degrade the model's general capabilities. The model may become very good at domain tasks but worse at general reasoning.

## Retrieval-Augmented Generation (RAG)

RAG retrieves relevant documents from an external knowledge base and includes them in the LLM's prompt context, grounding the model's response in retrieved evidence.

### RAG Pipeline Components

1. **Document ingestion**: Documents are chunked, embedded, and stored in a vector database.
2. **Query processing**: The user query is embedded using the same embedding model.
3. **Retrieval**: The top-k most similar document chunks are retrieved from the vector database.
4. **Augmentation**: Retrieved chunks are formatted and prepended to the query as context.
5. **Generation**: The LLM generates a response conditioned on the retrieved context.

### When to Use RAG

- **Dynamic or frequently updated knowledge**: When the information changes regularly—product catalogs, documentation, news, policies, pricing. Updating a document store is instantaneous compared to retraining.
- **Attribution and verifiability**: RAG naturally supports citations. The model can reference specific source documents, and users can verify the information. This is critical in legal, medical, and compliance contexts.
- **Large knowledge bases**: When the total knowledge exceeds what can be practically encoded in model weights. A company with 100,000 support articles cannot fine-tune all that content into a model, but RAG can retrieve the relevant subset per query.
- **Reducing hallucination**: By grounding responses in retrieved documents, RAG reduces (but does not eliminate) hallucination. The model is constrained to information present in the context.
- **Multi-tenant or access-controlled data**: RAG can apply per-user or per-role document filters at retrieval time, ensuring users only see information they are authorized to access.
- **Rapid prototyping**: A RAG system can be built and iterated on in days. Embedding documents and writing a retrieval pipeline requires no model training.

### Limitations of RAG

- **Retrieval quality ceiling**: The generated response can only be as good as the retrieved documents. If the retriever fails to find the relevant information, the generator cannot compensate.
- **Latency overhead**: The retrieval step adds latency—typically 50–200ms for a vector search plus embedding computation. For multi-stage retrieval (retrieve-then-rerank), latency can reach 500ms+.
- **Context window constraints**: Retrieved documents consume prompt tokens. With large documents or many retrieved chunks, the context window can be exhausted, leaving less room for the model's reasoning.
- **Complex reasoning over scattered information**: When answering a question requires synthesizing information from many documents that must be understood together, RAG's retrieve-top-k approach may not surface all necessary pieces.
- **Chunk boundary artifacts**: Information split across chunk boundaries may be partially retrieved, leading to incomplete context.

## Decision Framework

| Factor | Favors Fine-Tuning | Favors RAG |
|---|---|---|
| Knowledge update frequency | Static, rarely changes | Dynamic, frequently updated |
| Knowledge base size | Small to medium | Any size |
| Need for citations | Not required | Required |
| Latency requirements | Ultra-low latency | Moderate latency acceptable |
| Budget | Training compute available | Minimal upfront investment |
| Behavioral changes | Need consistent style/format changes | Need factual grounding |
| Hallucination tolerance | Can tolerate some | Must minimize |

## Combining Fine-Tuning and RAG

The most effective production systems often combine both approaches:

- **Fine-tune for style, RAG for knowledge**: Fine-tune the model to produce outputs in the desired format and tone, while using RAG to supply the factual content. For example, a medical chatbot fine-tuned to use clinical language and structured responses, with RAG providing the specific medical knowledge.
- **Fine-tune the retriever**: Train a domain-specific embedding model to improve retrieval quality for specialized vocabularies or domains where general-purpose embeddings underperform.
- **Fine-tune for RAG-specific behavior**: Train the model to better utilize retrieved context—learning to cite sources, ignore irrelevant retrieved passages, and say "I don't have enough information" when the retrieved context is insufficient.
- **RAFT (Retrieval Augmented Fine-Tuning)**: A technique that fine-tunes the model on examples that include both relevant and irrelevant retrieved documents, teaching the model to extract the correct answer from noisy context.

## Cost Comparison

### Fine-Tuning Costs

- Dataset preparation: Human time for curation (often the largest cost).
- Training compute: A LoRA fine-tune of a 7B model costs approximately $5–50 on cloud GPUs.
- Iteration: Each experiment requires re-training (hours to days).
- Hosting: The fine-tuned model must be served, with costs proportional to model size.

### RAG Costs

- Embedding generation: One-time cost of ~$0.01–0.10 per million tokens for commercial APIs.
- Vector database: Storage and query costs scale with document count. Managed services charge per million vectors per month.
- Retrieval latency: Per-query cost of embedding the query + vector search.
- LLM inference: Longer prompts (with retrieved context) cost more per query than non-RAG queries.

For most organizations, RAG has a lower barrier to entry and faster iteration cycle. Fine-tuning becomes cost-effective when the behavioral changes are well-defined and the domain is stable enough to justify the investment.
