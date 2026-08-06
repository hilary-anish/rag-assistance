# Transformer Architecture

## Overview

The Transformer is a neural network architecture introduced by Vaswani et al. in the 2017 paper "Attention Is All You Need." It replaced recurrent architectures (RNNs, LSTMs) as the dominant model for sequence processing tasks by enabling full parallelization during training and capturing long-range dependencies through self-attention. Virtually all modern large language models (GPT, Claude, LLaMA, PaLM) and embedding models (BERT, sentence-transformers) are based on the Transformer.

## Self-Attention Mechanism

Self-attention is the core operation of the Transformer. It allows each position in a sequence to attend to every other position, computing a weighted combination of all position representations where the weights reflect relevance.

### Scaled Dot-Product Attention

Given an input sequence of length $n$ with model dimension $d_{model}$, self-attention operates through three learned linear projections:

- **Query (Q)**: What each position is looking for. Computed as $Q = XW_Q$.
- **Key (K)**: What each position offers. Computed as $K = XW_K$.
- **Value (V)**: The information each position carries. Computed as $V = XW_V$.

The attention output is:

```
Attention(Q, K, V) = softmax(QK^T / √d_k) V
```

The term $QK^T$ computes a similarity score between every pair of positions. The scaling factor $\sqrt{d_k}$ prevents the dot products from growing large in magnitude as $d_k$ increases, which would push the softmax into regions with extremely small gradients.

The softmax normalizes each row of the attention matrix to a probability distribution, so the output for each position is a weighted average of all value vectors.

### Multi-Head Attention

Rather than performing a single attention function, the Transformer uses multi-head attention. The queries, keys, and values are linearly projected $h$ times with different learned projections, attention is computed in parallel for each head, and the results are concatenated and projected:

```
MultiHead(Q, K, V) = Concat(head_1, ..., head_h) W_O
where head_i = Attention(QW_Q^i, KW_K^i, VW_V^i)
```

Each head operates on a subspace of dimension $d_k = d_{model} / h$. Different heads can learn to attend to different types of relationships—one head might capture syntactic dependencies while another captures semantic similarity. A base Transformer typically uses 8–16 heads; large models use 32–128.

### Attention Complexity

Standard self-attention has $O(n^2 \cdot d)$ time and $O(n^2)$ memory complexity, where $n$ is the sequence length. This quadratic scaling is the primary bottleneck for long sequences and has motivated research into efficient attention variants.

## The Encoder-Decoder Architecture

The original Transformer follows an encoder-decoder structure designed for sequence-to-sequence tasks like machine translation.

### Encoder

The encoder consists of a stack of $N$ identical layers (6 in the original paper). Each layer has two sub-layers:

1. **Multi-head self-attention**: Each position attends to all positions in the input sequence (bidirectional).
2. **Position-wise feed-forward network (FFN)**: A two-layer MLP applied independently to each position: $FFN(x) = \max(0, xW_1 + b_1)W_2 + b_2$. The inner dimension is typically 4× the model dimension.

Each sub-layer is wrapped with a residual connection and layer normalization:

```
output = LayerNorm(x + SubLayer(x))
```

The encoder processes the entire input sequence in parallel and produces a sequence of contextualized representations.

### Decoder

The decoder also consists of $N$ identical layers, each with three sub-layers:

1. **Masked multi-head self-attention**: Same as encoder self-attention, but with a causal mask that prevents each position from attending to subsequent positions. This ensures the autoregressive property—the prediction for position $i$ depends only on positions $< i$.
2. **Cross-attention**: Multi-head attention where queries come from the decoder and keys/values come from the encoder output. This is how the decoder accesses the input representation.
3. **Position-wise feed-forward network**: Same as in the encoder.

### Architectural Variants

- **Encoder-only (BERT, RoBERTa)**: Uses only the encoder with bidirectional attention. Best suited for classification, token-level tasks, and producing embeddings. Cannot generate text autoregressively.
- **Decoder-only (GPT, Claude, LLaMA)**: Uses only the decoder with causal (left-to-right) masking. Generates text autoregressively. Dominates current LLM design because of scaling efficiency and the generality of next-token prediction.
- **Encoder-decoder (T5, BART, original Transformer)**: Full architecture with cross-attention between encoder and decoder. Used for sequence-to-sequence tasks like translation and summarization.

## Positional Encoding

Since self-attention is permutation-equivariant (it has no inherent notion of position), positional information must be explicitly injected. The original Transformer uses sinusoidal positional encodings:

```
PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

These encodings are added to the input embeddings. The sinusoidal pattern was chosen because it allows the model to learn to attend to relative positions, since $PE(pos+k)$ can be expressed as a linear function of $PE(pos)$.

### Modern Alternatives

- **Learned positional embeddings**: Trainable vectors for each position. Used in GPT-2 and BERT. Limited to a fixed maximum sequence length.
- **RoPE (Rotary Position Embedding)**: Encodes position by rotating the query and key vectors in 2D subspaces. The dot product between rotated queries and keys naturally depends on relative position. Used in LLaMA, Mistral, and most modern LLMs. Supports extrapolation to longer sequences than seen during training (with techniques like NTK-aware scaling or YaRN).
- **ALiBi (Attention with Linear Biases)**: Adds a linear bias to attention scores based on the distance between positions. No positional embedding is added to the input. Provides strong length extrapolation.

## Feed-Forward Networks and Expert Layers

The position-wise FFN in each Transformer layer typically uses a GeLU or SiLU activation and an expansion ratio of 4×. In a model with $d_{model} = 4096$, the FFN inner dimension is 16,384.

Modern architectures often replace the standard FFN with a **Gated Linear Unit (GLU)** variant:

```
FFN_GLU(x) = (xW_gate ⊙ σ(xW_up)) W_down
```

where $\sigma$ is the SiLU activation and $⊙$ is element-wise multiplication. This gating mechanism improves training stability and model quality.

**Mixture of Experts (MoE)** replaces each FFN with multiple expert FFNs and a routing mechanism that selects a subset (typically 2 of 8–128) of experts for each token. MoE allows scaling model capacity without proportionally scaling computation, as only the selected experts are activated for each token.

## Layer Normalization

The placement of layer normalization differs between architectures:

- **Post-norm** (original Transformer): $LayerNorm(x + SubLayer(x))$. Harder to train at scale without careful learning rate warmup.
- **Pre-norm** (GPT-2, most modern models): $x + SubLayer(LayerNorm(x))$. More stable training, especially for deep models. Used in virtually all modern LLMs.
- **RMSNorm**: A simplified variant that normalizes by the root mean square without centering: $RMSNorm(x) = x / \sqrt{mean(x^2) + \epsilon} \cdot \gamma$. Slightly faster than LayerNorm and used in LLaMA, Mistral, and other recent models.

## KV Cache and Inference Optimization

During autoregressive generation, the Transformer decoder recomputes attention over all previous tokens at each step. The **KV cache** stores the key and value projections of all previous tokens so they do not need to be recomputed. This reduces per-step computation from $O(n^2)$ to $O(n)$ but requires $O(n \cdot d \cdot L)$ memory, where $L$ is the number of layers.

For a model with 32 layers, 32 heads, head dimension 128, and float16 precision, the KV cache for a 4,096-token sequence requires approximately 1 GB. Techniques to reduce this include:

- **Multi-Query Attention (MQA)**: All heads share the same key and value projections, reducing KV cache by a factor of $h$.
- **Grouped-Query Attention (GQA)**: Groups of heads share key/value projections—a middle ground between full multi-head and multi-query attention.
- **Paged attention (vLLM)**: Manages KV cache in non-contiguous memory pages, reducing fragmentation and enabling batching of requests with different sequence lengths.
