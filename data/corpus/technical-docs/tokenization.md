# Tokenization Methods

## Overview

Tokenization is the process of converting raw text into a sequence of discrete units (tokens) that a language model can process. It is the first step in any NLP pipeline and directly impacts model vocabulary size, sequence length, and the ability to handle out-of-vocabulary (OOV) words. Modern language models universally use subword tokenization, which splits text into units that fall between character-level and word-level granularity.

## Why Subword Tokenization

### The Vocabulary Problem

- **Word-level tokenization** assigns one token per word. This requires a very large vocabulary to cover a language (English alone has over 170,000 words in current use), and any word not in the vocabulary is mapped to an unknown token [UNK]. Morphological variants (run, running, runs, runner) each consume a vocabulary slot despite sharing a root.
- **Character-level tokenization** uses individual characters as tokens, requiring a tiny vocabulary (a few hundred entries). However, sequences become extremely long (a 100-word sentence might be 500+ characters), increasing computational cost quadratically with attention mechanisms. The model must also learn spelling and morphology from scratch.
- **Subword tokenization** provides a middle ground. Common words are kept as single tokens, while rare or unknown words are split into meaningful subword units. "Unhappiness" might become ["un", "happiness"] or ["un", "happi", "ness"], allowing the model to generalize from known subwords to novel words.

## Byte Pair Encoding (BPE)

BPE, originally a data compression algorithm, was adapted for NLP by Sennrich et al. (2016) and is the most widely used tokenization method. GPT-2, GPT-3, GPT-4, LLaMA, and Claude all use BPE variants.

### Algorithm

1. **Initialize**: Start with a vocabulary of all individual characters (or bytes) present in the training corpus.
2. **Count pairs**: Count the frequency of every adjacent pair of tokens in the corpus.
3. **Merge most frequent pair**: Replace all occurrences of the most frequent pair with a new merged token. Add this token to the vocabulary.
4. **Repeat**: Continue merging until the desired vocabulary size is reached.

### Example

Starting text: "low lower lowest"

Initial tokens: ['l', 'o', 'w', ' ', 'l', 'o', 'w', 'e', 'r', ' ', 'l', 'o', 'w', 'e', 's', 't']

After merging 'l'+'o' → 'lo': ['lo', 'w', ' ', 'lo', 'w', 'e', 'r', ' ', 'lo', 'w', 'e', 's', 't']

After merging 'lo'+'w' → 'low': ['low', ' ', 'low', 'e', 'r', ' ', 'low', 'e', 's', 't']

The merge rules are learned from the training corpus and applied deterministically at inference time. A common vocabulary size for BPE is 32,000–100,000 tokens.

### Byte-Level BPE

GPT-2 introduced byte-level BPE, which operates on raw bytes (256 base tokens) rather than Unicode characters. This guarantees that any input text can be tokenized without OOV tokens, including code, URLs, and multilingual text. The base vocabulary of 256 bytes is extended through BPE merges to the target vocabulary size.

## WordPiece

WordPiece, developed by Schuster and Nakajima (2012) and used by BERT, is similar to BPE but uses a different merge criterion.

### Differences from BPE

- **Merge criterion**: BPE merges the most frequent pair. WordPiece merges the pair that maximizes the likelihood of the training corpus—effectively the pair where the merged token's frequency is highest relative to the product of the individual frequencies. This is equivalent to maximizing mutual information.
- **Prefix notation**: Non-initial subwords are prefixed with "##" to indicate they are continuations. "Embedding" might tokenize as ["em", "##bed", "##ding"].
- **Greedy longest-match**: At inference time, WordPiece uses a greedy longest-match-first algorithm to tokenize text, whereas BPE applies its learned merge rules in the exact order they were learned.

### Example

For the word "unhappiness", WordPiece with BERT's vocabulary produces:

```
["un", "##hap", "##pi", "##ness"]
```

The "##" prefix tells the model (and any post-processing) that this token continues the previous one rather than starting a new word.

## SentencePiece

SentencePiece, developed by Kudo and Richardson (2018), is a language-independent tokenization framework that treats the input as a raw stream of Unicode characters, including whitespace. Unlike BPE and WordPiece, which typically require pre-tokenization (splitting on whitespace and punctuation first), SentencePiece operates directly on raw text.

### Key Features

- **Whitespace as a character**: SentencePiece uses the Unicode character "▁" (U+2581) to represent spaces. This allows it to learn tokens that span word boundaries and to losslessly reconstruct the original text. "New York" might become ["▁New", "▁York"] or even ["▁New▁York"] if frequent enough.
- **Language agnostic**: Because it does not rely on whitespace for word boundaries, SentencePiece works equally well for languages without spaces between words (Chinese, Japanese, Thai).
- **Multiple algorithms**: SentencePiece supports both BPE and a unigram language model algorithm.

### Unigram Language Model

The unigram algorithm, also developed by Kudo (2018), takes the opposite approach from BPE:

1. **Initialize**: Start with a large seed vocabulary (e.g., all substrings up to a certain length that appear in the corpus).
2. **Compute loss**: For each token in the vocabulary, compute how much removing it would increase the overall corpus log-likelihood under a unigram language model.
3. **Prune**: Remove the tokens whose removal increases the loss the least, reducing the vocabulary by a fixed percentage.
4. **Repeat**: Continue pruning until the target vocabulary size is reached.

At inference time, the unigram model can produce multiple valid tokenizations for a given input. It selects the one with the highest probability under the unigram language model, or samples from the distribution for regularization (subword regularization).

## Tiktoken

Tiktoken is OpenAI's fast BPE tokenizer implementation, used for GPT-3.5 and GPT-4. It uses byte-level BPE with a pre-tokenization regex pattern that splits text into chunks before applying BPE merges:

```
pat = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""
```

This pattern ensures that tokens do not span certain boundaries (e.g., a letter token won't merge with a following digit). GPT-4's tokenizer (cl100k_base) has a vocabulary of 100,256 tokens.

## Tokenization Effects on Model Behavior

### Fertility

Tokenization fertility is the average number of tokens per word. A lower fertility means fewer tokens per word, which is computationally cheaper. English typically has a fertility of 1.2–1.5 with modern tokenizers, while some languages (Burmese, Khmer) can have fertilities of 5–10, meaning the model's effective context window is much shorter for those languages.

### Token Boundaries and Arithmetic

LLMs struggle with character-level reasoning (spelling, counting, arithmetic) partly because tokenization groups characters unpredictably. The number "12345" might be one token or three tokens ("123", "45") depending on context, making digit-level operations difficult for the model.

### Encoding of Whitespace and Special Characters

How whitespace is tokenized affects code generation. A tokenizer that produces a single token for four spaces (a common indentation unit) is more efficient for code than one that produces four individual space tokens.

## Practical Considerations

### Vocabulary Size Trade-offs

- **Smaller vocabulary** (8K–32K): More subword splits, longer sequences, but better generalization to rare words and less memory for the embedding matrix.
- **Larger vocabulary** (64K–256K): Fewer tokens per input, shorter sequences, but a larger embedding matrix and potentially poorer coverage of rare subwords.

### Pre-tokenization

Many tokenizers apply pre-tokenization rules before the core algorithm. These rules split on whitespace, punctuation, or regex patterns to prevent undesirable merges (e.g., preventing a token that spans a word boundary and a punctuation mark). The choice of pre-tokenization rules significantly affects the final tokenization.

### Special Tokens

All tokenizers include special tokens not derived from the training corpus:

- **[CLS], [SEP]** (BERT): Classification and separator tokens for sentence-pair tasks.
- **[PAD]**: Padding token for batching sequences of different lengths.
- **<|endoftext|>** (GPT): End-of-sequence marker.
- **<|im_start|>, <|im_end|>** (ChatML): Markers for message boundaries in chat models.

These tokens are assigned dedicated vocabulary IDs and are never split by the tokenizer.

### Token Counting

For cost estimation and context window management, accurate token counting is essential. Libraries like tiktoken (OpenAI), tokenizers (Hugging Face), and sentencepiece provide exact token counts for their respective models. Rough estimates (1 token ≈ 4 characters in English, or 1 token ≈ 0.75 words) are useful for quick approximations.
