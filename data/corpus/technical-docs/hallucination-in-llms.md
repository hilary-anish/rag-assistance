# Hallucination in Large Language Models

## Overview

Hallucination in LLMs refers to the generation of content that is fluent and plausible-sounding but factually incorrect, unsupported by the input context, or fabricated. It is one of the most significant challenges in deploying LLMs in production, particularly in domains where factual accuracy is critical—healthcare, legal, finance, and education. Understanding the types, causes, and mitigation strategies for hallucination is essential for building trustworthy AI systems.

## Types of Hallucination

### Intrinsic Hallucination

Intrinsic hallucination occurs when the generated output contradicts the provided source material. The model has the correct information available in its context but generates something inconsistent with it.

**Example**: Given a document stating "The company was founded in 2015," the model responds "Founded in 2013, the company has grown rapidly." The source is available but the model contradicts it.

Intrinsic hallucination is particularly problematic in RAG systems where the model is expected to faithfully represent retrieved documents. It can often be detected by comparing the generated output against the source context.

### Extrinsic Hallucination

Extrinsic hallucination occurs when the generated output contains information that cannot be verified from the source material—it is neither supported nor contradicted by the available context. The model fabricates plausible-sounding details.

**Example**: When asked "What are the side effects of drug X?" with a source document that lists three side effects, the model lists those three plus two additional side effects not mentioned in the source. The additional claims may or may not be true, but they are not grounded in the provided evidence.

### Factual Hallucination

The model generates statements that are factually false according to world knowledge, regardless of any provided context. This includes incorrect dates, fabricated statistics, non-existent citations, and wrong attributions.

**Example**: "The Eiffel Tower, completed in 1887, stands at 324 meters." (The Eiffel Tower was completed in 1889, not 1887.)

### Faithfulness Hallucination

In summarization and RAG tasks, faithfulness hallucination occurs when the output is not faithful to the input document. The model may rephrase content in a way that changes the meaning, omit critical qualifiers, or merge information from different parts of the document incorrectly.

**Example**: A source document says "The treatment showed promising results in preliminary trials but has not been validated in large-scale studies." The model summarizes this as "The treatment has been validated and shows promising results."

## Causes of Hallucination

### Training Data Issues

- **Factual errors in training data**: LLMs learn statistical patterns from their training corpus, which contains errors, outdated information, and contradictions. The model has no mechanism to distinguish accurate from inaccurate training examples.
- **Memorization gaps**: The model may have seen a fact during training but not frequently enough to reliably recall it. Partial memorization leads to confident but incorrect outputs—the model generates a plausible completion that fills in the gaps with fabricated details.
- **Distributional bias**: The training data overrepresents common patterns. The model may generate the most statistically common completion rather than the factually correct one. For example, associating a famous scientist with their most well-known work even when asked about a different contribution.

### Architectural Limitations

- **Compressed representation**: An LLM compresses all of its training data into a fixed set of parameters. This lossy compression means that specific facts may be stored imprecisely or conflated with similar facts. A model with 7 billion parameters has roughly 14 GB of weights—far less than the terabytes of training data.
- **Autoregressive generation**: The model generates one token at a time, each conditioned on all previous tokens. Once an incorrect token is generated, subsequent tokens are conditioned on the error, potentially amplifying the hallucination through a snowball effect.
- **Lack of retrieval mechanism**: Standard LLMs have no way to access external knowledge at inference time. They must generate answers entirely from their parametric memory, which is inherently incomplete and potentially outdated.

### Decoding Strategy

- **Temperature and sampling**: Higher temperatures increase randomness in token selection, which can cause the model to select less likely (and potentially incorrect) tokens. Temperature = 0 (greedy decoding) minimizes this but can lead to repetitive or degenerate output.
- **Top-p and top-k**: These sampling parameters control the diversity of generated text. Aggressive filtering (very low top-p) reduces hallucination risk but also reduces output quality and diversity.

### Prompt-Related Causes

- **Ambiguous instructions**: Vague prompts give the model latitude to generate plausible-sounding but unsupported content.
- **Pressure to answer**: Without explicit permission to say "I don't know," models tend to generate an answer for every question, even when they lack sufficient knowledge.
- **Leading questions**: Questions that contain or imply a false premise can cause the model to accept and build on the false premise rather than correcting it.

## Detection Methods

### Reference-Based Detection

Compare the generated output against a reference source (ground truth document, knowledge base, or retrieved context):

- **NLI-based detection**: Use a Natural Language Inference (NLI) model to classify each generated sentence as "entailed," "contradicted," or "neutral" with respect to the source document. Sentences classified as "contradicted" are hallucinations; "neutral" sentences may be extrinsic hallucinations.
- **Fact extraction and verification**: Extract atomic facts from the generated text, then verify each fact against the source. Tools like FActScore decompose text into individual factual claims and check each one independently.
- **QA-based consistency**: Generate questions from the source document, ask those questions of the generated text, and check if the answers are consistent. Inconsistencies indicate hallucination.

### Reference-Free Detection

Detect hallucination without a ground truth source:

- **Self-consistency checking**: Generate multiple responses to the same prompt and compare them. Facts that appear consistently across samples are more likely to be correct; facts that vary are potentially hallucinated. This leverages the observation that true knowledge is robust to sampling variation while hallucinations are not.
- **Confidence-based detection**: Examine the model's token-level probabilities. Hallucinated content often has lower average token probabilities or higher entropy than factual content. However, this is not reliable in isolation—confident hallucinations are common.
- **Internal state probes**: Train classifiers on the model's internal representations (hidden states) to predict whether a generated statement is truthful. Research has shown that LLMs often have internal representations of truthfulness that are not reflected in their output.

### LLM-as-Judge

Use a separate LLM (or the same LLM with a different prompt) to evaluate the factuality of generated content:

- **Direct evaluation**: "Is the following statement factually correct? Explain why or why not."
- **Pairwise comparison**: Present the generated output alongside the source and ask whether the output is faithful.
- **Structured rubric**: Provide a detailed scoring rubric for the evaluator model to follow.

LLM-as-judge approaches are scalable and correlate well with human judgments for well-defined tasks, but they are themselves susceptible to hallucination—the evaluator model may not catch errors it would also make.

## Mitigation Strategies

### Retrieval-Augmented Generation (RAG)

RAG reduces hallucination by grounding the model's responses in retrieved evidence. Rather than relying on parametric memory, the model generates answers from explicitly provided documents. RAG does not eliminate hallucination—the model can still ignore, misinterpret, or contradict the retrieved context—but it significantly reduces the rate of factual errors and enables verification against the source.

### Constrained Decoding

Restrict the model's output space to prevent certain types of hallucination:

- **Schema-constrained output**: Force the output to conform to a JSON schema, preventing the model from generating free-form text that might contain fabrications.
- **Vocabulary restriction**: Limit the output vocabulary to terms present in the source document.
- **Grounded generation**: At each generation step, constrain the model to tokens that can be traced to the input context.

### Chain-of-Verification (CoVe)

A multi-step process where the model generates a response, then generates verification questions about its own claims, answers those questions independently, and revises the original response based on the verification results. This self-correction loop catches some hallucinations that the model can identify when prompted to verify specific claims.

### Confidence Calibration

Train or prompt the model to express calibrated confidence in its statements:

- **Verbalized uncertainty**: Instruct the model to qualify uncertain statements with phrases like "I'm not certain, but..." or "Based on my training data, which may be outdated..."
- **Abstention**: Train the model to say "I don't know" when it lacks sufficient knowledge, rather than generating a plausible guess. This requires careful training to avoid excessive abstention.
- **Probability-based thresholding**: Use token probabilities to identify low-confidence generations and flag them for human review.

### Fine-Tuning for Faithfulness

- **Constitutional AI**: Train the model with explicit principles about truthfulness and faithfulness, using AI feedback to reinforce these behaviors.
- **RLHF with factuality rewards**: Incorporate factual accuracy as a reward signal during reinforcement learning, penalizing hallucination.
- **Knowledge-grounded training**: Fine-tune on tasks that require faithful summarization and grounded generation, teaching the model to stay close to source material.

### Architectural Approaches

- **Retrieval-augmented language models (REALM, RETRO, Atlas)**: Integrate retrieval directly into the model architecture, accessing an external knowledge store during both training and inference.
- **Tool-augmented models**: Allow the model to call external tools (search engines, calculators, databases) when it needs factual information, rather than relying on parametric memory.
- **Fact-verification modules**: Add a separate module that checks generated facts against a knowledge base before outputting them to the user.

## Measuring Hallucination

### Automated Metrics

- **FActScore**: Decomposes generated text into atomic facts and computes the percentage supported by a reference source. Ranges from 0 (all hallucinated) to 1 (all supported).
- **ROUGE/BERTScore against source**: Measures overlap between generated text and source document. Low overlap may indicate hallucination, but high overlap may indicate copying rather than faithful generation.
- **SelfCheckGPT**: Generates multiple samples and measures consistency. Hallucinated facts will vary across samples while factual statements remain consistent.

### Human Evaluation

Human evaluation remains the gold standard for hallucination detection. Annotators assess:

- **Factual accuracy**: Is each claim in the output correct?
- **Faithfulness**: Is the output consistent with the provided source?
- **Completeness**: Does the output omit critical information from the source?

Human evaluation is expensive ($0.50–$5.00 per evaluation depending on complexity and domain expertise required) and does not scale, but it is necessary for calibrating automated metrics and establishing benchmarks.

## Hallucination in Production Systems

### Risk Tiers

- **High risk** (medical, legal, financial advice): Hallucination can cause direct harm. Requires human-in-the-loop review, citation of sources, and explicit uncertainty signaling.
- **Medium risk** (customer support, technical documentation): Hallucination causes user frustration and erodes trust. Requires RAG grounding and automated fact-checking.
- **Low risk** (creative writing, brainstorming, entertainment): Hallucination is often acceptable or even desirable. Creative applications benefit from the model's ability to generate novel content.

### Monitoring in Production

- Track the rate of user-reported factual errors.
- Implement automated hallucination detection on a sample of production outputs.
- Monitor retrieval quality metrics (context relevance, context recall) as leading indicators—poor retrieval predicts downstream hallucination.
- A/B test mitigation strategies and measure their impact on both hallucination rate and user satisfaction.
