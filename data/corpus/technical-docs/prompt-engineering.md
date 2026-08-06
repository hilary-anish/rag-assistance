# Prompt Engineering Techniques

## Overview

Prompt engineering is the practice of designing and structuring inputs to large language models (LLMs) to elicit accurate, relevant, and well-formatted responses. Because LLMs are sensitive to input phrasing, structure, and context, the way a prompt is written can dramatically affect output quality—sometimes more than the choice of model itself. Prompt engineering encompasses techniques ranging from simple instruction writing to complex multi-step reasoning frameworks.

## Foundational Concepts

### The Prompt as a Program

An LLM prompt is functionally a program: it specifies the task, provides context, sets constraints, and defines the expected output format. Unlike traditional programs, prompts operate through natural language instructions interpreted probabilistically. The same underlying task can yield vastly different results depending on how the prompt is structured.

### System Prompts vs. User Prompts

Most chat-based LLMs distinguish between system prompts and user prompts:

- **System prompt**: Sets the model's persona, capabilities, constraints, and behavioral guidelines. It persists across turns in a conversation. System prompts are processed with higher priority by most models and are the right place for instructions that should always apply.
- **User prompt**: The specific request or question in each turn. Contains the task-specific input and any per-turn instructions.

The division allows separating persistent behavior (system) from per-request variation (user). In API usage, system prompts are set once per session or application, while user prompts change with each request.

## Zero-Shot Prompting

Zero-shot prompting provides the task description and input without any examples. The model relies entirely on its pre-training knowledge to understand the task.

```
Classify the sentiment of the following review as positive, negative, or neutral.

Review: "The battery life exceeded my expectations but the screen quality was disappointing."

Sentiment:
```

Zero-shot works well for tasks the model has seen extensively during pre-training (sentiment analysis, summarization, translation) but can fail on novel or ambiguous task definitions.

## Few-Shot Prompting

Few-shot prompting includes a small number of input-output examples (typically 2–8) before the actual task input. The examples demonstrate the expected behavior, format, and reasoning pattern.

```
Classify the sentiment of the following reviews:

Review: "Absolutely loved this product, works perfectly!"
Sentiment: positive

Review: "Terrible quality, broke after two days."
Sentiment: negative

Review: "It's okay, nothing special but does the job."
Sentiment: neutral

Review: "The battery life exceeded my expectations but the screen quality was disappointing."
Sentiment:
```

### Best Practices for Few-Shot Examples

- **Diversity**: Include examples that cover different cases and edge cases. If classifying sentiment, include positive, negative, neutral, and mixed examples.
- **Consistency**: Use the exact same format across all examples. Inconsistent formatting confuses the model about the expected output structure.
- **Ordering**: Place examples in a logical order. For classification, group by category. For generation, order from simple to complex. The model is slightly biased toward patterns in the most recent examples.
- **Representative difficulty**: Include examples that match the difficulty of the actual task. If the task involves nuance, show nuanced examples.

### Few-Shot Selection

For RAG systems and dynamic applications, examples can be selected dynamically based on similarity to the current input (retrieval-augmented few-shot). This provides the most relevant demonstrations without hardcoding a fixed set.

## Chain-of-Thought (CoT) Prompting

Chain-of-thought prompting instructs the model to show its reasoning step by step before producing a final answer. This significantly improves performance on tasks requiring multi-step reasoning: arithmetic, logic, commonsense reasoning, and complex analysis.

### Manual CoT

Provide examples with explicit reasoning chains:

```
Q: A store has 15 apples. 8 are sold in the morning and 3 more are delivered in the afternoon. How many apples are there at the end of the day?

A: Let me work through this step by step.
1. Starting apples: 15
2. After morning sales: 15 - 8 = 7
3. After afternoon delivery: 7 + 3 = 10
The store has 10 apples at the end of the day.
```

### Zero-Shot CoT

Simply appending "Let's think step by step" to the prompt triggers reasoning behavior without examples. This was shown by Kojima et al. (2022) to significantly improve accuracy on reasoning benchmarks.

```
Q: If a train travels at 60 mph for 2.5 hours, then at 40 mph for 1.5 hours, what is the total distance traveled?

Let's think step by step.
```

### When CoT Helps

- Mathematical reasoning and word problems.
- Multi-hop factual reasoning (questions requiring combining multiple facts).
- Code generation involving complex logic.
- Tasks where the model needs to consider multiple factors before deciding.

### When CoT Does Not Help

- Simple factual recall ("What is the capital of France?").
- Straightforward classification tasks with clear categories.
- Tasks where the final answer is obvious and does not require reasoning.
- Very short outputs where the reasoning overhead is not justified.

## Structured Output Prompting

Prompting the model to produce structured output (JSON, XML, YAML, tables) ensures the response can be programmatically parsed and processed.

```
Extract the following information from the text and return it as JSON:

Text: "Dr. Sarah Chen published a paper on neural networks in Nature on March 15, 2024."

Return JSON with fields: author, topic, publication, date.
```

### Techniques for Reliable Structured Output

- **Provide a schema**: Show the expected JSON structure with field descriptions and types.
- **Include an example**: One complete input-output example eliminates most formatting issues.
- **Use system prompt enforcement**: Instruct in the system prompt that the model should always respond in the specified format.
- **Constrained decoding**: Many inference frameworks (vLLM, Outlines, guidance) support constraining the model's output to match a JSON schema at the decoding level, guaranteeing valid structure.

## Prompt Chaining

Prompt chaining breaks a complex task into a sequence of simpler prompts, where each step's output feeds into the next step's input. This is more reliable than asking the model to perform a complex multi-part task in a single prompt.

### Example: Document Analysis Pipeline

1. **Step 1 (Extract)**: "Extract all factual claims from this document. Return as a numbered list."
2. **Step 2 (Classify)**: "For each claim, classify whether it is supported by the provided source documents. Return the claim and a verdict (supported/unsupported/partially supported)."
3. **Step 3 (Summarize)**: "Summarize the unsupported claims and explain what additional evidence would be needed to verify them."

### Advantages of Chaining

- Each step is simpler and more likely to succeed.
- Intermediate outputs can be inspected and validated.
- Different models or temperatures can be used for different steps.
- Failures in one step can be retried without re-running the entire pipeline.

## Retrieval-Augmented Prompting

In RAG systems, the prompt includes retrieved documents as context. The prompt structure must guide the model to use the retrieved context appropriately.

### Effective RAG Prompt Structure

```
Use the following context to answer the question. If the context does not contain enough information to answer, say so explicitly. Do not use information outside the provided context.

Context:
[Document 1]
[Document 2]
[Document 3]

Question: {user_question}

Answer:
```

### Key Considerations

- **Instruction placement**: Place the instruction to use only the provided context before the context itself. The model processes text sequentially, and early instructions frame how subsequent content is interpreted.
- **Context ordering**: More relevant documents should be placed first or last (not in the middle), as LLMs exhibit a "lost in the middle" effect where information in the center of a long context is attended to less.
- **Source attribution**: Instruct the model to cite specific documents (by number or title) to enable verification.
- **Handling irrelevant context**: Explicitly instruct the model to ignore irrelevant passages. Without this, the model may attempt to incorporate all provided context.

## Role Prompting

Assigning a role or persona to the model can improve response quality for domain-specific tasks:

```
You are an experienced database administrator with 15 years of PostgreSQL expertise. A junior developer has asked you the following question. Provide a clear, practical answer.
```

Role prompting works by activating domain-specific knowledge and communication patterns the model learned during pre-training. It is most effective when the role is well-represented in the training data (doctor, lawyer, engineer) and less effective for highly specialized or fictional roles.

## Self-Consistency

Self-consistency generates multiple responses to the same prompt (using temperature > 0) and selects the most common answer. For reasoning tasks, different sampling paths may lead to different intermediate steps but converge on the correct final answer. Majority voting across 5–40 samples can improve accuracy by 5–15% on reasoning benchmarks, at the cost of proportionally higher inference compute.

## Prompt Templates and Variables

Production systems use parameterized prompt templates rather than hardcoded prompts:

```python
template = """
Given the following customer support ticket, categorize it into one of these categories: {categories}.

Ticket: {ticket_text}

Category:
"""
```

Templates separate the prompt logic from the dynamic content, enabling:
- Consistent formatting across all requests.
- Easy A/B testing of prompt variations.
- Version control and review of prompt changes.
- Parameterized behavior without modifying the core instruction.

## Common Failure Modes

- **Prompt injection**: Malicious user input that overrides the system prompt. Mitigate with input sanitization, delimiter tokens, and output validation.
- **Instruction following degradation**: Very long prompts or contexts can cause the model to lose track of instructions. Place the most critical instructions at the beginning and end of the prompt.
- **Format instability**: Small changes in prompt wording can cause the model to switch output formats unpredictably. Use constrained decoding or output validation to enforce format.
- **Sycophancy**: The model agrees with the user's stated or implied beliefs rather than providing accurate information. Mitigate with explicit instructions to be objective and provide evidence.
