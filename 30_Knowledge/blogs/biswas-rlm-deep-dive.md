---
type: blog
title: "Recursive Language Models: An All-in-One Deep Dive"
author: "Avishek Biswas"
url: https://towardsdatascience.com/recursive-language-models-one-example-deep-dive-that-explains-everything/
rw_id: 01ks7cjhh1ed1jdb6zyap0fm6y
topics: [agentic-harnesses, long-context, recursive-reasoning, llm-scaffolds, codeact]
priority: medium
read_state: queued
added: 2026-05-22
last_updated: 2026-05-22
---

# Recursive Language Models: An All-in-One Deep Dive

**TL;DR** — A 6.5K-word walkthrough of **Recursive Language Models (RLMs)** — an agentic harness pattern in which the LLM operates inside a persistent Python REPL with an `llm_query(...)` function for spawning subagents. The user's prompt is exposed as a `context` variable, not loaded into the LLM's context window directly; subagent outputs land as REPL variables, not LLM messages. The core idea Biswas highlights: **pass-by-reference instead of pass-by-value** across agent boundaries. He contrasts RLMs with ReAct / CodeAct / vanilla subagents through a toy "count R's in 50 fruit names" example, walks through a 60M-character Lex Fridman transcript trajectory that cost $0.20, and shares the full system prompt from his open-source `fast-rlm` implementation.

## Context

The post sits in the agentic-scaffold design space — ReAct (LLM-defined tool calls), CodeAct (LLM-written code execution), subagents, file-system-augmented agents. Biswas spent "a decent chunk of last month implementing RLMs, running benchmarks, and producing a 50-minute tutorial video," then wrote this as the FAQ-distilled "what made me go a-ha." The RLM paper itself is referenced as `arxiv.org/abs/2512.24601` — i.e., the original is a December 2025 paper that I don't have in vault yet.

The audience is practitioners debating which agentic pattern to choose for long-context tasks.

## Core argument

### The pass-by-value problem

Every prior agentic pattern (ReAct, CodeAct, subagent stacks, even file-system-augmented ones) shares the same bottleneck:

> "The LLM still has to read the file in the end and reproduce it verbatim (assuming that's a strict requirement) ... The transmission problem still exists."

I.e., even when subagents do the heavy lifting, the orchestrator must eventually *regenerate* outputs token by token, paying for transmission of every byte of intermediate state through context windows. This becomes prohibitive when subagent outputs are large (a 50-item dict per subagent × 3 subagents).

### The RLM contract (§3, four points)

> 1. "A language model interacts with arbitrarily long prompts through an external programmable environment or an REPL."
> 2. "Printed outputs are truncated at the scaffold layer."
> 3. "The LLM can write code to programmatically explore and create new transformations of the prompt."
> 4. "It can recursively invoke sub-agents to complete smaller subtasks. The subagent responses do not get automatically loaded into the parent agent's context, it gets returned as symbols or variables inside the parent's REPL."

The two structural moves that matter:

**1. Context-by-reference**. The user's prompt lives as a Python variable `context` in a Pyodide-inside-Deno sandbox. The LLM only sees the fact that the variable exists; it explicitly chooses to read it (`print(context[:200])` etc), and `print` truncates at the scaffold layer regardless of LLM request. The LLM never has unbounded context dumped on it.

**2. Subagent outputs as REPL values, not messages**. `llm_query("sub-task")` spawns a fresh child RLM, which returns via `FINAL(value)`. The value lands as a Python object in the parent's REPL — `FRUIT_DICT = await llm_query(...)`. The parent can compose answers from variables it never actually read into its own context window. `FINAL(answer)` then returns the variable's *value*, not its serialised representation through the LLM.

The killer payoff (the Lex Fridman trajectory):

> "I passed in a CSV containing transcripts of 320 episodes of the Lex Fridman podcast and asked it to find what his first 10 ML guests had to say about AGI. The context had 60,855,062 characters. ... It took 4 minutes to crunch ... cost me 0.2$ with Minimax-M2.5. It read 1M tokens (825K was cache hits so it was quite cheap), produced just 69K tokens (19K were reasoning)."

60M characters of input, 4 minutes, $0.20 — *because the LLM only had to "see" the structure of the data and the slices it explicitly chose to inspect, not the full transcript*.

## Notable details

### The system prompt is the spec
Biswas reproduces his full system prompt verbatim. The interesting design choices baked in:

- **Print-truncates-at-the-scaffold** ("upto 200 words") — the LLM cannot force itself to over-consume context. The harness, not the LLM, enforces the budget.
- **`llm_query` is async** and must be awaited; subagents are explicitly encouraged to be parallelised with `asyncio.gather(*tasks)`.
- **Subagents inherit no parent state** — context is whatever you pass them, full stop.
- **`FINAL(variable_name)` returns the variable's value** — the prompt is at pains to warn against `FINAL("variable_name")` which would return the literal string. This is the pass-by-reference contract showing through at the LLM-prompt level.
- **Parallel subagents are explicitly the path to speed**: "Subagents that are parallel tend to finish 10x faster ... The value of your intelligence and thinking capability is how you design your method so that you maximize subagent parallelization."

### Five reasons Biswas claims RLMs work (§4)
1. **Focused attention**: load context by choice, not in bulk.
2. **Multi-step reasoning**: matches the recursive structure of long-context tasks (multi-hop QA, codebase search, multi-doc summarisation).
3. **Robustness to noise**: when 99% of input is irrelevant, "recursive search avoids attention dilution."
4. **Composable variable outputs**: subagent answers are Python objects, agent can use them without reading them.
5. **Arbitrarily long outputs**: not bounded by the LLM's context length; return a Python variable instead.

### Cost story
- Subagents hit KV caches "90% of the time" because their system prompt and history are fixed per role.
- "Low cost on prompt input tokens! And depending on the problem, low cost on completion tokens."

## Why it matters [analyst's view]

This is the clearest practitioner-level distillation of the RLM pattern I've seen. Three reasons it's worth keeping in the vault:

1. **The pass-by-reference framing is exportable.** The framing — "every prior agentic pattern is implicitly pass-by-value; RLMs are pass-by-reference" — is the cleanest single sentence summary of what's distinct about RLMs vs CodeAct + subagents. Useful for explaining the pattern to other engineers without needing to walk them through the paper.

2. **The Lex Fridman trajectory is a useful concrete anchor.** Concrete data points anchor the "long context costs you the world" intuition. 60M characters, $0.20, 4 minutes is the kind of number you can use to argue against feeding 10M tokens into a frontier model directly.

3. **The system prompt is reusable.** Biswas reproduces it in full. The prompt encodes a specific theory of how to make a base LLM behave under the RLM contract — useful as a starting point if I want to build my own RLM harness.

The conceptual overlap with [[papers/baek-2026-gram]] is interesting but loose: both are "recursive" — but GRAM does recursion in *latent state* via stochastic transitions; RLMs do recursion in *symbol space* via spawned subagents. Same word, very different mechanisms. A note worth holding in mind when the term "recursive reasoning" comes up.

## Open questions

[analyst's view]
- **What's in the original RLM paper (arxiv 2512.24601)?** Biswas's post is the practitioner gloss; the paper presumably has the formal scaffold definition and benchmark numbers. Worth saving when found.
- **Failure modes**. Biswas reports cost savings on a CSV-extraction-style task. What about tasks where the parent agent legitimately needs to *read* large fractions of the context (e.g., long-form summarisation with strict faithfulness)?
- **Caching mechanics**. The "90% KV cache hit" claim depends on subagent message structure being stable. How does that interact with custom prompt suffixes or tool-call diversification?
- **What does the LPRM / verifier story look like for RLMs?** [[papers/baek-2026-gram]] needs an LPRM to pick best-of-N latent samples. RLMs presumably need a way to handle subagent failures or disagreement. Not discussed in this post.

## Connections

- **Conceptually distinct cousin**: [[papers/baek-2026-gram]] — "recursive reasoning" in latent state, not in symbol space. Worth contrasting these two notions.
- **Original paper to chase**: arxiv.org/abs/2512.24601 — not yet in vault.
- **Implementation reference**: `github.com/avbiswas/fast-rlm` (Biswas's open-source impl).
- **Topic MOCs**: [[topics/agentic-harnesses]], [[topics/long-context]], [[topics/recursive-reasoning]]
- **Author indices**: [[authors/avishek-biswas]]

## Selected quotes

> "The idea of passing context around by reference, instead of replicating it. ... That is what RLMs do."

> "The way the user prompt makes it into the LLM's context window is not by how we pass it! The LLM makes a deliberate decision to read it from the environment."

> "[60,855,062 characters], 4 minutes to crunch, and the fun part is it cost me 0.2$ with Minimax-M2.5."

> "This would be basically impossible to do at this quality with a base LM. (context rot, since 99% of the data is useless). It will cost 20x more with ReAct model (too many tasks). It will cost 10x more with a React + Subagent model (read/write contexts instead of using symbolic variables)."
