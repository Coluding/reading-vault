---
type: paper
title: "POWERPLAY: Training an Increasingly General Problem Solver by Continually Searching for the Simplest Still Unsolvable Problem"
authors: ["Jürgen Schmidhuber"]
year: 2013
venue: "Frontiers in Psychology, 4:313"
url: https://arxiv.org/abs/1112.5309
rw_id: "manual:schmidhuber-2013"
topics: [open-ended-learning, exploration, general-intelligence, intrinsic-motivation, position-paper]
priority: high
read_state: queued
relevance: ""
added: 2026-05-22
last_updated: 2026-05-22
---

# POWERPLAY

**TL;DR** — Schmidhuber's 2011/2013 algorithmic framework for an agent that *invents its own training problems*. POWERPLAY continually searches the space of (new-task, solver-modification) pairs and accepts any modification that **provably solves all previously learned tasks plus the new one** without breaking any old ones. Candidate (task, modification) pairs are ordered by **conditional computational complexity** (time × space) given current experience — so the system always tackles "the simplest still unsolvable problem" next. The validation cost per new task is bounded regardless of repertoire size, so the system can in principle grow indefinitely without quadratic blowup. Schmidhuber frames it as a "greedy but practical implementation of basic principles of creativity."

> _Ingested manually as a citation chase from [[papers/jiang-2022-rethinking-exploration]] (cited as a foundational novelty-seeking-agent-that-generates-its-own-challenges proposal). Source: arXiv abstract page; not via Readwise. Only the abstract was fetched._

## Context & motivation

Schmidhuber's framing inverts the usual ML problem statement:

> "Most of computer science focuses on automatically solving given computational problems. I focus on automatically inventing or discovering problems in a way inspired by the playful behavior of animals and humans, to train a more and more general problem solver from scratch in an unsupervised fashion." — abstract

The lineage: this paper sits in Schmidhuber's long-running research program on intrinsic motivation, curiosity, formal creativity, and self-referential / self-improving systems. POWERPLAY (2011, published in Frontiers in Psychology 2013) is the algorithmic instantiation. The companion papers [53,54] mentioned in the abstract are the experimental analyses.

## Method (as described in the abstract)

### Search space
> "The novel algorithmic framework POWERPLAY (2011) continually searches the space of possible pairs of new tasks and modifications of the current problem solver, until it finds a more powerful problem solver that provably solves all previously learned tasks plus the new one, while the unmodified predecessor does not." — abstract

This is the core algorithmic move: every accepted modification is *constrained* to preserve prior competence. Adding capability cannot subtract from old capability.

### Search ordering
> "POWERPLAY's search orders candidate pairs of tasks and solver modifications by their conditional computational (time & space) complexity, given the stored experience so far." — abstract

I.e., POWERPLAY always tries to find the **simplest still-unsolvable problem** that the agent can newly learn given current experience. Conceptually: Kolmogorov-complexity-style ordering on (task, modification) pairs.

### Skill efficiency / reuse
> "Wow-effects are achieved by continually making previously learned skills more efficient such that they require less time and space. New skills may (partially) re-use previously learned skills." — abstract

Two kinds of improvement count: (i) acquiring a new skill, (ii) compressing an old skill (making it cheaper to invoke). Both are valid POWERPLAY steps.

### Validation cost
> "The computational costs of validating new tasks need not grow with task repertoire size." — abstract

This is the key practicality claim — without it, the cost of "validate that I still solve all prior tasks" would grow at least linearly in repertoire size.

## Experimental setup

_Not addressed in the abstract._ The abstract refers to companion papers [53,54] for "a first experimental analysis."

## Key results

_Not addressed in the abstract._ POWERPLAY is presented in this paper as a *framework*, not as an empirical result.

## Ablations

_Not addressed by the source_ at the abstract level.

## Limitations

_Not addressed by the source_ at the abstract level.

[analyst's view] Limitations a reader would flag:
- **"Provably solves all prior tasks" is hard in practice.** For neural-network function approximators this is essentially the catastrophic-forgetting problem with a "no forgetting allowed" constraint. The proof obligation is severe.
- **Conditional-complexity ordering is hand-wavy in practice.** Real complexity is uncomputable; in implementation this becomes some learned proxy.
- **No specification of what counts as a "task".** POWERPLAY assumes a well-defined task space; in open domains, even *describing* candidate tasks is itself a research problem.

## Why it matters [analyst's view]

Three reasons:

1. **It is the cleanest formalisation of self-curriculation pre-2020.** POWERPLAY says: "the agent searches the joint (task, solver-update) space; accept updates that strictly grow the repertoire." That's the structural shape of every modern self-improvement loop — RLHF-with-prompts, autocurricula, LLM-bootstrap-self-improvement — minus the implementation specifics.

2. **The "simplest unsolved problem" ordering anticipates ZPD-style curricula.** [[papers/jiang-2022-rethinking-exploration]]'s appeal to Vygotsky's Zone of Proximal Development is Schmidhuber's "simplest still unsolvable problem" idea, restated in developmental-psychology language. Same core intuition; different vocabulary.

3. **The "validation cost need not grow" claim is the right thing to target.** A practical open-ended learner has to amortise the cost of "do I still know what I knew?" Schmidhuber names this as a design objective in 2011 — most 2026 systems still don't achieve it cleanly (catastrophic forgetting, regression suites, eval drift).

## Open questions

[analyst's view]
- **Modern LLM analogue.** What does POWERPLAY look like in an LLM-post-training setting? RLHF iterates without provable backward-compatibility — explicit POWERPLAY-style "must not regress on prior evals" is closer to today's *eval regression suites* than to mainstream training methods.
- **Connection to [[papers/clune-2019-ai-gas]] pillar 3.** Clune's "generating effective learning environments" is essentially POWERPLAY's task-discovery half. Different vocabulary, overlapping intent. Worth comparing carefully.
- **The "wow-effect" framing (skill compression / efficiency improvement)** is rarely valued in modern benchmarking. Almost all modern eval is about new-task acquisition. A POWERPLAY-aware system would also be rewarded for *making old skills cheaper*. Worth bringing up the next time the topic of "what should we be evaluating?" comes up.

## Connections

- **Conceptual successor**: [[papers/clune-2019-ai-gas]] — AI-GAs' third pillar ("generate effective learning environments") is essentially POWERPLAY's task-invention loop, restated.
- **Conceptual descendant**: [[papers/jiang-2022-rethinking-exploration]] — Jiang explicitly cites Schmidhuber as a foundational proposal.
- **Adjacent intrinsic-reward work**: [[papers/guo-2022-byol-explore]] — BYOL-Explore is a curiosity-driven exploration method; POWERPLAY is a more ambitious "invent the task" cousin.
- **Topic MOCs**: [[topics/open-ended-learning]], [[topics/exploration]], [[topics/intrinsic-motivation]]
- **Author indices**: [[authors/jurgen-schmidhuber]]

## Selected quotes

> "Most of computer science focuses on automatically solving given computational problems. I focus on automatically inventing or discovering problems in a way inspired by the playful behavior of animals and humans, to train a more and more general problem solver from scratch in an unsupervised fashion." — abstract

> "POWERPLAY's search orders candidate pairs of tasks and solver modifications by their conditional computational (time & space) complexity, given the stored experience so far." — abstract

> "POWERPLAY may be viewed as a greedy but practical implementation of basic principles of creativity." — abstract

## Source caveats

- Ingested manually via citation chase, not Readwise.
- Only the arXiv abstract page was fetched. Body method, experiments, theoretical results — including the specific complexity bounds and the relation to Gödel-machine self-reference — are **not addressed by the source** at this triage depth.
