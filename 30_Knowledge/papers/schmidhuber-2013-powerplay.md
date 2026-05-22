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
read_state: skimmed
relevance: ""
added: 2026-05-22
last_updated: 2026-05-22
---

# POWERPLAY

**TL;DR** — Schmidhuber's 2011/2013 algorithmic framework for an agent that *invents its own training problems*. POWERPLAY iterates an epoch loop: a search algorithm proposes a candidate triple (new task $T$, modification of the solver $s \to q$, proof that $q$ solves $T_1,\ldots,T_i$ while the old $s_{i-1}$ cannot solve $T_i$). Candidate (task, modification) pairs are explored in order of **conditional computational complexity** (time × space) given current experience, implemented via Hutter-style universal search over programs with prior $P(p) = 2^{-L(p)}$ — so the system always tackles the *simplest still unsolvable* problem next. Validation cost is bounded *independent of repertoire size* via component-tracking lists and (in the prefix-code variant) by construction, so the system can grow indefinitely without quadratic blowup. The paper formalises a "greedy but practical" approximation to creativity that is distinct from the asymptotically-optimal but constant-impractical Gödel machine.

> _Ingested manually as a citation chase from [[papers/jiang-2022-rethinking-exploration]]. Source: full paper via ar5iv HTML render of arXiv:1112.5309. This revision reflects a full-paper read._

## Context & motivation

Schmidhuber's framing inverts the usual ML problem statement:

> "Most of computer science focuses on automatically solving given computational problems. I focus on automatically inventing or discovering problems in a way inspired by the playful behavior of animals and humans, to train a more and more general problem solver from scratch in an unsupervised fashion."

The lineage: POWERPLAY is one node in Schmidhuber's long-running program — universal search (Solomonoff/Levin), OOPS (Optimal Ordered Problem Solver), the Gödel machine (self-referential proof-based self-modification), artificial curiosity (1991+), and the formal theory of creativity. POWERPLAY's distinctive contribution is to give up asymptotic optimality in exchange for a tractable, practically-runnable self-improving loop.

The biological motivation is explicit:

> "Consider… how initially helpless human babies become rather general problem solvers over time? Apparently by playing."

POWERPLAY is "play" as a formal optimisation algorithm: invent tasks at the edge of competence, master them, repeat.

## Method — POWERPLAY Variant I (Algorithm 2)

The core loop. Given a current solver $s_{i-1}$ and stored history of solved tasks $T_1, \ldots, T_{i-1}$, each epoch $i$ executes a repeat-until loop seeking a triple $(p, T, q)$ satisfying three conditions:

1. **Task invention.** A candidate program $p \in \mathcal{P}$ computes a task description $T \in \mathcal{T}$.
2. **Solver modification.** Program $p$ computes a modified solver $q \in \mathcal{S}$ from $s_{i-1}$.
3. **Correctness demonstration.** Program $p$ verifies that $q$ solves *all* tasks $T_1, \ldots, T_{i-1}, T$, while the previous solver $s_{i-1}$ cannot solve the new $T$.

If the triple validates, store $(p_i, T_i, s_i \leftarrow q)$ and update the history. Otherwise, keep searching.

The acceptance criterion is **strict and monotone**: capability cannot be lost. Schmidhuber stages this as the essential structural property — every accepted update *strictly grows* the demonstrated repertoire.

## Search ordering — universal-search bias toward simplicity

The search over candidate programs $p$ is not flat enumeration. Algorithm 4.1 implements POWERPLAY using **OOPS**-style time allocation over the program space:

- Prior $P(p) = 2^{-L(p)}$ where $L(p)$ is program length in bits (Solomonoff-style, favouring short programs).
- Maintain a global time budget $t_{lim}$. Allocate program $p$ a runtime budget proportional to $P(p) \cdot t_{lim}$.
- Whenever the candidate set with sufficient probability mass is exhausted without success, double $t_{lim}$ and try again.

The effect: candidate (task, modification) pairs are explored in order of *conditional computational complexity given prior experience*. Short programs that solve the candidate task quickly are tried first. This implements the principle that:

> "The greedy search of typical PowerPlay variants uses time-optimal program search to order candidate pairs of tasks and solver modifications by their conditional computational (time & space) complexity, given the stored experience so far."

This is also why the paper's title says "simplest still unsolvable problem" — the first task that validates is by construction the simplest one POWERPLAY can newly handle, given current $s_{i-1}$.

## Validation cost — the practicality argument

The naive cost of "verify $q$ still solves $T_1, \ldots, T_{i-1}$" is at least linear in $i$ — and would dominate the loop for any large repertoire. POWERPLAY argues this need not be the case. Two mechanisms:

### Component tracking (Section 3.3.2)

For each component $s^k$ of the solver, maintain a list $L^k$ of which previously-solved tasks depend on $s^k$. When the new candidate $q$ differs from $s_{i-1}$ only on a subset of components, Correctness Demonstration only needs to re-verify the union of $L^k$ over modified components — *not* the full repertoire.

> "If the most recent task does not require changes of many components of $s$, and if the changed bits do not affect many previous tasks, then Correctness Demonstration may be very efficient."

### Prefix-code / self-delimiting solvers (Section 3.3.3)

If $\mathcal{S}$ is restricted to self-delimiting (prefix-code) programs and modifications can only *extend* the solver (append new bits / new modules, never modify old ones), then the prior repertoire $T_{<i}$ is *guaranteed* to remain solvable by construction. Correctness Demonstration on prior tasks becomes trivial — no re-verification needed.

The combination of these two mechanisms is the technical heart of POWERPLAY's "validation cost need not grow with task repertoire size" claim.

## Wow-effects and skill compression (Section 5)

A key concern: doesn't POWERPLAY degenerate into a trivial sequence of micro-tasks at the edge of competence? Schmidhuber's answer is to broaden what counts as an "accepted" task.

A new task $T_i$ is valid if it does *either*:

- Add a new capability the agent could not previously demonstrate, OR
- Require solving at least one previously-solved task $T_k$ **more efficiently** — less storage, less computation time, less energy. This is the **wow-effect**: a compression / speedup of prior skills counts as progress.

> "On realistic but general architectures such as PCs and RNNs, at least once the upper storage size limit of $s$ is reached, PowerPlay will start 'compressing' previous solutions, making $s$ generalize in the sense that the same relatively short piece of code helps to solve different tasks."

This is the route to *generalisation*: with a fixed-capacity solver, the only way to add new skills is to compress old ones, which forces the discovery of shared sub-routines.

## POWERPLAY Variant II — cost-based acceptance (Section 7.1)

Variant I's strict monotonicity (cannot forget anything) is impractical for many real settings. **Variant II** (Algorithm 7.1) relaxes this:

- Define a real-valued cost function $\text{Cost}(s, TSET)$ combining program length $L(s)$, per-task solve times $t'_s(T)$, and per-task rewards $r(T)$.
- Accept the candidate $s_i$ if $c^*_i - c_i > \epsilon$ where $c_i = \text{Cost}(s_i, T_{\leq i})$ and $c^*_i = \text{Cost}(s_{i-1}, T_{\leq i})$.
- Under this criterion, $s_i$ *may* forget some prior abilities, provided the overall cost improvement exceeds the threshold.

Section 7.2 mentions a probabilistic variant for stochastic environments where the strict guarantee is replaced by a confidence-bounded statement.

## Neural-network instantiation (Algorithm 4.3, Section 4.1.2)

POWERPLAY is not married to discrete programs. The paper sketches an RNN instantiation:

- The solver $s$ is a recurrent neural network RNN1 with weight matrix.
- A separate generator network RNN2 (defining the candidate-program space $\mathcal{P}$) outputs both task descriptions and proposed weight modifications.
- **SLIM NNs** (Self-delimiting neural networks, from prior Schmidhuber work, reference [47]) are used so that only the neurons/connections actually exercised during a task are "active," enabling efficient resets and component tracking even at the level of weights.

This gives a concrete picture: POWERPLAY isn't just symbolic-program search; the same loop applies if $\mathcal{S}$ is the space of RNN weight configurations and $\mathcal{P}$ is parameterised by a meta-RNN.

## Connection to foundational frameworks

### Gödel machines (Section 9.1)

Gödel machines: self-referential systems that rewrite their own code if and only if they can *prove* the rewrite is beneficial. Asymptotically optimal but with prohibitive constant factors. POWERPLAY explicitly trades the asymptotic guarantee for practical efficiency:

> "PowerPlay, on the other hand, is designed to incrementally build a practical more and more general problem solver that can solve numerous tasks quickly, not in the asymptotic sense, but by exploiting to the max its given particular search algorithm and computational architecture."

POWERPLAY is the engineering-pragmatic cousin of the Gödel machine: same self-improvement structure, weaker guarantees, much more tractable.

### Formal theory of creativity (Section 9.3)

Schmidhuber explicitly frames POWERPLAY as:

> "A greedy but practical implementation of basic principles of creativity."

The "creativity" angle: POWERPLAY's task-invention loop is what an artist/scientist does — propose problems at the edge of competence, solve them, repeat. The same mechanism Schmidhuber argues underlies aesthetic preference (data that is *learnable* and *novel*) is what drives POWERPLAY task selection.

### Universal search (Section 4.1)

The time allocation scheme implements Hutter's $H_\text{search}$: if some unknown optimal program for the current task requires $f(k)$ steps, the search finds it within $O(f(k)/P(p)) = O(f(k))$ steps — constant overhead, independent of problem size. This is what makes universal search "the right asymptotic engine" inside POWERPLAY.

## Experiments

Schmidhuber's 2011/2013 paper itself is primarily theoretical / algorithmic. Section 8 directs to companion papers — referenced as **[53]** and **[52]**, which are the Srivastava-Steunebrink-Schmidhuber experimental analyses of POWERPLAY on RNN-based and program-search-based setups. The abstract notes the experimental papers report:

- Interesting *developmental stages* — distinct phases of behavior across epochs.
- Automatic *self-modularisation* — the solver re-uses code/sub-circuits across tasks once skills are discovered.

The 2013 paper itself does not present headline numbers; the experimental analyses must be fetched separately.

## Limitations & open problems (Section 10, "Words of Caution")

Schmidhuber names several:

- **Scalability with very large repertoires** remains theoretically argued but not yet empirically validated at scale.
- **Self-generated vs externally-posed tasks.** Schmidhuber explicitly flags this as an open question: *Under what conditions does the self-invented curriculum accelerate solving externally-imposed tasks?* This is the transfer question — POWERPLAY is built around self-curriculation, but the practical motivation is usually some external benchmark.
- **Probabilistic variants are weaker.** Variant II + stochastic environments make weaker statements ("under assumptions about repeatability of trials…"), and the strict-monotone guarantee is lost.
- **Practical task-space specification.** $\mathcal{T}$ has to be defined; in open domains the description-of-candidate-tasks problem is itself a research problem (this is the same critique that hits [[papers/clune-2019-ai-gas]]' pillar 3).

## Why it matters [analyst's view]

Four reasons, after the full read:

1. **It is the cleanest pre-LLM formalisation of self-curriculation.** POWERPLAY says: search the joint (task, solver-update) space; accept updates that strictly grow the repertoire (Variant I) or improve a cost (Variant II). That is the structural shape of every modern self-improvement loop — RLHF-with-self-generated-prompts, RLAIF, autocurricula, LLM-bootstrap-self-improvement — minus the implementation specifics. Modern implementations almost universally use Variant II (cost-improvement, not strict-monotone), which is worth flagging when reading this in 2026.

2. **The "simplest unsolved problem" ordering anticipates ZPD-style curricula.** [[papers/jiang-2022-rethinking-exploration]]'s appeal to Vygotsky's Zone of Proximal Development is exactly Schmidhuber's "simplest still unsolvable problem" idea, restated in developmental-psychology language. Same core intuition, different vocabulary — POWERPLAY's framing is more algorithmically precise.

3. **The "validation cost need not grow" claim is the right design target — and almost nobody hits it.** A practical open-ended learner has to amortise "do I still know what I knew?" Schmidhuber names this as a design objective in 2011 and gives two concrete mechanisms (component tracking, prefix-code monotonicity). Most 2026 systems don't have either; they have *eval regression suites* that grow linearly. POWERPLAY is more architectural about this than the state of the art.

4. **The wow-effect / compression angle is rarely valued in modern benchmarking.** Modern eval is almost entirely about *new-task acquisition*. POWERPLAY says: making old skills cheaper should *also* count as progress. This is a useful counter-frame the next time the question of "what should we be evaluating?" comes up. Compression as progress is also the route by which POWERPLAY achieves generalisation in fixed-capacity solvers — a point worth lifting into LLM-post-training discussions.

## Open questions

[analyst's view]
- **Modern LLM analogue.** What does POWERPLAY look like in an LLM-post-training setting? RLHF iterates without provable backward-compatibility — explicit POWERPLAY-style "must not regress on prior evals" is closer to today's *eval regression suites* than to mainstream training methods. Is there a paper that explicitly does POWERPLAY-on-LLM? (Worth a citation search.)
- **Connection to [[papers/clune-2019-ai-gas]] pillar 3.** Clune's "generating effective learning environments" is essentially POWERPLAY's task-discovery half. Different vocabulary, overlapping intent. Worth comparing carefully.
- **Component tracking in modern neural nets.** Section 3.3.2's component tracking is conceptually clean but assumes you can identify *which weights/components* a task depends on. In dense transformer networks this is far harder than in modular symbolic programs. Have modular / mixture-of-experts architectures inadvertently re-enabled this property?
- **Companion experimental papers [52, 53].** The 2013 paper points to Srivastava et al. for the actual empirical results. Those should be tracked down before citing POWERPLAY-as-empirical.

## Connections

- **Conceptual descendant**: [[papers/clune-2019-ai-gas]] — AI-GAs' third pillar ("generate effective learning environments") is essentially POWERPLAY's task-invention loop, restated. Clune cites Schmidhuber.
- **Conceptual descendant**: [[papers/jiang-2022-rethinking-exploration]] — Jiang explicitly credits Schmidhuber as a foundational proposal.
- **Adjacent intrinsic-reward work**: [[papers/guo-2022-byol-explore]] — BYOL-Explore is a curiosity-driven exploration method; POWERPLAY is a more ambitious "invent the task itself" cousin.
- **Empirical leg of the same research program**: [[papers/lehman-2011-novelty-search]] — novelty search supplies the *empirical* demonstration ("objectives can deceive") that POWERPLAY supplies the *algorithmic framework* for.
- **Topic MOCs**: [[topics/open-ended-learning]], [[topics/exploration]], [[topics/intrinsic-motivation]]
- **Author indices**: [[authors/jurgen-schmidhuber]]

## Selected quotes

> "Most of computer science focuses on automatically solving given computational problems. I focus on automatically inventing or discovering problems in a way inspired by the playful behavior of animals and humans, to train a more and more general problem solver from scratch in an unsupervised fashion."

> "Given an old problem solver that can already solve a finite known set of previously learned tasks, a search algorithm is used to find a new pair that provably has the following properties: (1) The new task cannot be solved by the old problem solver."

> "PowerPlay's search orders candidate pairs of tasks and solver modifications by their conditional computational (time & space) complexity, given the stored experience so far."

> "The computational costs of validating new tasks need not grow with task repertoire size."

> "On realistic but general architectures such as PCs and RNNs, at least once the upper storage size limit of $s$ is reached, PowerPlay will start 'compressing' previous solutions, making $s$ generalize in the sense that the same relatively short piece of code helps to solve different tasks."

> "PowerPlay may be viewed as a greedy but practical implementation of basic principles of creativity."

> "Consider… how initially helpless human babies become rather general problem solvers over time? Apparently by playing."

## Source caveats

- Ingested manually via citation chase, not Readwise. Cite as Schmidhuber 2013, *Frontiers in Psychology* 4:313 (also arXiv:1112.5309).
- Content extracted from the ar5iv HTML render of the full paper. Specific algorithm numbers (Algorithm 2, Algorithm 4.1, Algorithm 4.3, Algorithm 7.1) are as named in the ar5iv render; cross-check against the PDF if citing precise algorithm numbers in a paper.
- The experimental analyses are in companion papers (referenced as [52, 53] in this paper, identified as Srivastava-Steunebrink-Schmidhuber work). Those have not been fetched as part of this triage; specific empirical claims about POWERPLAY behavior require fetching those separately.
