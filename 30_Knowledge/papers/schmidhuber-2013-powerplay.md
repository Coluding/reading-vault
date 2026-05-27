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
last_updated: 2026-05-25
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

## Worked example [analyst's view]

*Illustrative construction, not from the paper (the 2013 paper is theoretical). Domain: the solver $s$ is a small library of subroutines; a task $T$ is an input→output spec it must satisfy.*

Start with $s_0$ = empty solver, empty history. The search enumerates programs $p$ in $Kt$ order (short + fast-to-verify first).

- **Epoch 1 — invent `increment`.** $p_1$ outputs task $T_1$ = "given $n$, return $n+1$", a solver `inc(n)=n+1`, and the proof. Forward: $s_0$ can't increment ✅. Backward: no prior tasks ✅. Accept → $s_1=\{\texttt{inc}\}$. *Why not "multiply" first? Multiply is also unsolvable by $s_0$, but its build-and-verify program is longer/slower → higher $Kt$. Increment is the lowest-$Kt$ unsolvable task — "simplest still unsolvable."*
- **Epoch 2 — invent `add`, reusing frozen `inc`.** $p_2$ outputs $T_2$ = "add $a,b$" and `add(a,b)` = call `inc` $b$ times. Because `inc` is already stored and callable, $p_2$ is *short* (OOPS "conditional on experience"). Forward: $s_1$ can't add arbitrary $b$ ✅. Backward: `inc` untouched, $T_1$ still solved ✅ (guaranteed by construction in the append-only/prefix variant). Accept → $s_2=\{\texttt{inc},\texttt{add}\}$.
- **Epoch 3 — invent `multiply`, reusing `add`.** Same pattern: `mul(a,b)` = call `add` $a$ times. Forward ✅, backward ✅. The curriculum **self-orders** `inc → add → mul` — nobody scripted it; each step is just the cheapest new skill *given what's already known*.

**A rejected candidate** (makes the conditions vivid): some $p_\text{bad}$ invents `multiply` but its $q$ *rewrites* `inc` into a buggy $n+2$. Forward ✅, but **backward ❌** — $q$ no longer solves $T_1$ (and `add`, which calls `inc`, breaks too). Rejected. This is exactly the "learn a new trick, silently break an old skill" failure that strict monotonicity forbids. (Symmetrically, a task $s_2$ can already solve fails the *forward* check.)

**A wow-effect epoch:** with the solver near a capacity limit, the search instead finds $p_\text{wow}$ that invents *no new task* but refactors `inc`/`add`/`mul` into one shared counting primitive, proving all of $T_1,T_2,T_3$ still pass using *less total code*. Forward = solving prior tasks more cheaply (the wow-effect); backward ✅. Accept — the solver shrank while keeping every skill, and the shared primitive is the reusable structure that makes the *next* new skill cheap to invent.

Across every epoch the search returns one program $p$ that invents $T$, proposes $q$, and proves forward + backward — always the lowest-$Kt$ such $p$ first, which is what produces the smooth `inc → add → mul → compress` trajectory rather than a random jumble.

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

## Algo 4.1 Details

What the search actually has to do

  First, be clear on the search space. POWERPLAY isn't searching over tasks and over solver-modifications separately — it searches over programs $p$, where a single $p$ does three things when run: (1) emits a task description $T$,
   (2) emits a modified solver $q$, (3) runs the correctness proof (q solves $T_1..T_i$, $s_{i-1}$ fails $T_i$). A program either validates (produces an accepted triple) or it doesn't.

  Two hard facts make naïve enumeration impossible:

  - You can't bound runtime in advance. Some $p$ halt fast, some halt slow, some loop forever. You can't just "run each candidate to completion" — the halting problem says you'll hang on the first non-terminating one.
  - There are infinitely many candidates. You need to make progress on short/promising ones without starving them to test longer ones.

  Universal (Levin) search is the classic answer to exactly this, and OOPS is Schmidhuber's incremental version of it. That's what Algorithm 4.1 plugs in.

  The core: Levin search by time allocation

  Assign every program a prior $P(p) = 2^{-L(p)}$ ($L(p)$ = length in bits). Because programs are encoded as a prefix (self-delimiting) code, the Kraft inequality gives

  $$\sum_p P(p) = \sum_p 2^{-L(p)} \le 1.$$

  That $\le 1$ is the load-bearing fact. Now run a phase with total budget $t_{lim}$: give program $p$ a slice of

  $$\text{steps}(p) = \lfloor P(p)\cdot t_{lim}\rfloor = \lfloor 2^{-L(p)}, t_{lim}\rfloor$$

  steps, and run all programs interleaved ("dovetailed") up to their slice. Total work in the phase is $\sum_p \text{steps}(p) \le t_{lim}\sum_p P(p) \le t_{lim}$ — a whole phase costs $O(t_{lim})$ no matter how many candidates 
  exist. That's the entire trick of why slicing proportional to the prior is safe: the budget can't be overrun.

  A program that needs more steps than its current slice is simply suspended (not failed) — it resumes with a bigger slice in the next phase.

  The doubling schedule

  If a phase finishes with no validating $p$ — i.e. the candidates carrying meaningful probability mass were all run to their slice and none succeeded — double $t_{lim}$ and rerun. This is iterative deepening, but on time budget
  instead of search depth.

  The cost bookkeeping is the reason it's free to not know the right budget ahead of time. Phase $k$ costs $\le 2^k t_0$, and the phases form a geometric series:

  $$t_0 + 2t_0 + 4t_0 + \dots + 2^K t_0 < 2\cdot 2^K t_0,$$

  so all the failed phases together cost less than the final successful phase. You pay at most a factor of 2 for searching blind on the budget.

  Put the two together and you get Levin's bound: if some program $p^*$ would validate in $t(p^*)$ steps, the search finds a solution in time

  $$O!\big(t(p^*)/P(p^*)\big) = O!\big(2^{L(p^*)}, t(p^*)\big).$$

  The quantity being implicitly minimized is Levin complexity

  $$Kt(p) = L(p) + \log_2 t(p),$$

  a length-plus-log-runtime score. So the first program to validate is the one minimizing $Kt$ — short and fast to run-and-verify, jointly. That is the precise content of "explored in order of computational complexity (time &
  space)."

  What OOPS adds over plain Levin search: the "given prior experience" part

  Plain Levin search restarts from scratch each problem. OOPS (Optimal Ordered Problem Solver) makes it incremental, which is why the note says ordering is by conditional complexity "given the stored experience so far." Two
  additions:

  1. Reuse of frozen prior solutions. Programs found in earlier epochs are kept on a stack and are callable by new candidate programs. So a new $p$ that invokes an existing subroutine is short — its $L(p)$ is small because the
  heavy lifting is already encoded and just referenced. The effective prior becomes $P(p \mid \text{experience})$: programs that build on what's already known are cheap, so they're tried first. This is how "conditional
  computational complexity given prior experience" enters — the bias isn't a fixed $2^{-L(p)}$, it's $2^{-L(p,\mid,\text{stored code})}$.
  2. Bias-optimal split + protection of past solutions. OOPS splits its time budget (canonically half on extending the current best prefix, half on fresh search from scratch) and never overwrites already-found solutions. That
  protection is what keeps the incremental search from regressing — it dovetails with POWERPLAY's monotonicity requirement.

  Why this yields "the simplest still unsolvable problem"

  Now the title phrase falls out mechanically:

  - Acceptance condition (3) requires $s_{i-1}$ to fail $T_i$ — so any validated task is by definition currently unsolvable.
  - The search returns the first validating triple in $Kt$-order.
  - Therefore $T_i$ is the lowest-conditional-Levin-complexity task that the current solver can't yet do but some short, fast modification can solve-and-prove.

  "Simplest still unsolvable" = "minimal $Kt(p \mid s_{i-1}, \text{history})$ among triples whose task $s_{i-1}$ fails." It's not "simplest" in some intuitive task-difficulty sense — it's simplest in the
  invent-it-and-validate-it-cheaply sense, conditioned on everything already learned.

  The link back to the validation-cost section

  One connection the note leaves implicit: the runtime $t(p)$ in $Kt(p)$ includes the correctness-demonstration cost (re-checking $T_1..T_{i-1}$). So if validation cost grew linearly with repertoire size, the $\log t(p)$ term
  would creep up every epoch and progressively penalize every candidate — the search would grind down as the agent learns more. That's exactly why the component-tracking and prefix-code mechanisms (the next section of the note)
  matter: they hold the validation term roughly constant, which keeps the complexity ordering — and the per-epoch search cost — from degrading as the repertoire grows. The two sections are coupled: the search-ordering argument
  only stays practical because the validation-cost argument keeps $t(p)$ bounded.


