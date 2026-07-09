---
type: paper
title: "A Compositional Framework for Open-ended Intelligence"
authors: ["Ida Momennejad", "Roberta Raileanu"]
year: 2026
venue: "arXiv preprint"
url: https://arxiv.org/abs/2606.15386
rw_id: 01kvq3sahfs6f6dtk9ngy9m8zn
topics: [open-endedness, reinforcement-learning, compositional-generalization]
priority: medium
read_state: queued
relevance: ""
added: 2026-07-08
last_updated: 2026-07-08
---

## TL;DR

This is a **theoretical / position paper** (no experiments) that proposes a formal object for open-ended intelligence built from a finite set of primitives $P$, a set of composition operators $C$, and the induced closure $L(P, C)$. The central move is to relocate open-endedness away from *behavioral* repertoires (the dominant "stepping stone" framing of Novelty Search, QD/MAP-Elites, POET, XLand) and onto *compositional* structure: open-endedness is the property that a **minimal generating set** of primitives and composition rules yields an *infinite yet useful* closure across families of worlds. The paper's constructive proposal is **Next Primitive Prediction (NPP)** — an architectural objective analogous to next-token prediction, but where the model predicts the next primitive–operator pair in a reasoning trace under a parsimony (minimum-description-length) constraint, forcing a compact, role-flexible basis that transfers across "possible worlds" via **Transfer-as-Recomposition (TaR)**. It also proposes evaluation metrics (Primitive Reuse Index, Compositional Depth Generalization, Transfer-as-Recomposition) and grounds the framework in case studies from physics, neuroscience, collective intelligence, and evolutionary biology. No numbers are reported because the paper is conceptual; its contribution is a vocabulary and a research agenda, not an empirical result.

## Context & motivation

The paper attacks a gap in existing open-ended learning: current systems (POET, XLand, Voyager, FunSearch, AlphaEvolve, Novelty Search, QD) learn to handle variation *within* a single environment or procedurally-generated family, but their "reuse" is **episodic rather than compositional** — an agent replays a skill when it meets a similar scenario, but does not decompose that skill into transferable *algorithmic primitives*. The authors argue the dominant "stepping stone" framing (Lehman & Stanley 2011; Stanley et al. 2017; Clune 2019) captures how complex solutions emerge without a fixed objective but "does not natively capture compositional generalization," offers no guarantee of learning a minimal set of latent primitives, and no guarantee of efficient learning. The paper positions itself against reward-maximization ("Reward is Enough", Silver et al. 2021) and scale-only ("Bitter Lesson", Sutton 2019) views, arguing these deliver raw performance but not the mechanistic interpretability or sample efficiency needed to adapt to data-scarce novel world families. It draws heavily on computational-neuroscience evidence (mixed selectivity, disentangled/abstract geometry in prefrontal cortex and hippocampus) as an "existence proof" that biological intelligence solves compositional generalization through structured primitives and recombination, and on recent mechanistic-interpretability work (Lippl et al. 2026; Eberle et al. 2025) claiming that LLM latent representations decompose into algorithmic primitives and their compositional geometry.

## Method

This is a conceptual framework, so "method" is a set of formal definitions and an architectural proposal rather than an algorithm with training runs.

### Problem formulation
Open-ended intelligence is redefined as the capacity to **search over the closure** $L(P, C)$ induced by a minimal primitive set $P$ and composition operators $C$, generating an unbounded space of algorithms, representations, behaviors, and solutions across worlds. Exploration is framed as search over *the right primitives and the right transition motifs to compose them* — distinct from search over actions or skills alone.

### Core idea
Two pillars: (1) a minimal set of **representational primitives** (what the agent perceives as objects/relations/states — a world model is a combination of these) and **algorithmic primitives** (minimal computational operations: nearest-neighbor retrieval, distance computation, comparison, verification); (2) an acquired **compositional grammar** (sequencing, recursion, branching) plus recurring **transition motifs**. The closure of the two pillars is claimed to yield unbounded adaptive responses. Progress is measured by *parsimony*: a **smaller, more reusable basis** simultaneously with an *expansion* in the diversity and depth of reachable compositions.

### Architecture / algorithm
The formal core is the triple $(P, C, L)$:
- $P$ — a finite primitive library (representational: axioms, fields, objects; algorithmic: comparison, retrieval, verification).
- $C$ — composition operators / motifs that chain primitives (sequencing, recursion, branching).
- $L(P, C)$ — the **closure**: the set of all computations, representations, and strategies reachable by repeatedly applying $C$ to elements of $P$. Open-endedness is defined by whether $|L(P, C)| = \infty$.

Every solution is represented as a **Primitive Transition Graph (PTG)** $G = (P, C)$: nodes are primitives $p \in P$ invoked during the solution, edges are operator instances $c \in C$ that transform state between primitives. Recurring subgraph **motifs** ("meaningful phrases" — e.g. `[Calculate Distance] → [Threshold Check] → [Branch]` appearing in 90% of navigation tasks) are mined, "wrapped," and promoted to **higher-order primitives**, yielding a hierarchy (analogous to refactoring a recurring code block into a reusable function).

The architectural objective is **Next Primitive Prediction (NPP)**. Given a partial traversal $(p_{1:t}, c_{1:t})$ of a PTG in a world $W$, NPP predicts the next primitive–operator pair:
$$(p_{t+1}, c_{t+1}) \sim p_{\theta}(\cdot \mid p_{1:t}, c_{1:t}, W), \qquad (p_{t+1}, c_{t+1}) \in \mathcal{P} \times \mathcal{C}$$
with loss
$$\mathcal{L}_{\text{NPP}} = -\log P(p_{t+1}, c_{t+1} \mid p_{1:t}, c_{1:t}, W)$$
where $W$ is a "world vector" (the physics/constraints of the current domain) fused into the latent representation, $p_{t+1}$ is the predicted primitive, and $c_{t+1}$ the predicted composition operator. The predictor is proposed as a transformer-based graph neural network (2 levels: a backbone for multimodal fusion of state $S_t$ and graph history $G_t$; prediction over a structured vocabulary of primitives+operators rather than surface tokens). Crucially, because the vocabulary is primitives-and-operators, "new solutions emerge from recomposition of existing primitives rather than from task-specific fragments."

### Derivations / why it works
The paper states two informal propositions (asserted, not proved with full rigor in the fetched text):
- **Proposition 4.1 (Unbounded compositional closure).** If $P$ is finite and $C$ contains at least one recursive/generative operator, and compositions under $C$ preserve type-consistency and admit reuse of intermediate outputs, then $L(P, C)$ contains an unbounded number of distinct compositions. The analogy given: the natural numbers are the closure of $\{0\}$ under the successor function — an infinite set from a single compact rule. The authors stress that *unbounded closure alone is trivial*; the substantive criterion is the **minimal generating set problem** — the smallest $P, C$ whose closure covers the relevant task space — bounded by verification primitives and description-length pressure.
- **Proposition 5.2 (Compositional pressure toward a minimal basis).** If training enforces a parsimony constraint favoring reuse of a minimal primitive set, then optimizing $\mathcal{L}_{\text{NPP}}$ biases the system toward representing solutions as compositions over a shared basis $(P, C)$ rather than as task-specific sequences; consequently learned primitives/motifs admit reuse across distinct task families.

Parsimony is operationalized as a **Minimum Description Length (MDL)** objective — "what is the smallest set of functions $P$ that can reconstruct all successful traces?" — approximated in practice by a **contrastive** penalty (SimCLR/MoCo-style) that punishes the model for emitting a complex non-reusable black-box operation when a composition of existing primitives would suffice. This is explicitly framed as wake–sleep library learning in the spirit of **DreamCoder** (Ellis et al. 2021). The authors note sparse steerable directions already exist in trained transformers (function vectors, Todd et al. 2024), so the exploitable geometry is partly present before fine-tuning; whether NPP is ultimately cheaper than next-token training is flagged as an open empirical question.

### Training procedure
_Not addressed by the source_ — no datasets, optimizer, or hyperparameters are given; NPP is proposed, not trained. The paper suggests a **self-play / self-improvement** setting where an agent authors its own curriculum by generating "imagined worlds" (counterfactual variants of mastered ones), and searches not over policies but over the closure $L(P, C)$. The parsimony constraint is what makes self-play *self-improving rather than self-confirming*: a primitive earns its place only if it survives recomposition across the agent's own imagined variations (echoing the epistemic "neighborhood of nearby possible worlds" used to separate knowledge from lucky guesses).

### Inference / sampling
_Not addressed by the source_ (no generative sampling procedure; the architecture is proposed, not run).

## Experimental setup

_Not addressed by the source_ — this is a position/theory paper with **no experiments**. In place of experiments it offers Table 1 and Table 2, which taxonomize existing open-ended systems (Evolutionary Algorithms, POET, POET-Enhanced, XLand, AdA, FunSearch, AlphaEvolve, ASI/Hughes et al., Novelty Search, JEPA) against the proposed compositional framework, and four qualitative case studies (physics, neuroscience, collective intelligence, evolutionary biology).

## Key results

No empirical numbers. The "results" are conceptual claims and proposed **evaluation metrics** for compositional open-ended intelligence:
- **Primitive Reuse Index (PRI)** — frequency with which a learned primitive $p \in P$ appears across distinct task families; high PRI = general axiom, low PRI = brittle fragment; meant to separate memorization from induction of a minimal generating set.
- **Compositional Depth Generalization (CDG)** — success on PTGs of depth $d_{\text{test}} > d_{\text{train}}$, testing mastery of the closure.
- **Primitive Discovery Yield (PDY)** — rate at which newly discovered primitives improve utility / sample efficiency.
- **Open-ended Curriculum Robustness (OCR)** — robustness of primitive composition under shifted task distributions.
- **Transfer-as-Recomposition (TaR)** — role-flexibility measured by holding the library fixed while changing environmental constants; low TaR ⇒ primitives were memorized.
- Also proposed: **causal isolation/induction** in the residual stream — ablating a primitive vector $p$ should cause predicted failures, and patching it across contexts should transfer function (following the mechanistic-interpretability results of Lippl et al. 2026, who extract a primitive like `get-nearest-neighbor-city` from an LLM solving TSP and inject it into an AIME-solving run).

## Ablations

_Not addressed by the source_ (no experiments to ablate).

## Limitations

The authors themselves flag (Section 8.3, "Compositionality Spectrum") that they do **not** claim all intelligent computation is compositional: systems occupy a spectrum from memorization (episodic retrieval) → cached inference (e.g. the successor representation) → full compositional generalization. They concede composition is "essential and underdeveloped" but not subsuming the other modes, and that the field lacks the formal vocabulary to study it directly. An honest reader should add: (1) the two propositions are stated informally, without complete proofs in the fetched text; (2) NPP is entirely unvalidated — there is no demonstration that such an objective is trainable, competitive with next-token prediction, or that discovered primitives actually transfer; (3) the "minimal generating set problem" is acknowledged as central but no algorithm is given to solve it beyond MDL/contrastive heuristics; (4) the neuroscience/evolution case studies are analogies and existence proofs, not evidence that the proposed architecture works.

## Why it matters [analyst's view]

The paper is best read as a *conceptual reframing plus a bet on an architecture*. Its most useful contribution is the crisp distinction between **compression** (library learning: does a shared library better compress programs for the *same* task?) and **transfer** (does a primitive recovered in one task family reappear in the same functional role in a *structurally unrelated* one?) — the latter is what PRI and TaR are designed to measure, and it cleanly separates this proposal from DreamCoder/Stitch. The NPP objective is the load-bearing idea: if next-token prediction induced linguistic grammar, the bet is that next-*primitive* prediction over a graph-structured vocabulary induces a *grammar of computation*. That's an appealing analogy but currently unbacked by any run. For the vault this connects the open-endedness literature (POET/XLand/Voyager) to mechanistic interpretability (function vectors, residual-stream primitives) and to world models (representational primitives = a world model in their framing). The explicit hook to JEPA — that a JEPA backbone could in principle serve NPP if augmented with a minimal-set primitive library, a graph-structured target, and a parsimony constraint — is worth tracking. The paper is a research agenda; its value will be determined entirely by whether someone can instantiate NPP and show measurable transfer.

## Open questions / things to verify

- Is NPP actually trainable, and is it cheaper or more expensive than next-token training? (The authors explicitly call this "an open empirical question.")
- Do the two propositions have full formal proofs in the appendix (not present in the fetched HTML)? Proposition 4.1's preconditions (type-consistency + reuse of intermediate outputs) look sufficient for infinitude but the *useful/minimal* part is where the real difficulty hides.
- How is the "world vector" $W$ obtained in practice, and does conditioning on it actually enable re-binding (e.g. re-binding "jump" to a new gravity constant) as claimed?
- Does the Lippl et al. (2026) primitive-injection result (TSP primitive → AIME) replicate and generalize? Much of the framework's plausibility rests on it.
- Are PRI/CDG/TaR measurable on real trained models, or do they require access to a clean $P, C$ decomposition that current models don't expose?

## Connections

- Topic MOCs: [[topics/open-endedness]], [[topics/reinforcement-learning]]
- Author indices: [[authors/ida-momennejad]], [[authors/roberta-raileanu]]
- Contrasts with: JEPA / next-latent prediction (Assran et al. 2023), Voyager (Wang et al. 2023), FunSearch (Romera-Paredes et al. 2024), AlphaEvolve (Novikov et al. 2025), POET/XLand, DreamCoder & Stitch (library learning), eigenoptions (Machado et al. 2017)

## Selected quotes

> "Open-endedness is therefore not primarily a property of a behavioral repertoire; it is a property of a transition structure over primitives." — §3.3

> "Just as next-token prediction in transformer architectures led to the learning of the grammar of language, next-primitive prediction ... could lead to the discovery of the compositional grammar of open-ended intelligence." — §3.4

> "Library learning asks whether a shared library better compresses programs for the same task; we ask whether a primitive recovered in one task family reappears, in the same functional role, in a structurally unrelated one. That is a transfer question, not a compression question." — §8.1
