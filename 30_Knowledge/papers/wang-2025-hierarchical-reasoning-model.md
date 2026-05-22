---
type: paper
title: "Hierarchical Reasoning Model"
authors: ["Guan Wang", "Jin Li", "Yuhao Sun", "Xing Chen", "Changling Liu", "Yue Wu", "Meng Lu", "Sen Song", "Yasin Abbasi Yadkori"]
year: 2025
venue: arXiv
url: https://arxiv.org/abs/2506.21734
rw_id: 01ks7dra0pb5jsf6fx6jr11e9g
topics: [recursive-reasoning, latent-reasoning, brain-inspired, equilibrium-models, adaptive-computation]
priority: high
read_state: queued
relevance: ""
added: 2026-05-22
last_updated: 2026-05-22
---

# Hierarchical Reasoning Model

**TL;DR** — A brain-inspired recurrent architecture (Sapient Intelligence) with two interlocked modules — a *slow* high-level $f_H$ and a *fast* low-level $f_L$ — that performs $N \times T$ steps of internal latent computation per forward pass. With only ~27M params and ~1000 training examples per task, HRM solves Sudoku-Extreme, Maze-Hard, and ARC-AGI puzzles end-to-end without pretraining or CoT data, beating much larger LLMs that use chain-of-thought. The key tricks: (i) "hierarchical convergence" where $f_L$ converges within a cycle and $f_H$ then resets its target, preventing the premature-convergence pathology of plain RNNs; (ii) a 1-step DEQ-style approximate gradient that needs $O(1)$ memory; (iii) deep supervision across segments; (iv) adaptive computational time (Q-learning halting).

## Context & motivation

The framing target is the limitation of fixed-depth Transformers:

> "The fixed depth of standard Transformers places them in computational complexity classes such as $AC^0$ or $TC^0$, preventing them from solving problems that require polynomial time" (§1).

And the practical limits of CoT:

> "CoT for reasoning is a crutch, not a satisfactory solution. It relies on brittle, human-defined decompositions where a single misstep or a misorder of the steps can derail the reasoning process entirely. This dependency on explicit linguistic steps tethers reasoning to patterns at the token level" (§1).

The pitch: "latent reasoning" — perform extended computation in a recurrent hidden state, not in token space. HRM positions itself against Universal Transformers / Looped Transformers (recurrent depth without hierarchy) and against the deep-RNN convergence-collapse problem.

Brain-inspired motivation (§2):
- Hierarchical cortical processing across timescales.
- Slow theta-band ($f_H$) gates fast gamma-band ($f_L$) computations.
- Recurrent feedback loops without BPTT, since BPTT is biologically implausible.

## Method

### Architecture (§2)
Four learnable components: input network $f_I$, low-level recurrent $f_L$, high-level recurrent $f_H$, output network $f_O$. The forward pass unfolds over $N$ high-level cycles of $T$ low-level timesteps each ($N \times T$ total steps).

$$\tilde{x} = f_I(x; \theta_I)$$
$$z_L^i = f_L(z_L^{i-1}, z_H^{i-1}, \tilde{x}; \theta_L)$$
$$z_H^i = \begin{cases} f_H(z_H^{i-1}, z_L^{i-1}; \theta_H) & \text{if } i \equiv 0 \pmod{T} \\ z_H^{i-1} & \text{otherwise} \end{cases}$$
$$\hat{y} = f_O(z_H^{NT}; \theta_O)$$

$f_L$ updates every step conditioned on the *frozen* high-level state for that cycle; $f_H$ only updates at the end of each cycle.

### Core idea — hierarchical convergence (§2)

> "During each cycle, the L-module (an RNN) exhibits stable convergence to a local equilibrium. This equilibrium, however, depends on the high-level state $z_H$ supplied during that cycle. After completing the $T$ steps, the H-module incorporates the sub-computation's outcome (the final state $z_L$) and performs its own update. This $z_H$ update establishes a fresh context for the L-module, essentially 'restarting' its computational path and initiating a new convergence phase toward a different local equilibrium."

This is the central architectural insight: $f_L$ would otherwise saturate, but $f_H$ periodically perturbs its target, keeping the system in a non-trivial computation regime. Effective depth is $NT$ but with stable per-cycle dynamics.

### Approximate gradient (§2)
Standard BPTT needs $O(T)$ memory. HRM uses a **1-step gradient approximation** rooted in **Deep Equilibrium Models** and the Implicit Function Theorem. If $f_L$ converges to a fixed point $z_L^\star$ within a cycle, IFT gives:

$$\frac{\partial z_H^\star}{\partial \theta} = (I - J_\mathcal{F}|_{z_H^\star})^{-1} \frac{\partial \mathcal{F}}{\partial \theta}\bigg|_{z_H^\star}$$

The 1-step approximation drops the Neumann series after the identity term: $(I - J_\mathcal{F})^{-1} \approx I$. This yields gradients that flow only through the *last* state of each module — memory cost $O(1)$, no unrolling through time. Pseudocode (§2):

```
with torch.no_grad():
    for _i in range(NT):
        zL = L_net(zL, zH, x)
        if _i % T == T-1: zH = H_net(zH, zL)
# 1-step grad
zL = L_net(zL, zH, x)
zH = H_net(zH, zL)
return zH, zL, output_head(zH)
```

### Deep supervision (§2)
Training stacks $M$ "segments" of $N \times T$ steps. After each segment, compute loss, take an optimizer step, then *detach* $z^m$ before the next segment. Equivalent to a 1-step gradient at the segment level. The detach prevents gradients from flowing across segments — "more frequent feedback to the H-module and serves as a regularization mechanism" (§2).

### Adaptive Computational Time (ACT)
HRM augments deep supervision with **Q-learning halting**: the model learns to decide when to stop computing per example. Brain motivation (§2): the model "dynamically alternates between automatic thinking (System 1) and deliberate reasoning (System 2)."

## Experimental setup (§3)

- **Sudoku-Extreme**: 9×9 with mean difficulty 22 backtracks per puzzle (vs ~0.45 for Sudoku-Bench). 1000 train examples (or full 3.8M for analysis).
- **Maze-Hard**: 30×30 mazes with shortest-path length > 110. 1000 train / 1000 test.
- **ARC-AGI**: ARC-AGI-1 and ARC-AGI-2 fluid-intelligence benchmark. Standard augmentation (translations, rotations, flips, colour permutations); test-time inference generates 1000 augmented variants per test input; top-2 majority predictions submitted.
- All tasks: random init, no pretraining, no CoT labels.
- Optimizer: Adam-atan2 (scale-invariant Adam variant) + constant LR + linear warmup.
- 27M parameters total.

## Key results (§3, Fig. 1)

- **Sudoku-Extreme**: HRM achieves "nearly perfect performance" with 1000 training examples. Direct-prediction baselines and CoT LLMs "fail entirely." Direct-pred with full 3.8M Sudoku-Extreme reaches only 16.9% accuracy.
- **Maze-Hard**: similar story — HRM solves it; CoT baselines "almost never manage to solve the tasks." 175M-param Transformer on 1M examples (Lehnert et al.) "marginal success ... below 20% using the pass@64 evaluation metric."
- **ARC-AGI**: HRM outperforms "much larger models with significantly longer context windows."

Specific headline numbers (figures, not always quoted in body):
- Direct-pred swapped with HRM "more than a twofold performance improvement" on ARC-AGI-1.

## Ablations & analysis

### Brain correspondence (§4)
- The **Participation Ratio** (effective dimensionality) of HRM's modules: $z_L$ PR = 30.22, $z_H$ PR = 89.95. Ratio $z_H / z_L \approx 2.98$.
- Mouse cortex empirical ratio: $\approx 2.25$.
- Untrained network: $z_L$ PR = 42.09, $z_H$ PR = 40.75 — *no* hierarchy. The dimensionality split is an *emergent property of training*, not architectural.
- "HRM therefore departs from the collapse pattern and instead fosters a high-dimensional representation in its higher module."
- The paper is careful: "this evidence is correlational ... the causal necessity of this emergent hierarchy remains an important question."

### Solution-strategy visualization (§3.3)
- Maze: HRM "appears to initially explore several potential paths simultaneously, subsequently eliminating blocked or inefficient routes, then constructing a preliminary solution outline followed by multiple refinement iterations."
- Sudoku: "resembles a depth-first search approach, where the model appears to explore potential solutions and backtracks when it hits dead ends."
- ARC: "incremental adjustments to the board and iteratively improving it until reaching a solution ... similar to hill-climbing."
- "The model shows that it can adapt to different reasoning approaches, likely choosing an effective strategy for each particular task."

## Limitations

[analyst's view] — the paper doesn't have a stand-alone limitations section in the fetched body, so this is partly synthesised:

- **Single-task training, single-task evaluation.** Each benchmark is a separate trained model. The paper doesn't show one HRM solving Sudoku, ARC, and Maze simultaneously.
- **Causal necessity of the dimensionality hierarchy is open** (acknowledged in §4).
- **The 1-step gradient approximation is theoretically motivated by DEQ convergence** but HRM doesn't *force* $f_L$ to converge; it relies on $f_L$ converging in practice within $T$ steps. The implicit assumption deserves more scrutiny.
- **No scaling-up beyond 27M.** Whether the hierarchical-convergence mechanism survives at, say, 1B parameters is open.
- **GRAM ([[papers/baek-2026-gram]]) shows HRM is deterministic** — single latent trajectory per input. For multi-solution problems (N-Queens), HRM has fundamentally limited coverage.

## Why it matters [analyst's view]

Three things land:

1. **Latent reasoning > CoT, for specific problems.** HRM is a clean existence proof that token-level CoT is not the only way to scale reasoning depth. 27M params, no pretraining, no CoT data, solves problems that CoT-based 100B+ LLMs fail completely on. This is the strongest empirical evidence to date that *architectural* depth (recurrent latent computation) is a legitimate alternative axis to *generation* depth (longer CoT traces).

2. **Hierarchical convergence is the load-bearing idea.** Most prior recurrent-Transformer work (Universal Transformer, Looped Transformer) just adds a recurrent loop and a halt mechanism. HRM's contribution is the observation that *plain* recurrence converges prematurely and stalls — and that a slow-modulator architecture cleanly fixes this. The brain-inspired framing is rhetorical; the actual mechanism is well-defined and architecturally distinct.

3. **The DEQ-IFT-1-step-grad recipe is a powerful template.** Memory $O(1)$ gradient through a deep recurrent stack is exactly what makes HRM cheap to train. This is the same idea (1-step gradient at equilibrium) that [[papers/maes-2026-leworldmodel]] sidesteps by using SIGReg, and that [[papers/guo-2022-byol-explore]]'s BYOL-style methods replace with EMA. Three lineages, three solutions to the same "how do you train a recurrent system without BPTT" problem.

The 2026 follow-ups already in vault — GRAM and the broader recursive-reasoning line — build directly on this paper. HRM is the foundational entry.

## Open questions

[analyst's view]
- **Why exactly do the brain-PR ratios match?** Is the parallel a deep claim about the universality of hierarchical-dimensionality optimisation pressure, or selection bias in benchmark choice?
- **Does ACT actually learn to allocate compute well?** The paper claims it does but a careful budget-vs-difficulty ablation is left to follow-ups.
- **The "1000 training examples per task" line is striking** — but each task uses heavy augmentation. The effective training-set size is orders of magnitude larger. Worth a careful reader-side audit.
- **Can HRM's hierarchical mechanism be wrapped around a Transformer?** The paper uses small per-module nets. A scaled-up version with full Transformer blocks as $f_L$ and $f_H$ is the obvious next test.
- **GRAM extends HRM with stochasticity** ([[papers/baek-2026-gram]]). Are there other extension axes (more than 2 levels, learned cycle lengths, dynamic $T$)?

## Connections

- **Extended by**: [[papers/baek-2026-gram]] — GRAM adds stochastic latent transitions on top of HRM's deterministic hierarchical recursion; addresses HRM's single-trajectory limitation on multi-solution tasks.
- **Cousin / contrast**: Universal Transformer, Looped Transformer, TRM (Tiny Recursive Models) — referenced as baselines GRAM beats and HRM extends conceptually.
- **Adjacent**: [[papers/maes-2026-leworldmodel]] — different recurrent-latent computation pattern (no $f_H/f_L$ separation but same "compact latent dynamics" framing) for world modeling rather than reasoning.
- **Topic MOCs**: [[topics/recursive-reasoning]], [[topics/latent-reasoning]], [[topics/brain-inspired]]
- **Author indices**: [[authors/guan-wang]]

## Selected quotes

> "We propose the Hierarchical Reasoning Model (HRM), a novel recurrent architecture that attains significant computational depth while maintaining both training stability and efficiency. HRM executes sequential reasoning tasks in a single forward pass without explicit supervision of the intermediate process, through two interdependent recurrent modules: a high-level module responsible for slow, abstract planning, and a low-level module handling rapid, detailed computations." — abstract

> "With only 27 million parameters, HRM achieves exceptional performance on complex reasoning tasks using only 1000 training samples. The model operates without pre-training or CoT data, yet achieves nearly perfect performance on challenging tasks including complex Sudoku puzzles and optimal path finding in large mazes." — abstract

> "HRM therefore departs from the collapse pattern and instead fosters a high-dimensional representation in its higher module ... HRM autonomously discovers an organizational principle that is thought to be fundamental for achieving robust and flexible reasoning in biological systems." — §4
