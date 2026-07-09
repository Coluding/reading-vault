---
type: paper
title: "Fixed-Point Reasoners: Stable and Adaptive Deep Looped Transformers"
authors: ["Sajad Movahedi", "Vera Milovanović", "Shlomo Libo Feigin", "Alexander Theus", "Thomas Hofmann", "Valentina Boeva", "T. Konstantin Rusch", "Antonio Orvieto"]
year: 2026
venue: "arXiv preprint"
url: https://arxiv.org/abs/2606.18206
rw_id: 01kvq3q6xr5eya7shankynepch
topics: [looped-transformers, reasoning, recurrence, latent-reasoning]
priority: high
read_state: queued
relevance: ""
added: 2026-07-08
last_updated: 2026-07-08
---

## TL;DR

FPRM (Fixed-Point Reasoning Model) is a non-hierarchical Looped Transformer for latent reasoning that (a) makes deep looping trainable by switching from post-norm to **pre-norm plus residual scaling**, and (b) halts adaptively by **iterating until the hidden state converges to a fixed-point**, using the convergence residual itself as the stop signal — no separate ACT halting module. The core insight is that "a Looped Transformer is, in part, a very deep transformer," so it inherits deep-net signal-propagation pathologies; pre-norm fixes signal propagation but blows up activations, and learnable residual-scaling scalars ($\alpha_1,\beta_1,\alpha_2,\beta_2$) restore boundedness while keeping the map contractive enough to converge. With only 7M parameters and no hierarchy, FPRM reaches **94.2% on Sudoku-Extreme, 87.0% on Maze-Hard, and 47.5% on ARC-AGI-1 (pass@2)** — beating the hierarchical HRM/TRM baselines of equal or larger size on three of four tasks. It is, the authors claim, the first Transformer-based reasoning model to genuinely scale test-time compute to task difficulty (unlike TRM's ACT, which halts too early or too late).

## Context & motivation

Test-time reasoning needs two ingredients: **flexibility** (spend a variable amount of compute) and **adaptivity** (decide when to stop). Chain-of-Thought supplies both but requires special training regimes and hand-crafted traces, undermining end-to-end training. Looped architectures (Universal Transformers, Neural GPUs, and recently HRM and TRM) supply flexibility for free by scaling compute along the *depth* dimension: $\mathbf{z}_{i+1} = f_\theta(\mathbf{z}_i; \mathbf{x})$, unrolling the same weight-tied block. They have an inductive bias toward learning algorithms and have beaten much larger LLM reasoners on puzzles.

But two design axes were left open. **First, halting**: most looped models fix or randomly sample the loop count (no adaptivity), or bolt on an Adaptive Computation Time (ACT) module — a separate learned halting network that requires a continuous relaxation of a discrete stopping decision and is hard to optimize. The paper (and prior analyses of HRM) shows ACT does *not* reliably scale compute with actual difficulty. **Second, depth utilization**: unrolling many loops yields a very deep effective network, so looped models suffer the "curse of depth" (signal propagation problems, rank collapse). Curiously, looped models mostly use **post-norm** — the opposite of the pre-norm standard in deep non-looped transformers — because post-norm keeps activations bounded as the loop iterates. The paper's driving question: *can we switch post-norm → pre-norm (for good signal propagation) while keeping activations bounded some other way?* FPRM answers yes, via residual scaling, and shows the resulting model no longer needs the hierarchy that HRM/TRM relied on.

## Method

### Problem formulation
A Looped Transformer applies a weight-tied block $f_\theta(\cdot;\mathbf{x})$ repeatedly to a latent state $\mathbf{z}$, conditioned on input $\mathbf{x}$, producing a prediction $\hat{y}=h_\phi(\mathbf{z})$ from the (converged) latent via a head $h_\phi$. Objective: cross-entropy on exact-sequence puzzle targets, trained with deep supervision. The design goals are (i) trainability at large effective depth, (ii) bounded/stable iterates, (iii) convergence to a fixed-point usable as a halting signal.

### Core idea
Treat the unrolled loop as a deep network: use **pre-norm** for signal propagation, add **learnable residual-scaling scalars** to keep the recurrence bounded and contractive so it converges to a fixed-point $\mathbf{z}^\star = f_\theta(\mathbf{z}^\star;\mathbf{x})$, and **halt when consecutive iterates stop changing** (the convergence residual drops below a tolerance).

### Architecture / algorithm

**Normalization placement.** A Transformer layer has two sub-layers (attention, FFN) with maps $f^\ell_{\theta^\ell}$. Post-norm normalizes *after* the residual add; pre-norm normalizes the *input* of each sub-layer:
$$\mathbf{z}^{\ell} = \operatorname{Norm}_{\text{post}}\!\left(\mathbf{z}^{\ell-1} + f^{\ell}_{\theta^{\ell}}(\operatorname{Norm}_{\text{pre}}(\mathbf{z}^{\ell-1}))\right),\qquad \ell = 1,\dots,2L.$$
Here $L$ is the number of Transformer blocks (so $2L$ sub-layers). Post-norm bounds activation magnitude but blocks signal propagation; pre-norm propagates signal but lets residual magnitude grow exponentially with depth. FPRM keeps only $\operatorname{Norm}_{\text{pre}}$ and controls magnitude with scaling.

**Layer-wise residual scaling.** Within one application of $f_\theta$, residual stream and sub-layer output are weighted by tied scalars $(\alpha_1,\beta_1)$ shared across all $L$ layers:
$$\mathbf{z}^{\ell} = \alpha_1\,\mathbf{z}^{\ell-1} + \beta_1\, f^{\ell}_{\theta^{\ell}}\!\left(\operatorname{Norm}_{\text{pre}}(\mathbf{z}^{\ell-1})\right),\qquad \ell=1,\dots,2L. \tag{2}$$
$\alpha_1$ weights how much of the residual stream is preserved (keeping it dominant, i.e. $\alpha_1$ high, is the classic signal-propagation remedy); $\beta_1$ weights the sub-layer's contribution.

**Iteration-wise input mixing.** Between consecutive loop iterations, the input $\mathbf{x}$ is re-injected with tied scalars $(\alpha_2,\beta_2)$ shared across iterations (à la DEQ input injection):
$$\mathbf{z}^0_{i+1} = \alpha_2\,\mathbf{z}^{2L}_i + \beta_2\,\mathbf{x}. \tag{3}$$
$\alpha_2$ controls how much of the previous loop's output persists (smaller $\alpha_2$ = more contractive), $\beta_2$ the input-injection strength.

### Derivations / why it works

**Boundedness (Theorem 1).** The two scaling schemes are coupled. Assuming each sub-layer map is bounded, $\|f^\ell_{\theta^\ell}(\mathbf{u})\| \le c_f$, and $0\le\alpha_1,\alpha_2<1$, choosing
$$\beta_2 = 1 - \alpha_2\alpha_1^{2L},\qquad \beta_1 = \frac{\beta_2(1-\alpha_1)}{1-\alpha_1^{2L}}$$
makes the fixed-point iterates $\{\mathbf{z}^0_i\}$ bounded, and if they converge to $\mathbf{z}^0_\infty$ then $\|\mathbf{z}^0_\infty\| \le \|\mathbf{x}\| + \alpha_2 c_f$. The per-sub-layer boundedness assumption $\|f^\ell\|\le c_f$ is itself satisfied by pre-norm (Kim et al. 2021). So pre-norm + this coupling gives bounded iterates *for any input* — recovering the property post-norm was providing.

**Convergence (Theorem 2).** Let $\lambda_f$ be the Lipschitz constant of the $L$-layer map $\mathbf{z}^0 \mapsto \mathbf{z}^{2L}$ from Eq. 2. Then the full looped step $f_\theta(\cdot;\mathbf{x})$ (Eqs. 2–3) is Lipschitz with constant $\alpha_2\lambda_f$. If $\alpha_2\lambda_f < 1$ the map is a **contraction**, so by Banach the iteration converges to a *unique* fixed-point $\mathbf{z}^\star$ at a linear rate:
$$\|f_\theta(\mathbf{z}_i;\mathbf{x}) - \mathbf{z}_i\| \le (\alpha_2\lambda_f)^i\,\|f_\theta(\mathbf{z}_0;\mathbf{x}) - \mathbf{z}_0\|.$$
Hence a small $\alpha_2$ buys convergence. But an *overly* contractive map kills expressivity, so $\alpha_1,\alpha_2$ are made **learnable**; empirically, initializing $\alpha_2$ small (more contractive) then letting it adapt works best, and after training the $\alpha$ distributions widen while medians stay near init.

**Oscillation and damping (Theorem 3).** Theorem 2 is only sufficient; in practice the true contraction factor isn't guaranteed and some inputs make the iteration *oscillate* around $\mathbf{z}^\star$ without converging. Linearizing near the fixed-point, $\mathbf{z}_{i+1}-\mathbf{z}^\star \approx J(\mathbf{z}_i-\mathbf{z}^\star)$ with $J=\partial f_\theta/\partial\mathbf{z}|_{\mathbf{z}^\star}$; oscillation occurs when an eigenvalue has $\Re(\lambda_i)<1$ but $|\lambda_i|\ge 1$ (spiraling). The fix is a **damped iteration map**:
$$g_{\eta,\theta}(\mathbf{z};\mathbf{x}) := \eta f_\theta(\mathbf{z};\mathbf{x}) + (1-\eta)\mathbf{z}.$$
If every eigenvalue satisfies $\Re(\lambda_i)<1$, there is an $\eta_0\in(0,1)$ such that for all $\eta\in(0,\eta_0)$ the damped iteration converges locally, and $g_{\eta,\theta}$ has the **same fixed-points** as $f_\theta$. So damping suppresses oscillation without changing the solution.

**Halting signal.** Convergence at iteration $i$ is measured by the relative residual
$$r_i = \frac{\|\mathbf{z}_i - f_\theta(\mathbf{z}_i;\mathbf{x})\|_\infty}{\|f_\theta(\mathbf{z}_i;\mathbf{x})\|_\infty + \epsilon},$$
and the loop halts once $r_i < \tau$. The damping $\eta$ is chosen adaptively by a **patience mechanism** (Algorithm 1, "FPOPT"): track the best residual so far $r^\star=\min_{j\le i} r_j$; if no improvement for $P$ consecutive steps (and $r>\tau$), geometrically decay $\eta \leftarrow \gamma\eta$ with $\gamma\in(0,1)$ and reset patience. So $\gamma$ (decay rate) and $P$ (patience) become the compute–accuracy dials.

### Training procedure
Built on the public **TRM codebase**, adopting its **deep-supervision** loop (Algorithm 2). Training loops in windows of $k$ iterations (the truncated-BPTT depth, a hyperparameter); at inference $k=1$. Each forward pass calls FPOPT to damp the step; the head produces a prediction, cross-entropy is computed and back-propagated (truncated BPTT), then the state is **detached** from the graph before the next deep-supervision step. Looping stops when the optimizer detects a fixed-point (residual below tolerance, or step-size too small). Models are 7M parameters; Sudoku uses 1M augmented training samples (1000 puzzles × 1000 augmentations); state-tracking trains on sequences up to length 32.

**Optimization / truncated BPTT (Prop. 1).** Contractive fixed-point models can be trained with truncated BPTT. With $J=\partial f_\theta/\partial\mathbf{z}(\mathbf{z}^\star;\mathbf{x})$ and $P=\partial f_\theta/\partial\theta(\mathbf{z}^\star;\mathbf{x})$, the implicit function theorem gives $\frac{d\mathbf{z}^\star}{d\theta} = (\mathbf{I}-J)^{-1}P$. Under contractivity the Neumann series $(\mathbf{I}-J)^{-1}=\sum_{j\ge 0} J^j$ converges, so truncating at depth $k$ gives $\frac{d\mathbf{z}^\star}{d\theta}\approx\sum_{j=0}^{k-1} J^j P$ (related to Jacobian-free backprop). Proposition 1: if $\|J\|_2=\sigma<1$, the truncation error is $\|(\mathbf{I}-J)^{-1}-\sum_{j=0}^{k-1}J^j\|_F \le \sqrt{D}\,\frac{\sigma^k}{1-\sigma}$ — exponential decay in $k$. This **decouples loop count from memory** (fixed memory footprint regardless of iterations).

### Inference / sampling
At test time, iterate $f_\theta$ with damping until $r_i<\tau$ (residual threshold, e.g. 0.1) or the step-size collapses. Compute spent scales with input difficulty. $\gamma$ and $P$ select a point on the compute–accuracy Pareto front.

## Experimental setup

- **Datasets:** Sudoku-Extreme (9×9, ~1M train / 422,786 test, exact-sequence accuracy), Maze-Hard (30×30 shortest path, 1000 train/1000 test, no augmentation), ARC-AGI-1 and ARC-AGI-2 (few-shot 2D grid transformations, pass@2), state-tracking $A_5$ (alternating group) and $S_5$ (symmetric group) on 5 elements (train ≤32 updates, OOD eval up to 128).
- **Baselines:** TRM (7M), HRM (27M), URM (14M), Attractor Model (7M and 27M), EqR (7M) — all hierarchical or non-looped except where noted; FPRM is the only single-loop / no-hierarchy model.
- **Metrics:** exact sequence accuracy (pass@1 for Sudoku/Maze, pass@2 for ARC), final-state accuracy for state-tracking; effective layers (loop count) as the compute measure.

## Key results

- **Table 1 (7M-param class):** FPRM is best on Sudoku-Extreme **94.2%** (beats even 27M HRM at 55% and 7M EqR at 93.0%, TRM 74.7%), Maze-Hard **87.0%** (beats 7M TRM 85.3%; loses to 27M Attractor Model 93.1%), ARC-1 pass@2 **47.5%** (beats TRM 44.6%). On ARC-2 it gets **6.2%**, on par with reproduced TRM checkpoints but below TRM's reported 7.8% and URM. Notably ARC benchmarks appear far more sensitive to parameter count, so that comparison is flagged as possibly unfair.
- For scale reference, DeepSeek-R1 (671B, CoT) gets 15.8% on ARC-1 / 1.3% on ARC-2; Claude 3.7 Sonnet 28.6% / 0.7% — i.e. these tiny looped models are competitive with giant LLM reasoners on ARC.
- **Adaptivity (state-tracking, length 128):** plain TRM fails to extrapolate (45.8% $A_5$, 39.4% $S_5$). Adding a causal 1D conv helps TRM greatly (91.4% / 97.2%) but ACT remains unreliable (drops to 65.3% on $A_5$; only some seeds learn to adapt). FPRM scales compute smoothly and reaches **98.1% $A_5$ / 98.8% $S_5$** at length 128.
- **Adaptivity (Sudoku):** both FPRM and TRM+ACT adapt loop count to difficulty (empty-cell count), but FPRM is more accurate *and* more compute-efficient; default TRM (ACT off) always exhausts the max budget regardless of difficulty.
- **Depth utilization:** across matched effective-layer budgets (Figure 6), FPRM beats TRM and the gap *widens* at higher compute — evidence FPRM uses its depth better. Pre-norm+scaling model's Sudoku accuracy saturates at ~2× the compute of the post-norm variant before plateauing.

## Ablations

- **Residual-scaling init (Table 2, Sudoku):** best is **high $\alpha_1$, low $\alpha_2$** — e.g. $(\alpha_1{=}0.75,\alpha_2{=}0.25)\to 94.23\%$ vs $(0.25,0.25)\to 83.44\%$. High $\alpha_1$ (dominant residual stream) matches classic signal-propagation fixes; low $\alpha_2$ matches the contractivity requirement of Theorem 2. Increasing $\alpha_1$ helps slightly more than decreasing $\alpha_2$ (hypothesized: $\alpha_1$ gradients are noisier, coming from both attention and FFN sub-layers, so a good $\alpha_1$ init matters more).
- **Post-supported hierarchy hypothesis:** replacing TRM's H/L hierarchy compute with more deep-supervision steps *improves* TRM — consistent with the claim that hierarchy's real benefit is masking the post-norm signal-propagation problem, not the biological "hierarchy" story.
- **Decay rate $\gamma$ / patience $P$:** larger $\gamma$ → more compute → higher accuracy (direct trade-off); $P$ has minimal effect that vanishes as $\gamma\to1$. Applying FPRM's own architectural mods to the *original* TRM layer hurt TRM (attributed to un-retuned hyperparameters / loop design).

## Limitations

- **Author-acknowledged:** tested only on **algorithmic/puzzle tasks, not natural language** — whether latent compositional reasoning transfers to other domains is open. Experiments use only Transformers (though the framework could use CNN/MLP/SSM backbones).
- **Author-flagged suboptimality:** looped models don't guarantee optimal learned algorithms; e.g. solving $A_5$ via these iterations is super-logarithmic where an optimal algorithm is logarithmic — FPRM inherits this suboptimal scaling.
- **[analyst's view]** The "enable ACT at inference" comparison for TRM is a reconstruction by the authors (halting when halting-prob > 0.5), not TRM's native behavior, so head-to-head adaptivity numbers depend on that choice. Also, halted samples stay in the batch until the last sample halts — the efficient batched implementation is left to future work, so the compute-savings are somewhat theoretical in current code. Maze-Hard and ARC results were partly non-reproducible for the Attractor Model / TRM baselines, adding noise to the comparison.

## Why it matters [analyst's view]

This is a clean "looped transformers are just deep transformers" reframing that yields two concrete, transferable wins: (1) pre-norm + learnable residual scaling as a *drop-in* recipe for training deep loops, and (2) fixed-point convergence as a *free* halting signal that dissolves the fragile ACT machinery. The most provocative claim is that **the hierarchy in HRM/TRM was a crutch for post-norm's signal-propagation failure** — if true, a whole line of "biologically motivated hierarchical reasoning" simplifies to "fix your normalization." That connects latent-reasoning research directly to the deep-net signal-propagation / rank-collapse literature (Noci, Sun, Dong), and to Deep Equilibrium Models (this is essentially a DEQ trained with fixed-point iteration + input injection rather than Broyden/Anderson solves). For test-time-compute scaling, it's a compelling alternative to CoT: adaptivity without verbalization, halting without a learned head.

## Open questions / things to verify

- Does the pre-norm+scaling recipe hold up on **natural-language** reasoning, or is it specific to fixed-length symbolic grids?
- The proposed open challenge: a latent-reasoning architecture that solves state-tracking in **logarithmic** iterations while remaining Turing-complete.
- How much of FPRM's Sudoku win is the normalization change vs. the fixed-point halting vs. dropping hierarchy? Table 3's negative transfer to TRM suggests the pieces are entangled with hyperparameters.
- Verify the batched-halting efficiency claim once an efficient implementation exists (currently the slowest sample gates the batch).
- The Jacobian eigenvalue condition $\Re(\lambda_i)<1$ for Theorem 3 — how often is it actually satisfied in practice, and what happens to inputs where it fails?

## Connections

- Builds on: Deep Equilibrium Models (Bai et al. 2019 — input injection, implicit-function-theorem gradient), Universal Transformers (Dehghani et al. 2019), truncated-BPTT for equilibrium models (Geng et al. 2021), Jacobian-free backprop (Fung et al. 2022).
- Contrasts with / improves on: HRM (Wang et al. 2025) and TRM (Jolicoeur-Martineau 2025) — beats them without hierarchy; concurrent Attractor Models (Fein-Ashley & Rashidinejad 2026) and Equilibrium Reasoners / EqR (Huang et al. 2026), described as orthogonal (they use Anderson acceleration / breadth-search initial guesses).
- Topic MOCs: [[topics/looped-transformers]], [[topics/reasoning]], [[topics/recurrence]], [[topics/latent-reasoning]]
- Author indices: [[authors/sajad-movahedi]], [[authors/antonio-orvieto]], [[authors/thomas-hofmann]], [[authors/t-konstantin-rusch]]

## Selected quotes

> "Looped Transformers are, in part, very deep transformers." — §1

> "let the model loop until its hidden state converges to a fixed-point, and use the convergence itself as the halting signal. Unlike ACT, the fixed-point halting mechanism requires no external module and lets the model spend as much compute as a given input demands." — §1

> "we hypothesize that there might be a simpler explanation for the success of hierarchical models: the hierarchy improves signal propagation." — §5
