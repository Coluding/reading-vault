---
type: paper
title: "You Need Better Attention Priors"
authors: ["Elon Litman", "Gabe Guo"]
year: 2026
venue: arXiv
url: https://arxiv.org/abs/2601.15380
rw_id: 01ktsphqxxppep2p35hh0revkm
topics: [attention, transformers, positional-encoding, optimal-transport, attention-sinks]
priority: medium
read_state: queued
relevance: ""
added: 2026-06-10
last_updated: 2026-06-17
---

# You Need Better Attention Priors

## TL;DR

The paper recasts scaled dot-product attention as an **Entropic Optimal Transport (EOT)** problem and shows that the standard softmax corresponds to a transport plan regularized toward an **implicit uniform prior** over key positions. Replacing the Shannon-entropy regularizer with a KL divergence against an *arbitrary* prior $\pi$ yields a closed form: the optimal attention is a softmax over content scores shifted additively by the **log-prior**, $p^* = \text{softmax}(s/\tau + \log\pi)$. The authors introduce **GOAT** (Generalized Optimal transport Attention with Trainable priors), which parameterizes this log-prior as a learnable, continuous function of position — a truncated Fourier (spectral) series for the shift-invariant relative component plus a dedicated key-only bias $u(j)$ for attention sinks — and packs it into a single unmodified SDPA call so it stays FlashAttention-compatible with no extra L×L bias matrix. GOAT lowers in-distribution C4 perplexity by **1.55 points over ALiBi** at 125M params, extrapolates to **16× training length** where RoPE collapses, keeps near-perfect Passkey/NIAH retrieval far beyond the training window, and (on DNA modeling) cuts peak CUDA memory **36%** (2.86 GB → 1.83 GB) vs RoPE. Its EOT analysis also gives a first-principles explanation of attention sinks as the optimal default in low-signal regimes and a way to model them *explicitly* instead of entangling them with content.

## Context & motivation

Self-attention is the core primitive of the Transformer (Vaswani et al., 2017), but the paper argues it remains "largely motivated by heuristics": the dot product is an intuitive similarity, the softmax a smooth argmax surrogate. Prior work refines attention along three axes — efficiency (linear attention, sparse attention, FlashAttention), expressivity (relative position encodings and attention biases: Shaw et al. 2018, Transformer-XL, RoPE/Su et al. 2021, ALiBi/Press et al. 2022), and behavioral stability (attention sinks: Xiao et al. 2024). The authors' complaint is that positional encodings (PEs) are bolted on as separate heuristics without a unified derivation, which leads to **structural entanglement** (positional bias coupled to semantic norms) and instability (length-generalization failure, emergent sinks).

The paper builds directly on Litman (2025), "Scaled-Dot-Product Attention as One-Sided Entropic Optimal Transport" (arXiv:2508.08369), and on the EOT/Sinkhorn line (Cuturi 2013; Sinkhorn & Knopp 1967). Its thesis: if attention *is* a KL-regularized transport problem, then the regularizer's reference distribution (the prior) is the natural, principled home for inductive biases — and standard PEs are just crude approximations of a "platonic" EOT-derived prior $\log\pi$. The contribution is a generalized mechanism that is simultaneously efficient, expressive, and offers direct control over emergent behaviors like sinks.

## Method

### Problem formulation

Treat a single query $i$ as a unit mass (a Dirac delta $\delta_i$) that must be transported across the $L$ keys $\{j\}_{j=1}^L$. The cost of moving mass from query $i$ to key $j$ is the negative affinity $c_{ij} = -s_{ij}$, where $s_{ij}$ is the (unscaled) dot-product score. Attention seeks a transport plan $p \in \Delta^{L-1}$ (the probability simplex) minimizing expected cost while staying maximally entropic.

**Definition 2.1 (EOT objective).** The attention weights are the unique minimizer
$$p^* = \arg\min_{p\in\Delta^{L-1}}\Big\{ \underbrace{\langle p, -s\rangle}_{\text{transport cost}} - \underbrace{\tau\,H(p)}_{\text{regularization}} \Big\},$$
where $H(p) \triangleq -\sum_j p_j \log p_j$ is the Shannon entropy and $\tau>0$ is a temperature. **Proposition 2.2** states this recovers standard softmax attention (full derivation in their Appendix A). So standard attention is the **maximum-entropy** distribution subject to matching the expected score (a Jaynes-style max-ent argument).

### Core idea

The entropy regularizer is, up to a constant, the *negative* KL divergence to the uniform distribution $\mathcal{U}$:
$$-H(p) = \mathrm{KL}(p\,\|\,\mathcal{U}) - \log L. \tag{3}$$
So standard attention silently assumes a **flat prior** and penalizes any non-uniformity not justified by content. The generalization: swap $\mathcal{U}$ for an arbitrary prior $\pi \in \Delta^{L-1}$ that encodes structural expectations (locality, periodicity, sinks) directly into the transport objective.

### Architecture / algorithm

**Closed-form solution (Proposition 3.1).** With KL against $\pi$ as the regularizer,
$$p^* = \arg\min_{p\in\Delta^{L-1}}\big\{ -\langle p, s\rangle + \tau\,\mathrm{KL}(p\,\|\,\pi) \big\},$$
the unique optimum is
$$p_j^* = \text{softmax}\!\left(\frac{s_j}{\tau} + \log\pi_j\right). \tag{5}$$
This identifies the "missing term": **standard PEs are heuristic approximations of the EOT prior $\log\pi$.** The log-prior additively contours the transport cost rather than altering content representations.

Identifying $\tau$ with the usual $\sqrt{d_c}$ scaling and writing the unnormalized log-prior as $\mathcal{K}_{ij}$ gives the operational GOAT score:
$$p_{ij} = \text{softmax}_j\!\left( \frac{\langle q_{c,i}, k_{c,j}\rangle}{\sqrt{d_c}} + \mathcal{K}_{ij} \right), \tag{11}$$
where $q_{c,i}, k_{c,j}$ are *content* query/key vectors of dimension $d_c$ and $\mathcal{K}_{ij}$ is the unnormalized log-prior (the softmax normalizes $\exp(\mathcal{K})$ into a proper prior). GOAT's whole job is to parameterize $\mathcal{K}_{ij}$ so it (a) expresses shift-invariant relative relations including direction, (b) supports global defaults (sinks), and (c) is computable inside standard kernels without materializing an $L\times L$ bias.

**Relative component — spectral / Fourier parameterization.** The shift-invariant part is a truncated Fourier series in the offset $(i-j)$:
$$\mathcal{K}_{ij}^{\text{rel}} = \sum_{r=1}^R \big[ \alpha_r \cos(\omega_r(i-j)) + \beta_r \sin(\omega_r(i-j)) \big], \tag{12}$$
where $\{\omega_r\}_{r=1}^R$ are *fixed geometric* frequencies, and $\alpha_r$ (symmetric) and $\beta_r$ (antisymmetric/directional) are *learnable* spectral weights. Because the formulation works on unnormalized logits, $\alpha_r, \beta_r$ may be **negative**, letting the model learn not only local attraction but explicit **repulsion/suppression** at chosen frequencies (motivated by spectral representations of shift-invariant kernels à la Rahimi & Recht 2007 / Bochner, in their Appendix C).

To make this an inner product (so SDPA can compute it), expand with angle-difference identities $\cos(\omega_r(i-j)) = \cos\omega_r i \cos\omega_r j + \sin\omega_r i \sin\omega_r j$ and similarly for sine. This factorizes per frequency into a 2-D positional key (a plain Fourier feature of position $j$)
$$k_{\text{rel},j}^{(r)} = [\cos(\omega_r j),\ \sin(\omega_r j)]^\top, \tag{17}$$
and a 2-D query that is a **spectral rotation** parameterized by $\alpha_r, \beta_r$:
$$q_{\text{rel},i}^{(r)} = [\,\alpha_r\cos(\omega_r i) + \beta_r\sin(\omega_r i),\ \ \alpha_r\sin(\omega_r i) - \beta_r\cos(\omega_r i)\,]^\top. \tag{18}$$
Their dot product reconstructs the $r$-th spectral term; concatenating across all $R$ frequencies gives $q_{\text{rel},i}, k_{\text{rel},j}$ in a positional subspace of size $2R$.

**Sink component — explicit key-only bias.** Attention sinks (mass dumped on a token, usually the first, regardless of query) are normally induced by learning a high-norm *content* key, which entangles a structural default with semantics. GOAT instead adds a dedicated **key-only** logit bias $u(j)$, implemented with one extra SDPA dimension:
$$\langle q_{\text{sink},i}, k_{\text{sink},j}\rangle = \langle 1, u(j)\rangle = u(j)\quad \forall i. \tag{19}$$
$u(j)$ is a learnable linear decay plus an MLP over sinusoidal and length-normalized scalar inputs (for length extrapolation). It is a query-independent broadcast bias that dominates in low-signal regimes without touching content.

**Unified parameterization and the scaling trick.** The full log-prior is $\mathcal{K}_{ij} = \mathcal{K}_{ij}^{\text{rel}} + u(j)$. GOAT reserves a prior subspace of $d_p = 2R+2$ dimensions (padded) and uses the remaining $d_c = d_h - d_p$ for content, with **block-diagonal** Q/K projections so semantic and structural subspaces stay strictly independent. Because real kernels compute $\text{softmax}(q^\top k/\sqrt{d_h})$, the inputs are *pre-scaled* so the prior enters unscaled at effective temperature 1:
$$q'_i = \big[\, q_{c,i}\sqrt{\tfrac{d_h}{d_c}},\ \ q_{\text{rel},i}\sqrt{d_h},\ \ \sqrt{d_h}\,\big]^\top,\qquad k'_j = \big[\, k_{c,j},\ \ k_{\text{rel},j},\ \ u(j)\,\big]^\top. \tag{20–21}$$
Then $\langle q'_i, k'_j\rangle/\sqrt{d_h} = \langle q_{c,i},k_{c,j}\rangle/\sqrt{d_c} + \mathcal{K}_{ij} = s_{ij} + \log\pi_{ij}$ (Eq. 22–24). Content is scaled by $1/\sqrt{d_c}$ while the prior is unscaled — deliberate, so the prior isn't attenuated at high head dimension and structural biases stay stable. Value vectors $v_j$ stay purely content-based on the full head dimension, so spatial info affects only routing, not the mixed representation.

**Algorithm 1 (GOAT forward pass):** (1) generate prior components — `q_rel ← SpectralRotate(i, α, β)`, `k_rel ← FourierFeat(j)`, `u(j) ← ϕ(SinusoidalEnc(j))`; (2) compose $q', k'$ with the scaling trick above; (3) `return FlashAttention(q', k', v)`. It is a drop-in MHA replacement with no kernel modification and no claimed overhead.

**Disentanglement vs. PEs.** RoPE injects position *multiplicatively* via rotations, $z_{ij} = (R_i q)^\top (R_j k)$, so the positional bias magnitude is coupled to semantic norms $\|q\|\|k\|$ — to force a sink such a model must inflate content vectors, corrupting semantics. By *adding* the log-prior, GOAT keeps prior and content disentangled. ALiBi is additive but rigidly monotonic; GOAT replaces that with a learnable, EOT-derived prior that's both disentangled and expressive while keeping fixed-encoding extrapolation.

### Derivations / why it works

**Proof of Eq. 5.** Expand $\mathrm{KL}(p\|\pi) = \sum_j p_j\log p_j - \sum_j p_j\log\pi_j$ and substitute into the objective $J(p) = -\sum_j p_j s_j + \tau\sum_j p_j\log p_j - \tau\sum_j p_j\log\pi_j$. Grouping gives an *effective cost* $(-s_j - \tau\log\pi_j)$ plus a standard Shannon term, i.e. the KL-regularized problem is a Shannon-regularized problem with modified cost. The optimum is $p_j^* \propto \exp\big((s_j + \tau\log\pi_j)/\tau\big) = \pi_j \exp(s_j/\tau)$; normalizing recovers Eq. 5. (Eqs. 6–10.)

**Why sinks are inevitable (Theorem 5.1, Collapse to Prior).** Fix query $i$; let $\omega_i \triangleq \max_k s_{ik} - \min_k s_{ik}$ be the content-score dynamic range. Then
$$\pi_{ij}\exp(-\omega_i) \le p_{ij} \le \pi_{ij}\exp(\omega_i), \tag{25}$$
so as content signal vanishes ($\omega_i \to 0$) the posterior converges pointwise to the prior, $p_i \to \pi_i$. Every head must revert to its prior when semantic signal is lost — what distinguishes stable from unstable models is the *shape* of that prior.

**Sinks as margins (Def. 5.2).** $j^*$ is a sink with margin $m_i(j^*) \triangleq \min_{k\ne j^*}(z_{ij^*} - z_{ik}) > 0$, where $z_{ij} = s_{ij} + \mathcal{K}_{ij}$. The margin splits into a *content* part $(s_{ij}-s_{ik})$ and a *prior* part $(\mathcal{K}_{ij}-\mathcal{K}_{ik})$ (Eq. 27). Standard attention has uniform prior ($\mathcal{K}_{ij} = -\log L$), so it must build sinks through content (high-norm generic key) — entangling structure with semantics. GOAT builds them through the prior via a large $u(j^*)$, and the unscaled parameterization gives the default channel a gradient roughly $\sqrt{d_c}$× larger than content.

**Stability bound (Theorem 5.4, Context Sensitivity).** Define total context sensitivity $\Psi(\mathcal{C}) \triangleq \sum_{k\in\mathcal{C}} p_{ik} = 1 - p_{ij^*}$; a perturbation $\|\Delta v_k\|_2 \le \varepsilon$ to context values bounds the output change by $\|\Delta o_i\|_2 \le \varepsilon\,\Psi(\mathcal{C})$. In the low-signal limit: with a **uniform** prior, $\lim_{L\to\infty}\lim_{\omega_i\to0}\Psi_{\text{uni}}(\mathcal{C}) = 1$ (asymptotically unstable — context noise dominates). With a **peaked** prior of sink margin $\delta = \min_{k\in\mathcal{C}}(\mathcal{K}_{ij^*}-\mathcal{K}_{ik}) > 0$,
$$\lim_{\omega_i\to0}\Psi_{\text{sink}}(\mathcal{C}) \le \frac{L-1}{\exp(\delta)+L-1}. \tag{31}$$
So sensitivity is suppressed exponentially in $\delta$, and maintaining stability needs only **logarithmic margin growth** $\delta \sim \ln L$ — trivially achievable through the unconstrained bias $u(j)$.

**Admissible priors.** They note (their Appendix F) that practical constraints — SDPA-compatibility, translation equivariance, stability — uniquely restrict admissible priors to a **finite trigonometric family**, which motivates the Fourier parameterization.

### Training procedure

- Language modeling: 125M-parameter models trained on the **C4** dataset (Raffel et al., 2020), **4B tokens**, training context length $L_{\text{train}} = 2048$.
- Long-context retrieval: GPT-style LMs trained up to $L_{\text{train}} = 1024$ for Passkey / NIAH.
- DNA: 125M decoder-only LM on the Human Reference Genome, next-token modeling under identical training budgets vs RoPE.
- Vision: ViT on ImageNet-1k (Deng et al., 2009), trained at 224×224.
- Exact optimizer / learning-rate / schedule details: _not addressed by the source text fetched (likely in appendices)._

### Inference / sampling

Not a generative-modeling method in the diffusion sense; inference is standard autoregressive / forward attention. GOAT changes only the routing (attention weights) via the additive prior. The DNA experiment does report generated 200-nt continuations, but there's no special sampler — _no dedicated sampling procedure described._

## Experimental setup

- **Datasets:** C4 (LM); synthetic Passkey Retrieval and Needle-in-a-Haystack (long-context, following Liu et al. 2023); Human Reference Genome (DNA); ImageNet-1k (vision); a synthetic "copy-mixture" toy task (attend to first token $j=0$, previous token $j=i-1$, or noise).
- **Baselines:** RoPE (Su et al. 2021), position-interpolated RoPE (Chen et al. 2023), ALiBi (Press et al. 2022), sinusoidal absolute PEs, ViT with absolute positional embeddings (Dosovitskiy et al. 2021).
- **Metrics:** perplexity (in-distribution and extrapolated length), span-token / Passkey / NIAH accuracy, validation NLL and bits/base (DNA), top-1 accuracy vs input resolution (ViT), training throughput and peak CUDA memory, GC% trajectory Pearson correlation (DNA generative quality).

## Key results

- **C4 LM:** GOAT improves in-distribution perplexity by **1.55 points over ALiBi** while extrapolating robustly to **16× $L_{\text{train}}$**, where RoPE "degrades catastrophically" past the training window and ALiBi extrapolates flat but underfits the training window (Fig. 2).
- **Learned bias is interpretable:** the model spontaneously discovers a sharp spike at $j=0$ (explicit sink) and a rise near $j\approx 2000$ (local recency) in $u(j)$ — validating that structural needs decouple from content (Fig. 2b).
- **Sink-mass dynamics match theory:** as signal $\omega$ increases, GOAT smoothly sheds sink mass from $\approx 1$ (prior-dominated) to $\approx 0$ (content-driven); ALiBi stays over-defaulted, RoPE only partially disengages (Fig. 2c, validating Theorem 5.1).
- **Long-context retrieval:** near-perfect Passkey and NIAH accuracy at lengths far beyond training, while RoPE / interpolated-RoPE / sinusoidal baselines degrade sharply with length and needle depth (Fig. 3).
- **DNA modeling:** lower validation NLL and bits/base than RoPE under identical budgets; comparable training throughput but **peak CUDA memory drops 36% (2.86 GB → 1.83 GB)**; generated continuations track ground-truth GC% better (**Pearson r = 0.466 vs 0.320**, averaged over N=3) (Fig. 4).
- **Vision:** trained at 224×224, GOAT spontaneously learns a 2D shift-invariant local prior and maintains substantially higher top-1 accuracy at higher evaluation resolutions zero-shot, while the absolute-PE ViT degrades (Fig. 5).

## Ablations

The **copy-mixture toy task** acts as the central ablation of the prior's structure (Fig. 1). GOAT recovers the ground-truth structure in a disentangled way: the **key-only sink component** $\mathcal{K}_{\text{sink}}(i,j)=u(j)$ captures the global preference for $j=0$, and the **relative component** $\mathcal{K}_{\text{rel}}(i,j)=\kappa(i-j)$ learns a periodic diagonal bias peaking at $j=i-1$. Crucially, **deep negative spectral troughs** (negative $\alpha_r, \beta_r$) actively *suppress* attention to intervening positions, isolating the target diagonal from background noise — direct evidence that allowing negative spectral weights (repulsion) matters. The decomposition figure separates sink (a), relative (b), row-centered total log-prior (c), and the induced causal prior after masking + row-normalization (d). The interpretability visualizations (Figs. 2b, 3b, 5b) function as qualitative ablations showing the prior implements explicit inductive bias without modifying content scores. _Quantitative leave-one-component-out ablations (e.g. relative-only vs sink-only vs full) are not described in the fetched main text._

## Limitations

The authors' own Impact Statement is generic and flags no specific limitation. Honest-reader flags **[analyst's view]**:

- The prior family is restricted to a **finite trigonometric / Fourier basis** with *fixed* geometric frequencies; only the spectral weights $\alpha_r, \beta_r$ are learned. Phenomena not well represented by a few sinusoids may be hard to capture, and $R$ (number of frequencies) is an unexamined hyperparameter in the fetched text.
- Stability guarantees (Theorems 5.1, 5.4) are derived in the **low-signal limit** ($\omega_i\to0$) and for sequence length "smaller than the prior's period" — behavior near or beyond the period is less characterized.
- Headline LM results are at a single small scale (**125M params, 4B tokens**); no evidence at modern frontier scale.
- "No computational overhead" is asserted from the kernel-compatibility construction; the only measured efficiency number is the DNA memory drop, with throughput merely "comparable."

## Why it matters [analyst's view]

This is a *unifying-theory* paper: it takes a grab-bag of position-encoding heuristics (sinusoidal, RoPE, ALiBi) and re-derives them as approximations of one object — the EOT prior $\log\pi$ — then says "so just learn the prior, properly, and add it." The appeal is that a single mechanism simultaneously addresses three problems usually treated separately: length extrapolation, sink stability, and expressivity, while staying inside FlashAttention. If the 125M results hold at scale, GOAT is a candidate drop-in for the position-encoding slot in production decoders, with the bonus of *interpretable, controllable* sinks (you can read $u(j)$ off the model). The EOT-as-attention framing also connects attention to the optimal-transport / Sinkhorn literature, which could seed transfer of OT theory (regularization, duality, stability bounds) into transformer design. The sink result is the most conceptually satisfying piece: it turns "attention sinks" from an empirical curiosity into the *predicted* optimum of a KL-regularized transport problem in low-signal regimes, with a clean stability bound (Eq. 31) explaining why models need them and why doing it via a disentangled bias beats doing it via inflated content keys.

## Open questions / things to verify

- Does the 1.55-ppl / 16× extrapolation advantage survive at 1B+ params and longer training? Single-scale evidence.
- How sensitive is performance to $R$ (number of frequencies) and the fixed geometric frequency schedule? Not reported in the main text.
- The "no overhead" claim deserves wall-clock numbers; the prior adds $d_p = 2R+2$ dimensions to Q/K and an MLP for $u(j)$ — small, but nonzero.
- Appendix F's claim that SDPA-compatibility + translation-equivariance + stability *uniquely* restricts priors to a finite trigonometric family is load-bearing for the design; worth checking the proof.
- How does GOAT interact with the actual FlashAttention sink/streaming-LLM machinery (Xiao et al. 2024) it aims to subsume?
- DNA "36% memory reduction" vs RoPE is surprising for a method that *adds* dimensions — verify the comparison is apples-to-apples (e.g. RoPE's rotation overhead vs GOAT's block-diagonal projections).

## Connections

- Builds on: Litman (2025), "Scaled-Dot-Product Attention as One-Sided Entropic Optimal Transport" (arXiv:2508.08369) — the EOT-attention framing this paper generalizes.
- Contrasts with: RoPE (Su et al. 2021, multiplicative/entangled), ALiBi (Press et al. 2022, additive but rigid monotonic), interpolated RoPE (Chen et al. 2023), sinusoidal/learned absolute PEs.
- Related machinery: Sinkhorn & Knopp (1967), Cuturi (2013) entropic OT; Rahimi & Recht (2007) random/Fourier features; FlashAttention (Dao 2023; Shah et al. 2024); attention sinks (Xiao et al. 2024; Gu et al. 2024).
- Topic MOCs: [[topics/attention]], [[topics/transformers]], [[topics/positional-encoding]], [[topics/optimal-transport]], [[topics/attention-sinks]]
- Author indices: [[authors/elon-litman]], [[authors/gabe-guo]]

## Selected quotes (optional)

> "We generalize the attention mechanism by viewing it through the lens of Entropic Optimal Transport, revealing that standard attention corresponds to a transport problem regularized by an implicit uniform prior." — Abstract

> "This result formally identifies the missing term in the attention mechanism: standard PEs are merely heuristic approximations of this EOT-derived prior $\log\pi$." — §3, after Eq. 5

> "Standard attention typically induces sinks by learning high-norm content keys, entangling a structural default with semantic representation. We instead introduce a dedicated key-only logit bias $u(j)$ in the prior." — §4
