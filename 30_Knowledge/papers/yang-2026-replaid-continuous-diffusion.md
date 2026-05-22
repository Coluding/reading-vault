---
type: paper
title: "Continuous Diffusion Scales Competitively with Discrete Diffusion for Language"
authors: ["Zhihan Yang", "Wei Guo", "Subham Sekhar Sahoo", "Yongxin Chen", "Morteza Mardani", "John Thickstun", "Shuibai Zhang", "Arash Vahdat"]
year: 2026
venue: arXiv
url: https://arxiv.org/abs/2605.18530
rw_id: 01ks7ds32gk7a714w9my9bw261
topics: [diffusion-language-models, continuous-diffusion, scaling-laws, generative-models]
priority: high
read_state: queued
relevance: ""
added: 2026-05-22
last_updated: 2026-05-22
---

# Continuous Diffusion Scales Competitively with Discrete Diffusion for Language (RePlaid)

**TL;DR** — Revisits the Plaid likelihood-based continuous diffusion language model and argues its prior "continuous diffusion doesn't scale" reputation was a *protocol* artifact, not an intrinsic limitation. By aligning Plaid's architecture with modern discrete-DLM practice (LayerNorm, AdaLN-Zero, bidirectional attention, GELU-tanh, RoPE) the authors get **RePlaid**, which closes the AR-vs-continuous-DLM compute gap from a frequently-cited 64× down to **20×** (with self-conditioning), reaching state-of-the-art 22.1 PPL on OpenWebText *among continuous DLMs*, and beating MDLM in the over-trained regime. Two theoretical results: (i) minimising ELBO variance via the noise-schedule yields a linear cross-entropy schedule "for free"; (ii) ELBO-based embedding optimisation creates structured latent geometry that drives most of the likelihood gain.

## Context & motivation (§1)

The standing story before this paper:

> "Previous studies show empirical evidence that continuous diffusion underperforms discrete diffusion and AR models when evaluated at equivalent training FLOPs and parameter counts; the original Plaid study reported a significant 64× compute overhead compared to autoregressive models to achieve matching likelihood. This figure is frequently cited as evidence of the poor scalability of continuous DLMs."

The authors' suspicion: "we aim to determine whether this performance gap is an inherent limitation of continuous diffusion or a consequence of incompatible scaling analyses."

Why care about continuous diffusion at all (§1)?
- Smooth latent geometry enables "structured editing" and "ODE-based solvers ... facilitating distillation of complex sampling trajectories into high-fidelity, single-step generators."
- Continuous diffusion handles low sampling budgets gracefully where discrete diffusion suffers "severe quality degradation."

## Background — Plaid (§2)

Plaid is a Variational Diffusion Model (VDM) for text. Vocabulary $[V]$, length-$L$ sequence $x \in \{0,1\}^{L \times V}$ as one-hot. Embedding $e = xE \in \mathbb{R}^{L \times d_e}$ with $E$ a learnable embedding matrix (unit-norm rows). Low embedding dim by default: $d_e = 16$.

### Forward / reverse process
Forward process is Gaussian noising on the embedded sequence:
$$q(z_t \mid x) = \mathcal{N}(\alpha_t e, \sigma_t^2 I), \quad t \in [0, 1]$$

Variance-preserving: $\alpha_t^2 + \sigma_t^2 = 1$, SNR $:= \alpha_t^2 / \sigma_t^2$ strictly decreasing.

True reverse posterior:
$$q(z_s \mid z_t, x) = \mathcal{N}\big((\alpha_{t|s}\sigma_s^2/\sigma_t^2)z_t + (\alpha_s \sigma_{t|s}^2/\sigma_t^2) e, (\sigma_{t|s}^2 \sigma_s^2/\sigma_t^2) I\big)$$

### Training objective — NELBO

$$-\log p_\theta(x) \leq \mathrm{KL}(q(z_1 \mid x) \| p(z_1)) + \mathbb{E}_{z_0}\left[\sum_{l=1}^L -\log \langle x_\theta^l(z_0, 0), x^l \rangle\right] - \frac{1}{2} \mathbb{E}_{t, z_t}\left[\mathrm{SNR}'(t) \| e_\theta(z_t, t) - e\|^2\right]$$

Three terms: prior loss (Gaussian KL at $t=1$), reconstruction loss (negative log-likelihood at $t=0$), diffusion loss (SNR-weighted denoising MSE on embeddings).

### Learnable noise schedule
$\gamma(t) = \gamma_0 + (\gamma_1 - \gamma_0)\tilde\gamma(t)$ where $\tilde\gamma: [0,1] \to [0,1]$ is a monotone neural net; $\alpha_t^2 = \sigma(-\gamma(t))$, $\sigma_t^2 = \sigma(\gamma(t))$. Endpoints minimise diffusion loss, interior minimises diffusion-loss Monte-Carlo *variance*.

### Plaid's specific additions over standard VDMs
- Categorical reparameterization $e_\theta = x_\theta E$ (predict tokens, embed predicted tokens) instead of predicting in embedding space directly.
- Self-conditioning [Chen 2023]: for 25% of each batch, run a forward pass with no-grad, feed the prediction back as conditioning input for the actual prediction.
- Cheaper embedding ops: $L \times d_e$ vs $L \times V$ matmuls (~50× FLOP savings at $V \approx 50K$, $d_e = 16$).

## Method — RePlaid (§3.1)

Architectural alignment with MDLM/Duo:
- LayerNorm (replaces Plaid's RMSNorm).
- MLP biases.
- GELU(tanh) activations.
- AdaLN-Zero modulation with learnable gating for residual connections.
- Removes FP32 final-logit head (was a Plaid numerical confounder).

Otherwise Plaid's algorithmic core is unchanged — same NELBO loss, same categorical reparameterization, same low-dim embedding $d_e = 16$, same self-conditioning trick.

## Experimental setup (§3.2)

- **Data**: SlimPajama (deduplicated). Llama-2 tokenizer ($V = 32{,}001$ with a special mask token). Batch 256, sequence length 2048.
- **Optimizer**: AdamW, cosine LR $2e{-4} \to 2e{-5}$, $\beta_1 = 0.9$, $\beta_2 = 0.95$, 1% warmup. Weight decay 0.1, no dropout.
- **Compute reference**: 231M MDLM at 1e20 FLOPs ≈ 32 hours on 8× NVIDIA GB200.
- **Baselines under same protocol**: AR (Diffusion Transformer with causal attention), MDLM, Duo. Same protocol as Sahoo et al. — same dataset, optimization, compute budgets.

## Key results

### Headline scaling-law numbers (§3, Fig. 1)
- AR vs **MDLM** compute gap: **14×**.
- AR vs **RePlaid (with self-conditioning)** compute gap: **20×**.
- AR vs RePlaid (without self-conditioning): 27×.
- AR vs the *original* Plaid figure: 64×.

> "By adopting the exact protocol from Sahoo et al. ... we perform the first unified scaling comparison between continuous and discrete DLMs. Crucially, in this controlled environment, the performance gap narrows to 20× for RePlaid versus AR, placing continuous diffusion on par with discrete diffusion and challenging the narrative of its inherent unscalability."

The "20× vs 14×" gap to MDLM is real but small relative to the 64× starting point.

### Over-trained regime (Fig. 1c)
> "In the over-trained regime below the green line, RePlaid (s.c.) consistently outperforms MDLM."

I.e., past the compute-optimal frontier, continuous *wins* against discrete diffusion.

### OpenWebText benchmark (§1, abstract)
- **RePlaid PPL: 22.1** — state-of-the-art among continuous DLMs.
- Outperforms Duo with fewer parameters.
- Outperforms MDLM in over-trained regime.

### Comparisons with other continuous DLMs
> "RePlaid achieves consistent improvements in generation quality over other recent continuous models like FLM and LangFlow."

(Specific numeric comparisons against FLM / LangFlow not in fetched body excerpt.)

## Theoretical insights (§1, partial)

Two claims framed as "why likelihood-based training works":

1. **Linear cross-entropy schedule emerges from ELBO variance minimization.**
   > "Optimizing the noise schedule via ELBO variance naturally recovers a near-linear cross-entropy schedule that evenly distributes denoising difficulty without heuristic reparameterization."
   The implication: the heuristic time-reparameterizations used in most continuous-diffusion work (cosine, sigmoid, etc.) are *approximating* what ELBO-variance minimization solves directly.

2. **ELBO-based embedding optimization creates structured geometry.**
   > "ELBO-based embedding optimization inherently regularizes the latent geometry to improve likelihood, creating a structured, low-entropy space that prevents potential token dispersion in cross-entropy-based training."

   Cross-entropy training (used by some continuous-DLM competitors) doesn't put this geometric pressure on the embedding matrix.

The full proofs / derivations are in the body — not fully extracted in this triage.

## Ablations

_Partial — fetched body covers method and scaling-law results but didn't reach the explicit ablation tables in this excerpt._ Key ablation surfaces named in §3:
- Self-conditioning: removed → 27× gap instead of 20×.
- The architectural-alignment changes (LayerNorm vs RMSNorm, AdaLN-Zero, GELU-tanh) — each presumably ablated in App. J.

## Limitations

_Not addressed in the fetched body excerpt at the limitations level._

[analyst's view]
- 20× compute overhead vs AR is still a lot. RePlaid closes the gap to discrete diffusion (14×) but doesn't close it to AR.
- All evaluation is on perplexity / scaling-law metrics. Downstream generation quality / instruction-following / capabilities not benchmarked against AR or MDLM.
- The "over-trained regime" win is the most useful empirical result but its scope is narrow — only matters if you intend to train past compute-optimal.

## Why it matters [analyst's view]

Three things:

1. **Falsifies "continuous diffusion doesn't scale."** The 64× compute-gap number has been the standard reason to ignore continuous DLMs. Bringing it to 20× changes the prior. *Now* the question becomes "which side of the discrete/continuous trade-off do you want?" rather than "which is intrinsically better at scale?"

2. **The argument structure is exemplary.** This is a careful *protocol-corrected* re-evaluation, in the same spirit as Chinchilla revising Kaplan, or [[papers/huang-2026-semantic-tube-prediction]] challenging Chinchilla's data term. The lesson is methodological — when a "scaling law" is established with one protocol, you should expect the headline number to move when you align the protocol.

3. **Continuous diffusion's downstream advantages get back on the menu.** ODE solvers, single-step distillation, smooth-geometry editing — all the things continuous diffusion can do that discrete diffusion can't — were being mortgaged against the scaling claim. With the gap closed, applications-driven choices (controllability, edit-ability, low-sample-budget generation) become reasonable again.

The "structured embedding geometry" theoretical claim is the most interesting follow-up direction. It connects to the same broader picture as [[papers/huang-2026-semantic-tube-prediction]]'s Geodesic Hypothesis and [[papers/maes-2026-leworldmodel]]'s SIGReg-driven latent structure — three different ways of saying "the geometry of the latent matters more than the architecture details."

## Open questions

[analyst's view]
- **How does the linear cross-entropy schedule claim compare quantitatively against the most common heuristic schedules (cosine, sigmoid)?** The "natural recovery" framing suggests a derivation worth chasing.
- **Embedding structure analysis.** The "structured, low-entropy space" claim begs a follow-up interpretability paper. What does the learned $E$ matrix look like? Anisotropic? Manifold-structured?
- **Does the 20× → "rapidly improving" trend continue with newer architectures (Mamba, RWKV, hybrid attention)?** RePlaid sticks with DiT.
- **Inference-time advantages**. The story for *why you'd want* continuous diffusion is mostly inference-side. The paper benchmarks training-side scaling. A side-by-side inference benchmark vs AR/MDLM would complete the picture.
- **Connection to other ELBO-based latent generative models.** GRAM ([[papers/baek-2026-gram]]) is also ELBO-trained on Gaussian latents but for reasoning. The shared "ELBO + structured Gaussian latent" recipe is becoming a pattern.

## Connections

- **Closest prior work**: original Plaid (the model RePlaid revises), MDLM, Duo (the discrete-DLM baselines), FLM, LangFlow (other continuous-DLM competitors).
- **Methodological cousins**:
  - [[papers/baek-2026-gram]] — ELBO + Gaussian latent training; reasoning rather than language modelling.
  - [[papers/huang-2026-semantic-tube-prediction]] — also targets LLM data-efficiency via geometric priors on the latent.
  - [[papers/maes-2026-leworldmodel]] — also argues structured latent geometry is the right inductive bias (via SIGReg + Cramér–Wold).
- **Topic MOCs**: [[topics/scaling-laws]], [[topics/generative-models]]
- **Author indices**: [[authors/zhihan-yang]]

## Selected quotes

> "While diffusion has drawn considerable recent attention from the language modeling community, continuous diffusion has appeared less scalable than discrete approaches. To challenge this belief we revisit Plaid, a likelihood-based continuous diffusion language model (DLM), and construct RePlaid by aligning the architecture of Plaid with modern discrete DLMs." — abstract

> "RePlaid exhibits a compute gap of only 20× compared to autoregressive models, outperforms Duo while using fewer parameters, and outperforms MDLM in the over-trained regime." — abstract

> "We show that optimizing the noise schedule to minimize the ELBO's variance naturally yields linear cross-entropy (information loss) over time. This evenly distributes denoising difficulty without any case-specific time reparameterization." — abstract
