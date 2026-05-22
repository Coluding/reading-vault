---
type: paper
title: "Generative Modeling with Flux Matching"
authors: ["Peter Pao-Huang", "Xiaojie Qiu", "Stefano Ermon"]
year: 2026
venue: arXiv
url: https://arxiv.org/abs/2605.07319
rw_id: 01ks7dtbdm9rgh311b3qb2g7yd
topics: [generative-models, score-matching, diffusion-models, vector-fields, fokker-planck]
priority: high
read_state: queued
relevance: ""
added: 2026-05-22
last_updated: 2026-05-22
---

# Generative Modeling with Flux Matching

**TL;DR** — Generalizes score matching from "match the unique vector field $\nabla \log p_{\mathrm{data}}$" to "match *any* vector field whose stationary distribution is $p_{\mathrm{data}}$." Built on the Fokker–Planck equation: many vector fields produce the same stationary distribution, the score is just one. Flux Matching replaces the pointwise score-equality objective with a weaker condition that only the *divergence of the probability flux* matches — leaving an infinite nullspace of valid generative fields. Trained via a **projected Fisher divergence** that stays in the same $L^2(p_{\mathrm{data}})$ geometry as standard score matching but is invariant to flux-preserving perturbations. The flexibility unlocks faster-mixing samplers, interpretable fields (RNA velocity), and structured generative dynamics — at a 2–4× training cost vs DSM.

## Context & motivation

Modern generative modelling fixes its target on a single point in vector-field space:

> "Modern generative modeling ... canonically targets one particular vector field called the (Stein) score function, typically fit via score matching, whose population loss is the Fisher divergence." (§1)

The paper's core observation:

> "The narrow focus on the score overlooks a large space of alternative vector fields whose diffusion processes share the same target distribution. We refer to these vector fields as generative vector fields, or generative fields for short, with the Fokker–Planck equation (FPE) characterizing the full family." (§1)

The pitch: when the *dynamics* of the generative process matter (faster mixing, interpretability, structured dependencies), the score is a strict subset of what you can ask the model to learn.

## Method

### Setup (§2)
Data distribution $p_{\mathrm{data}}$ on $\mathbb{R}^d$, observed via samples. Standard score-matching learns $f_\theta \approx \nabla \log p_{\mathrm{data}}$ by minimising the Fisher divergence
$$\mathcal{J}(\theta) = \mathbb{E}_{x \sim p_{\mathrm{data}}}\left[\|f_\theta(x) - \nabla \log p_{\mathrm{data}}(x)\|^2\right]$$
and at inference time samples via Langevin dynamics or other gradient-based MCMC. Once learned, $f_\theta$ enters the SDE $dx_t = f_\theta(x_t) dt + \sqrt{2} dW_t$.

### The Fokker–Planck characterization (§2.2)

Density $p_t$ under $dx_t = f_\theta(x_t) dt + \sqrt{2} dW_t$ evolves as
$$\frac{\partial p_t(x)}{\partial t} = -\nabla \cdot \big(p_t(x) f_\theta(x)\big) + \nabla \cdot \big(p_t(x) \nabla \log p_t(x)\big)$$

At stationarity ($\partial_t p_t = 0$, $p_t = p_{\mathrm{data}}$), this gives **Proposition 2.1**:

$$\nabla \cdot \big(p_{\mathrm{data}}(x) f_\theta(x)\big) - \nabla \cdot \big(p_{\mathrm{data}}(x) \nabla \log p_{\mathrm{data}}(x)\big) = 0 \quad \forall x$$

This condition is *necessary and sufficient* for $f_\theta$ to generate $p_{\mathrm{data}}$ — and crucially, the score $\nabla \log p_{\mathrm{data}}$ satisfies it trivially but is *not* the only solution. Any field of the form
$$f_\theta(x) = \nabla \log p_{\mathrm{data}}(x) + v(x), \quad \text{with } \nabla \cdot \big(p_{\mathrm{data}}(x) v(x)\big) = 0 \quad (3)$$
has the same stationary distribution. The score is one point in a high-dimensional family; the family parameterised by $v$ is the "extra freedom."

### Flux Matching objective (§3)

Replace pointwise score equality with flux-divergence equality:

> "It is only necessary to match the flux divergence $\nabla \cdot (p_{\mathrm{data}} f_\theta)$ rather than to match the vector field pointwise. Consequently, Flux Matching allows learning the family of generative vector fields that need not be the score." (§3)

To compare flux divergences in the same $L^2(p_{\mathrm{data}})$ geometry as Fisher, define $\Pi_{\mathrm{flux}} f$ as the unique gradient field with $\nabla \cdot (p_{\mathrm{data}} \Pi_{\mathrm{flux}} f_\theta) = \nabla \cdot (p_{\mathrm{data}} f_\theta)$. The **projected Fisher divergence** is:

$$\widetilde{\mathcal{J}}(\theta) := \mathbb{E}_{x \sim p_{\mathrm{data}}}\big[\|\Pi_{\mathrm{flux}} f_\theta(x) - \nabla \log p_{\mathrm{data}}(x)\|^2\big] \quad (4)$$

This is invariant to any $v$ with $\nabla \cdot (p_{\mathrm{data}} v) = 0$, by construction — precisely the freedom the paper wants. Since $\nabla \log p_{\mathrm{data}}$ is already a gradient field, score matching is recovered as the special case where $f_\theta$ is required to be the score.

### Tractable Flux Matching loss (§3.2)
Direct computation of $\Pi_{\mathrm{flux}}$ is intractable in high dimensions. The paper derives an equivalent objective (same gradients) using auxiliary quantities $u_\theta := f_\theta - \nabla \log p_{\mathrm{data}}$ and $r_\theta := p_{\mathrm{data}}^{-1} \nabla \cdot (p_{\mathrm{data}} u_\theta) = \nabla \cdot u_\theta + u_\theta \cdot \nabla \log p_{\mathrm{data}}$. The loss is computable via standard automatic differentiation. (Full derivation in §3.2; not extracted here.)

### Noise-annealed extension (§3.3)
For practical generative modelling, the paper extends Flux Matching to the noise-annealed setting used by diffusion models — learning a continuum of fields for increasingly noised distributions, rather than one field for $p_{\mathrm{data}}$ directly. The learned models can be used out-of-the-box with existing diffusion samplers and likelihood computations.

### Selecting application-specific solutions (§3.4)
Two strategies for picking *which* member of the flux family to learn:
- **Architectural constraints** — design $f_\theta$ to bake in a structural property (e.g. a directed graph constraint).
- **Regularizers** — add penalties that prefer specific attributes (mixing speed, smoothness, interpretability).

## Experimental setup (§4)

Four applications:
1. **High-dimensional image generation**: CIFAR-10, CelebA 64×64. Standalone Flux Matching objective.
2. **Fast-mixing fields for accelerated sampling**.
3. **RNA velocity in single-cell genomics**: interpretable generative field for biology.
4. **Directed dependencies between variables**: structured generative dynamics.

## Key results

### Unconditional image generation (Table 1)

| Dataset | Model | FID ↓ | IS ↑ | NLL (bpd) ↓ |
|---|---|---|---|---|
| CIFAR-10 | DSM | 4.74 | 8.52 | 3.16 |
| CIFAR-10 | Flux | 9.06 | 8.54 | 3.26 |
| CelebA 64 | DSM | 2.41 | — | 2.03 |
| CelebA 64 | Flux | 7.07 | — | 2.17 |

Training efficiency (also Table 1):

| Dataset | Model | Speed (it/s) | Memory/GPU (G) |
|---|---|---|---|
| CIFAR-10 | DSM | 11.63 | 2.79 |
| CIFAR-10 | Flux | 4.01 | 5.69 |
| CelebA 64 | DSM | 7.20 | 7.79 |
| CelebA 64 | Flux | 1.77 | 22.67 |

The authors are transparent:

> "The remaining FID gap to DSM is unsurprising since DSM has benefited from many engineering iterations specifically aimed at optimizing FID, whereas Flux Matching is evaluated here as a first-pass implementation of a new learning objective. The bottom half shows that Flux Matching is roughly 3–4× slower than DSM during training and uses about 2–3× more memory." (§4.3)

> "At sampling time, the learned field can be used in the same samplers as a score model." (§4.3)

### Faster mixing (§4.4)
Score-based Langevin dynamics is reversible and known to mix slowly. Flux Matching can learn non-reversible fields that mix faster, requiring fewer sampling steps to reach the target. Figure 6 shows FID-vs-steps curves where Flux Matching converges in fewer steps than DSM. (Specific step-count numbers in figure; not extracted.)

### RNA velocity (§4.5)
Flux Matching fits *interpretable* vector fields directly to single-cell genomics data — the learned field has biological-meaning-structure that score matching's score-field doesn't preserve.

### Directed dependencies (§4.6)
By baking in architectural constraints (e.g. lower-triangular Jacobian for a topological ordering), Flux Matching produces generative fields with prescribed directed dependencies — useful for causal modelling and structured generation.

## Ablations

_Body presents component-level results in §4.2 (synthetic 2D), §4.3 (CIFAR/CelebA Table 1), §4.4–4.6 (per-application showcases)._ Specific ablations on the projected-Fisher-vs-direct-flux-divergence choice and on the noise-annealed extension are in body sections not fully extracted. Worth chasing on a re-read.

## Limitations (§5)

Stated explicitly by the authors:

> "Flux Matching is most useful when the goal is not only to model a distribution, but also to learn a generative vector field with additional desired structure. If one only cares about unrestricted generative modeling, then standard DSM already provides a highly optimized objective. Flux Matching should be viewed less as a replacement for DSM and more as a paradigm that enables use cases where the vector field itself matters."

> "The main practical limitation is computational cost, with our implementation being roughly 2–4× more expensive than DSM in both runtime and memory."

> "The flexibility that Flux Matching provides is only useful when paired with a meaningful inductive bias or auxiliary objective. Flux Matching expands the class of learnable generative vector fields but does not automatically identify the best member for a given application, and designing architectures or regularizers that exploit this extra freedom remains problem dependent."

The authors are unusually candid: this isn't a "we beat the state of the art" paper, it's a "we open a new design dimension."

## Why it matters [analyst's view]

Three things land:

1. **It exposes a hidden constraint in score-based generative modelling.** Every diffusion / score-matching paper since 2019 has *implicitly* targeted the score because nobody wrote down that the FPE-stationary condition admits a family. The paper's contribution is partly conceptual: "you were never solving the generation problem; you were solving the much more constrained score-matching problem." Once you see it, the score's privileged status looks accidental.

2. **The "vector field as design choice" framing is genuinely new.** In score matching, you choose the *architecture*; the *target* is fixed. In Flux Matching, you choose the target. This is closer to how engineers design optimisers (where the loss is a design choice) than how generative modellers typically work. It opens the door to encoding application-side constraints (interpretability, mixing speed, mechanistic structure) directly into the generative objective rather than as post-hoc fixes.

3. **It connects diffusion to the broader "structured latent geometry" trend.** [[papers/maes-2026-leworldmodel]]'s SIGReg, [[papers/huang-2026-semantic-tube-prediction]]'s Geodesic Hypothesis, [[papers/yang-2026-replaid-continuous-diffusion]]'s ELBO-based embedding structure, and now Flux Matching's "vector-field-as-choice" — four 2026 papers that, from very different angles, all argue that *what the latent does* matters more than *how much you scale*. Worth flagging as a cluster.

The 2–4× cost is the headline blocker for practical adoption. If someone reduces that overhead by a constant factor, Flux Matching becomes the default for any task where the generative dynamics matter (biology, causal modelling, controllable generation).

## Open questions

[analyst's view]
- **How does Flux Matching compare to flow matching / rectified flow / consistency models?** These also generalise score matching but along different axes. The paper situates itself relative to DSM, not these recent neighbours — a direct comparison would be useful.
- **What's the FID gap *after* engineering parity?** "First-pass implementation" leaves the door open that engineered Flux Matching could match DSM on FID. Worth tracking.
- **Can the projected Fisher divergence be computed without the auxiliary $r_\theta$ machinery?** A cleaner objective might cut the 3–4× cost.
- **Connection to non-equilibrium statistical mechanics.** Non-reversible Markov processes mix faster (well-known in stat mech). Flux Matching is essentially a way of learning those processes from data. Is there a cleaner formulation in terms of Helmholtz decomposition?
- **Cross-link to [[papers/yang-2026-replaid-continuous-diffusion]]**: both papers exploit the "ELBO/divergence-based" structure of latent geometry in generative modelling. Yang's paper claims ELBO-based training produces structured embeddings naturally; Flux Matching claims structured fields require explicit choice. Tension or complementarity?

## Connections

- **Direct neighbourhood**:
  - [[papers/yang-2026-replaid-continuous-diffusion]] — continuous-diffusion-for-language; both papers in the broader "rethink score-matching" cluster.
  - [[papers/baek-2026-gram]] — also a generative model with chosen latent dynamics (stochastic guidance in recursive reasoning); same general philosophy of "the dynamics are the design choice."
  - [[papers/maes-2026-leworldmodel]] — SIGReg makes latent geometry a design choice; Flux Matching makes vector-field geometry a design choice. Same instinct, different layer.
- **Topic MOCs**: [[topics/generative-models]], [[topics/score-matching]], [[topics/diffusion-models]]
- **Author indices**: [[authors/peter-pao-huang]]
- **Code**: `github.com/peterpaohuang/flux_matching`

## Selected quotes

> "Many different vector fields produce diffusion processes with the same stationary distribution. Modern generative modeling, however, canonically targets one particular vector field called the (Stein) score function, typically fit via score matching." — §1

> "We introduce Flux Matching, a new paradigm for generative modeling that generalizes existing score-based models to a broader family of vector fields that need not be conservative. Rather than requiring the model to equal the data score, the Flux Matching objective imposes a weaker condition that admits infinitely many vector fields whose stationary distribution is the data." — abstract

> "Flux Matching opens a new dimension in generative modeling by turning the vector field itself into a design choice rather than a fixed target." — abstract
