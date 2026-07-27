---
type: blog
title: "SIGReg from First Principles"
author: "Reza Bayat"
url: https://rezabyt.github.io/blogposts/sigreg-tutorial.html
rw_id: 01kyhvdwdbgzxfv5mdxctq9q9b
topics: [jepa, self-supervised-learning, representation-learning, world-models]
priority: high
read_state: queued
added: 2026-07-27
last_updated: 2026-07-27
---

# SIGReg from First Principles

## TL;DR

A ~9k-word step-by-step construction of **SIGReg** — the anti-collapse regulariser introduced in LeJEPA (Balestriero & LeCun, 2025) and used by [[papers/maes-2026-leworldmodel]] — built from scratch rather than quoted. The chain: you want a differentiable, sample-only scalar whose population minimum is exactly the isotropic Gaussian $\mathcal{N}(0, I_D)$; KL is unusable (needs a density), so compare **characteristic functions** instead; Fourier uniqueness makes CF-matching equivalent to distribution-matching; the **Epps–Pulley statistic** turns "two CFs agree" into one integral; **trapezoidal quadrature** turns the integral into a finite sum; and the **Cramér–Wold theorem** reduces the $D$-dimensional test to an average of 1-D tests over random unit directions. The payoff line: since SIGReg's population zero is the isotropic Gaussian and the isotropic Gaussian is full rank, *collapse is incompatible with the optimum* — an anti-collapse guarantee by construction rather than by heuristic. Roughly 15 lines of NumPy, $O(NMT)$ per step, defaults $M=1024$ directions, $T=16$ knots, bandwidth $\lambda=1$, mixing weight $\lambda_{\text{reg}}=0.1$.

## Context

Bayat (already in the vault via [[papers/bayat-2026-tapered-language-models]]) writes this as a tutorial bridging two papers: **LeJEPA** (Balestriero & LeCun, arXiv:2511.08544, 2025), which introduced SIGReg and the argument that isotropic Gaussian embeddings minimise worst-case downstream prediction risk, and **LeWorldModel / LeWM** (Maes, Le Lidec, Scieur, LeCun, Balestriero, arXiv:2603.19312, 2026), which applies it to latent world modelling. The post develops SIGReg in the general JEPA setting and uses LeWM as the concrete temporal-prediction instance. The framing target is explicit: read this and "the formulas should click much faster" in either paper.

The problem it opens with is the standard JEPA failure. Encoder $f_\theta$ maps observation to embedding, predictor $g_\phi$ maps $z_t \mapsto \hat{z}_{t+1}$, loss compares $\hat{z}_{t+1}$ against $f_\theta(o_{t+1})$. Both sides pass through the *same* encoder, so a constant encoder gives $z_t = z_{t+1} = c$ and any predictor outputting $c$ achieves exactly zero loss. Bayat is blunt that this is not hypothetical: "training LeWM on $\mathcal{L}_{\text{pred}}$ alone can converge to trivial encoders. Gradient descent takes the easiest path to zero loss, and 'encode everything the same' is the easiest path."

## Core argument

**Three requirements** fix the design space (§1). The regulariser must (i) be zero in the population limit iff $z \sim \mathcal{N}(0,I)$ and positive otherwise, (ii) be differentiable — which immediately kills the classical Gaussianity tests, since Kolmogorov–Smirnov takes a supremum over CDFs and Shapiro–Wilk uses sorted order statistics, neither of which has usable gradients — and (iii) scale at most linearly in embedding dimension $D$ and batch size $N$, both in the hundreds, evaluated every step.

**Why isotropic Gaussian is the target.** Three reasons, of which only the first is about collapse: full rank by construction ($\Sigma = \sigma^2 I$, all eigenvalues equal, so a lower-dimensional optimum is impossible); no preferred direction (rotational symmetry means the regulariser doesn't bias which axes carry information); and maximum entropy for a fixed second moment (the least-committal full-rank target). The *primary* justification is imported from LeJEPA: under broad downstream-probing assumptions, isotropic Gaussian embeddings minimise worst-case prediction risk. So SIGReg isn't just making the cloud "look nice" — it targets a distribution the theory calls optimal for later tasks.

**The instructive failure (§1).** The obvious idea — at each step draw a fresh $z^* \sim \mathcal{N}(0,I)$ per embedding and penalise $\|z - z^*\|^2$ — is *exactly wrong*, and the post's best pedagogical moment. Taking the expectation over the random target, the gradient in expectation is a pure pull toward the origin: the random directions average out and what remains is precisely the collapse you were trying to prevent. The conceptual error is "conflating *samples from* $\mathcal{N}(0,I)$ with *the distribution itself*." You don't want any individual $z$ near any particular $z^*$; you want the *empirical distribution* of the batch to be statistically indistinguishable from Gaussian samples. That's a property of the collection, not of points.

**Characteristic functions (§2).** KL needs $p_z$, which you'd have to estimate in high dimensions and couldn't backprop through cleanly. Instead use the CF $\varphi_z(t) = \mathbb{E}[e^{itz}]$ — the Fourier transform of the measure, estimable from samples as a plain mean of $\cos/\sin$ terms, hence differentiable. **Fourier uniqueness**: two distributions are equal iff their CFs are equal, so the CF is a complete fingerprint. The target is $\varphi_0(t) = e^{-t^2/2}$ (real-valued, since $\mathcal{N}(0,1)$ is symmetric). Nice intuition throughout: $t$ is a frequency knob the analyst chooses — small $t$ packs all sample angles near $1$ on the unit circle and probes coarse structure (mean, variance), large $t$ scatters them and probes tails and fine shape.

**Epps–Pulley (§3).** The scalar is the weighted $L^2$ gap between empirical and target CFs, scaled by $N$. The weight is essential, not cosmetic: the bare squared gap $|\varphi_N(t) - \varphi_0(t)|^2$ oscillates around a $1/N$ noise floor forever, so the unweighted integral **diverges**. Multiplying by $w(t) = e^{-t^2/(2\lambda^2)}$ makes the area finite and concentrates it where the comparison is informative. The bandwidth $\lambda$ controls what the test can see: at $\lambda=0.3$ it only checks low moments, so a bimodal sample (which already matches mean and variance) survives untouched; at $\lambda=1$ or $3$ it sees higher-order shape and smooths the bimodal into a Gaussian. Standard choice $\lambda=1$.

**Quadrature (§4).** The integrand is a product of $\cos$, $\sin$ and Gaussians — infinitely differentiable — so trapezoidal error decays as $O(K^{-2})$ and few knots suffice. Practical detail: start the grid at $t=0.2$, not $0$, because at $t=0$ both CFs equal $1$ exactly and the integrand vanishes regardless of the distribution — a knot there is wasted. The tutorial's convergence sweep: $K=8$ lands within 7% of ground truth, $K=16$ within **0.04%**, and past that it's computational waste. (LeJEPA instead uses a symmetric grid, ~17 knots on $[-4,4]$, exploiting the even symmetry; the tutorial uses positive frequencies only, which captures the same information up to a constant absorbable into $\lambda_{\text{reg}}$.)

**Cramér–Wold (§5)** is the load-bearing step. The theorem: $X \stackrel{d}{=} Y$ iff $u^\top X \stackrel{d}{=} u^\top Y$ for every unit $u \in S^{D-1}$ — a multivariate distribution is fully determined by its 1-D shadows. The proof is one line in CF language: the CF of a projection is $\varphi_{u^\top X}(t) = \varphi_X(tu)$, so if all projection CFs match then $\varphi_X = \varphi_Y$ everywhere (any point is $tu$ for some $t$, $u$), and Fourier uniqueness closes it. The convenient consequence: for $z \sim \mathcal{N}(0, I_D)$, $u^\top z \sim \mathcal{N}(0, \|u\|^2)$, so **unit-norm** directions all share the identical 1-D target $\mathcal{N}(0,1)$ — one target CF for every projection. Since you can't test the whole sphere, sample $M$ unit vectors uniformly (draw Gaussian, normalise) and average, giving an unbiased Monte Carlo estimate.

**Why averaging matters** — the sharpest empirical point in the post. For an anisotropic 2-D Gaussian ($\sigma_x = 2$, $\sigma_y = 0.5$), the per-direction Epps–Pulley statistic varies by **more than a factor of three** across directions: the diagonal projections mix the over-wide $x$ with the too-narrow $y$ and land near variance 1, so they "look Gaussian" while the joint clearly isn't. The true sphere-averaged SIGReg value is $0.0674$ — *no single direction reports anything close to it.*

**The guarantee (§6).** Read the chain bottom-up in the population limit: SIGReg $= 0$ ⟹ every projection has zero Epps–Pulley ⟹ (Fourier uniqueness) every $u^\top z \sim \mathcal{N}(0,1)$ ⟹ (Cramér–Wold) $z \sim \mathcal{N}(0,I)$, which is full rank. Collapse cannot be a minimum. Bayat is careful about what this does *not* say: "The guarantee is about the minimum, not the loss landscape." Cramér–Wold says nothing about local minima, saddles, or reachability; the empirical evidence (a rank-1 initialisation whose covariance spreads back out to $\approx I$) suggests a benign landscape, but that's observation, not theorem. The honest summary he gives: "clean global minimum + empirically tractable landscape is what makes it work in practice."

## Notable details

- **Total objective** (§7): $\mathcal{L} = \mathcal{L}_{\text{pred}} + \lambda_{\text{reg}}\,\text{SIGReg}(Z)$, with LeWM using $\lambda_{\text{reg}} = 0.1$. Encoder params feel both gradients; predictor params feel only $\mathcal{L}_{\text{pred}}$ (SIGReg doesn't depend on $\phi$).
- **Explicit notation warning** — two unrelated $\lambda$s: the Epps–Pulley bandwidth ($\lambda=1$, inside the weight function) and the outer mixing coefficient ($\lambda_{\text{reg}}=0.1$). When LeWM writes "$\lambda_{\text{reg}}=0.1$" it means the latter.
- **Each term alone fails**: prediction loss alone has the trivial constant-encoder optimum; SIGReg alone shapes the marginal but nothing ties embeddings to dynamics. The pairing is what works.
- **No tricks**: "No alternating updates, no two-timescale dynamics, no stop-gradient tricks." Both losses jointly differentiable, single scalar, one optimiser — the selling point versus EMA/stop-gradient recipes.
- **Reference implementation**: `sigreg(Z, M=1024, lam=1.0, T=16)` — build knots+weights, draw $M$ unit vectors, `H = U @ Z.T`, then vectorised $\cos/\sin$ means give empirical CFs, squared gap against $G = e^{-t^2/2}$, weighted sum, mean over directions. Largest intermediate is $(M, T, N)$; at typical sizes ~4M floats.
- **Collapse escape is bimodal first**: starting from all mass at the centre, the optimiser splits into two clumps at $\pm 0.78$ by step 50 before filling in to a Gaussian by step 2000. Bimodality is the loss's easiest first move out of a delta spike, not something it's told to do.
- **Finite-$M$ caveat**: with $M$ directions per step a batch could miss a malicious non-Gaussian direction; rescued by fresh re-sampling each step (covering the sphere densely over training) plus the difficulty of staying near zero on random directions while being anomalous somewhere.
- The maths is old — **Cramér–Wold is 1936, Epps–Pulley is 1983**; only the application to deep learning is recent.

## Why it matters [analyst's view]

This is the missing connective tissue for a thread the vault has been accumulating from three directions without ever holding the mechanism itself. [[papers/maes-2026-leworldmodel]] is noted as "the first end-to-end stable JEPA from pixels with only a two-term loss (next-embedding MSE + SIGReg)" — this post *is* the derivation of that second term. [[papers/ivashkov-2026-sensorimotor-world-models]] reports 84% vs 59% for inverse-dynamics anti-collapse **over SIGReg** on OGBench-Cube, and that comparison is much easier to reason about now: SIGReg enforces a property of the *marginal* embedding distribution (be isotropic), whereas SMWM's inverse-dynamics head enforces a property of the *information content* (retain what's controllable). Those are different targets, and SMWM's win suggests that on control tasks "spread out" is a weaker constraint than "keep the actionable bits" — isotropy is satisfiable by spreading noise as happily as by spreading signal. The post half-concedes this: SIGReg alone "does not force those embeddings to preserve information useful for prediction."

The framing I'd keep is Bayat's closing one: anti-collapse has historically been "an empirical art" — stop-gradients, EMA targets, contrastive negatives, tricks that work but resist analysis. SIGReg's claim to be structurally different rests entirely on the population-minimum statement, and the post is unusually disciplined about not overselling it (guarantee is about the minimum, not the landscape; finite $M$ and $K$ are approximations to an ideal chain). That's the right level of confidence, and it's the standard I'd hold the [[topics/jepa]] MOC's other anti-collapse recipes to.

Also worth noting as a *methodological* artifact: the "pull each $z$ toward a random Gaussian sample" failure is a genuinely general trap — matching samples when you meant to match distributions — and the two-line expectation argument that exposes it transfers well beyond JEPAs.

## Connections

- Topic MOCs: [[topics/jepa]], [[topics/self-supervised-learning]], [[topics/representation-learning]], [[topics/world-models]]
- Related in vault:
  - [[papers/maes-2026-leworldmodel]] — the direct application; this post supplies the derivation behind LeWM's SIGReg term and shares its notation.
  - [[papers/ivashkov-2026-sensorimotor-world-models]] — the competing anti-collapse mechanism (inverse dynamics), benchmarked *against* SIGReg; read together for the marginal-shape vs information-content contrast.
  - [[papers/baldassarre-2025-dino-world-models]] — the third strategy: dodge collapse entirely by freezing a pretrained encoder.
  - [[papers/bayat-2026-tapered-language-models]] — same author.
- Sources cited by the post, both `_needs note_`:
  - **LeJEPA** (Balestriero & LeCun, arXiv:2511.08544, 2025) — introduces SIGReg and the worst-case-downstream-risk argument for isotropic embeddings. The [[topics/jepa]] MOC already flags JEPA foundations as a gap; this is the highest-value one to close.
  - **LeWorldModel** (Maes et al., arXiv:2603.19312, 2026) — already noted in the vault.
- Author index: [[authors/reza-bayat]]

## Selected quotes

> "Gradient descent takes the easiest path to zero loss, and 'encode everything the same' is the easiest path."

> "The conceptual mistake is conflating 'samples from $\mathcal{N}(0,I)$' with 'the distribution itself.' We don't want any single $z$ to equal any particular sample. We want the empirical distribution of $z$ to be statistically indistinguishable from samples drawn from $\mathcal{N}(0,I)$."

> "If you know what every 'shadow' looks like, you know the body that cast them." — on Cramér–Wold

> "The guarantee is about the minimum, not the loss landscape." — §6
