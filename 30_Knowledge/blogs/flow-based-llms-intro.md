---
type: blog
title: "An Intuitive Introduction to Flow-Based Language Generation"
author: "Floor Eijkelboom"
url: https://flow-based-llms.github.io/
rw_id: 01ktspm69d6ej7cdz4aytbsrg7
topics: [flow-matching, diffusion-language-models, generative-models]
priority: high
read_state: queued
added: 2026-06-10
last_updated: 2026-06-10
---

# An Intuitive Introduction to Flow-Based Language Generation

## TL;DR

A pedagogical, animation-heavy walkthrough of **flow-based language models** — generative models that transform Gaussian noise into text by following a continuous velocity field, refining *every* token position in parallel and committing to nothing until the end. The post's central claim: flow matching becomes the right tool for discrete data once you (a) reframe the flow-matching target as a **posterior distribution over tokens on the probability simplex** trained with cross-entropy (Variational Flow Matching, VFM), and (b) **distill the continuous flow into a flow map** that jumps across time in one or two steps (Categorical Flow Maps, CFM). The surprising punchline is that softmax + cross-entropy — the machinery LLMs already use — turns out to be *exactly* the geometrically correct objective VFM prescribes. Distillability is also presented as a structural advantage flows have that discrete-diffusion LLMs fundamentally lack: you can't distill a staircase.

## Context

A companion / explainer post for the author's own line of flow-based LLM research, signed "Floor" (Floor Eijkelboom, first author of the cited VFM paper [1] and a co-author of Categorical Flow Maps [2]). Intended audience: ML researchers who know autoregressive and diffusion LMs and want an intuitive on-ramp into flow matching, the variational reframing, and categorical flow maps. The framing is explicitly recruiting: the foundations are "just settling," nobody has yet matched a tuned AR baseline at scale, so there is open research room. It is structured in two phases — **Phase 1: learn a flow** (noise→text trajectory, the conceptual core enabling parallel generation) and **Phase 2: distill that flow into a few-step generator** (the practical payoff). [analyst's view] Thematically a sibling of the Dieleman flow-maps explainer being triaged the same day; both push the "continuous transport for discrete data" instinct, but this post centers the variational/simplex framing rather than embedding-space diffusion.

## Core argument

**1. Why AR and discrete diffusion both fall short.** Autoregression is structurally sequential: token $n$ cannot be computed until tokens $1,\dots,n-1$ exist. Discrete-diffusion LLMs (dLLMs) escape left-to-right ordering but still make *hard, irreversible* token commitments one (batch) at a time — "an autoregressive model that has learned its own generation order." What's wanted is **true parallel generation**: every position continuously refineable, nothing finalized until the end.

**2. Flow matching.** Move points from noise $p_0$ to data $p_1$ along a chosen probability path $(p_t)_{t\in[0,1]}$. Define the bridge by linear interpolation of a sampled pair $(x_0,x_1)$:
$$x_t = (1-t)\,x_0 + t\,x_1.$$
The **ideal velocity field** is the conditional expectation of the straight-line direction over all couplings through $x_t$:
$$v_t(x_t) = \mathbb{E}[\,x_1 - x_0 \mid x_t\,] = \mathbb{E}\!\left[\tfrac{x_1 - x_t}{1-t}\;\middle|\;x_t\right].$$
Individual couplings are straight; the averaged trajectory is curved. It's learned by plain MSE regression onto the target $x_1-x_0$ — by regression logic the optimum is the conditional expectation. Sampling integrates the field, e.g. Euler: $x_{t+\Delta t} = x_t + \Delta t\, v_\theta(x_t,t)$.

**3. The variational / denoiser view.** Pulling the fixed conditioning out, $v_t(x_t) = \frac{\mathbb{E}[x_1\mid x_t] - x_t}{1-t}$, so equivalently predict the **clean endpoint** $\hat{x}_1 = \mathbb{E}[x_1\mid x_t]$. The key insight: minimizing $\|\hat{x}_1 - x_1\|^2$ *is* maximum likelihood under a fixed-variance Gaussian conditional. So flow matching was always probabilistic — MSE just encodes a Gaussian assumption. Making the assumption explicit lets us **swap the conditional family**. Treat the network output $\mu_\theta(x_t)$ as the mean of a predictive distribution $q_\theta(x_1\mid x_t)$ and fit by KL to the true posterior:
$$\mathcal{L}_{\text{VFM}} = \mathbb{E}_{t,x_t}\big[-\log q_\theta(x_1\mid x_t)\big].$$
Gaussian $q$ → MSE (continuous data); **categorical $q$ → cross-entropy** (discrete data). This is Variational Flow Matching [1].

**4. The simplex geometry.** Each token is a one-hot vertex of the probability simplex $\Delta^{K-1}$; $\mu_\theta(x_t)\in\Delta^{K-1}$ is produced by a **softmax head** and represents a posterior over the vocabulary. The VFM loss reduces to **cross-entropy against the clean token, applied at every noise level $t$** — exactly the loss every LM already uses. The conditional mean is just the probability vector, giving induced velocity
$$v_t(x_t) = \frac{\mu_\theta(x_t) - x_t}{1-t},$$
an arrow pointing from the current state toward the predicted simplex point. For sequences, linearity of expectation decomposes the posterior coordinate-wise: $n$ independent softmax heads (one per position), all conditioned on the full noisy sequence $x_t$, so the transformer carries the joint structure while the heads read off per-position marginals. (Numerical note: $1/(1-t)$ blows up as $t\to1$, so integration stops just short and reads tokens directly off $\mu_\theta$.)

**5. Flow maps for few-step generation.** Integrating the velocity needs dozens–hundreds of forward passes. A **flow map** $X_{s,t}(x_s)$ directly outputs where a point at $(x_s,s)$ lands at time $t$, parameterized as a secant step $X_{s,t}(x_s) = x_s + (t-s)\,v_{s,t}(x_s)$. The defining identity (no simulation needed) is the **Lagrangian / self-distillation** condition: $X_{s,t}$ is the flow map of $v_t$ iff its time-derivative agrees with the velocity along its own trajectory, with $\partial_t X_{s,t}\to v_t$ as $s\to t$. Training minimizes the squared gap between the network's two self-readings (its $\partial_t X$ and its $v_t$), with a stop-gradient on the velocity target. Because the identity is exact at $s=t$, short jumps bootstrap long ones — **self-distillation, one network, no separate teacher**. Empirically flow-map samplers hit many-step quality with **10–20× fewer function evaluations**, sometimes a single step.

**6. Categorical Flow Maps (CFM).** Combine both fixes. Since every infinitesimal velocity points toward the simplex, the whole trajectory and endpoint stay on it — so bake the constraint in: predict $\mu_{s,t}\in\Delta^{K-1}$ via softmax and write the flow map as a **convex combination** of $x_s$ and $\mu_{s,t}$ with non-negative weights summing to one. As $t\to1$ the prediction *is* the final sample, and the simplex is preserved by construction. All self-distillation machinery transfers, plus the continuous-domain control toolkit (classifier guidance, reward tilting, SMC reweighting).

## Notable details

- **The distillability argument against dLLMs is structural, not engineering.** A dLLM's "trajectory" is a sequence of discrete commitments with no continuous time parameter to shortcut — "you can't distil a staircase." Flow trajectories are smooth, and smoothness is exactly what makes them compressible into a flow map.
- **Flow maps buy more than speed — they buy look-ahead.** A flow map learns the *solution operator*, so you can evaluate where a point will land, check a constraint/reward there, and pull the gradient back to bend the current step (test-time guidance / planning). A plain velocity sampler only sees the local tangent.
- **Covariance $\Sigma_t$ is optional.** The VFM objective pins only the conditional mean, so the sampler runs off $\mu_\theta$ alone; modeling $\Sigma_t$ doesn't change sampled marginals but can reshape gradients and give per-step uncertainty (adaptive step size, calibration).
- **Conditional generation comes for free** — conditioning variable $y$ just enters the posterior $q_\theta(x_1\mid x_t, y)$; velocity, denoiser, and flow map go through unchanged.
- **Beyond text:** the same machinery applies to any categorical/structured data — molecular graphs (nodes + edges as categorical) are flagged as a *clearer near-term payoff* than language, since one-step generation with property guidance (binding affinity, solubility) is feasible and useful today.
- **Open problems named:** no frontier-scale flow-based pretrain matching a tuned AR baseline yet; no settled post-training stack (RLHF acting on velocity vs. flow map vs. jointly breaks consistency); architectural priors for a "Text-DiT"; interpolant/prior/noise-schedule choices; two-stage distillation vs. joint self-distillation (recent work hints two-stage may be cleaner).
- **Convergent work:** Flow Map Language Models [4] and Discrete Flow Maps [5] independently rederive a close formulation and rewrite the self-distillation objective itself in KL form — "KL all the way down." CDCD (Dieleman et al. [3]) is the embedding-space continuous-diffusion cousin without flow-map distillation.

## Why it matters [analyst's view]

This is the clearest single-document case I've seen for the thesis that **softmax + cross-entropy were "geometrically correct all along"** — it retroactively justifies standard LM machinery as the right denoiser for categorical flow matching, which is a genuinely satisfying unification. The load-bearing move is the variational reframing: it dissolves the apparent mismatch between continuous flows and discrete tokens by recognizing MSE as just a Gaussian likelihood, then swapping in a categorical one. Combined with flow maps, it gives a coherent story for *parallel, controllable, distillable, non-autoregressive* text generation. The candor about the gap between recipe and scaled engineering (no AR-matching pretrain exists) is the right caveat — this is a research-program manifesto, not a results paper. For the vault, it ties together the looped/diffusion-LM cluster and the flow-matching paper notes under one conceptual roof, and the molecule-generation aside is a useful reminder that the near-term payoff may be non-text categorical data.

## Connections

- Topic MOCs: [[topics/flow-matching]], [[topics/diffusion-language-models]], [[topics/generative-models]]
- Related papers (in vault): [[papers/yang-2026-replaid-continuous-diffusion]] (continuous diffusion LMs — adjacent embedding/continuous route), [[papers/lee-2026-looped-diffusion-lm]] (masked/looped diffusion LMs — the "hard commitment" family this post argues against), [[papers/pao-huang-2026-flux-matching]] (flow matching applied to LMs), [[papers/bartosh-2026-dual-rate-diffusion]] (diffusion-LM sampling dynamics)
- Sibling explainer triaged same day: the Dieleman flow-maps post (same "continuous transport for discrete data" instinct; CDCD [3] above is Dieleman's own line)
- Author index: [[authors/floor-eijkelboom]]
- Key external references cited: Variational Flow Matching (Eijkelboom et al., NeurIPS 2024, arXiv:2406.04843); Categorical Flow Maps (Roos et al., 2026); Flow Map Matching (Boffi et al., 2024, arXiv:2406.07507); Flow Matching for Generative Modeling (Lipman et al., ICLR 2023, arXiv:2210.02747); Rectified Flow (Liu et al., ICLR 2023, arXiv:2209.03003)

## Selected quotes

> "Flow-based language models are the first practical, truly parallel alternative, refining every position continuously, committing nothing until the end. And surprisingly, they fit naturally with the same softmax-and-cross-entropy geometry LLMs already use, and are naturally distillable to one-step generators."

> "In that sense, a dLLM is still fundamentally autoregressive — just an autoregressive model that has learned its own generation order."

> "The softmax-and-cross-entropy machinery of language modelling was already geometrically correct all along. VFM explains why."

> "A flow map learns where will this trajectory land? — a question that only makes sense when the trajectory is a smooth function of time. dLLMs don't have that... You can't distil a staircase."
