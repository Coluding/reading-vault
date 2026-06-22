---
type: blog
title: "Autoregression vs Diffusion - Understanding Sampling via Optimal Transport"
author: "jiha"
url: https://jiha-kim.github.io/posts/autoregression-vs-diffusion-understanding-sampling-via-optimal-transport/
rw_id: 01ktspp43h5azbrhrnf692x02v
topics: [optimal-transport, generative-models, diffusion-models, autoregressive-models]
priority: medium
read_state: queued
added: 2026-06-10
last_updated: 2026-06-17
---

## TL;DR

The post reframes autoregressive (AR) and diffusion/flow generative models as two computational strategies for the *same* geometric task: transporting a tractable noise law $P_\text{noise}$ to the data law $P_\text{data}$. The unifying tool is **optimal transport (OT)**. The author's central claim: AR sampling is exactly the **Knothe–Rosenblatt rearrangement** — a triangular, coordinate-ordered map built from stacked 1D inverse-CDF (quantile-matching) steps, which is *exact* and tractable but carries an *ordering bias* and is generally not $W_2$-optimal. Diffusion/flow models instead learn an *unconstrained* transport map (ideally the symmetric **Brenier map** $T=\nabla\psi$), removing the ordering constraint at the cost of having to learn the map/dynamics numerically. The OT lens does not claim the two are literally the same algorithm — it shows they are different parameterizations and relaxations of one source-to-target problem, and predicts the AR-text / diffusion-image split will keep blurring as transport solvers improve.

## Context

This is a long (≈6,400-word, "25 min") expository / survey post on Ji-Ha's Blog (published 2026-04-28), pitched at readers who want the mathematical scaffolding connecting sampling methods. It is heavy on OT theory (Monge, Kantorovich, duality, Wasserstein, Benamou–Brenier, Brenier's theorem) and threads recent ML papers through that scaffolding (Flow Matching, Rectified Flow, VAR, frequency-space AR, Schrödinger bridges, one-step generators, polar factorization). The stated route is explicit: "define the transport problem, solve it exactly in 1D, show how autoregression reuses that 1D solution conditionally, then compare it with Brenier maps, Wasserstein geodesics, and the learned flow or one-step approximations used in practice." It positions itself against the common framing that AR and diffusion are fundamentally different strategies.

## Core argument

**1. Generative modeling = transport.** Sampling from a complex $P_\text{data}$ directly is intractable, so we sample easy noise $Z\sim P_\text{noise}$ and push it through a map. A transport map must satisfy $T_\sharp P_\text{noise}=P_\text{data}$ (pushforward), equivalent to the change-of-variables $p_\text{data}(x)=p_\text{noise}(z)\lvert\det J_T(z)\rvert^{-1}$. Gaussian/Uniform noise is standard because they are the maximum-entropy laws under a fixed second moment (Gaussian on $\mathbb{R}$) or on a bounded interval (Uniform).

**2. The OT machinery.** Monge seeks a single-valued map minimizing $\min_T \mathbb{E}_{Z}[c(Z,T(Z))]$ s.t. $T_\sharp P_\text{noise}=P_\text{data}$; Kantorovich relaxes to a coupling $\pi\in\Pi(P_\text{noise},P_\text{data})$ minimizing $\int c\,d\pi$, which is *linear* in $\pi$ and hence has a dual: $\max_{f,g}\sum_i a_i f_i+\sum_j b_j g_j$ s.t. $f_i+g_j\le c_{ij}$ (prices that certify lower bounds; complementary slackness $\pi_{ij}>0\Rightarrow f_i+g_j=c_{ij}$). With a ground metric this value becomes the $p$-Wasserstein distance $W_p(P,Q)=(\inf_\pi\int d(x,y)^p d\pi)^{1/p}$. $W_1$ collapses to the 1-Lipschitz dual $\sup_{\lVert f\rVert_\text{Lip}\le1}\mathbb{E}_P f-\mathbb{E}_Q f$ — the reason $W_1$ matters in adversarial (Wasserstein-GAN-style) modeling.

**3. The 1D solution is closed-form.** Inverse transform sampling: $X=F^{-1}(U)$ with $U\sim\mathcal{U}(0,1)$ already *is* a transport map. For 1D convex costs $c(x,y)=h(x-y)$, the optimal map is **quantile matching** — "sort both sides and pair equal ranks" — justified by the quadrangle inequality (uncrossing matching lines never increases cost). Continuously: $u=F(x)$ (pullback to uniform noise via the probability integral transform), $T(x)=G^{-1}(u)$ (pushforward).

**4. AR = Knothe–Rosenblatt = stacked conditional 1D transports.** Apply quantile matching one coordinate at a time:
$$T(x_1,\dots,x_D)=\bigl(T_1(x_1),\,T_2(x_1,x_2),\,\dots,\,T_D(x_1,\dots,x_D)\bigr),$$
with each $T_i(x_i\mid x_{<i})=F^{-1}_{Q_i\mid Q_{<i}}\bigl(F_{P_i\mid P_{<i}}(x_i\mid x_{<i})\bigr)$. This guarantees $T_\sharp P_\text{noise}=P_\text{data}$ but is **greedy / order-dependent** — it is $W_2$-optimal only as the limit of a heavily skewed cost $c(x,y)=\sum_i\lambda_i(x_i-y_i)^2$ with $\lambda_1\gg\dots\gg\lambda_D$. In density language this is the AR factorization $p(x)=\prod_i p(x_i\mid x_{<i})$, sampled by per-coordinate inverse-CDF. **Next-token prediction is exactly conditional inverse transform sampling**: from predicted probs $(0.6,0.3,0.1)$ the discrete CDF is $(0.6,0.9,1.0)$, and $u=0.75$ selects "banana"; cross-entropy NLL $\mathcal{L}=-\sum_i\log P(x_i^\text{true}\mid x_{<i})$ learns exactly the conditionals inducing these 1D transports.

**5. Brenier = the symmetric, globally optimal alternative.** Abandoning the greedy order, the true $W_2^2$ optimizer (for absolutely continuous $P$, finite second moments) is unique and has the form $T=\nabla\psi$ with $\psi$ convex. The higher-dim analogue of 1D monotonicity is **cyclic monotonicity** (no finite cyclic reassignment lowers cost), which Rockafellar turns into a convex potential — so $T=\nabla\psi$ is a *structural consequence of optimality*, not a modeling choice. Pairwise monotonicity $\langle T(x_1)-T(x_2),x_1-x_2\rangle\ge0$ is only the 2-cycle special case.

**6. Diffusion/flow = learned unconstrained transport.** Parameterize a velocity field $\tfrac{dx}{dt}=v_\theta(x,t)$ and integrate it (flow map $\phi_1$). Flow Matching / Rectified Flow pick a coupling $q(x_0,x_1)$ and regress toward the straight-line conditional velocity:
$$\mathcal{L}_\text{CFM}=\mathbb{E}_{t,(X_0,X_1)\sim q}\bigl\lVert v_\theta(X_t,t)-(X_1-X_0)\bigr\rVert_2^2.$$
The *global* geometry depends entirely on the coupling: $q=P_\text{noise}\otimes P_\text{data}$ gives independent pairing; an OT plan gives an OT-informed target; Brownian-bridge paths weighted by entropic-OT couplings give Schrödinger-bridge variants. So FM is a framework for turning a coupling+path choice into a regression loss — **not automatically the Benamou–Brenier geodesic.**

## Notable details

- **Benamou–Brenier dynamic OT:** $W_2^2(\mu_0,\mu_1)=\inf_{\rho_t,v_t}\int_0^1\!\int\lVert v_t\rVert^2\rho_t\,dx\,dt$ subject to the continuity equation $\partial_t\rho_t+\operatorname{div}(\rho_t v_t)=0$. The Wasserstein geodesic is **displacement interpolation** $X_t=(1-t)X_0+tT(X_0)$ (mass travels), contrasted with the linear mixture $\rho_t=(1-t)\rho_0+t\rho_1$ (mass fades in/out, doesn't move).
- **Gaussian closed form** (rare clean high-dim case): for $P=\mathcal{N}(m_0,\Sigma_0)$, $Q=\mathcal{N}(m_1,\Sigma_1)$, the Brenier map is affine $T(x)=m_1+A(x-m_0)$ with $A=\Sigma_0^{-1/2}(\Sigma_0^{1/2}\Sigma_1\Sigma_0^{1/2})^{1/2}\Sigma_0^{-1/2}$, and $W_2^2=\lVert m_0-m_1\rVert^2+\operatorname{tr}(\Sigma_0+\Sigma_1-2(\Sigma_0^{1/2}\Sigma_1\Sigma_0^{1/2})^{1/2})$ (the covariance term is the **Bures metric**). For standard-Gaussian source, $T(z)=m+\Sigma^{1/2}z$. The Cholesky map $z\mapsto m+Lz$ (the Gaussian KR rearrangement) reaches the *same target law* but a *different coupling/cost* than the symmetric-root Brenier map — the cleanest concrete instance of the whole AR-vs-OT distinction.
- **Sliced Wasserstein** as the multi-line generalization of the 1D trick: $\operatorname{SW}_p^p(P,Q)=\int_{\mathbb{S}^{D-1}}W_p^p(P_\theta,Q_\theta)\,d\theta$, estimated by averaging over random directions. Caveat (Kitagawa & Takatsu): topologically equivalent to $W_p$ but generally not bi-Lipschitz, and for $p>1$, $D\ge2$ the sliced spaces are not geodesic — a cheap geometry-aware *comparison* metric, not a single coherent high-dim transport map.
- **Reparameterized AR:** VAR (Tian et al., 2024) does next-*scale* prediction (coarse→fine resolution) instead of raster-scan; frequency-space AR (Yu et al., 2026) autoregresses in a wavelet/Fourier basis. Dieleman's "diffusion is spectral autoregression" reading (DDPM ≈ low→high frequency ordering, valid in expectation not per-sample); Falck (2025) argues that spectral hierarchy is *not necessary* — hierarchy-free diffusion can match DDPM and improve high-freq generation.
- **Entropic OT / mini-batch coupling:** $\min_\pi\int c\,d\pi+\varepsilon\,\mathrm{KL}(\pi\Vert\pi_\text{ref})$; with product reference $\mu\otimes\nu$, $\mathrm{KL}(\pi\Vert\mu\otimes\nu)=H(\mu)+H(\nu)-H(\pi)$, so it's entropy regularization up to constants (solved via Sinkhorn). Mini-batch OT pairing reduces crossing paths so velocity targets are more compatible. **Freulon et al. (2026)** caveat: endpoint matching is a two-marginal problem and doesn't build temporal coherence; a non-product (Gaussian-process) reference makes local transitions fit one coherent dynamical model.
- **One-step generators:** Drifting Models (Deng et al., 2026) learn a one-step map via an attraction–repulsion drift field $V_{p,q}(x)=V_p^+(x)-V_q^-(x)$ (kernel $k(x,y)=\exp(-\lVert x-y\rVert/\tau)$), anti-symmetric so $V_{p,p}=0$ (matched dists are equilibria); trains $\mathcal{L}=\mathbb{E}\lVert x-\operatorname{stopgrad}(x+V_{P_\text{data},q_\theta}(x))\rVert^2$ — **no paired endpoints needed**. Optimal Flow Matching (Kornilov et al., 2024) recovers the straight OT displacement in one FM step via convex-function parameterization.
- **Polar factorization (Vesseron & Cuturi, 2025; Morel et al., 2023):** any exact generator factors $P_\text{noise}$-a.e. as $\nabla\psi\circ M$ where $M$ preserves the source law — the infinite-dim analogue of matrix polar decomposition. Same density, different pairing: you can search Gaussian-preserving latent rearrangements (volume-preserving, $\lvert\det J_\phi\rvert=1$, via divergence-free ODEs + Liouville) to lower a trained flow's quadratic transport cost *without changing its density*, pushing the coupling toward the Monge map.

## Why it matters [analyst's view]

This is a strong "Rosetta Stone" note for the generative-modeling shelf of the vault: it gives a single coordinate system (OT) in which next-token LLM sampling, normalizing flows, diffusion, flow matching, and one-step distillation are all special cases differing only in *which coupling/path/cost* they implicitly pick. The most actionable mental model is the triangular-vs-symmetric split: AR buys exactness and exact likelihood with an ordering bias (KR rearrangement); flows/diffusion buy order-invariance and access to symmetric $W_2$ geometry but must learn the map numerically. The post's closing prediction — that "text uses AR and images use diffusion mostly because of inductive bias and computational convenience" and the boundary will keep blurring — is directly relevant to the flow-based-LLM thread already in the vault. The Gaussian Cholesky-vs-Brenier contrast is the cleanest worked example I've seen of "same marginal, different coupling," and is worth keeping as a teaching anchor. Caveat for future-me: this is an *expository* post, so the equations are standard textbook OT (Brenier, Benamou–Brenier, Rockafellar) rather than novel results — its value is synthesis and the curated bridge to 2024–2026 ML papers, several of which (Drifting Models, Freulon et al., Optimal Flow Matching) are worth their own paper notes.

## Connections

- Topic MOCs: [[topics/optimal-transport]], [[topics/generative-models]], [[topics/diffusion-models]], [[topics/autoregressive-models]], [[topics/flow-matching]]
- Related notes: [[blogs/flow-based-llms-intro]] (AR/diffusion boundary for LLMs), [[papers/pao-huang-2026-flux-matching]], [[papers/yang-2026-replaid-continuous-diffusion]], [[blogs/shing-diffusionblocks]]
- Papers referenced (candidates for future notes): Flow Matching (Lipman et al., 2023), Rectified Flow (Liu et al., 2022), VAR (Tian et al., 2024), Optimal Flow Matching (Kornilov et al., 2024), Drifting Models (Deng et al., 2026), Brenier polar factorization (Vesseron & Cuturi, 2025), Monge-map normalizing flows (Morel et al., 2023), entropic OT beyond product reference (Freulon et al., 2026)
- Author index: [[authors/jiha]]

## Selected quotes

> "autoregression does it through a sequence of conditional transports, while diffusion does it through a time-dependent denoising dynamics. Optimal transport is therefore a useful geometric lens for comparing them, even when the underlying training objectives are not literally the same."

> "So in 1D, optimal transport is just 'sort both sides and pair equal ranks.'"

> "Through the lens of optimal transport, autoregression and diffusion are two computational strategies for the same geometric task: transporting $P_\text{noise}$ to $P_\text{data}$. ... Text uses autoregression and images use diffusion mostly because of inductive bias and computational convenience; better transport parameterizations and solvers will keep blurring that boundary."
