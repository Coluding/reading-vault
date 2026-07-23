---
type: paper
title: "Intelligence from Learnable Novelty"
authors: ["Yanbo Zhang", "Michael Levin"]
year: 2026
venue: arXiv preprint
url: https://arxiv.org/abs/2607.18433
rw_id: 01ky7gbazfa8a21a9bxtnwxnx9
topics: [open-ended-learning, exploration, representation-learning, predictive-processing]
priority: high
read_state: queued
relevance: ""
added: 2026-07-23
last_updated: 2026-07-23
---

# Intelligence from Learnable Novelty

## TL;DR

Zhang and Levin (Allen Discovery Center, Tufts / Wyss Institute) argue that three classically separate faces of intelligence — complexity generation in dynamical systems, abstraction in representation learning, and exploration in agents — are all projections of one quantity: **learnable novelty**, the part of a data stream's cumulative surprise that a compute-bounded observer can actually convert into a reusable model. This quantity is exactly the *epiplexity* of Finzi et al. (2026): the program length of the best model a bounded learner can fit. The paper's technical contribution is a **closed-form, differentiable estimator** of it: fix a random reservoir computer as the bounded observer, so that all learnable capacity sits in a linear readout solvable by ridge regression, and score the readout's spectral description length. Used as a measure with zero supervision, the estimator ranks the Turing-complete rule 110 highest among all 88 elementary cellular automata. Used as a gradient-ascent objective, it drives a neural cellular automaton into a soliton-producing regime and organizes an unsupervised MNIST encoder into digit clusters (linear probe 0.53 → 0.89, no labels). Used as an intrinsic RL reward, it improves on the task baseline in nine of ten environments and collapses in none — while a naive state-magnitude bonus collapses on two.

## Context & motivation

The two most influential intrinsic drives fail in mirror image. **Novelty search** (Lehman & Stanley 2011) rewards surprise and escapes deceptive optima, but a maximizer of surprise is transfixed by a *noisy television*: pure noise is unpredictable forever, hence forever "novel", yet teaches nothing (the classic failure of prediction-error curiosity — Pathak et al. 2017, Burda et al. 2019). The **free-energy principle** (Friston 2010) minimizes surprise, and its degenerate optimum is the *dark room*: nothing is easier to predict than nothing happening (Sun & Firestone 2020). The paper's diagnosis: both objectives treat as one quantity the surprise a learner can convert into knowledge and the surprise it never can. Novelty search maximizes the sum and is dragged toward the unlearnable residual; free energy minimizes the sum and discards structure along with noise.

The learnable part alone was recently formalized as **epiplexity** by Finzi et al. (2026) — but their construction trains a full neural network for every system scored, which is expensive and, crucially, *not differentiable in the system being scored*: it works as a measure but not as an objective. This paper turns epiplexity into an optimizable quantity, and closes both pathologies by construction: a noisy TV is all residual (contributes nothing learnable), a dark room contributes nothing at all.

## Method

### Problem formulation

Score the relationship between inputs $X = (x_1,\dots,x_N)$ and targets $Y = (y_1,\dots,y_N)$ by how much *learnable structure* it contains for a fixed bounded observer $\varphi$; then, in the "inverse" setting, backpropagate that score through $Y$ into the parameters of whatever system generated it (a cellular-automaton rule, an encoder, indirectly a policy) and ascend it.

### Core idea

Cumulative surprise splits into a learnable part (model program length) and an unlearnable residual; isolate the learnable part, and make it cheap and differentiable by choosing a bounded observer whose optimal model has a closed form — a reservoir computer with a ridge-regression readout.

### From prequential surprise to epiplexity

A receiver holding $X$ observes $Y$ one symbol at a time, predicting each before it arrives and paying surprise $\ell_i = -\log_2 p(y_i \mid y_{<i}, X)$ (bits paid = negative log probability the receiver assigned to what actually arrived). The chain rule collapses the total to the **prequential description length** (Dawid 1984, Blier & Ollivier 2018):

$$L = \sum_{i=1}^{N} \ell_i = -\log_2 p(Y \mid X).$$

Novelty search maximizes $L$; the free-energy principle minimizes it. The MDL principle (Rissanen 1978) splits $L$ over models $M$:

$$L \approx \min_{M \in \mathcal{M}} \big[\, |M| - \log_2 p(Y \mid X, M) \,\big],$$

where $|M|$ is the model's description length (the part of the $X\!\to\!Y$ relationship an observer can *learn and reuse*) and the second term is the residual the best model cannot account for. The unrestricted minimum is uncomputable, so restrict to a finite model class $\mathcal{M}_\varphi$ defined by a fixed bounded observer $\varphi$, and define **learnable novelty** as the program length of the restricted optimum:

$$S^\varphi(Y \mid X) = |M^*_\varphi(Y \mid X)|, \qquad M^*_\varphi = \arg\min_{M \in \mathcal{M}_\varphi}\big[\,|M| - \log_2 p(Y \mid X, M)\,\big].$$

This is exactly Finzi et al.'s epiplexity. The boundedness is load-bearing: it is *why* both the noisy TV and the dark room contribute zero to $|M^*_\varphi|$, and — unlike novelty search's hand-picked behavior descriptor — the observer, not a designer, decides what counts as novel.

### The closed-form reservoir estimator

The epiplexity definition fixes only that the learner is bounded, not its architecture. A **reservoir computer** (Jaeger & Haas 2004; Maass et al. 2002) — a fixed random nonlinear feature map $\varphi$ followed by a single linear readout — makes the model search trivial: with feature matrix $H = \varphi(X) \in \mathbb{R}^{N \times m}$ ($m$ = reservoir feature dimension) and targets $Y \in \mathbb{R}^{N \times D}$, all learnable capacity resides in the readout $W \in \mathbb{R}^{m \times D}$, so "program length of the model" reduces to "description length of the linear operator $W$".

Assuming Gaussian residual noise $Y = HW + \epsilon$, the total description length is

$$L(W) = \frac{\|Y - HW\|_F^2}{2\sigma^2 \ln 2} + \mathcal{C}(W, \varphi),$$

where the first term converts mean-squared residual into bits at noise scale $\sigma^2$, and $\mathcal{C}$ prices the weights. For the weight part they take a **spectral description length** over the singular values $s_i(W)$:

$$\mathcal{C}_{\mathrm{spec}}(W) = \alpha \log_2 \det(I_m + \eta W W^\top) = \alpha \sum_i \log_2\!\big(1 + \eta\, s_i(W)^2\big),$$

with $\eta$ a resolution parameter (the precision at which a readout direction is priced) and $\alpha$ an overall scale (fixed to $1/2$; it changes neither rankings nor gradient direction). Why the log-det form: scaling a direction adds only *logarithmic* cost (a direction costs roughly as many bits as its magnitude has digits), and redundant/coincident readout directions add no new singular value, hence almost no cost — so the score counts *independent structure learned*, not raw weight magnitude. The same form matches the coding-rate objectives of Yu et al. (2020) and can be derived exactly as the marginal of a matrix-Gaussian prior on $W$ with a Wishart hyperprior on its row precision (Appendix C): integrating out the precision $\Lambda$ via the Wishart integral identity yields $-\log_2 p(W) = C + \frac{a+D}{2}\log_2\det(I_m + \eta WW^\top)$ — ridge = fixed precision, log-det = marginalized precision.

The log-det objective has no closed-form minimizer, so they approximate it with ridge regression, justified by a Taylor expansion: for small $W$, $\log_2\det(I_m + \eta WW^\top) \approx \eta \|W\|_F^2 / \ln 2$, so the spectral cost reduces to the quadratic ridge penalty, with $\sigma^2$, $\alpha$, $\eta$ merging into a single ridge parameter $\lambda$; the ridge's own shrinkage keeps $W$ in the small-norm regime where the expansion holds. Features are standardized per column ($\tilde{H}_c = (H_c - \mu_c)/(\hat\sigma_c \sqrt{m})$; the $\sqrt{m}$ keeps random-readout output scale invariant to width), while the target is centered and divided by a *fixed, pre-posited* scale $u_Y$ rather than its empirical std — deliberately, because target magnitude itself carries information: a larger target demands a larger readout, which the spectral cost prices at more bits; $u_Y$ acts as the posited measurement precision. The estimator is then fully closed-form:

$$S^\varphi(Y \mid X) = \tfrac{1}{2} \sum_i \log_2\!\big(1 + \eta\, s_i(W_\lambda)^2\big), \qquad W_\lambda = (\tilde{H}^\top \tilde{H} + \lambda I_m)^{-1} \tilde{H}^\top \tilde{Y}.$$

Deterministic, unique optimum, and **differentiable in $(X, Y)$** — so the score's gradient can flow back into whatever generated the data, with $\varphi$ frozen.

Three engineering points that matter:

- **Numerical stability (App. E):** the normal equations square the design matrix's condition number, which corrupts the score exactly in the low-epiplexity regime (nearly collinear features). They instead solve the augmented least-squares system $[\tilde H;\ \sqrt{\lambda} I_m]\,w_j = [\tilde y_j;\ 0]$ by reduced QR + triangular solve — same $W_\lambda$, square-root conditioning, still differentiable (QR and triangular solves have well-defined derivatives).
- **Reservoir criticality (App. A.3):** a plain random deep reservoir sits in the ordered phase (per-layer perturbation multiplier $\chi \approx 0.55$, defined as $\chi = \mathbb{E}\,\|\delta h^{(\ell+1)}\| / \|\delta h^{(\ell)}\|$; $\chi\!<\!1$ ordered, $\chi\!>\!1$ chaotic), so its nonlinearities act nearly linearly. One rule fixes this: normalize pre-activations over the feature axis before every nonlinearity, pinning $\chi \approx 1$ (edge of chaos) independent of depth, input scale, and architecture.
- **Ridge vs. exact MDL (App. D):** minimizing the log-det objective exactly via a majorize–minimize iteration ($W_{k+1} = (\tilde H^\top \tilde H + \lambda M_k)^{-1}\tilde H^\top \tilde Y$ with $M_k = (I_m + \eta W_k W_k^\top)^{-1}$) extracts up to ~2.3× more bits, but ranks systems almost identically (Spearman $\rho = 0.997$ over the ECA rules; Pearson $r = 0.996$ on NCA training trajectories). The approximation changes *how many bits*, not *which systems are structured* — so they keep the ridge.

### Why maximizing it should yield the three faces of intelligence

- **Dynamics:** trivial dynamics offer nothing to learn, fully chaotic dynamics nothing learnable — the maximum lies between order and chaos, where universal computation lives (only a system that can compute arbitrarily keeps producing learnable structure without bound).
- **Representations:** a bounded observer forces the encoder to shed redundancy and emit as many independent, recoverable distinctions as the data support — implicit compression, from which category structure should emerge.
- **Policies:** maximizing the learnable novelty of reachable futures rewards actions that keep the future rich and penalizes termination/stagnation — a sharpened causal-entropic / empowerment account: futures must be not merely diverse but *learnable*.

### Training / experimental procedure

Reservoirs are matched to data geometry (Table 3): circular 1D conv (256 ch, kernel 3, ELU, depth 3, $\lambda{=}0.03$) for ECA; depth-4 conv, $\lambda{=}0.3$ for inverse NCA; random MLPs elsewhere (MNIST: depth 4, width 2048, $\lambda{=}3$, $\eta{=}30$; RL: depth 4, width 32, $\lambda{=}0.3$; flows: width 64, $\lambda{=}0.1$). Inverse experiments: NCA trained 2,000 steps with AdamW (lr $10^{-4}$, cosine annealing, grad clip 0.5, batch 2048, score scaled by 1/100); MNIST encoder is an MLP (hidden 64/128/256, code $D{=}64$, unit-norm) trained 500 steps with AdamW, batch 128. RL: PPO (Stable-Baselines3 defaults), 8 parallel envs, rollout 1024, minibatch 256, $\gamma = 0.999$, GAE $\lambda = 0.98$, lr $3\times 10^{-4}$, 600k steps, 10 seeds.

## Experimental setup

- **Datasets/systems:** 88 locally unique elementary CA rules (width-64 ring, Bernoulli(1/2) init, 1000-step burn-in to the attractor, target = next 32 states stacked, $N{=}512$, 10 reservoir/data redraws); continuous flows (Lorenz, Rössler, Thomas vs. 3 linear systems); 1D two-channel neural CA (unit-norm per site, direct and residual normalized updates, $\tau{=}8$, Gaussian state noise 0.1 after burn-in — without it a uniform fixed point gives zero score and zero gradient); MNIST; 10 RL tasks (Acrobot, MountainCarContinuous, Hopper, BipedalWalker, HalfCheetah, Walker2d, Swimmer, PointMaze, LunarLander, Pendulum).
- **Baselines:** for RL — task reward alone; epiplexity-only (agent never sees the task); task + state-magnitude bonus (squared norm of input-normalized state, identically calibrated — a "cheating" control); task + epiplexity bonus. Per-step bonus is the increment $S^\varphi_t - S^\varphi_{t-1}$, maintained online by Sherman–Morrison recursive least squares ($O(m^2)$/step), weight $\beta$ calibrated so the episode-level bonus is 0.1× the random-policy return scale.
- **Metrics:** epiplexity score/ranking vs. Wolfram classes; t-SNE + linear-probe and 5-NN accuracy on MNIST codes; mean task return over 10 seeds at 600k steps.

## Key results

- **ECA ranking:** rule 110 (the only proven Turing-complete elementary rule) ranks first over the whole space, margin 20.7 bits over the runner-up; constant-attractor rules score exactly zero; chaotic rule 30 sits below complex rule 54 ($S^\varphi_{54} - S^\varphi_{30} = 10.9 \pm 1.0$ bits) at the reference observer. Lorenz (46) ≫ linear flows (6.9–8.4) on continuous systems.
- **Inverse NCA:** gradient ascent on the single scalar $S^\varphi$ carries an initially simple rule into a soliton regime — localized structures traveling at fixed velocity that interact on collision (how rule 110 computes) — in 9/9 seeds for both update rules, plateauing at $S^\varphi \approx 86$–89; the learned rule generalizes to wider lattices.
- **MNIST:** with no labels ever, digit clusters emerge in the code; linear probe 0.53 → 0.89, 5-NN 0.66 → 0.89, rising together with epiplexity.
- **RL:** epiplexity bonus improves on the task baseline on 9/10 tasks and collapses on none (worst case: Walker2d, −4%). Headlines: Hopper 1879 → 2192, LunarLander 169 → 208, Acrobot −167±166 → −83±2 and MountainCar 28±43 → 93±1 (unreliable exploration → reliable solving on every seed; PointMaze seed-variance 77 → 22). The magnitude control collapses on Hopper (1879 → 516) and LunarLander (169 → −171, hijacked by a near-constant leg-contact coordinate). Epiplexity-only agents don't solve tasks (on Acrobot they *avoid* the goal, since success ends the episode) — except Hopper/Walker2d, where locomotion itself is the richest trajectory and epiplexity-only *beats* the task reward on Walker2d (327 vs. 296).

## Ablations

- **ECA robustness (App. B):** one-at-a-time scans of $\eta \in [0.03, 30]$, $\lambda \in [0.001, 1]$, $\tau \in \{4..64\}$, depth 1–5, kernel {3,5,7}, channels {64..512}, $N$ {128..1024}: rule 110 stays first everywhere except when the observer or window is cut below the scale of its structure ($\tau{=}4$; depth ≤ 2). Spearman with the reference ranking > 0.90 except at depth ≤ 2. The rule 30 vs. 54 order flips with the observer's receptive-field radius (radius ≤ 2: structured 54 above chaotic 30; radius ≥ 3: reversed) — a concrete demonstration that complexity is observer-relative. Rule 110 is first in both regimes.
- **MNIST (App. G.2):** learning rate, batch size, code dim, reservoir depth all land at 0.80–0.90 probe accuracy; but weak ridge ($\lambda \le 0.3$) or low resolution ($\eta \le 0.1$) drops accuracy below 0.5. A *tight* readout bound ($\lambda{=}3$, far above reservoir-computing norms) is what forces compression toward natural abstractions; enlarging code/reservoir raises the raw score without sharpening class structure.
- **RL reservoir size:** deliberately small (width 32) — a wide reservoir can fit chaos and would reward it; a narrow one can only fit structured trajectories. Input normalization per coordinate is essential (unbounded observations otherwise saturate $\varphi$).

## Limitations

Paper's own (both acknowledged in Discussion): **(1) The observer never grows.** Freezing the reservoir buys the closed form but fixes the boundary of the learnable forever — the estimator sees only shallow structure linearly readable from random features, and the score saturates once a system exhausts its observer. They propose co-evolving observer and observed (suggesting LLMs via in-context learning as a substrate where generator and bounded sequential learner coincide). **(2) In RL the drive is only a bonus** anchored by an external task reward; and nearly all tested environments are finite games, where the surest novelty-preserving strategy is avoiding terminal states — so epiplexity-only agents learn to survive and even defer goals.

[Honest-reader flags — analyst's view]: all testbeds are small (MNIST, 1D CA, classic-control/MuJoCo); no comparison against actual learned-curiosity baselines (ICM/RND) in the RL table, only the magnitude control; the "intelligence" framing is far grander than the evidence; several hyperparameters ($u_Y$, $\tau$, reservoir width, per-env $\beta$) encode nontrivial prior knowledge of the domain, which softens the "no supervision of any kind" claim.

## Why it matters [analyst's view]

This is the cleanest operationalization I've seen of the intuition that intrinsic motivation should target *learnable* surprise, not raw surprise — the same intuition behind learning-progress/compression-progress drives, but here in closed form, differentiable, and with an explicit MDL decomposition explaining both the noisy-TV and dark-room failures as two sides of one conflation. The reservoir trick is the real contribution: it converts epiplexity from a measurement (Finzi et al.'s per-system training run) into a first-class objective you can put a gradient through. For the vault's open-endedness thread, this sits directly between [[papers/lehman-2011-novelty-search]] (which it subsumes: novelty without the hand-designed behavior descriptor) and [[papers/schmidhuber-2013-powerplay]] (the Discussion explicitly argues compression *progress* rebuilds the dark-room problem, while learnable novelty does not, since a dark room's extractable program length is near zero and stays there). Against prediction-error bonuses like [[papers/guo-2022-byol-explore]], the pitch is: no inner model trained alongside exploration, and structural immunity to noise — noise inflates per-step surprise but adds nothing to the learnable part. The frozen-observer limitation is the interesting open edge: the co-evolution proposal is essentially the observer-side version of what open-ended systems like [[papers/momennejad-2026-compositional-open-ended]] want from the environment side.

## Open questions / things to verify

- How does the epiplexity bonus compare head-to-head with ICM/RND on actually noisy environments (noisy-TV gridworlds, Atari with sticky actions)? The immunity claim is argued structurally, not tested against those baselines.
- Does the score's saturation under a frozen observer bite in longer training runs? The NCA plateaus by step ~1,500 — is that the system's ceiling or the observer's?
- The rule 30/54 rank flip with receptive field shows the measure is observer-relative by design; how sensitive are the *inverse* results (which regime the NCA lands in) to observer geometry?
- Read the epiplexity source paper (Finzi et al., arXiv 2601.03220) — _needs note_.
- Musat 2026, "Neural weight norm = Kolmogorov complexity" (arXiv 2605.10878), cited as support for the ridge-as-MDL move — _needs note_.
- Code: https://github.com/Zhangyanbo/learnable-novelty — worth a look; the estimator is small enough to reimplement from this note.

## Connections

- Builds on / subsumes: [[papers/lehman-2011-novelty-search]] — replaces the hand-picked behavior descriptor with observer-relative learnable surprise
- Contrasts with: [[papers/guo-2022-byol-explore]] — prediction-error exploration with a co-trained world model vs. closed-form frozen-observer score; [[papers/schmidhuber-2013-powerplay]] — compression progress, which the paper argues re-admits the dark room
- Adjacent open-endedness: [[papers/momennejad-2026-compositional-open-ended]]
- Topic MOCs: [[topics/open-ended-learning]], [[topics/exploration]], [[topics/representation-learning]], [[topics/predictive-processing]]
- Authors: no vault index yet for Yanbo Zhang or Michael Levin (Levin also co-authored the Growing NCA Distill article this paper builds its NCA testbed on); [[authors/kenneth-stanley]] is tracked via novelty search

## Selected quotes

> "Both failures have a single cause: each objective treats as one quantity the surprise a learner can convert into knowledge and the surprise it never can." — Abstract

> "The two pathologies are mirror images with a common cause: both objectives conflate the total novelty in the data with the structure a bounded mind can absorb." — §1

> "Removing the approximation changes how many bits the observer extracts, not which systems it finds structured; the estimator therefore keeps the ridge readout." — App. D

> "What sets learnable novelty apart is not a higher ceiling on any one task but its stability: it rewards state the bounded observer can compress, never the raw magnitude that aids one environment and ruins another." — §4.3

> "Learnable novelty is the part both objectives needed and neither isolates." — §5
