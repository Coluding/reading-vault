---
type: paper
title: "Latent Thought Flow: Efficient Latent Reasoning in Large Language Models"
authors: ["Xiandong Zou", "Jing Huang", "Jianshu Li", "Pan Zhou"]
year: 2026
venue: "arXiv preprint"
url: https://arxiv.org/abs/2606.16222
rw_id: 01kvq3rm1sca5gwpbhnp0e82rw
topics: [latent-reasoning, reasoning, language-models]
priority: high
read_state: queued
relevance: ""
added: 2026-07-08
last_updated: 2026-07-08
---

## TL;DR

Latent Thought Flow (LTF) makes LLM reasoning happen inside the model's continuous hidden space instead of decoded chain-of-thought tokens, and — unlike prior latent-reasoning methods that learn a single deterministic or reward-maximizing path — it learns a **distribution** over variable-length latent trajectories whose probability is proportional to a reward combining answer quality and computation cost. It does this with a **continuous GFlowNet**: a stochastic sampler trained so terminal trajectories are sampled $\propto$ reward, which preserves diverse high-utility paths rather than collapsing onto one mode (as RL tends to). Two ingredients make it work under sparse answer-only supervision: an **Entropy-Weighted Subtrajectory Balance (EW-SubTB)** objective that propagates the terminal reward to intermediate latent prefixes and up-weights high-entropy subtrajectories, and a **reference-prior regularizer** that anchors early exploration to teacher rationales then anneals away. Only a LoRA adapter and a latent head are trained. Across LLaMA-3.2 1B/3B, LLaMA-3.1 8B, and DeepSeek-R1-Distill-Qwen-1.5B, LTF beats explicit CoT and strong latent baselines (CoLaR, ReGuLaR): the abstract reports **+9.5% accuracy while cutting reasoning length 27.2%** on average; the intro's per-setting numbers are +12.9% acc / -34.5% length (finetuning) and +6.0% / -19.9% (transfer) vs. the strongest baselines.

## Context & motivation

Explicit CoT / Self-Consistency / Tree-of-Thoughts decompose problems into text rationales, but decoding every intermediate thought into tokens creates a "linguistic space bottleneck": high inference overhead and an accuracy–efficiency trade-off. Latent reasoning (Pause tokens, iCoT/stepwise internalization, Compressed CoT, Coconut, CoLaR, ReGuLaR) moves deliberation into continuous hidden states. But the paper argues these methods all learn a *single* deterministic path via compression, distillation, or reward maximization under a fixed budget — they never specify **how probability mass should be allocated across reasoning trajectories of differing correctness and cost**. Maximum-likelihood objectives inherit the verbosity of training rationales; RL causes posterior collapse onto a few high-reward modes. What's missing is a principled sampler that models the correctness-vs-cost trade-off, concentrating mass on concise+accurate trajectories. GFlowNets provide exactly a reward-proportional (not reward-maximizing) sampler; prior GFlowNet-for-reasoning work operated on discrete text/proof/action traces — LTF is the first to bring reward-proportional trajectory learning into the *continuous latent* space.

## Method

### Problem formulation
Training data $\mathcal{D}=\{(x,y,r)\}$: input $x$, target answer $y$, optional reference rationale $r$ (training only). Model $p_\Theta$ with $\Theta=\{\phi,\psi\}$: $p_\phi$ = embedding+decoder layers producing contextual latents, $p_\psi$ = language head. Given $x$, sample a variable-length latent trajectory
$$\tau=(\mathbf{z}_{1:T},\bot)\in\mathcal{T},\quad \mathbf{z}_t\in\mathbb{R}^{d_z},\quad 0\le T\le T_{\max},$$
where $\mathbf{z}_t$ is a continuous thought and $\bot$ is termination ($T=0$ = answer with no reasoning). The target answer defines a utility $\mathcal{R}_{x,y}(\tau)$ inducing the desired posterior $p^*(\tau\mid x,y)\propto\mathcal{R}_{x,y}(\tau)$ (Eq. 1); train the sampler to match it.

### Core idea
Sample a trajectory with high probability iff it solves the problem accurately with minimal computation, low probability if incorrect or redundant — i.e. learn a reward-proportional distribution over variable-length latent paths, not one optimal path.

### Architecture / algorithm
**Sampler (Sec 3.1).** Let $h_x=p_\phi(x)$, state $s_t=(h_x,\mathbf{z}_{1:t})$, $s_0=h_x$. A latent head (3-layer MLP) parameterizes a Gaussian policy:
$$q_\varphi(\mathbf{z}_{t+1}\mid s_t)=\mathcal{N}\!\bigl(\bm\mu_\varphi(s_t),\operatorname{diag}(\bm\sigma^2_\varphi(s_t))\bigr),\quad \pi^\bot(s_t)=p_\psi(\langle\mathrm{eos_r}\rangle\mid s_t), \quad (3)$$
sampled by reparameterization $\mathbf{z}_{t+1}=\bm\mu_\varphi(s_t)+\bm\sigma_\varphi(s_t)\odot\bm\epsilon$, $\bm\epsilon\sim\mathcal{N}(0,I)$, so answer-loss gradients flow through the latents. The stop probability $\pi^\bot(s_t)$ is the decoded end-of-reasoning token probability; forced to 1 at $T_{\max}$. Trajectory density:
$$q_\varphi(\tau\mid x)=\Bigl[\prod_{t=0}^{T-1}(1-\pi^\bot(s_t))\,q_\varphi(\mathbf{z}_{t+1}\mid s_t)\Bigr]\pi^\bot(s_T). \quad (4)$$
The answer is then decoded $p_\psi(y\mid x,\tau)=\prod_m p_\psi(y_m\mid\tau,y_{<m})$ (Eq. 5).

**Reward (Sec 3.2).** Terminal accuracy–efficiency utility:
$$\mathcal{R}_{x,y}(\tau)=V_{x,y}(\tau)\exp(-\lambda_c C(\tau)), \quad (6)$$
with quality
$$V_{x,y}(\tau)=\operatorname{Ver}(y,\hat y_\tau)+\exp\!\bigl(\tfrac{1}{|y|}\log p_\psi(y\mid x,\tau)\bigr), \quad (7)$$
where $\operatorname{Ver}$ is task accuracy (aligns reward with the metric) and the normalized-likelihood term is a dense signal; cost $C(\tau)=T$ (Eq. 8), penalizing length. This induces $p^*(\tau\mid x,y)=\mathcal{R}_{x,y}(\tau)/Z_\mathcal{R}(x,y)$ (Eq. 9); the partition function integrates over continuous trajectories of all lengths, hence intractable — motivating the amortized GFlowNet sampler.

### Derivations / why it works
**Continuous GFlowNet (Sec 3.3).** Flow-balance for a subtrajectory $s_i\to\cdots\to s_j$:
$$F(s_i)\prod_{t=i}^{j-1}P_F(s_{t+1}\mid s_t)=F(s_j)\prod_{t=i}^{j-1}P_B(s_t\mid s_{t+1}). \quad (10)$$
Because the state $s_t=(h_x,\mathbf{z}_{1:t})$ stores the full prefix, each state has a **unique parent** (drop the last latent), so the backward transition is deterministic and contributes zero log-density. Forward density to a non-terminal state:
$$P_F(s_{t+1}\mid s_t)=(1-\pi^\bot(s_t))\,q_\varphi(\mathbf{z}_{t+1}\mid s_t), \quad (11)$$
giving forward edge log-density $\ell_t^\varphi=\log q_\varphi(\mathbf{z}_{t+1}\mid s_t)+\log(1-\pi^\bot(s_t))$ (Eq. 12).

Key trick: allow **every prefix to terminate**, so the flow of an intermediate prefix is computed analytically from its immediate-stop reward — no separate learned flow network:
$$F(s_t)\pi^\bot(s_t)=\mathcal{R}_{x,y}(s_t\to\bot)\;\Rightarrow\; F(s_t)=\frac{\mathcal{R}_{x,y}(s_t\to\bot)}{\pi^\bot(s_t)}. \quad (13)$$

**Subtrajectory Balance residual** (Eq. 14), substituting Eq. 13:
$$\chi_{i:j}=\log\frac{\mathcal{R}_{x,y}(s_i\to\bot)\bigl[\prod_{t=i}^{j-1}q_\varphi(\mathbf{z}_{t+1}\mid s_t)(1-\pi^\bot(s_t))\bigr]\pi^\bot(s_j)}{\mathcal{R}_{x,y}(s_j\to\bot)\pi^\bot(s_i)},$$
with loss $\mathcal{L}_{\mathrm{SubTB}}=\mathbb{E}_\tau[\sum_{0\le i<j\le T}\chi_{i:j}^2]$ (Eq. 15). Minimizing it makes the sampler put more density on prefixes/completions with higher answer reward while keeping multiple valid paths.

**Entropy-Weighted SubTB (EW-SubTB).** Treating all subtrajectories equally is suboptimal — high-entropy paths carry richer, more spread-out information and need stronger supervision. Define per-step entropy $h_t^{(s)}=\mathcal{H}[q_\varphi(\mathbf{z}^{(s)}_{t+1}\mid s^{(s)}_t)]$, length-normalized $\bar h^{(s)}_{i:j}$, and a stop-gradient softmax-style weight
$$\omega_{i:j}^{(s)}=\operatorname{sg}\!\Bigl[\exp(\bar h^{(s)}_{i:j})\big/\bigl(\tfrac{1}{|\mathcal{S}_{i:j}|}\sum_{r}\exp(\bar h^{(r)}_{i:j})\bigr)\Bigr] \quad (16)$$
over $S$ sampled trajectories, giving $\mathcal{L}_{\mathrm{flow}}=\tfrac1S\sum_s\sum_{i<j}\omega^{(s)}_{i:j}(\chi^{(s)}_{i:j})^2$ (Eq. 18). Because $\omega$ only reweights the squared residual (not the balance ratio), it **preserves the reward-proportional target** while changing optimization emphasis.

**Reference-prior regularization (Sec 3.4).** Early exploration in high-dim continuous space drifts to meaningless states. A reference branch $p^{\mathrm{ref}}_{\theta'}(\mathbf{z}_t\mid s_{t-1})=\mathcal{N}(\bm\mu_{\theta'},\operatorname{diag}\bm\sigma^2_{\theta'})$ (Eq. 19) is anchored to teacher rationales via $\mathcal{L}_{\mathrm{prior}}=-\mathbb{E}_{(x,r)}[\log p^{\mathrm{ref}}_{\theta'}(r\mid x)]$ (Eq. 20). Omitted if no rationale.

**Final objective** (Eq. 22): $\mathcal{L}=\mathcal{L}_{\mathrm{flow}}+\lambda_{\mathrm{ans}}\mathcal{L}_{\mathrm{ans}}+\lambda_{\mathrm{prior}}\mathcal{L}_{\mathrm{prior}}$, with answer loss $\mathcal{L}_{\mathrm{ans}}=-\mathbb{E}_\tau[\log p_\psi(y\mid\tau)]$ (Eq. 21). Gradients are **stopped through the scalar reward** in the flow loss so the decoder can't cheat by rescaling reward.

### Training procedure
LoRA rank $r=128$, $\alpha=32$; AdamW, lr $1\times10^{-4}$, weight decay $1\times10^{-2}$, global batch 256, 100 epochs on RTX PRO 6000 GPUs. Only LoRA + latent head trainable. Rollout samples $S$ = rollout_n = 20, $T_{\max}=32$. Reward cost weight $\lambda_c=0.03$ (stable in $(0.01,0.04)$); $\lambda_{\mathrm{ans}}=1.0$; $\lambda_{\mathrm{prior}}$ annealed piecewise-linearly $3.0\to0.1$ over 100 epochs (updated every 5). Reference $r$ = embedding of the golden explicit rationale. Best-validation-accuracy checkpoint selected.

### Inference / sampling
Given $x$, sample latent thoughts via Eq. 3 and decode the answer (no gold rationale, no explicit CoT). Default = single trajectory. Test-time scaling: sample $N$ latent chains (tree of latent thoughts, Monte-Carlo) — cheap because chains are short. Generation temperature 0.9, top-$p$ 0.95.

## Experimental setup

- **Backbones:** LLaMA-3.2 1B/3B Instruct, LLaMA-3.1 8B Instruct, DeepSeek-R1-Distill-Qwen-1.5B.
- **Finetuning datasets:** GSM8K-Aug, ASDiv-Aug, DU (math, word problems, data understanding).
- **Transfer (OOD):** GSM-Hard, SVAMP, MultiArith (main tables); also AQUA-RAT, MATH for extreme compression.
- **Metrics:** Accuracy (%), Reasoning Length #L (# reasoning steps; 0 for one-pass iCoT). Latent step ≈ one explicit decode step in compute, so #L is a fair efficiency proxy. Averaged over 5 seeds.
- **Baselines:** explicit — CoT, Assist-CoT, SoftCoT++; latent — iCoT, CODI, Coconut, SoftCoT++, CoLaR, ReGuLaR. Ablation exploration objectives — GRPO (RL), Detailed Balance, Trajectory Balance.

## Key results

- **Finetuning (Table 1):** LTF best accuracy–efficiency on all backbones. LLaMA-1B avg 55.56 (ReGuLaR) → **59.68**; on GSM8K-Aug 34.58→37.09 acc with length 3.69→3.34. LLaMA-8B 74.40→77.08 avg. On ASDiv-Aug 92.00→94.02. DeepSeek-1.5B 58.06→62.16.
- **Extreme compression (Table 2):** on MATH, LTF +2.72% over ReGuLaR at #L=1.00; AQUA-RAT +3.61% at #L=1.00 (CoLaR needs 20–60 steps to reach lower accuracy).
- **Transfer (Table 3):** highest avg accuracy on all four backbones, +1.89 / +1.84 / +2.11 / +2.38% over ReGuLaR, with shorter length. E.g. LLaMA-1B avg 48.47→50.36; SVAMP 48.19→52.01.
- Headline abstract numbers: **+9.5% acc, −27.2% length** avg vs. strong latent baselines.

## Ablations

- **Entropy weighting (Table 4):** consistent accuracy gains that grow with $S$ (+0.40 / +0.67 / +0.95% for $S=5/10/20$); little length change. Larger $S$ helps: avg 57.41→59.68 as $S$: 5→20.
- **Exploration objective (Table 5):** GFlowNet >> RL. GRPO 47.49% avg at length 12.25; DB 55.98 @ 7.28; TB 56.80 @ 7.51; **LTF 59.68 @ 1.91** — best accuracy *and* far shorter.
- **Reference prior (Table 8):** RPR mainly shortens reasoning (avg length 2.75→1.91) with a small accuracy gain (59.32→59.68); w/o RPR slightly wins GSM8K-Aug (37.21 vs 37.09), showing latent-space local optima.
- **Test-time scaling (Table 9):** $N$ latent chains 1→10 lifts avg accuracy 59.68→62.13 (+2.45) with length essentially flat (1.91→1.93); most gain by $N=5$.
- **Reasoning entropy (Table 10):** CoLaR 0.013 (near-collapse), ReGuLaR 0.019, LTF-w/o-EW 0.030 (over-stochastic), **LTF 0.024** — EW regulates entropy into an "effective regime," neither collapsed nor too random.

## Limitations

Paper's own: experiments are textual (math/symbolic) only — vision/speech left to future work; theoretical generalization analysis is incomplete. Honest-reader flags: gains over ReGuLaR are modest in absolute terms (often 2–3%); requires teacher rationales for the prior to work best; the $S=20$ rollout and all-pairs SubTB sum ($O(T^2)$ residuals) add training cost; reward uses a task verifier ($\operatorname{Ver}$), so it assumes a checkable answer.

## Why it matters [analyst's view]

LTF's real contribution is conceptual: reframing efficient latent reasoning as **distribution matching over variable-length trajectories** rather than path optimization, and showing GFlowNets port cleanly into continuous latent space via the unique-parent trick that makes intermediate flows analytic (Eq. 13). The GRPO-vs-GFlowNet ablation is the most striking result — RL collapses to long, lower-accuracy trajectories while the flow objective yields both higher accuracy and ~6× shorter reasoning, a concrete case where reward-proportional beats reward-maximizing. The entropy analysis (an "effective entropy regime" between collapse and noise) is a nice diagnostic that could generalize beyond this paper. It sits at the intersection of latent-reasoning, inference-time-scaling (test-time latent-chain sampling), and GFlowNets/flow-matching-style distribution matching, and is a natural companion to CoLaR/ReGuLaR in a latent-reasoning MOC.

## Open questions / things to verify

- How much of the win depends on the teacher-rationale prior vs. the GFlowNet objective in the no-rationale regime?
- Does the $O(T^2)$ EW-SubTB scale to longer latent budgets ($T_{\max}\gg32$) or harder reasoning where more steps are genuinely needed?
- The abstract (+9.5% / −27.2%) vs. intro (+12.9% / −34.5% finetune) numbers differ — which averaging set each refers to.
- Whether "reasoning length" (#L) is truly compute-comparable across latent vs. explicit methods, as claimed.

## Connections

- Topic MOCs: [[topics/latent-reasoning]], [[topics/reasoning]], [[topics/language-models]]
- Author indices: [[authors/xiandong-zou]], [[authors/pan-zhou]]
- Related: extends latent-reasoning baselines Coconut, CoLaR, ReGuLaR; builds on GFlowNet theory (Bengio et al.; continuous GFlowNets, Lahlou et al. 2023) and Subtrajectory/Trajectory Balance credit assignment; contrasts with RL-based latent reasoning (GRPO, LEPO) that maximizes reward.

## Selected quotes

> "existing paradigms often fail to optimize the latent reasoning manifold: maximum-likelihood objectives inherit the verbosity of training rationales, while reinforcement learning tends to trigger posterior collapse onto a few high-reward modes." — §1

> "a trajectory should be sampled with high probability when it solves the problem accurately using minimal computation, and with low probability when it is incorrect or redundant." — §3

> "the flow of an intermediate prefix does not need to be learned by a separate network; it can be computed from the reward obtained by immediately stopping at that prefix." — §3.3
