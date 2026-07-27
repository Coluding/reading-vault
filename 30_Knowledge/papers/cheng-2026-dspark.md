---
type: paper
title: "DSpark: Confidence-Scheduled Speculative Decoding with Semi-Autoregressive Generation"
authors: ["Xin Cheng", "Xingkai Yu", "Chenze Shao", "Jiashi Li", "Yunfan Xiong", "Yi Qian", "Jiaqi Zhu", "Shirong Ma", "Xiaokang Zhang", "Jiasheng Ye", "Qinyu Chen", "Chengqi Deng", "Jiping Yu", "Damai Dai", "Zhengyan Zhang", "Yixuan Wei", "Yixuan Tan", "Wenkai Yang", "Runxin Xu", "Yu Wu", "Zhean Xu", "Xuanyu Wang", "Muyang Chen", "Rui Tian", "Xiao Bi", "Zhewen Hao", "Shaoyuan Chen", "Huanqi Cao", "Wentao Zhang", "Anyi Xu", "Huishuai Zhang", "Dongyan Zhao", "Wenfeng Liang"]
year: 2026
venue: arXiv preprint (DeepSeek-AI + Peking University)
url: https://arxiv.org/abs/2607.05147
rw_id: 01kyhj32x1szpnp992g5216gf2
topics: [llm-inference, gpu-optimization, language-models]
priority: high
read_state: queued
relevance: ""
added: 2026-07-27
last_updated: 2026-07-27
---

# DSpark: Confidence-Scheduled Speculative Decoding with Semi-Autoregressive Generation

## TL;DR

DSpark is DeepSeek's production speculative-decoding framework, built on the observation that the two dominant drafter families each fail in a different way: autoregressive drafters (Eagle3) pay $T_{\text{draft}} \propto \gamma$ so they must stay shallow, while parallel drafters (DFlash) collapse drafting to one forward pass but predict every block position independently, so acceptance decays badly along the block. DSpark's first contribution is a **semi-autoregressive** drafter — keep the heavy parallel backbone, then bolt a *lightweight* sequential head (a rank-256 Markov transition bias, or an RNN variant) on top that conditions each position on the token actually sampled at the previous position. Its second contribution is **confidence-scheduled verification**: a calibrated confidence head predicts per-position *conditional* survival probability, and a hardware-aware scheduler turns "how many tokens should each request verify?" into a global throughput-maximisation problem solved greedily against a profiled `SPS(B)` engine-capacity table. Offline, DSpark beats Eagle3 by **30.9% / 26.7% / 30.0%** macro-average accepted length on Qwen3-4B/8B/14B and DFlash by **16.3% / 18.4% / 18.3%**, while adding only **0.2–1.3%** to per-round latency. Deployed in DeepSeek-V4's live serving stack it replaced the MTP-1 baseline two weeks after the V4-preview release, delivering **60–85%** faster per-user generation (V4-Flash) at matched aggregate throughput. Checkpoints and a training repo (**DeepSpec**) are open-sourced.

## Context & motivation

Speculative decoding (Chen et al., 2023; Leviathan et al., 2023) accelerates LLM inference by splitting generation into a cheap **draft model** $q$ that proposes a block of $\gamma$ candidate tokens and an expensive **target model** $p$ that verifies the whole block in one forward pass via rejection sampling — accepting the longest prefix consistent with $p$ and appending one bonus token. Because the acceptance rule provably preserves the target distribution, the speedup is *lossless*.

The paper frames the whole design space through one equation. Let $\tau$ be the number of tokens accepted per cycle and $T_{\text{draft}}, T_{\text{verify}}$ the wall-clock costs of the two passes:

$$\mathcal{L} = \frac{T_{\text{draft}} + T_{\text{verify}}}{\tau} \tag{1}$$

$\mathcal{L}$ is the average latency per *generated* token. Three levers follow: draft faster (lower $T_{\text{draft}}$), draft better (raise $\tau$), or verify smarter (lower the effective $T_{\text{verify}}$). Existing work picks one:

- **Autoregressive drafters** (Eagle3, DeepSeek MTP) condition position $k$ on tokens sampled at $<k$, giving strong $\tau$ — but $T_{\text{draft}} \propto \gamma$ forces small $\gamma$ and shallow (often 1-layer) networks. Tree verification recovers some $\tau$ at the cost of many more verification tokens, which hurts serving throughput.
- **Parallel drafters** (Medusa, PARD, DART, DFlash) emit all $\gamma$ logits in one pass, so $T_{\text{draft}}$ is nearly independent of $\gamma$ and the drafter can be deep. But each position marginalises over all possible predecessors rather than conditioning on the one actually sampled — the classic **multi-modal collision** of non-autoregressive translation (Gu et al., 2018). Given contexts admitting both "of course" and "no problem", the drafter happily emits "of problem". Acceptance therefore decays along the block.

The second, less-studied bottleneck is *how much to verify*. Verifying more tokens is nearly free under light load but, under high concurrency, every low-confidence token occupies target-model batch capacity that another active request could use. And the right length varies by **data** too — code sustains far higher acceptance than open-ended chat. Static block lengths cannot straddle both axes, which is exactly why DeepSeek's production system had been stuck on the conservative single-token **MTP-1**: a static MTP-3/5 drafter *strictly degrades* aggregate throughput under high concurrency.

DSpark attacks both bottlenecks: semi-autoregressive generation for $\tau$, confidence-scheduled verification for $T_{\text{verify}}$.

## Method

### Problem formulation

Given a frozen target LM $p$ and an anchor token $x_0$ (the bonus token the target produced in the previous cycle), produce a length-$\gamma$ draft block $X = (x_1,\dots,x_\gamma)$ together with **exact per-token probabilities** $p_k^d(\cdot)$, plus per-position confidence scores $c_k$ used to decide how much of the block to send for verification. The exactness requirement is load-bearing: rejection sampling accepts $x_k$ with probability $\min(1, p_k^t(x_k)/p_k^d(x_k))$, so a drafter that cannot report a normalised per-token probability cannot be used losslessly.

### Core idea

Put the compute where it parallelises and the *dependency* where it's cheap: a deep parallel backbone supplies base logits for every position in one pass (high capacity at position 1, where a rejection is most costly), and a tiny sequential module adds a prefix-dependent **bias** to those logits so later positions stop colliding across modes. Then only verify the prefix whose calibrated survival probability justifies the batch capacity it consumes.

### Architecture / algorithm

**Parallel stage.** The backbone is DFlash (Chen et al., 2026), which conditions the drafter on the target's internal state via *KV injection*. During prefill, hidden states from $m$ chosen target layers $\{l_1,\dots,l_m\}$ are concatenated and projected into the draft hidden space:

$$H_{\text{ctx}} = \text{RMSNorm}\big(W_c[H^{(l_1)}; \dots; H^{(l_m)}]\big) \tag{2}$$

where $W_c$ is a single shared projection matrix mapping the concatenated $m \cdot d_{\text{target}}$ features down to the draft width. RMSNorm is there to keep the injected features on the same scale as the draft activations — without it the injected context would dominate or vanish relative to the draft block's own representations. These context features are then spliced into **every** draft layer along the sequence dimension of keys and values:

$$K_i = [W_i^K H_{\text{ctx}};\, W_i^K H_d], \qquad V_i = [W_i^V H_{\text{ctx}};\, W_i^V H_d] \tag{3}$$

with $H_d$ the draft block's own hidden states at layer $i$. The point of concatenating rather than adding is that draft positions attend *bidirectionally* to each other **and** to the frozen target context — the drafter never has to re-derive context the target already computed, which is what lets it be a small model without losing much. The drafter shares (and freezes) the target's embedding table and LM head, so its logits live in the target's vocabulary space for free.

DSpark makes one change to DFlash: instead of feeding anchor + $\gamma$ mask tokens and predicting only the mask positions, it treats **the anchor itself as the first prediction position**, so $\gamma$ input tokens (anchor + $\gamma-1$ masks) yield $\gamma$ draft logits. This shaves one position of draft compute at similar quality.

**Sequential stage.** The sequential module supplies a prefix-dependent transition bias $B_k(x_0, x_{<k}, \cdot)$ that is *added to the backbone logits before the softmax*. Crucially it is **not** a globally normalised energy model — it induces a causal block distribution by ordinary autoregressive factorisation:

$$P(X \mid x_0) = \prod_{k=1}^{\gamma} p_k(x_k \mid x_0, x_{<k}), \qquad p_k(\nu \mid x_0, x_{<k}) = \frac{\exp\big(U_k(\nu) + B_k(x_0, x_{<k}, \nu)\big)}{\sum_{u \in \mathcal{V}} \exp\big(U_k(u) + B_k(x_0, x_{<k}, u)\big)} \tag{4}$$

Here $U_k \in \mathbb{R}^{|\mathcal{V}|}$ is the base logit vector the parallel backbone produced at position $k$, $\mathcal{V}$ is the vocabulary, and $x_0$ is the anchor. **Why local normalisation matters:** each $p_k$ is a plain softmax over the vocabulary, so an exact per-token probability is available in $O(|\mathcal{V}|)$ — which is precisely what the rejection-sampling rule needs. This is the paper's stated differentiator from CRF-NAT (Sun et al., 2019), which also stacks a sequential module over parallel hidden states but whose *global* partition function makes exact per-token probabilities intractable, and from CTC-drafter (Wen et al., 2024), whose latent alignment marginalisation restricts it to greedy verification.

At inference the sequential block samples left to right, so it must satisfy $T_{\text{sequential}} \ll T_{\text{parallel}}$ or the whole point is lost. Two instantiations:

- **Markov head (default).** Restrict $B_k$ to depend only on the immediately preceding token: $B(x_{k-1}, \cdot)$. In principle this is a full $|\mathcal{V}| \times |\mathcal{V}|$ transition matrix — impossible to store for $|\mathcal{V}| \approx 10^5$ — so it is factorised low-rank as $W_1 W_2$ with $W_1 \in \mathbb{R}^{|\mathcal{V}| \times r}$, $W_2 \in \mathbb{R}^{r \times |\mathcal{V}|}$:

  $$B(x_{k-1}, \cdot) = W_1[x_{k-1}]\, W_2 \in \mathbb{R}^{|\mathcal{V}|} \tag{5}$$

  $W_1$ acts as an embedding lookup (one row read, no matmul) and $W_2$ as a logit projection; with $r=256$ by default, both storage and per-step compute stay tiny even at large vocabulary. Back to the example: once position 1 samples "of", the Markov head boosts "course" and suppresses "problem" at position 2.

- **RNN head.** The Markov head is memoryless beyond one step. The RNN head keeps a recurrent state $s_k \in \mathbb{R}^{d_s}$ accumulating the whole within-block prefix. At step $k$ it concatenates the previous state, the previous token's Markov embedding, and the backbone hidden state — $z_k = [s_{k-1};\, W_1[x_{k-1}];\, h_k] \in \mathbb{R}^{2d_s + d}$ — and applies one gated update:

  $$s_k = \sigma(W_g z_k) \odot s_{k-1} + \big(1 - \sigma(W_g z_k)\big) \odot \tanh(W_c z_k), \qquad B_k(x_{<k}, \cdot) = W_2^\top \tanh(W_o z_k) \tag{6}$$

  $W_g, W_c, W_o$ are slices of a single fused linear projection (gate / candidate / output), and $s_0 = 0$. The convex gate is a GRU-style leak: $\sigma(W_g z_k)$ decides per-dimension how much history to retain vs. overwrite. Note $W_2$ is *reused* from the Markov head to map back to vocabulary space, keeping the output projection cheap.

### Confidence head

For each draft position the confidence head emits a scalar $c_k \in (0,1)$ modelling the **conditional** probability that $x_k$ survives verification *given that all of $x_1,\dots,x_{k-1}$ were accepted*. It is a single linear projection plus sigmoid:

$$c_k = \sigma\big(w^\top [h_k;\, W_1[x_{k-1}]]\big) \tag{7}$$

where $h_k$ is the backbone hidden state and $W_1[x_{k-1}]$ the Markov embedding of the previously sampled token — so the confidence estimate sees both context and the realised prefix. Supervision is the **analytical** per-step acceptance rate, which for the standard rejection rule equals one minus the total variation distance between draft and target distributions:

$$c_k^* = 1 - \tfrac{1}{2}\,\|p_k^d - p_k^t\|_1 \tag{8}$$

This is a soft label available in closed form during training, so no rollout is needed.

**Sequential Temperature Scaling (STS).** Threshold-based heuristics only need confidences to *rank* correctly; DSpark's scheduler needs their **absolute magnitudes**, because it computes an expected accepted-token count from them. Neural confidences are systematically overconfident (Guo et al., 2017), which would corrupt the throughput estimate. Since each $c_k$ is conditional, the chain rule makes the joint prefix-survival probability the cumulative product $\prod_{j \le k} c_j$. STS calibrates that product left to right: at each position $k$, a 1-D grid search finds the temperature scalar minimising Expected Calibration Error of the cumulative product, holding already-calibrated earlier positions fixed. Temperature scaling is order-preserving, so it fixes the probabilities without disturbing the ranking the head learned.

### Hardware-aware prefix scheduler

For a batch of $N$ active requests with per-position confidences $c_{i,k}$, define the prefix survival probability $s_{i,k} = \prod_{j \le k} c_{i,j}$. If request $i$ is scheduled a verification length $\ell_i \in \{0,\dots,\gamma\}$, then the total token batch sent to the target is $B = \sum_{i=1}^{N} (1+\ell_i)$ and the expected number of accepted tokens is $A = \sum_{i=1}^{N}\big(1 + \sum_{k \le \ell_i} s_{i,k}\big)$ (the $1+$ is the guaranteed bonus token). Letting $\text{SPS}(B)$ be the engine's profiled steps-per-second at forward-pass batch size $B$ — measured once at engine init and stored as a lightweight cost table — the scheduler maximises expected system token throughput:

$$\Theta = A \cdot \text{SPS}(B)$$

**Why a greedy solution is exact here.** Since $s_{i,k}$ is monotonically non-increasing in $k$, the marginal gain from extending request $i$ from $\ell_i = k-1$ to $k$ is exactly $s_{i,k}$. That monotonicity means globally sorting all candidate extensions $(i,k)$ by $s_{i,k}$ automatically respects intra-block prefix dependencies — you can never admit position $k$ before position $k-1$ of the same request. So for *fixed* $B$, greedy admission by survival probability is optimal; the algorithm then walks that admission path, recomputing $\Theta$ from the cost table at each step and keeping the best.

**The causality subtlety (Appendix A).** Losslessness requires the **non-anticipating property**: the decision to admit position $k$ must not depend on the realisation of $x_k$. But the confidence head is Markov — computing $s_{i,k+1}$ *requires* having sampled $x_{i,k}$. A retrospective global search therefore leaks future tokens into past admission decisions. The paper's counterexample: one request, $\gamma=2$, $c_1 = 0.8$, capacity curve $\text{SPS}(1)=1.0$, $\text{SPS}(2)=0.5$, $\text{SPS}(3)=0.45$. Then $\Theta_0 = 1 \cdot 1.0 = 1.0$ and $\Theta_1 = (1+0.8)\cdot 0.5 = 0.9$. If the realised $x_1$ yields $c_2 = 0.9$, then $s_2 = 0.72$ and $\Theta_2 = (1+0.8+0.72)\cdot 0.45 = 1.134$ — the global max, so $\ell = 2$ and $x_1$ *is* verified. If instead $x_1$ yields $c_2 = 0$, then $\Theta_2 = (1+0.8+0)\cdot 0.45 = 0.81$, the max stays $\Theta_0 = 1.0$, and $\ell = 0$ — $x_1$ is *not* verified. Admission of $x_1$ thus depends on $x_1$ itself. With $p_t = (0.7, 0.3)$ and $p_d = (0.5, 0.5)$ over $\{A,B\}$ the induced output distribution works out to $\Pr(Y{=}A) = 0.5 + 0.5 \times 0.7 = 0.85$ vs. the target's $0.7$ — demonstrably not lossless.

The fix in Algorithm 1 is the **early-stopping break**: halt the greedy walk the moment $\Theta$ fails to improve. In the counterexample $\Theta_1 < \Theta_0$, so the scheduler returns $\ell=0$ *before ever evaluating* $c_2$, and the decision rests only on pre-token information. Caveat the paper states plainly: stepwise early-stopping attains the global throughput maximum **iff** $\Theta$ is unimodal, i.e. iff the hardware capacity curve decays smoothly — which real hardware does not do (see §5.2 below).

### Training procedure

Random anchor positions are sampled from each target sequence to form $\gamma$-token training blocks. The **target model is frozen**; the drafter shares and freezes its embedding layer and LM head, updating only the backbone drafter, the sequential block, and the confidence head. Three losses, all position-weighted by $w_k = \exp(-(k-1)/\lambda)$ (from DFlash) to emphasise early positions — justified because under prefix-based verification an early rejection kills the whole block, so early positions carry disproportionate leverage on expected accepted length:

$$\mathcal{L}_{ce} = -\sum_{k=1}^{\gamma} w_k \log p_k^d(x_k^*) \tag{9}$$

standard next-token cross-entropy against the ground-truth token $x_k^*$;

$$\mathcal{L}_{tv} = \sum_{k=1}^{\gamma} w_k \|p_k^d - p_k^t\|_1 \tag{10}$$

a distribution-matching term. This is the term that actually matters: since per-step acceptance probability equals $1 - \frac{1}{2}\|p^d - p^t\|_1$ (Leviathan et al., 2023), minimising $\mathcal{L}_{tv}$ *directly maximises expected acceptance rate*, whereas cross-entropy only maximises agreement with the ground truth — which is not the same objective when the target model itself is wrong;

$$\mathcal{L}_{\text{conf}} = -\sum_{k=1}^{\gamma} w_k \big[c_k^* \log c_k + (1-c_k^*)\log(1-c_k)\big] \tag{11}$$

binary cross-entropy fitting the confidence head to the soft label of Eq. 8. Total:

$$\mathcal{L} = \alpha_{ce}\mathcal{L}_{ce} + \alpha_{tv}\mathcal{L}_{tv} + \alpha_{\text{conf}}\mathcal{L}_{\text{conf}} \tag{12}$$

with defaults $\alpha_{ce}=0.1$, $\alpha_{tv}=0.9$, $\alpha_{\text{conf}}=1.0$ — note the 9:1 tilt toward TV over CE, consistent with the argument above.

Data: **Open-PerfectBlend**, 1.3M samples (math 39.4%, code 38.9%, chat 17.6%, instruction-following 4.1%). Only the *prompts* are used; responses are regenerated by each target model with its recommended sampling parameters, so the drafter learns the target's actual output distribution. 10 epochs, non-thinking mode. _Optimizer, learning rate, and schedule are not stated in the fetched text._

### Inference / sampling

Standard chain-based (not tree) speculative decoding at temperature 1.0. Per cycle: target emits anchor $x_0$ → parallel backbone one forward pass → sequential head samples $x_1 \ldots x_\gamma$ left to right while the confidence head scores them → scheduler truncates to $\ell$ → target verifies the prefix in parallel → accepted prefix + one corrected/bonus token committed.

## Experimental setup

- **Targets**: Qwen3-4B / 8B / 14B, plus Gemma4-12B for cross-family generalisation.
- **Baselines**: **Eagle3** (autoregressive, Training-Time Test) and **DFlash** (SOTA parallel). All drafters retrained in the same framework on the same data for fairness; Eagle3's TTT horizon aligned to the block size (7); same target feature layers for all. Depth: 1 layer for Eagle3 (its latency budget forces this), 5 layers for DFlash and DSpark.
- **Benchmarks**: math (GSM8K, MATH500, AIME25), code (MBPP, HumanEval, LiveCodeBench), chat (MT-Bench, Alpaca, Arena-Hard).
- **Metric**: average accepted length $\tau$ per decoding round, inclusive of the target-generated bonus token. Confidence scheduler **disabled** offline, so Table 1 isolates raw draft quality.

## Key results

**Offline (Table 1, Qwen3-4B row):** DSpark 6.11 / 5.70 / 4.89 on GSM8K/MATH/AIME25, 5.13 / 5.38 / 4.86 on MBPP/HumanEval/LCB, 3.64 / 3.54 / 3.29 on MT-Bench/Alpaca/Arena-Hard — vs DFlash 5.40 / 4.85 / 4.15, 4.40 / 4.74 / 4.18, 3.07 / 2.96 / 2.83 and Eagle3 5.14 / 4.62 / 3.92, 3.69 / 4.16 / 3.77, 2.39 / 2.26 / 2.55. Macro-average: **+30.9% / +26.7% / +30.0% over Eagle3** and **+16.3% / +18.4% / +18.3% over DFlash** at 4B/8B/14B, with the advantage holding on Gemma4-12B.

**The domain effect that motivates scheduling:** DSpark on Qwen3-4B averages 5.57 (math) and 5.12 (code) but only 3.49 (chat). A static verification length cannot serve both regimes.

**Why parallel beats autoregressive at all** (§4.3.1) — the paper's most interesting analysis. Using *position-wise conditional acceptance* (denominator counts only rollouts where positions $1..k-1$ were all accepted, so position $k$ isn't penalised for earlier failures), the two families invert:

- **Position 1**: DFlash 0.88 vs Eagle3 0.81 on math; **0.72 vs 0.53** on chat. Both predict from target context alone here, so the gap is pure architectural capacity — Eagle3's $O(\gamma)$ latency confines it to 1 layer, DFlash's $O(1)$ affords 5. Since prefix survival is multiplicative, position 1 carries the most leverage.
- **Positions 2–7**: the ordering flips. Eagle3 *rises* (0.53 → 0.74 on chat) as the prefix disambiguates the continuation; DFlash decays (0.87 → 0.78 on code, 0.72 → 0.63 on chat).
- **DSpark** starts at DFlash-level (0.93 on math) *and* stays flat — it gets the capacity advantage at position 1 and the conditioning advantage after.

**Online (§5.4), DSpark-5 vs MTP-1 under live DeepSeek-V4 traffic:** on V4-Flash, +51% aggregate throughput at the 80 tok/s/user SLA; on V4-Pro, +52% at 35 tok/s/user. At matched throughput levels, **60–85%** (Flash) and **57–78%** (Pro) faster per-user generation. At the strict SLAs (120 Flash / 50 Pro) MTP-1 collapses into a tiny-batch regime and the reported ratios balloon to 661% / 406% — the paper is unusually candid that these should be read as *"DSpark extends the feasible interactivity frontier"* rather than as a representative speedup over a well-utilised baseline.

**Mechanism (Figure 8):** below ~200 concurrent requests (Flash) / ~150 (Pro) the scheduler expands the verification budget from MTP-1's static 2 tokens to roughly **4–6 tokens/request**; as concurrency saturates target capacity it smoothly contracts again. That adaptivity is exactly what MTP-3/5 could not do and why DeepSeek had stayed on MTP-1.

## Ablations

- **Depth** (§4.3.2, $\gamma=7$ fixed, 1→5 layers): monotone improvement, steepest from 1→2. A **2-layer DSpark beats the 5-layer DFlash** across all domains — injecting local autoregression is a better use of parameters than stacking more parallel depth.
- **Proposal length** (5 layers fixed, draft length $\in \{4,8,12,16\}$): the DSpark–DFlash gap *widens* with $\gamma$, since DFlash's marginal utility decays. At $\gamma=7$: +16% math / +15% code / +18% chat; at $\gamma=15$: **+30% / +26% / +22%**.
- **Markov vs RNN head**: RNN gives only marginal extra gains, mostly at long $\gamma$. Given higher implementation complexity and worse deployment properties, **Markov is the default** — a nice negative result.
- **Latency overhead**: at batch 128, averaged over context lengths {512, 1024, 2048, 4096}, scaling draft length 4→16 adds only **0.2%–1.3%** to full-round latency vs DFlash, while buying up to 30% more accepted length. The target's verification pass dominates, so the serial loop hides.
- **Confidence-head threshold sweep** (Qwen3-4B): raising the threshold lifts overall acceptance from **45.7% → 95.7%** on chat, **76.9% → 92.5%** on math, **67.6% → 92.0%** on code. Pruning bites hardest exactly where entropy is highest.
- **Calibration**: raw head has strong discrimination (ROC-AUC **0.81–0.90**) but is overconfident (**ECE 3–8%**); STS drops average ECE to **~1%**.

## Production engineering (§5)

Worth capturing separately because it's the part most papers omit:

- **Deployment config**: backbone = 3 MoE layers with mHC and sliding-window attention of 128; $\gamma=5$; Markov head; confidence head trained end-to-end then STS-calibrated.
- **Training scale-out (HAI-LLM)**: (a) *hidden-state communication* — instead of shipping full-vocabulary target logits ($|\mathcal{V}| \approx 10^5$) across workers, cache target activations and communicate only the pre-LM-head hidden states, running the LM head locally on the sampled positions; per-token communication drops from $O(|\mathcal{V}|)$ to $O(d)$. (b) *anchor-bounded sequence packing* — sample a fixed number of anchors per sequence and pack isolated prediction blocks densely, using token-level attention indices rather than 2-D masks, which preserves exact causal masking across packed sequences without padding overhead.
- **Async scheduling**: Algorithm 1 conflicts with production infra twice — real `SPS(B)` is jagged and step-wise, not smooth/unimodal; and Zero-Overhead Scheduling plus CUDA-graph replay require the next step's batch size *before* the current step finishes. The fix: approximate the capacity limit using confidence outputs **from two steps prior** (candidates in the current step are still sorted by up-to-date scores; only the truncation length $K$ comes from history), turning admission into a dynamic top-$K$ selection. This also lets them **remove the early-stopping break** and do an unconstrained global search across the jagged SPS cliffs — the two-step lag is itself a causal barrier, so losslessness survives without early stopping. Elegant: the systems compromise buys back the algorithmic guarantee it appeared to threaten.
- **Variable-length kernels**: dynamic per-request verification lengths break fixed-query-length decode kernels. They flatten all tokens across requests as independent elements and convey intra-sequence structure through a marker tensor in the sparse-attention implementation; on DeepSeek-V4 only the index-attention and compress kernels needed modification.

## Limitations

**Paper's own:** the scheduler minimises wasted *target* compute but DSpark still pays a fixed *draft-side* cost to generate the initial $\gamma$-token block through the parallel backbone. For hard queries with inherently low acceptance, that upfront drafting compute is unrecoverable; they propose difficulty-aware early exiting inside the drafter as future work.

**An honest reader would add:**
- The headline 661% / 406% figures are baseline-collapse artifacts. The paper says so itself, but the abstract's framing ("shifting the Pareto frontier") still leans on them.
- The online comparison is against **MTP-1**, DeepSeek's own prior production setup — not against a tuned Eagle3 or DFlash deployment. The offline and online evaluations therefore never share a baseline, so it's hard to attribute the production win between the drafter architecture and the scheduler.
- The greedy scheduler's optimality argument assumes unimodal $\Theta$; in production they *knowingly* break that assumption and rely on the async lag for correctness rather than on the Algorithm-1 proof. Losslessness is argued, not empirically verified in the fetched text (no distribution-equivalence test is reported).
- Everything is at temperature 1.0 with chain-based drafting. Behaviour under greedy/low-temperature decoding — where acceptance dynamics differ substantially — is unreported.
- The confidence head is calibrated on a held-out set; whether STS temperatures hold under distribution shift in live traffic isn't examined.

## Why it matters [analyst's view]

The genuinely transferable idea here is **"a little autoregression goes a long way."** The 2-layer-DSpark-beats-5-layer-DFlash result says the field's parallel-drafter arms race has been buying capacity where it's cheap and ignoring dependency where it's cheap *too*. A rank-256 lookup table recovers most of what full autoregression provides, because within a 5–16 token block the dependency structure that matters is mostly local. That's a strong prior for anyone building blockwise generators — including the diffusion/parallel-decoding lines this vault tracks, where the multi-modal collision failure is the same one Gu et al. named in 2018.

The second idea is **framing verification length as a scheduling problem rather than a modelling one.** Almost all prior adaptive-draft-length work sets a threshold on drafter confidence; DSpark's insight is that the *right* threshold is a function of current GPU load, and that this is computable if (and only if) you have calibrated absolute survival probabilities and a profiled capacity curve. The STS calibration step is the unglamorous piece that makes the whole thing work — ranking-quality confidence is useless when you need to multiply probabilities into an expected-throughput estimate. That "calibration is a systems requirement, not an ML nicety" argument generalises well beyond speculative decoding.

This also lands squarely on the axis [[blogs/bansal-kv-cache]] establishes: LLM decode is memory-bound and batch capacity is the scarce resource. DSpark is essentially an *allocator* for target-model batch capacity, the same way PagedAttention is an allocator for KV memory — both replace static provisioning with load-aware dynamic allocation, and both report their biggest wins at the tail (high concurrency / high fragmentation). Pairing the two notes gives a fairly complete picture of where 2026-era serving throughput actually comes from.

Finally, the paper is a rare artifact: a **production** report with live-traffic telemetry and named engineering compromises (ZOS, CUDA graphs, jagged SPS curves, variable-length kernels), plus open checkpoints and a training repo (**DeepSpec**). The gap between "works on Qwen3-4B offline" and "survived contact with DeepSeek-V4's serving stack" is where most inference papers quietly die.

## Open questions / things to verify

- Does the two-step-lagged async scheduler *actually* preserve the target distribution end-to-end? The causal-barrier argument is plausible but the paper reports no empirical distribution-equivalence check.
- How does the Markov head interact with tree/multi-path verification? DSpark is chain-only; the local-normalisation property should permit trees, and DDTree/TAPS/JetSpec are cited as doing exactly that.
- What is $\lambda$ in the position weighting $w_k = \exp(-(k-1)/\lambda)$? Inherited from DFlash and never given in the fetched text — it directly controls how much the objective discounts late positions, which is the central tension of the whole paper.
- Optimizer / LR / schedule are absent from the fetched source — a re-fetch task if re-implementation is ever attempted.
- Would the semi-autoregressive trick help *non*-speculative parallel decoders (diffusion LMs, NAT), where exact per-token probabilities aren't required and a globally-normalised head would be admissible?
- **DFlash** (Chen et al., 2026) is the backbone and the main baseline; **Eagle3** (Li et al., 2026b) the AR reference; **DeepSeek MTP** (DeepSeek-AI, 2024) the production baseline. All three are `_needs note_` — the vault can't situate this paper properly without at least DFlash.

## Connections

- Builds on: DFlash (parallel backbone + KV injection), Eagle3, DeepSeek MTP — _needs note_ for all three.
- Contrasts with: CRF-NAT (Sun et al., 2019) and CTC-drafter (Wen et al., 2024) — both stack sequential structure over parallel hidden states but lose exact per-token probabilities to global normalisation / latent marginalisation. Concurrent: Domino's CausalEncoder (≈ DSpark's RNN head), DFlare. _needs note_.
- Topic MOCs: [[topics/llm-inference]], [[topics/gpu-optimization]], [[topics/language-models]]
- Related in vault:
  - [[blogs/bansal-kv-cache]] — the other half of the serving-throughput story: KV cache as the binding memory constraint. DSpark allocates *batch capacity*; PagedAttention et al. allocate *memory*. Bansal's Era-3 note explicitly flags speculative decoding's separate draft/target caches as a KV-management complication.
  - [[papers/bayat-2026-tapered-language-models]] — same "don't provision uniformly" instinct, applied to per-layer MLP width instead of per-request verification length.
- Author index: [[authors/xin-cheng]]

## Selected quotes

> "Improving speedup therefore reduces to three levers: lowering $T_{\text{draft}}$ (draft faster), raising $\tau$ (draft better), or reducing the effective $T_{\text{verify}}$ (verify smarter)." — §2.1

> "Because each parallel position marginalizes over all possible prior tokens rather than conditioning on an exact sampled prefix, the model frequently proposes inconsistent suffix combinations—a mode known as multi-modal collision." — §4.3.1

> "We therefore interpret this high-SLA point primarily as evidence that DSpark extends the feasible interactivity frontier, rather than as a representative multiplicative speedup over a well-utilized baseline." — §5.4

> "DSpark circumvents these limitations by keeping the sequential correction local, so per-token probabilities remain exact softmax evaluations." — §6
