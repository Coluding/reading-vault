---
type: paper
title: "Maximal Brain Damage Without Data or Optimization: Disrupting Neural Networks via Sign-Bit Flips"
authors: ["Ido Galil", "Moshe Kimhi", "Ran El-Yaniv"]
year: 2025
venue: arXiv
url: https://arxiv.org/abs/2502.07408
rw_id: 01ktspptnt4ew2m0d3wcvk8sc1
topics: [adversarial-robustness, quantization, hardware-faults]
priority: high
read_state: queued
relevance: ""
added: 2026-06-10
last_updated: 2026-06-10
---

# Maximal Brain Damage Without Data or Optimization: Disrupting Neural Networks via Sign-Bit Flips

## TL;DR

The paper shows that a deployed DNN can be catastrophically destroyed by flipping a *handful* of weight bits — often **one or two** — chosen with no training data, no validation data, and no optimization. Their method, **Deep Neural Lesion (DNL)**, ranks candidate weights by a magnitude-based score, restricts the search to early layers (and, for CNNs, to at most one flip per kernel), and flips the IEEE-754 **sign bit** of the top-scoring weights (which negates the weight, $\theta_i \mapsto -\theta_i$). A one-pass variant, **1P-DNL**, refines the ranking using a single forward+backward pass on *random* inputs to inject a Taylor/OBD-style gradient saliency term. Two sign flips in ResNet-50 reduce ImageNet accuracy by 99.8%; one backbone flip collapses Mask R-CNN COCO detection/segmentation; two flips into two different experts take Qwen3-30B-A3B from 78% to 0% on MATH-500. They also show selectively protecting the ~0.001–1% most critical sign bits is a cheap, effective defense.

## Context & motivation

Most weight-space attacks (BFA [Rakin et al. 2019], DeepHammer [Yao et al. 2020], ZeBRA [Park et al. 2021], Terminal Brain Damage [Hong et al. 2019]) require either real data, costly synthetic/pseudo-data, or iterative gradient-based bit search with many forward/backward passes. That makes them slow, detectable, and dependent on having full run access to the victim model. The paper's premise is that *hardware* bit-flip vectors (Rowhammer, DMA from untrusted peripherals, firmware/SSD-controller exploits, rootkits, GPU cache tampering, voltage/clock glitching) only let an attacker corrupt a small number of bits at known offsets — so the realistic question is: what is the *minimum* access and compute needed to wreck a model? Their answer: once you can write to stored parameters, almost nothing. The contribution is a **data-free, optimization-free** criterion for which bits to flip, validated across image classification (60 classifiers, 48 of them ImageNet models from `timm`/`torchvision`), object detection + instance segmentation (COCO), and reasoning LLMs including a Mixture-of-Experts model.

The title nods to LeCun et al.'s *Optimal Brain Damage* (OBD) — and that is not just branding: the bit-selection criterion is derived from the *same* second-order Taylor/Hessian saliency that OBD/OBS used for pruning, repurposed adversarially.

## Method

### Problem formulation

A model $f_\theta$ is trained on data $(X,Y)\sim\mathcal{D}$ to minimize expected risk $\min_\theta \mathbb{E}_{(X,Y)\sim\mathcal{D}}[\mathcal{L}(f_\theta(X),Y)]$. Parameters are stored in **IEEE-754 FP32**: 1 sign bit, 8 exponent bits, 23 mantissa bits, decoding to

$$ \theta = (-1)^s \times 2^{(e-127)} \times \left(1 + \frac{m}{2^{23}}\right). $$

Let $\mathrm{bits}(\theta)\in\{0,1\}^B$ be the $B$ memory bits encoding all of $\theta$. A **$k$-bit flip** picks $k$ distinct indices $j_1,\dots,j_k$ and produces $\theta'_{(k)}$ with

$$ \mathrm{bits}(\theta'_{(k)})_j = \begin{cases} 1 - \mathrm{bits}(\theta)_j, & j \in \{j_1,\dots,j_k\}, \\ \mathrm{bits}(\theta)_j, & \text{otherwise.} \end{cases} $$

The adversarial objective is $\min_k \max \; \mathbb{E}_{(X,Y)\sim\mathcal{D}}[\mathcal{L}(f_{\theta'_{(k)}}(X),Y)]$ — i.e. maximize damage with minimal $k$ (both for stealth and because Rowhammer-style faults can only manage a few flips). Damage is measured by the relative **accuracy reduction**

$$ \mathrm{AR}(k) = \frac{\mathrm{Acc}(\theta) - \mathrm{Acc}(\theta'_{(k)})}{\mathrm{Acc}(\theta)}, \qquad \mathrm{mAR}(N) = \frac{1}{N}\sum_{k=1}^{N}\mathrm{AR}(k). $$

### Threat model

The attacker has **write access to stored parameters** but: (i) no samples from $\mathcal{D}$, no access to $P(X)$ or $P(Y)$; (ii) cannot evaluate the model on *any* input — no forward or backward pass on any data (real, synthetic, or random). This is strictly more restrictive than all prior weight attacks ("To our knowledge, no prior method satisfies this restrictive threat model"). For the **1-pass** relaxation, the attacker is additionally allowed a *single* forward and a *single* backward pass on *one random* input $z$ (e.g. Gaussian noise shaped like an image, or random tokens) — still no access to the real data distribution.

### Core idea — why sign bits

Among FP32 bits, flipping the **sign bit** (always the MSB, trivially locatable in memory) instantly negates the weight: $\theta_i \mapsto -\theta_i$, a clean perturbation of magnitude $\Delta\theta_i = -2\theta_i$. Exponent flips rescale magnitude and can be even more destructive for LLMs, but they are far less *selective* (random exponent flips already wreck models, so they make a poor stealthy targeted attack). Sign-bit targeting also aligns with hardware reality: Rowhammer/glitching reliably flip the *same bit offset* across many addresses, and the sign bit is a fixed offset. So sign bits are the cleanest, most-reproducible, easy-to-locate failure mode.

The empirical springboard: flipping **random** sign bits — even up to 100,000 of them — barely dents accuracy in most architectures. The vast majority of weights are robust. Only a tiny "critical" subset matters, motivating a targeted ranking.

### Architecture / algorithm — the bit-selection criterion

**Pass-free score (DNL).** Borrowing from magnitude pruning (high-magnitude weights matter most), the score is simply the absolute value:

$$ \mathcal{S}(\theta_i) = |\theta_i|. \tag{2} $$

Three heuristics shape the candidate set:

1. **Magnitude:** flip the sign of the top-$k$ largest-$|\theta_i|$ weights. Flipping a high-magnitude weight's sign is maximally disruptive (analogue of "don't prune the big weights").
2. **One-flip-per-kernel (CNN-specific):** for convolutional models, allow at most one selected weight per kernel. Flipping a *single* sign bit in an early Sobel/Gabor-like kernel destroys its feature-extraction; flipping *two* in the same kernel can partially cancel and merely reorient it (see derivation below). So spread flips across kernels.
3. **Early-layer targeting (generic):** restrict candidates to the first $L$ layers ($1\le L\le 10$; they use $L=10$). Early-layer corruptions propagate and compound through all downstream computation. (Conveniently, the largest-magnitude weights also tend to sit in early layers for most models — though exceptions like ShuffleNetV2 concentrate big weights late, where naive magnitude attack underperforms unless redirected to early layers.)

**Algorithm 1 (DNL, pass-free):**
1. Take $\theta_L$ = parameters in the first $L$ layers.
2. Sort $\theta_L$ descending by $|\theta_i|$.
3. $K \leftarrow$ top-$k$ entries.
4. For CNNs: enforce at most one selected entry per kernel.
5. For each $\theta_i \in K$: set $\theta_i \leftarrow -\theta_i$ (flip the sign bit).

No data, no passes, no gradients — just sort-by-magnitude inside early layers and negate.

**One-pass hybrid score (1P-DNL).** When one forward+backward pass on random input is allowed, augment magnitude with second-order/gradient saliency. With tunable $\alpha,\beta$ (they set $\alpha=\beta=1$):

$$ \mathcal{S}(\theta_i) = \alpha\,|\theta_i| + \beta\left| \frac{\partial \mathcal{R}}{\partial \theta_i}\,\theta_i + \tfrac{1}{2}H_{ii}\theta_i^2 + \sum_{j\neq i} H_{ij}\theta_i\theta_j \right|. \tag{3} $$

Here $\mathcal{R}(\theta)$ is defined as the **sum of model outputs on a random input** (sum of class scores for a Gaussian "image", or sum of logits for random tokens) — note this needs *no labels and no real data*. $H$ is the Hessian of $\mathcal{R}$. Two standard approximations make it one-pass-cheap:
- **Diagonal Hessian:** $H_{ij}\approx 0$ for $j\neq i$, dropping the coupling sum.
- **Gauss-Newton:** $H_{ii}\approx (\partial\mathcal{R}/\partial\theta_i)^2$, so no explicit second derivatives are needed — everything comes from one gradient.

Limits of Eq. (3): if $\partial\mathcal{R}/\partial\theta_i=0$ and $H_{ii}=0$ it collapses to $\mathcal{S}=\alpha|\theta_i|$ (= Eq. 2, pure magnitude); if $\alpha=0$ it becomes a pure OBD-style second-order score $\beta|\partial\mathcal{R}/\partial\theta_i\,\theta_i + \tfrac12 H_{ii}\theta_i^2|$.

**Algorithm 2 (1P-DNL):** same as Algorithm 1 but rank by Eq. (3) (computed from the single random-input pass) instead of by $|\theta_i|$.

### Derivations / why it works

**Why magnitude is the right zero-pass criterion (OBD reduction).** Take the second-order Taylor expansion of the risk around the trained weights:

$$ \Delta\mathcal{R} \approx g^\top \Delta\theta + \tfrac{1}{2}\Delta\theta^\top H\,\Delta\theta, \qquad g = \nabla_\theta \mathcal{R}(\theta). $$

At convergence $g\approx 0$, so curvature dominates. A sign flip on weight $i$ is $\Delta\theta_i = -2\theta_i$ (all other coordinates unchanged). Under a **diagonal-Hessian** approximation,

$$ \Delta\mathcal{R}_i \approx \tfrac{1}{2}(-2\theta_i)^2 H_{ii} = 2\,\theta_i^2 H_{ii}. $$

So the greedy maximizer for a $k$-flip budget is the $k$ indices with largest $\theta_i^2 H_{ii}$. **If $H_{ii}$ is roughly constant within a layer** (empirically common in early conv layers), this reduces to picking the $k$ largest $|\theta_i|$ — *exactly* the pass-free criterion (2). Even without that assumption, if $H \succeq \mu I$ then $\Delta\mathcal{R} \ge 2\mu\sum_{i\in S}\theta_i^2$, so choosing largest magnitudes maximizes a certified lower bound on loss damage. The 1P-DNL hybrid (3) is just this same saliency with $H_{ii}$ replaced by Gauss-Newton gradient estimates — recovering the classical Taylor pruning saliency $\propto |\theta_i g_i|$. In short: magnitude- and gradient-flips are the *adversarial duals* of OBD/OBS pruning importance.

**Why one flip per kernel (cancellation).** For a conv response $y = w^\top x$ on patch $x$, flipping signs at indices $i,j$ gives $\Delta y = -2(w_i x_i + w_j x_j)$. The second flip can offset the first when the two contributions have opposite sign. Averaged over patches with covariance $\Sigma$:

$$ \mathbb{E}[(\Delta y)^2] = 4\left(w_i^2 \Sigma_{ii} + w_j^2 \Sigma_{jj} + 2 w_i w_j \Sigma_{ij}\right). $$

Natural-image patches are locally positively correlated ($\Sigma_{ij}>0$) and early edge-detector kernels have opposite-signed lobes ($w_i w_j < 0$), so the cross-term is negative — the second in-kernel flip partially *cancels* the first. Hence spread flips across kernels.

**Why early layers.** Empirically, restricting flips to early layers consistently raises impact. Motivation (not a proof): a perturbation inserted at layer 1 is processed by all later layers; under a Lipschitz composition bound its worst-case amplification is $\le \prod_{\ell>1} L_\ell$. Neuroscience analogy: lesions in the retina/optic nerve cause more global failure than late-stage ones.

### Procedure — to re-implement

1. Choose bit type = sign (MSB of FP32). Choose $L$ (default 10 layers) and budget $k$.
2. **DNL:** gather all weights in first $L$ layers; sort by $|\theta_i|$ descending; take top-$k$; for CNNs drop any pick that shares a kernel with an already-selected pick (one per kernel); negate each selected weight (flip its sign bit). For transformers/LLMs, *omit* the one-per-kernel constraint.
3. **1P-DNL:** feed one random input (Gaussian image or random token ids), define $\mathcal{R}$ = sum of outputs, run one backward pass to get $\partial\mathcal{R}/\partial\theta_i$, score by Eq. (3) with $\alpha=\beta=1$ and the diagonal + Gauss-Newton approximations, then select/flip as above.
4. LLM note: best layer scope is model-dependent — first-five-blocks is strongest for Qwen3-30B-A3B and Nemotron Nano; all-layers is better for Qwen3-4B.

## Experimental setup

- **Image classification:** 60 classifiers total, 48 ImageNet-1K models from `timm`/`torchvision`; plus transfer datasets DTD, FGVC-Aircraft, Food101, Stanford Cars (on EfficientNet-B0, MobileNetV3-Large, ResNet-50). Architecture-size sweep over ResNet, RegNet, EfficientNet, ConvNeXt, ViT families. Prior-work comparison done on INT8-quantized ImageNet models (VGG-11, ResNet-50, MobileNet-V2, ViT-B/16).
- **Detection/segmentation:** COCO 2017; Mask R-CNN with ResNet-50 / ResNet-101 backbones (torchvision) and YOLOv8-seg (Ultralytics); metrics bbox/segm AP@[0.50:0.95] and AP@0.50. **Only backbone attacked; heads untouched.**
- **LLMs:** Qwen3-4B, Qwen3-30B-A3B-Thinking (MoE), Llama-3.1-Nemotron-Nano-8B, evaluated on a fixed 50-question MATH-500 subset with the canonical verifier. Encoder LMs (BERT, DistilBERT, RoBERTa) fine-tuned on GLUE (MRPC, QNLI, SST-2).
- **Baselines:** BFA (Rakin 2019), DeepHammer (Yao 2020), ZeBRA (Park 2021), Terminal Brain Damage (Hong 2019). Defenses tested: bit replication + majority vote, Hamming/ECC, DeepNcode encoding, weight-scaling.

## Key results

- **ImageNet:** flipping **two** sign bits in ResNet-50 reduces accuracy by 99.8%. With 10 sign flips, 43 of 48 models show AR > 60%. vs prior art (INT8): 1P-DNL collapses ResNet-50 by **99.4% with a single sign flip** (BFA needs ~5, ZeBRA ~5, DeepHammer ~23); VGG-11 99.8% with 2 flips (BFA 17, ZeBRA 8); ViT-B/16 99.3% with 5 (DNL) — where BFA needs 10 and ZeBRA fails to reach 50%. DNL/1P-DNL match or beat prior attacks while using fewer flips *and* needing no data and no optimization.
- **Transfer datasets:** DNL reaches AR(5) ≥ 85% on every model/dataset shown; 1P-DNL reaches AR(4) ≥ 90%. One or two flips already cause sharp collapse on DTD/Aircraft/Food101/Cars.
- **Detection/segmentation (backbone only):** Mask R-CNN R50/R101 — a single backbone sign flip drives bbox AP to ~0.01 and mask AP to 0.00; two flips collapse all metrics to ~100% AR. YOLOv8-seg is more resilient but 1–2 flips still remove >77% of both detection and segmentation performance. Distinct failure modes: Mask R-CNN keeps good localization but assigns the wrong class (heads intact); YOLOv8 fails to detect the dog and hallucinates a bird.
- **LLMs:** Qwen3-30B-A3B (MoE): **two DNL sign flips → 100% AR (78%→0%)**; the two flips hit *expert down-projection* weights in two different experts (layer 3 expert 82, layer 1 expert 68). 1P-DNL gets 71.8% AR from a *single* flip. Nemotron Nano: DNL 32 flips → 100% (first-5-block); 1P-DNL 17 → 100%. Qwen3-4B: 1P-DNL all-layers 4 flips → 95.3%. Targeted ≫ random (Qwen3-30B-A3B keeps 70% after 27 random flips). Corrupted generations degenerate into repetitive boilerplate ("I am a student"). MoE finding is notable: corrupting a couple of experts poisons latent token representations that propagate even through tokens not routed to those experts.
- **Exponent flips on LLMs:** a single targeted exponent-MSB flip reduces all three LLMs to 0% — even *random* exponent flips are highly destructive (Qwen3-30B-A3B → 6% at k=1), but they're too unselective to be a stealthy targeted attack.
- **Encoder LMs (GLUE):** mAR(10) ranges 69.99%–83.07% across nine model/task pairs (worst-protected: DistilBERT/SST-2 at 83.07%; most robust: RoBERTa/MRPC at 69.99%).

## Ablations

- **Strategy comparison (Fig 2):** random sign flips ≈ harmless even at 100k flips; magnitude-based top-$k$ disrupts most models; early-layer + second-order (1P-DNL) is strongest. AR(10) under different strategies confirms ordering random < magnitude < +early-layer < +gradient.
- **Layer scope:** early-layer restriction beats both random and pure-magnitude; any $L\in[1,10]$ helps; ShuffleNetV2 (big weights live late) shows magnitude-only attack underperforms unless redirected to early layers.
- **One- vs two-flips-per-kernel:** two flips in one kernel can restore edge detection (cancellation), confirming the spread-across-kernels rule.
- **Model size:** no clear correlation between parameter count and vulnerability across ResNet/RegNet/EfficientNet/ConvNeXt/ViT — big models collapse too.
- **Bit type:** sign cleanest/strongest for vision at low budgets; exponent more destructive but less selective for LLMs — "most harmful bit type is domain-dependent."
- (Appendix D) 1P-DNL compared against other one-pass pruning-saliency selectors and found the most potent.

## Limitations

- **Paper's own:** DNL assumes the adversary can directly modify a small number of stored parameters; if the model is sharded/compartmentalized/only partially writable, a global magnitude search isn't available and the attack weakens.
- **Honest-reader flags [analyst's view]:** (1) The "two flips destroy ResNet-50" headline depends on FP32 storage and direct write access — real Rowhammer can't necessarily flip an *arbitrary chosen* bit, only bits at hammerable offsets, so the gap between "we negated the weight in software" and "we flipped this exact memory cell via Rowhammer" is not closed experimentally. (2) Early-layer/Lipschitz justifications are explicitly motivation, not proof. (3) LLM evaluation is a single 50-question MATH-500 subset, so accuracy numbers are coarse. (4) The MoE "latent poisoning" propagation story is plausible but evidentially light (one rank-check example).

## Why it matters [analyst's view]

This reframes weight-space attacks from a data/optimization problem into a pure *memory-integrity* problem, and it does so with a strikingly elegant theoretical hook: the optimal bits to *attack* are the optimal weights *not to prune* — OBD/OBS saliency run in reverse. That duality is the intellectual payload. Practically, the result raises the bar for anyone deploying models on shared/edge/cloud hardware: if your supply chain or hypervisor neighbor can induce even a couple of bit flips, ECC on the *whole* model is overkill but ECC on the *DNL-identified critical sign bits* (~0.001–1% of params) is a cheap, principled mitigation — the same criterion that finds the attack surface defines the defense. The MoE finding is the scariest for current frontier deployment: two flips into two experts zeroing a 30B reasoning model, with corruption propagating through tokens that never route through the damaged experts, suggests routing sparsity does *not* localize fault impact. It connects to the broader story that "robustness to random perturbation" (most weights are fine) is a totally different property from "robustness to adversarially chosen perturbation" (a couple of weights are fatal).

## Open questions / things to verify

- Can the *exact* DNL-selected bits actually be flipped end-to-end via real Rowhammer/glitching, given offset constraints — or only in a software-write threat model?
- Does the attack survive non-FP32 storage in practice (bf16, FP8, INT4) at frontier scale? INT8 results exist; modern LLM-native formats are less tested here.
- The MoE latent-poisoning hypothesis — how broadly does corrupting non-universal experts derail generation? Needs more than the single rank-check example.
- How does the selective ECC defense trade off against attackers who target the *next*-most-critical unprotected bits (an arms race the paper hints at but doesn't fully chase)?
- Generalization of the MATH-500-subset LLM numbers to broader generation benchmarks.

## Connections

- Builds on (conceptually): Optimal Brain Damage (LeCun et al. 1989) and Optimal Brain Surgeon (Hassibi et al. 1992) saliency; magnitude pruning / Lottery Ticket (Frankle & Carbin 2018); gradient pruning (SNIP, GraSP, SynFlow, Molchanov Taylor).
- Contrasts with: BFA (Rakin 2019), DeepHammer (Yao 2020), ZeBRA (Park 2021), Terminal Brain Damage (Hong 2019) — all data/optimization-dependent.
- Topic MOCs: [[topics/adversarial-robustness]] · [[topics/quantization]] · [[topics/hardware-faults]]
- Author indices: [[authors/ido-galil]] · [[authors/moshe-kimhi]] · [[authors/ran-el-yaniv]]
- No genuinely related notes currently in the vault (existing notes are generative/world-model/predictive-processing) — this opens a fresh security/robustness cluster.

## Selected quotes

> "In image classification, flipping just two sign bits in ResNet-50 on ImageNet reduces accuracy by 99.8%." — Abstract

> "for Qwen3-30B-A3B, two sign flips into two different experts are sufficient to reduce accuracy from 78% to 0%." — §1

> "Thus, with a budget of k flips, the greedy maximizer is the set of k indices with largest $\theta_i^2 H_{ii}$. If $H_{ii}$ is approximately constant within a layer ... this reduces to choosing the k largest $|\theta_i|$, precisely our zero-pass criterion." — §3 (Theoretical motivation)

> "Sign flips, however, are multiplicative ($\theta \mapsto -\theta$); after the rescaling they remain $-\theta$ ... Hence, the scaling defense has no effect on DNL." — §6

> "only a tiny subset of sign bits is truly catastrophic." — §6
