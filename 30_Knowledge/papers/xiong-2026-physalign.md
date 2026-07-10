---
type: paper
title: "PhysAlign: Physics-Coherent Image-to-Video Generation through Feature and 3D Representation Alignment"
authors: ["Zhexiao Xiong", "Yizhi Song", "Liu He", "Wei Xiong", "Yu Yuan", "Feng Qiao", "Nathan Jacobs"]
year: 2026
venue: "arXiv preprint"
url: https://arxiv.org/abs/2603.13770
rw_id: ""
topics: [world-models, video-generation, generative-models]
priority: medium
read_state: skimmed
relevance: ""
added: 2026-07-10
last_updated: 2026-07-10
---
<!-- ingested via /paper-search from arXiv (not via Readwise; no rw_id) -->

## TL;DR

PhysAlign is an efficient LoRA-adapter framework that makes image-to-video (I2V) generation physically coherent without retraining the backbone. It rests on two pillars: (1) a **fully controllable synthetic data pipeline** built on Blender rigid-body simulation that yields 3K clips with paired RGB, dense depth, and fine-grained physics annotations (mass, size, elasticity, initial velocity, drop height); and (2) a **unified physical latent space** that couples explicit 3D geometry constraints (a multi-term depth-prediction loss) with a **Gram-based spatio-temporal relational alignment** that distills *kinematic priors* from a frozen video foundation model (V-JEPA2) by matching pairwise token-similarity structure rather than raw features. Fine-tuning Wan2.2-14B on just 3K synthetic clips, PhysAlign beats CogVideoX-5B, HunyuanVideo-I2V, and the Wan2.2 base on physics-invariance metrics and on the PhysicsIQ benchmark (38.1% vs 29.6% for base Wan2.2, 34.5% for Wan2.2+LoRA), while *also* improving VBench I2V visual quality rather than trading it off. At inference the teacher and depth head are discarded, so runtime is identical to a standard Wan2.2+LoRA.

## Context & motivation

Video diffusion models (VDMs) are promising simulators of dynamic scenes for robotics and media, but "existing models often generate temporally incoherent content that violates basic physical intuition," limiting practical use. The paper's diagnosis: "Current VDMs are typically trained on large-scale video datasets and thus learn physical coherence only implicitly. Yet explicit physical cues ... are rarely available in real video corpora." So the physics knowledge that would fix incoherent motion is exactly what the training data lacks.

PhysAlign attacks this on both sides. First, it *manufactures* the missing supervision with a controllable simulator that produces exact physics and 3D annotations. Second, instead of naively regressing to those annotations (which risks overfitting a narrow synthetic distribution and killing generative diversity), it injects physical structure through **relational** alignment to a video foundation model plus geometry constraints — a "structural knowledge" transfer rather than "rigid token matching." The stated contributions are: (i) the PhysAlign adapter framework with Gram-based spatio-temporal relational alignment coupled to 3D geometry constraints; (ii) the open synthetic rigid-body data pipeline with paired RGB / dense 3D / physics annotations; and (iii) data efficiency — "By fine-tuning on merely 3K synthetic clips, PhysAlign robustly internalizes physical laws and generalizes to complex real-world dynamics."

## Method

### Problem formulation

Standard I2V: given a conditioning first frame (plus text prompt), generate a temporally coherent video. The backbone is **Wan2.2-14B**, a DiT-based I2V model that "employs Flow Matching for generation," so the primary reconstruction objective is the flow-matching loss $\mathcal{L}_{\text{FM}}$. The added objective is *physical coherence*: generated motions and object interactions should obey intuitive rigid-body physics (consistent acceleration, velocity, size under motion/occlusion). PhysAlign trains only lightweight modules (LoRA + projector + depth head) on top of the frozen backbone.

### Core idea

Build a **unified physical latent space** by coupling two complementary supervisory signals that both derive from the annotated synthetic data: (a) explicit **3D geometry constraints** via depth prediction, giving the model a metric-structural grounding; and (b) a **Gram-based spatio-temporal relational alignment** to a frozen video teacher (V-JEPA2), which transfers *kinematic* / motion-trajectory structure. The relational (Gram) form is deliberate: aligning the *pairwise similarity structure* of tokens, not their absolute values, transfers "structural knowledge of the physics world" while "preserving generative diversity" and avoiding "overly rigid feature matching."

### Synthetic data pipeline

Built on **Blender** with low-level rigid-body control. Output: **3,000 training videos** at 512×512, 90 frames each, ray-traced. Each scene contains **3–7 objects** with physically grounded initial states. Randomized/annotated physical parameters:

- Mass, size, elastic coefficients (sampled from realistic ranges).
- Initial horizontal velocity via a random force direction (0°–360°).
- Random drop height.
- Camera orientation sampled within predefined ranges.

Multi-modal paired outputs per clip: RGB frames, **metric depth maps** (distance from camera), and **text annotations** that fuse the physics parameters into the prompt. Object vocabulary spans balls of distinct categories (basketball, soccer, tennis, bowling) and primitives (cubes, cylinders, cones). Collisions and occlusions arise naturally from the simulation. This is the supervision that real video corpora lack.

### Unified physical latent space

**(A) Gram-based spatio-temporal relational alignment ($\mathcal{L}_{\text{Phys}}$).**

Hidden tokens are read out from DiT block $b$:

$$H^{(b)} \in \mathbb{R}^{B \times (T_f h_p w_p) \times d}$$

where $B$ is batch, $T_f$ is the number of temporal anchors, $h_p = H/p_h$ and $w_p = W/p_w$ are spatial patch-grid dimensions for patch size $(p_h, p_w)$, and $d$ is the DiT hidden dimension. A lightweight trainable MLP $\phi^{(b)}: \mathbb{R}^d \to \mathbb{R}^D$ projects tokens into the teacher's embedding space of dimension $D$:

$$\hat{Y}^{(b)} = \phi^{(b)}(H^{(b)}) \in \mathbb{R}^{B \times (T_f h_p w_p) \times D}$$

The projected tokens are reshaped to a 5D volume and **parameter-free trilinear interpolated** to the teacher grid $(t_g, h_g, w_g)$:

$$\hat{Z}^{(b)} \in \mathbb{R}^{B \times t_g \times h_g \times w_g \times D}$$

(chosen over "parameterized convolutional downsamplers" to avoid distortion). Let $N_v = t_g h_g w_g$ be the total number of spatio-temporal tokens, and let $s, t \in \mathbb{R}^{B \times N_v \times D}$ be the flattened student and teacher features. A **cosine-similarity Gram matrix** is computed over token pairs:

$$G_{i,j} = \frac{\langle s_i, s_j \rangle}{\|s_i\|_2 \, \|s_j\|_2}$$

$G \in \mathbb{R}^{N_v \times N_v}$ "simultaneously captures intra-frame spatial geometry and inter-frame causalities" — i.e., spatial layout *and* motion trajectory relations. The alignment is a **margin-based L1 penalty** between the student's Gram matrix $G^{(b)}_S$ (from the DiT) and the frozen teacher's $G_T$ (from V-JEPA2):

$$\mathcal{L}^{(b)}_{\text{Phys}} = \frac{1}{N_v^2} \sum_{i=1}^{N_v} \sum_{j=1}^{N_v} \max\!\left(0,\ \left| [G^{(b)}_S]_{i,j} - [G_T]_{i,j} \right| - m \right)$$

with margin $m$ (default $0.1$). Symbols: $G^{(b)}_S$ = student relation matrix at DiT block $b$; $G_T$ = teacher (V-JEPA2, frozen) relation matrix; the hinge $\max(0, \cdot - m)$ only penalizes discrepancies exceeding the tolerance $m$, so small structural differences are permitted (preserving diversity). The teacher encodes video into $Z \in \mathbb{R}^{B \times t_g \times h_g \times w_g \times D}$; the whole mechanism is "joint spatio-temporal relational distillation" — this is how *kinematic priors* are extracted from the video foundation model. Because it matches *relations* not values, it transfers structure without forcing the generator onto the teacher's exact manifold.

**(B) 3D geometry constraint ($\mathcal{L}_{\text{3D}}$).**

The 3D signal is **depth**. A depth head (3D convolution) predicts depth latents/maps, supervised against the simulator's ground-truth depth by a four-term loss:

$$\mathcal{L}_{\text{3D}} = \beta_{\ell}\,\mathcal{L}_{\text{latent}} + \beta_p\,\mathcal{L}_{\text{pixel}} + \beta_s\,\mathcal{L}_{\text{st}} + \beta_t\,\mathcal{L}_{\text{temp}}$$

with hyperparameter weights $\beta_\ell, \beta_p, \beta_s, \beta_t$ (values not specified in source). The terms:

1. **Latent loss** — MSE on depth latents:
$$\mathcal{L}_{\text{latent}} = \frac{1}{BCTHW} \sum_{b,c,t,i,j} \left\| \hat{Z}^d_{b,c,t,i,j} - Z^{d\star}_{b,c,t,i,j} \right\|_2^2$$
where $\hat{Z}^d \in \mathbb{R}^{B \times C \times T \times H \times W}$ are predicted depth latents, $Z^{d\star}$ the ground-truth depth latents, and $(B,C,T,H,W)$ = batch, channels, temporal length, spatial height/width.

2. **Pixel (scale-shift-invariant) loss** — decouples absolute depth from relative structure via a per-frame optimal affine fit:
$$\mathcal{L}_{\text{pixel}} = \frac{1}{BT} \sum_{b=1}^{B} \sum_{t=1}^{T} \min_{s_{b,t}, t_{b,t}} \frac{1}{H_p W_p} \sum_{i,j} \left\| s_{b,t}\,\hat{D}_{b,t,i,j} + t_{b,t} - D^{\star}_{b,t,i,j} \right\|_2^2$$
where $\hat{D}, D^{\star} \in \mathbb{R}^{B \times 1 \times T \times H_p \times W_p}$ are decoded predicted / target depth at pixel resolution $(H_p, W_p)$, and $s_{b,t}, t_{b,t}$ are per-frame optimal scale and shift (solved by least squares).

3. **Structure / gradient-matching loss** — preserves depth discontinuities (object boundaries):
$$\mathcal{L}_{\text{st}} = \frac{1}{B T H_p W_p} \sum_{b,t,i,j} \left( \left|\partial_x \hat{D}_{b,t,i,j} - \partial_x D^{\star}_{b,t,i,j}\right| + \left|\partial_y \hat{D}_{b,t,i,j} - \partial_y D^{\star}_{b,t,i,j}\right| \right)$$
with $\partial_x \hat{D} = \hat{D}_{i,j+1} - \hat{D}_{i,j}$ (horizontal difference) and $\partial_y \hat{D} = \hat{D}_{i+1,j} - \hat{D}_{i,j}$ (vertical).

4. **Temporal loss** — enforces coherent depth evolution / reduces flicker:
$$\mathcal{L}_{\text{temp}} = \frac{1}{B(T-1)H_p W_p} \sum_{b=1}^{B} \sum_{t=1}^{T-1} \left\| \Delta_t \hat{D}_{b,t} - \Delta_t D^{\star}_{b,t} \right\|_1$$
with frame-difference $\Delta_t D_{b,t} := D_{b,t+1} - D_{b,t}$.

**Total objective.** The three signals combine into:

$$\mathcal{L} = \mathcal{L}_{\text{FM}} + \lambda_{\text{Phys}}\,\mathcal{L}_{\text{Phys}} + \lambda_{\text{3D}}\,\mathcal{L}_{\text{3D}}$$

where $\mathcal{L}_{\text{FM}}$ is the Wan2.2 flow-matching reconstruction loss (its exact form is not written out in the paper), $\mathcal{L}_{\text{Phys}}$ is the Gram alignment loss, $\mathcal{L}_{\text{3D}}$ the depth constraint, and $\lambda_{\text{Phys}}, \lambda_{\text{3D}}$ are weighting coefficients ($\lambda_{\text{Phys}} = 0.25$; $\lambda_{\text{3D}}$ not specified). Intuitively: $\mathcal{L}_{\text{FM}}$ keeps generation on-distribution, $\mathcal{L}_{\text{3D}}$ grounds metric geometry and its temporal evolution, and $\mathcal{L}_{\text{Phys}}$ injects motion-relational structure from the video teacher — together defining the "unified physical latent space."

### Training procedure

- Backbone: **Wan2.2-14B** (frozen); V-JEPA2 teacher frozen.
- Trained modules: **LoRA adapters** (rank 32) inserted into DiT `q, k, v, o` and `ffn.0/ffn.2`; the lightweight MLP projectors $\phi^{(b)}$; and the 3D depth-prediction head (3D conv).
- Optimizer: **AdamW**, lr $1\times10^{-4}$, weight decay $0.01$, **bfloat16**.
- Data: **3,000 synthetic clips only**. Training clip length 49 frames (truncated to 48 for V-JEPA2).
- Loss config: $\lambda_{\text{Phys}} = 0.25$, margin $m = 0.1$.
- Hardware: **4× H100**, ~**24 hours**.

### Inference / sampling

At inference PhysAlign "does not require the video teacher nor the 3D/depth head. We discard all auxiliary branches and directly run the frozen I2V backbone with the learned LoRA adapters." So runtime/memory are "essentially identical to a standard Wan2.2+LoRA." Reported inference config: **480×832, 50 steps, 49 frames**. The alignment machinery is purely a training-time regularizer.

## Experimental setup

- **Evaluation datasets/benchmarks:** a Blender synthetic test set; **WISA-Test** (200 real videos from WISA-80K covering collision, rolling, bouncing); the **PhysicsIQ** benchmark (physical-principle understanding); and **VBench I2V** (standard visual-quality suite).
- **Baselines:** CogVideoX-5B; HunyuanVideo-I2V (13B); Wan2.2-I2V (14B, the base); plus Wan2.2+LoRA (for PhysicsIQ) and reference/ground-truth video.
- **Metrics:** the **Physical Invariance Score (PIS)** measuring acceleration/velocity/size consistency across five quantities $(a_x, a_y, v_x, v_y, \Delta l)$; VBench I2V quality metrics (i2v_subject, i2v_background, subject/background consistency, motion smoothness, dynamic degree, aesthetic, imaging); the PhysicsIQ score (%); and a 1–4 human user study (Motion Physics, Identity Physics, Overall Quality).

## Key results

*Summary — full tables in the paper.*

- **Physical Invariance (Blender test):** PhysAlign leads all baselines on every quantity, e.g. $a_x$ 0.632 vs Wan2.2 0.520 / Hunyuan 0.571; $v_y$ 0.798 vs Wan2.2 0.661; $\Delta l$ 0.641 vs 0.546.
- **PIS on WISA-Test (real-world, zero-shot generalization from 3K synthetic clips):** again best across the board, e.g. $a_x$ 0.604 vs Wan2.2 0.444; $v_x$ 0.775 vs 0.669; $\Delta l$ 0.451 vs 0.317.
- **PhysicsIQ:** **38.1%** vs Wan2.2 29.6%, Wan2.2+LoRA 34.5%, CogVideoX 32.3% — the LoRA-only ablation (34.5) shows the alignment losses add ~3.6 points over plain LoRA finetuning.
- **VBench I2V (visual quality — the key "no tradeoff" result):** PhysAlign is *best on every metric* among the generative baselines and approaches the reference — e.g. i2v_subject 0.911 (Wan2.2 0.879, ref 0.931), background consistency 0.955 (ref 0.971), motion smoothness 0.996, dynamic degree 0.730 (Wan2.2 0.676), aesthetic 0.467, imaging 0.655. So the physics gains do **not** come at the cost of visual fidelity — they improve it.
- **User study (1–4):** Motion Physics 3.236, Identity Physics 3.343, Overall 3.286 — far above Wan2.2 (~2.55) and CogVideoX/Hunyuan (~1.6–1.8).

## Ablations

- **Component removal (PIS, both test sets):** starting from a plain LoRA-I2V baseline, adding each alignment helps and the **full model** is best. On Blender, LoRA-I2V gives $a_x$ 0.531 → w/o 3D-alignment 0.611 → w/o feature(Gram)-alignment 0.597 → full 0.632; similar monotone pattern on all five quantities and on WISA-Test (full 0.604 / 0.611 / 0.775 / 0.739 / 0.451). Both the Gram feature alignment (FA) and the 3D geometry alignment (3D-A) are load-bearing; removing either degrades PIS across metrics.
- **Teacher choice (WISA-Test, $\Delta l$):** video-native **V-JEPA2** (0.451) > image-only **DINOv3** (0.436) > training directly on 200 WISA videos (0.399). Confirms a *video* foundation model is the better source of kinematic priors than an image encoder, and that the synthetic-data + alignment recipe beats simply training on real physics videos.

## Limitations

- Scope is **intuitive physics** — "motion patterns that appear more plausible to human observers," not verified quantitative physics. "Scenarios requiring more complex physical understanding or high-level reasoning remain challenging."
- Strong on **solid / rigid-body mechanics**; **fluid dynamics, deformable bodies, and complex interactions** are flagged as future work — consistent with a data pipeline built purely on rigid-body simulation.
- [analyst's view] The synthetic distribution is narrow (balls/primitives, ballistic drops and collisions); real-world generalization is shown on WISA's similar phenomena, so transfer to structurally different dynamics (articulated bodies, contact-rich manipulation) is untested. PIS is a proxy for invariance, not a ground-truth physics check.

## Why it matters [analyst's view]

PhysAlign's central bet is that physical coherence is best injected as a **training-time relational regularizer distilled from a self-supervised video model**, not baked into the sampler or learned implicitly from web video. The Gram-matrix trick is the clever part: by aligning pairwise token-similarity *structure* to V-JEPA2 rather than regressing raw features, it borrows the teacher's motion/kinematic priors without collapsing the generator's diversity — and it costs nothing at inference (teacher and depth head are dropped). That "free at test time, better on both physics *and* visual quality" profile is the strongest selling point; most physics-for-video methods trade fidelity for coherence, and this one claims to improve both.

It sits in the emerging cluster of "make video generation physically coherent" work in the vault. The cleanest contrast is with [[papers/zhao-2026-phyworld]], which pursues the same goal through a different mechanism (flow-matching + preference optimization / DPO-style signal) rather than 3D + feature alignment. It also contrasts sharply with [[papers/yuan-2026-physics-alignment]], which does physics alignment at **inference time** — PhysAlign instead front-loads all the physics into LoRA training and keeps inference vanilla. The synthetic-data-from-simulation angle connects to the broader question of whether physical world models should learn from curated simulation or scale on real video.

## Open questions / things to verify

- Exact $\lambda_{\text{3D}}$ and the four $\beta$ depth-loss weights are unspecified — would want them for reproduction.
- $\mathcal{L}_{\text{FM}}$ is only named, not written; assumed to be Wan2.2's standard rectified-flow objective.
- How many DiT blocks $b$ get Gram alignment (single vs multi-block sum)? The per-block $\mathcal{L}^{(b)}_{\text{Phys}}$ suggests multiple, but the aggregation isn't stated in the extracted text.
- PIS is a custom metric; how it correlates with ground-truth physical accuracy (vs. mere human plausibility) is unclear.
- Does the improvement hold beyond ballistic rigid bodies? The teacher-ablation shows synthetic+alignment beats training on 200 real clips, but both are within the same physics regime.

## Connections

- Direct method sibling / contrast (physics-coherent video, different mechanism — flow-matching + DPO vs 3D + feature alignment): [[papers/zhao-2026-phyworld]]
- Physics alignment for video, inference-time (strong contrast — PhysAlign is training-time): [[papers/yuan-2026-physics-alignment]]
- Physics-in-video cluster (benchmarks / phenomena): [[papers/lin-2026-phyground]], [[papers/esmati-2026-invisible-hand-physics]]
- Feature-space video world modeling (V-JEPA-style representations): [[papers/porcher-2026-flowwm]]
- World/action models: [[papers/ye-2026-world-action-models]]
- Topic MOCs: [[topics/world-models]], [[topics/video-generation]], [[topics/generative-models]], [[topics/diffusion-models]]

## Selected quotes

> "PhysAlign constructs a unified physical latent space by coupling explicit 3D geometry constraints with a Gram-based spatio-temporal relational alignment that extracts kinematic priors from video foundation models." — Abstract

> "Instead of rigid token matching, we introduce a Gram-based spatio-temporal relational alignment to extract kinematic priors from video foundation models (e.g., V-JEPA2), and synergistically couple it with explicit 3D geometry constraints to build a unified physical latent space." — Contributions

> "At inference time, PhysAlign does not require the video teacher nor the 3D/depth head. We discard all auxiliary branches and directly run the frozen I2V backbone with the learned LoRA adapters." — Method
