---
type: paper
title: "RoboTTT: Context Scaling for Robot Policies"
authors: ["Yunfan Jiang", "Yevgen Chebotar", "Ruijie Zheng", "Fengyuan Hu", "Yunhao Ge", "Jimmy Wu", "Tianyuan Dai", "Scott Reed", "Li Fei-Fei", "Yuke Zhu", "Linxi Fan"]
year: 2026
venue: "arXiv preprint"
url: https://arxiv.org/abs/2607.15275
rw_id: 01ky7gcrkej4y7hnf0hp9xy7n0
topics: [robotics, imitation-learning, recurrence, test-time-training]
priority: high
read_state: queued
relevance: ""
added: 2026-07-23
last_updated: 2026-07-23
---

# RoboTTT: Context Scaling for Robot Policies

NVIDIA GEAR + Stanford + UT Austin (equal advising: Li Fei-Fei, Yuke Zhu, Linxi "Jim" Fan). Project page: research.nvidia.com/labs/gear/robottt

## TL;DR

RoboTTT integrates Test-Time Training (TTT) layers into a VLA robot foundation model (GR00T N1.7), turning the policy into a recurrent sequence model whose state is a set of *fast weights* — a small MLP updated by gradient descent at every timestep, during both training and inference. This compresses visuomotor history into weight space at constant inference cost, letting context scale to 8K timesteps (~4–5 minutes at 30 Hz) — three orders of magnitude beyond typical single-step/short-history VLAs — without growing latency. A training recipe of sequence action forcing (per-chunk noise levels in the flow-matching loss) plus truncated BPTT (fast weights carried across segments, gradients truncated) makes long-sequence training tractable in fixed GPU memory. On three real-robot bimanual assembly tasks, RoboTTT scores 79% average task completion, 87% above the single-step GR00T N1.7 baseline (42%), and is the only method to fully complete the five-minute, ten-stage Gear Bot task. Long context unlocks one-shot imitation from a single in-context human video (6/10 vs 0/10 for the best recurrent baseline), on-the-fly self-correction via "DAgger Distillation" (+36%), and perturbation robustness (83% vs 53% recovery). Closed-loop performance improves *steadily* as pretraining context grows from 128 to 8K timesteps with no saturation — the paper's headline claim that context length is a new scaling axis for robot foundation models.

## Context & motivation

State-of-the-art robot foundation models (GR00T N1.7, π0, RT-2-style VLAs, some World Action Models) condition on the current observation only or on 2–8 frames of history. Meanwhile, in LLMs, context length has become a major scaling axis. The paper argues long *visuomotor* context (not just delegating memory to external banks or language summaries) is required for: one-shot in-context imitation from human videos, on-the-fly improvement from a robot's own deployment history (à la Algorithm Distillation), and multi-stage long-horizon tasks where visually similar stages cause state aliasing.

Three obstacles block naive context scaling: (1) encoding long histories with sufficient capacity — RNN vector states are too small; (2) actually *exploiting* the context — appended history introduces spurious correlations (causal confusion: the policy overfits to past actions implicitly encoded in past observations); (3) keeping inference cost constant — Transformer decoding grows with history even with a KV cache. RoboTTT's answer: fast weights are a higher-capacity state than an RNN vector, gradient-descent updates at test time learn what to retain and discard, and propagating a fixed-size weight state keeps inference O(1) in context length.

## Method

### Problem formulation

Robot trajectories are sequences $\xi = \{(l, o_t, q_t, A_t)\}_{t=1}^{T}$: shared language instruction $l$, image $o_t$, proprioception $q_t$, and an $H$-step action chunk $A_t = [a_t, \ldots, a_{t+H-1}]$. The goal is a policy $\pi(A_t \mid \xi_{<t}, o_t, q_t)$ that conditions on history $\xi_{<t}$, with the aim of scaling context length $|\xi_{<t}|$ far beyond current practice.

### Core idea

Replace attention-over-history with a *learned compressor*: a small "fast model" whose weights are updated by gradient descent on a self-supervised association loss at every timestep — during deployment too — so the history lives in parameter space, retrieval is a forward pass, and inference cost is constant in context length.

### The TTT mechanism (the recurrent core)

Given a sequence of $d$-dimensional tokens $X$, learned projection matrices $\theta_Q, \theta_K, \theta_V$ produce query/key/value sequences $Q, K, V$ with per-timestep projections $Q_t, K_t, V_t$. The fast weights $W$ parameterize a small network $f_W(\cdot): \mathbb{R}^d \to \mathbb{R}^d$ (here a two-layer MLP). At each timestep, an **update step** trains the fast model to associate keys with values:

$$W_t \leftarrow W_{t-1} - \eta \nabla_W \mathcal{L}_{\mathrm{FW}}\big(f_{W_{t-1}}(K_t), V_t\big), \qquad \mathcal{L}_{\mathrm{FW}}(\hat{v}, v) = \|\hat{v} - v\|^2,$$

where $\eta$ is a *learnable* inner learning rate (learned on top of a constant base rate of 0.1). Then an **apply step** retrieves context for the downstream prediction:

$$O_t = f_{W_t}(Q_t).$$

"Update then apply" runs at both training and inference — this is what "test-time training" means. Why each piece exists: the MSE association loss forces the fast model to memorize the current token's key→value mapping, i.e. it *writes* the token into weight space; the apply step *reads* by querying. Because $\theta_Q, \theta_K, \theta_V$ and the fast-weight initialization $W_0$ are learned via the *outer* task loss (gradients flow through the inner gradient-descent steps — meta-learning through gradients-of-gradients), the compression mechanism itself is optimized for the control task, not fixed a priori. Contrast with full attention, which keeps every past K/V in memory and attends over all of them each step (cost grows with history), and with linear-recurrent memories like Gated DeltaNet, whose linear associative update has no test-time gradient descent and — the paper's ablation-supported hypothesis — is a less expressive compressor for dense, repetitive robot streams.

### Architecture: TTT inside a VLA

RoboTTT is instantiated on pretrained GR00T N1.7: an Eagle VLM backbone plus a 16-layer Diffusion Transformer (DiT) action head (538M params). A TTT layer is added *after* the self-/cross-attention layers in each of the 16 DiT layers (~10M params each; 690M total). Division of labor: **attention operates within a timestep, TTT layers operate across timesteps.**

The DiT input over $T$ timesteps is $[R_1, \Phi_1, q_1, \tilde{A}_1, \ldots, R_T, \Phi_T, q_T, \tilde{A}_T]$, where $\Phi_t$ are the VLM's vision-language tokens, $q_t$ the encoded proprioception token, $\tilde{A}_t$ the noised action tokens, and $R_t$ are $N{=}16$ *learned register tokens* prepended per timestep that attend to all other tokens. Attention layers process the single-step tokens $R_t, q_t, \tilde{A}_t$ and cross-attend to that timestep's $\Phi_t$. The per-timestep attention outputs are then concatenated along time, $X = [R_1, q_1, \tilde{A}_1, \ldots, R_T, q_T, \tilde{A}_T]$, and passed through the TTT layers (update Eq. above, then apply). The VL tokens $\Phi$ are deliberately *not* passed through TTT for compute efficiency — the 16 register tokens carry VL information across time instead. Positional embeddings are RoPE with $\theta_{\mathrm{rope}} = 10000$.

**Gating to preserve pretrained capabilities.** RoboTTT initializes from the base model weights and gates the TTT contribution per DiT layer with a learned $\alpha \in \mathbb{R}^d$ initialized near zero (0.001):

$$O = \tanh(\alpha) \odot O_{\mathrm{TTT}} + O_{\mathrm{attn}},$$

so TTT starts as a near-no-op and its influence is learned gradually, without disrupting the pretrained VLA's computation (Flamingo-style gating).

### Sequence training and the outer loss

Training runs TTT over each sequence in the inner loop and computes the outer task loss at every timestep:

$$\mathcal{L}_{\mathrm{fm}}(\xi; W_0) = \frac{1}{T} \sum_{t=1}^{T} \ell_t\big((l, o_t, q_t, A_t); W_{t-1}\big),$$

where $W_{t-1}$ is the fast-weight state entering timestep $t$ (updated inside the TTT layers). One optimization step updates both slow weights and $W_0$ — so $W_0$ and the update dynamics are meta-learned for robot trajectories.

**Sequence action forcing.** The per-step objective is flow matching: the DiT denoises $\tilde{A}_t = A_t^{\tau} = \tau A_t + (1 - \tau)\epsilon$, with flow timestep $\tau \in [0,1]$ and noise $\epsilon \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$. The key trick: sample the noise level **independently per action chunk** in the sequence rather than one level for the whole sequence. Full loss, with DiT head $v_\theta$:

$$\mathcal{L}_{\mathrm{fm}}(\xi; W_0) = \frac{1}{T} \sum_{t=1}^{T} \mathbb{E}_{\tau_t, \epsilon}\Big[\big\|v_\theta(\Phi_t, A_t^{\tau_t}, q_t; W_{t-1}) - (A_t - \epsilon)\big\|^2\Big], \qquad \tau_t = s(1-u),\ u \sim \mathrm{Beta}(1.5, 1),\ s = 0.999.$$

Why: sharing one noise level makes entire sequences uniformly easy (low noise) or uniformly hard (high noise), destabilizing training — echoing Diffusion Forcing (Chen et al. 2024). Without it, training is unstable and closed-loop motion is too inaccurate to progress (ablation).

**Truncated BPTT.** Full BPTT stores activations for every timestep, so memory grows with sequence length. TBPTT splits the sequence into segments; gradients flow only within a segment, but **fast weights carry over across segment boundaries** (values propagated, gradients detached), so TTT continues over the entire sequence. Memory is set by segment length, not total length — arbitrarily long training contexts under a fixed budget. $W_0$ still receives gradients through the first segment.

### Learning from heterogeneous context (loss masking)

Because fast-weight updates and slow-weight (imitation) updates are decoupled, masking the flow-matching loss on chosen timesteps turns them into *pure context*: they update the fast weights without serving as imitation targets. Two uses:

- **One-shot imitation from in-context human video**: pair a human video sequence $\xi^{\mathrm{video}}$ with a robot trajectory $\xi^{\mathrm{robot}}$ of the same task; the video portion updates fast weights only (loss masked), the action loss is computed on the robot portion conditioned on the updated fast weights. At test time, a single human video of an unseen configuration yields one-shot imitation.
- **DAgger Distillation**: given DAgger-style rollouts $\xi^{\mathrm{DAgger}}$ where each executed chunk is either a robot action $A_t^{\mathrm{R}}$ or a human correction $A_t^{\mathrm{H}}$, standard DAgger fine-tunes on corrections and discards the robot's failures. RoboTTT instead updates fast weights on the *full* interleaved history (failures included) while masking the imitation loss to human corrections only. Asymmetry is the point: failures-as-context, corrections-as-targets distills the human's failure→correction mapping into the fast-weight adaptation — the model learns to *respond to* failures, not merely imitate corrections in isolation. The paper frames it as Algorithm Distillation instantiated in robotics. At test time the model self-corrects online; its own corrections enter the history and are absorbed into the fast weights exactly as the human's were.

### Derivations / why it works

_No formal derivation; the paper's justification is the meta-learning structure (outer loss shapes $W_0$, $\eta$, and projections through the inner GD steps) plus ablations. The claimed mechanism for beating linear recurrences: a nonlinear MLP fast model updated by test-time gradient descent is a more expressive compressor of dense, repetitive robot streams than a linear associative state, and its initialization/dynamics can be meta-learned — GDN's linear state "admits no such meta-learning."_

### Training procedure

- Pretraining: mixture of tabletop bimanual robot data and egocentric human video data, curated for long trajectories. Only the new sequence-modeling layers (TTT or GDN) are tuned; the rest of GR00T N1.7 is frozen. Context length is *gradually increased* to the target (e.g. 8K for RoboTTT-8K). 30K steps on 16 NVIDIA GB200 GPUs; per-device batch 4 (global 64) for context ≤4K, per-device 1 (global 16) above. AdamW, weight decay $1{\times}10^{-5}$, Warmup-Stable-Decay schedule, peak LR $2{\times}10^{-5}$.
- Post-training: per downstream task at 1K context, 20K steps on 8 GPUs, all parameters fine-tuned, per-device batch 1, cosine schedule, peak LR $5{\times}10^{-5}$.

### Inference

Each rollout starts from the learned $W_0$; at every timestep the fast weights update on the current observation and propagate forward. Action chunks are generated by $k$-step denoising (the paper does not state $k$). Deployment: YAM bimanual tabletop robot, four RealSense D405 cameras (top, bottom, left wrist, right wrist), 480p RGB, RTX 5090 workstation, 30 Hz control.

## Experimental setup

- **Tasks** (real robot, bimanual, long-horizon assembly): *Pup Go Car* (toy-car roof + wheel assembly with screwing/drilling/handoffs/car flip; ~2 min episodes; 8h data), *Circuit* (~80 configurations of 2–3 circuit components with order constraints; train on 20, test on 60; ~1 min; 6h data), *Gear Bot* (ten-stage full toy assembly incl. two chassis flips and remote-control drive; ~5 min; 5h data).
- **Baselines**: GR00T N1.7 (single-step context), GR00T N1.7 Hist. (one history frame), GDN (TTT layers replaced by Gated DeltaNet — matched placement, gating, and parameter count; linear recurrent memory without test-time gradient descent). All post-trained on the same task data; sequence models at 1K context; non-sequence models at matched compute.
- **Metrics**: fully-successful trials out of 20 (10 for Gear Bot) and a rubric-based task-completion score in [0,1] (rubrics per stage in Appendix B).

## Key results

- **Main evaluation**: RoboTTT averages 79% task completion — +87% relative over GR00T N1.7 (42%), +41% over best baseline GDN (56%). Full successes: 9/20 Pup Go Car, 13/20 Circuit, 2/10 Gear Bot; no baseline ever fully completes Gear Bot. Qualitative wins: stage disambiguation under state aliasing, strategic retry (re-aligning a missed drill instead of skipping ahead), and precision under occlusion (past observations inform actions on currently occluded objects).
- **Naive history can hurt**: GR00T N1.7 Hist. scores 39.5% on Pup Go Car vs 57% for its no-history counterpart — appended frames introduce spurious correlations and put the robot temporally out of distribution.
- **Context scaling** (the headline): pretraining context from 128 → 8K timesteps gives steady closed-loop gains with no saturation; RoboTTT-8K reaches 71.5% average, +63% over the same model at 1K (43.9%) and +57% over the best short-context baseline (45.6%). GDN shows *no* scaling trend. Below 1K, RoboTTT underperforms its longer variants — 1K ≈ half a minute, shorter than the shortest episode, so inference runs the fast weights far past the trained window and RoPE extrapolates to unseen positions.
- **One-shot imitation** (Circuit, prompt identical across configs so the target is identifiable only from the in-context human video): RoboTTT 65% score, 6/10 successes; GDN 33%, 0/10 — recurrent memories can *encode* context but struggle to *use* it.
- **Perturbation robustness** (part removed mid-episode after installation): RoboTTT recovers 15/20 (roof) and 18/20 (tire) vs at most 10/20 and 11/20 for short-context baselines; GDN close behind (13/20, 18/20). 30 min of perturbation data co-trained.
- **DAgger Distillation** (100 pooled DAgger trajectories on Pup Go Car): standard DAgger gives +9% average across methods; DAgger Distillation gives +33% average (+36% RoboTTT, +29% GDN). Crucially, the suboptimal robot actions are worthless as *imitation targets* (fine-tuning on full trajectories = corrections-only, both 57%) — their value is as *context*.

## Ablations

- **Sequence action forcing**: removing it (single shared noise level per sequence) significantly hurts — motions become too inaccurate to make progress. Load-bearing.
- **Fast-model expressivity**: linear fast model beats GR00T N1.7 but is 27% worse than the two-layer MLP — nonlinear fast models matter most (consistent with TTT results in vision/language).
- **Development roadmap**: TTT on state tokens only → +action tokens (+23% relative; knowing past actions helps capture dynamics) → +register tokens (+18% relative). Register tokens added to plain GR00T N1.7 do *not* help — they are only useful paired with TTT's temporal modeling, i.e. the gain is not just extra capacity.

## Limitations

Paper's own: (1) scaling training context raises training cost (points to newer TTT training techniques like TNT); (2) TTT layers use a generic association objective — robotics-specific inner objectives unexplored; (3) failures remain in deployment; combining with RL to optimize task success directly is the stated next step. [analyst's view] Additional flags: all results are on one backbone (GR00T N1.7) and one embodiment (YAM bimanual tabletop) across three assembly tasks with 5–8h of task data each — the "new scaling axis" claim rests on a three-task average; full-success rates are still modest in absolute terms (9/20, 13/20, 2/10); the scaling-curve evaluations predate the DAgger training used in the main results, so the two headline tables aren't directly comparable; and inference-time denoising steps $k$ and TBPTT segment length are not reported in the fetched text.

## Why it matters [analyst's view]

This is the cleanest demonstration yet that the TTT/fast-weights line (Sun et al., Titans, Gated DeltaNet-adjacent work) transfers to real-robot control, and the first closed-loop evidence that *context length scales* for robot policies the way it does for LLMs. Two ideas look exportable beyond robotics: (1) the loss-masking trick that decouples "context that updates fast weights" from "targets that update slow weights" — DAgger Distillation is essentially Algorithm Distillation made practical with human corrections, and the finding that failures are valuable *as context but not as targets* is a crisp, reusable design principle; (2) the head-to-head vs Gated DeltaNet at matched parameters/placement is a rare controlled comparison of test-time-gradient-descent memory vs linear associative memory on a real sequential decision task — GDN encodes but can't exploit (0/10 one-shot imitation), supporting the "expressivity of the compressor" story. For the vault, this connects the recurrence/fast-weights thread to the robot-foundation-model thread: it's a concrete counterpoint to pure-VLA scaling, and a sibling to the GEAR lab's WAM work ([[papers/ye-2026-world-action-models]]) — same group, complementary bet (context scaling vs zero-shot world-action pretraining).

## Open questions / things to verify

- TBPTT segment length and denoising steps $k$ — check Appendix/official code when released.
- Does the scaling trend hold past 8K, and on non-assembly task families / other embodiments?
- How much of the GDN gap closes with meta-learned initialization tricks for linear recurrences, or with Muon-style inner optimizers (explicitly left to future work)?
- Latency numbers: "constant inference cost" is asserted architecturally; actual per-step wall-clock vs baselines is _not addressed by the source_.
- The closest prior work (Ziakas & Russo — fast weights for VLM value functions) vs this policy-level integration: worth reading to see what the value-function route misses.

## Connections

- Contrasts with: [[papers/karcini-2026-robots-beyond-vla]] — position paper arguing VLAs/world models are not enough; RoboTTT is a concrete architectural answer to one of the gaps (memory/adaptation).
- Same lab / complementary: [[papers/ye-2026-world-action-models]] — NVIDIA GEAR, overlapping authors (Chebotar, Reed, Zhu, Fan, Hu, Zheng, Wu); WAMs are the other GEAR bet on robot foundation models, cited in RoboTTT's related work.
- Related data/task regime: [[papers/jiang-2024-dexmimicgen]] — bimanual dexterous manipulation via imitation, overlapping authors (Fan, Zhu); the data-generation side of the same GEAR manipulation stack.
- Topic MOCs: [[topics/robotics]], [[topics/imitation-learning]], [[topics/recurrence]], [[topics/test-time-training]]
- Author indices: [[authors/yunfan-jiang]], [[authors/linxi-fan]], [[authors/yuke-zhu]], [[authors/li-fei-fei]]

## Selected quotes

> "RoboTTT is a sequence model whose recurrent state consists of fast weights: unlike slow weights, which are frozen at inference, fast weights are updated by gradient descent during both training and inference." — §1

> "This asymmetry, failures as context and corrections as targets, is precisely how the human's failure-to-correction mapping is distilled into the fast weights: the model learns to produce corrections in response to failures, rather than to imitate corrections in isolation." — §3.3

> "RoboTTT exhibits a clear scaling trend: closed-loop performance increases steadily with pretraining context length, reaching 71.5% at 8K … with no sign of saturation. GDN shows no such trend." — §4

> "Notably, the suboptimal robot actions carry no value as imitation targets … Their value is as context." — §4

> "The relevant observation window is difficult to specify a priori; RoboTTT instead learns what to retain, suggesting that with a sufficiently expressive sequence model, the use of long context can be learned rather than hand-designed." — §4
