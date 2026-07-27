---
type: paper
title: "WEAVER, Better, Faster, Longer: An Effective World Model for Robotic Manipulation"
authors: ["Arnav Kumar Jain", "Yilin Wu", "Jesse Farebrother", "Gokul Swamy", "Andrea Bajcsy"]
year: 2026
venue: arXiv preprint
url: https://arxiv.org/abs/2606.13672
rw_id: 01kyhvfjqef5gkdb4qyrtd4scc
topics: [world-models, robotics, flow-matching]
priority: high
read_state: queued
relevance: ""
added: 2026-07-27
last_updated: 2026-07-27
---

# WEAVER: World Estimation Across Views for Embodied Reasoning

## TL;DR

WEAVER is a 928M-parameter multi-view latent world model for robot manipulation whose organising argument is a **trifecta of desiderata that no prior WM satisfies at once**: (i) *fidelity* — predictions good enough to evaluate arbitrary visuomotor policies, which rules out JEPA-style models whose latents can't be decoded to images; (ii) *long-horizon consistency* — coherence across 40+ autoregressive calls under occlusion; and (iii) *efficiency* — fast enough for real-time test-time planning, which rules out video-diffusion WMs. The recipe is explicitly a synthesis rather than a new mechanism: flow matching + diffusion forcing from video generation, a frozen pretrained SD3 VAE encoder for OOD robustness, latent reward and critic heads from Dreamer-v4 (so no external VLM judge is needed), future-latent prediction from JEPA, and multi-view + memory conditioning from Ctrl-World. On five real Franka manipulation tasks it hits **ρ = 0.870** correlation with real success rate as an evaluator, lifts the π0.5 VLA's real success rate by **38%** via synthetic-data finetuning, and adds **15%** average success via best-of-N test-time planning while running **~20× faster** than Ctrl-World's inference pipeline. It Pareto-dominates Ctrl-World on FVD-vs-inference-time at every NFE budget tested.

## Context & motivation

World models promise three things for robotics — policy evaluation, policy improvement, and test-time planning — all without costly or unsafe real-world interaction. The paper's framing contribution is noticing that these three downstream uses impose *three different and mutually tensioned* upstream requirements, and then auditing prior work against all three at once:

- **Video generation models** produce high-fidelity rollouts but are far too slow.
- **JEPA-style WMs** have latents "that may not be decodable into the images required to evaluate arbitrary visuomotor robot policies" — a sharp and underrated point: if your policy consumes pixels, your world model must produce pixels, whatever it does internally.
- **Dreamer-v4** looks promising but learns its encoder from scratch, which the authors argue harms out-of-distribution robustness.
- **Ctrl-World**, the SOTA multi-view manipulation WM, operates "at far slower speeds than the real world," precluding planning and making policy improvement computationally painful.

Manipulation specifically compounds the problem: multiple camera views, objects occluded by the gripper that must be inferred from history, and a need for *physically* rather than merely *aesthetically* plausible predictions. The paper is candid that WEAVER "fuses together key design decisions from prior world modeling approaches" — the claim is the gestalt, not the parts.

## Method

### Setup

Long-horizon manipulation specified by a language instruction $\ell$. Proprioceptive state $q \in \mathbb{R}^8$ (joint angles + gripper). $n$ RGB views $I := (I^1,\dots,I^n)$ from wrist and third-person cameras. Each view is encoded to $H \times W$ patch tokens by a **frozen pretrained Stable Diffusion 3 VAE encoder**; proprioception is projected and concatenated. The model predicts an $h$-step future of latents $\hat{z}_{t+1:t+h+1}$ and rewards $\hat{r}_{t+1:t+h+1}$; a pretrained decoder recovers observations $\hat{o}_{t+1:t+h+1}$ so the last one can be fed back to the policy for the next action chunk.

### Core idea

Predict *future latents* (not pixels) with a flow-matching objective, keep the encoder/decoder pretrained and frozen so arbitrary pixel-consuming policies can still be evaluated, and attach reward/critic heads that score latents directly — so the whole evaluate-imagine-select loop stays inside latent space and never calls an external judge.

### Training objective

The latent dynamics model $f_\phi$ is trained by flow matching. Let $x_t^1 := z_{t+1:t+h+1}$ be the ground-truth next $h$ latents and $x_t^0 \sim \mathcal{N}(0,I)$ noise of the same shape. Define the linear interpolant

$$x_t^\tau = \tau x_t^1 + (1-\tau)x_t^0, \qquad \tau \in [0,1)$$

and train $f_\phi$ to regress the **velocity** $x_t^1 - x_t^0$:

$$\mathcal{L}_{\text{WM}}(\phi) = \mathbb{E}_{x_t^0, x_t^1, \tau}\Big[\big\|(x_t^1 - x_t^0) - f_\phi\big(z_t^{\text{hist}},\, z_t^{\text{mem}},\, a_t,\, x_t^\tau,\, \tau\big)\big\|_2^2\Big]$$

Term by term: $\tau$ is the flow time (0 = pure noise, 1 = clean latents); the target $x_t^1 - x_t^0$ is the constant velocity of the straight-line path between noise and data, which is what makes rectified-flow-style few-step sampling possible later; and the conditioning set is the whole point — $z^{\text{hist}}$ and $z^{\text{mem}}$ supply what the scene looked like, $a_t$ supplies what the robot is about to do, and $x_t^\tau$ is the current noisy iterate. Because the network predicts a velocity field rather than a denoised sample, generation is ODE integration and the number of function evaluations (NFE) becomes a tunable latency knob.

**Diffusion Forcing** is layered on top: independently sampled noise levels across future timesteps rather than one shared level. This is the long-horizon-consistency lever — it teaches the model to condition on partially-denoised neighbours, which is exactly the regime autoregressive rollout puts it in, so errors don't compound the way they do when training only ever shows uniformly-noised sequences.

### Memory and history

Long-horizon consistency "requires the WM to understand both what changes and what stays the same." Occlusions and a moving wrist camera constantly push objects out of view. Following Ctrl-World, WEAVER conditions on two sets:

- **Memory** $z_t^{\text{mem}} := (\dots, z_{t-2k}, z_{t-k})$ — every $k$-th encoded observation, long-term and *sparse*, to carry context about objects currently occluded.
- **History** $z_t^{\text{hist}} := (z_{t-1}, z_t)$ — the last two frames, dense and short-term, to capture the immediate consequences of actions.

The sparse/dense split is a compute argument as much as a modelling one: you need long reach without paying quadratic attention over every past frame.

### Architecture

An efficient **2D transformer**: $L$ dynamics blocks alternating spatial attention and *causal* temporal attention, conditioning on latent tokens, action tokens, and flow-timestep embeddings to autoregressively generate an $h$-step chunk. Stability kit: RMSNorm, RoPE, QK-Norm, SwiGLU. **SPRINT blocks** aggressively drop patch tokens in the latents for efficiency.

**Multi-view prediction** (external + wrist) is emphasised because most WMs predict a single view; the extra view is what disambiguates gripper–object contact under occlusion.

**Proprioceptive state prediction** is called out as a distinguishing choice: WEAVER predicts future $q$ explicitly, unlike Ctrl-World which predicts only visual observations. The stated reason is concrete — contact-rich deformable manipulation needs "the precise position of the arm and width of the gripper," which is not reliably readable off a generated image.

### Inference acceleration

Latency has two sources — (a) the forward pass and (b) iterative denoising — and both are attacked:

- **(a) KV caching** of memory and history tokens across denoising steps. These don't change between steps, so recomputing them is pure waste.
- **(b) Progressive noise schedule** with a **cosine** rather than linear schedule (as used by Ctrl-World and Dreamer-v4), reported as higher fidelity.
- **Rectified-flow distillation**: post-train by first generating a high-quality latent trajectory with the full denoising process, then using it as the target for a distillation step, enabling few-forward-pass generation. This is what brings latency into test-time-planning range.

### Latent value estimation

- **Reward head $R$**: rather than decoding to pixels and querying a VLM judge, WEAVER distils scores from an off-the-shelf reward model (**RoboMeter**) into a lightweight head operating directly on latents plus the instruction $\ell$. It aggregates latent tokens with AdaPool then MLPs, trained with plain MSE. Avoiding the decode-and-judge round trip is a large part of the speed win.
- **Critic $V$**: same latent-space design, MSE against bootstrapped $\lambda$-returns, so truncated-horizon rollouts can be valued beyond the imagined window.

### The three downstream loops

- **Policy evaluation** — replay recorded real action trajectories open-loop inside WEAVER, accumulating predicted reward. Tasks require 40+ iterative dynamics calls.
- **Policy improvement** — sample an $h$-step chunk from the policy, roll forward $K$ times for $H = Kh$ steps, compute advantage from reward + critic, and distil segments back into the policy **only when advantage exceeds a threshold** $\epsilon_{\text{adv}}$ (set to 0.1). The filter matters: it prevents updating at states where every sampled plan is predicted worse than current behaviour.
- **Test-time planning** — single-chunk **best-of-$N$**: sample $B$ candidate chunks, imagine each, execute the highest-advantage one. Deliberately avoids iterative dynamics calls. Uses $B=4$, horizon $h=12$.

## Experimental setup

- **Hardware/policy**: DROID setup — one Franka Emika Panda, two external Zed 2i cameras, wrist-mounted Zed Mini. Base policy is **π0.5**, a SOTA VLA trained on DROID.
- **Tasks** (5, chosen so the base policy already achieves ≥20%): Stack Bowls, PnP Bag (deformable chip bag), PnP Marker (reorient + insert), PnP Towel (soft towel into basket), Pour Beans (granular pouring).
- **Training**: pretrain on DROID for 1M steps, batch 32, lr 1e-4, **4×H100 for 10 days**; finetune on collected task data (50 rollouts/task) at lr 2e-5 for 16k steps. Timesteps downsampled by 3 → 5 Hz imagination. Actions represented as joint-position differences, with a learned velocity-to-position adapter.
- **Baseline**: **Ctrl-World**, a 1.5B diffusion WM initialised from Stable Video Diffusion, on a 256-trajectory DROID val split and a 100-trajectory OOD set. 10s autoregressive rollouts predicting 15-step (1s) action chunks. Metrics FID / FVD, plus Pearson $\rho$ and MMRV (maximum matrix ranking violation) for evaluation quality.

## Key results

- **Fidelity vs speed**: WEAVER beats Ctrl-World on FID and FVD at lower inference time, and degrades more gracefully as NFE drops (8/16/32/50). Wall-clock for a 10s chunk on one H100: **~10–30 s for WEAVER vs ~30–50 s for Ctrl-World**, with up to **16× less inference time** in the FVD-vs-time comparison. Both models are worst on wrist-camera views.
- **Long horizon**: over 150-step (10s) rollouts, WEAVER holds lower FID throughout, even when the budget is cut from 50 to 16 NFE. On OOD data the gap persists on the exterior view and is comparable on the wrist view.
- **Policy evaluation**: pretrained WEAVER beats Ctrl-World on both $\rho$ and MMRV; after finetuning, **WEAVER-FT reaches ρ = 0.870**. Pouring is the hardest task for pretrained models — granular dynamics are underrepresented in DROID.
- **Policy improvement**: finetuning π0.5 on WEAVER-generated synthetic data (1,000 advantage-filtered segments/task) comes within a **4% average gap** of finetuning on filtered *real* data; mixing real + synthetic beats real-only by **+11%**, for the headline **+38%** over the base policy. Success improves monotonically as synthetic segments scale 1,000 → 2,000 → 5,000.
- **Test-time planning**: **+15% average** success (max +20%), larger gains when the base policy is weaker. **~20× faster than Ctrl-World's pipeline** on an RTX A6000 Ada; batched candidate sampling scales sublinearly. Dynamics prediction remains the dominant cost.
- **Latent reward**: predicted reward tracks RoboMeter's ground truth across a rollout, correctly registering grasping and stacking events, and the derived advantage separates good from bad action samples.

## Ablations

The paper is structured as a design-decision distillation rather than a clean ablation table, and the appendix carries most of the isolated comparisons (NFE sweeps A3.2, noise schedules A3.4, KV-caching speedup A3.3). What is reported in the body:

- **NFE sweep** is the central ablation: Ctrl-World's quality collapses faster than WEAVER's as NFE falls — i.e. the flow-matching + rectified-distillation combination is what buys the Pareto frontier, not raw capacity (WEAVER is *smaller*: 928M vs 1.5B).
- **Cosine vs linear noise schedule** — cosine claimed higher fidelity (details deferred to A3.4).
- **Synthetic data scaling** — 1k → 5k segments improves monotonically on Pour Beans.
- **Data source comparison** — base / real-filtered / synthetic-filtered / mixed, with mixed best.
- Proprioception prediction and multi-view are motivated by failure analysis rather than ablated head-to-head in the main text; the claim that proprioception is "critical" for deformable manipulation is asserted from experience rather than isolated.

## Limitations

**Paper's own:** (1) visual observation is partial state — contacts, grasp stability, applied forces, and occluded geometry may be invisible in every view, so tactile/force-torque/depth sensing may be needed; (2) physics priors would likely help on deformable and granular tasks; (3) generation latency confines test-time planning to short-horizon reasoning over a *single* action chunk; (4) RoboMeter reward supervision is noisy — they observe it failing to distinguish fine-grained placement accuracy on PnP Marker, mitigated by the 0.1 advantage threshold.

**An honest reader would add:**
- Every headline number rests on **five tasks with 20 trials each** on one robot in one lab. That is normal for real-hardware work and still a thin base for "state-of-the-art results on robotic manipulation tasks."
- Policy evaluation success is **human-labelled on imagined rollouts** — a judgement call about whether a generated video depicts success, which is precisely where a WM's failure modes (plausible-looking but wrong) are hardest to catch.
- The ρ = 0.870 is for **WEAVER-FT**, finetuned on rollouts from the same policy family it then evaluates. The pretrained model is substantially worse, so the evaluator's accuracy is partly a function of having seen this exact setup — which limits the "evaluate arbitrary policies" pitch.
- Reward labels come from RoboMeter, so the WM's notion of success is inherited wholesale from another learned model; errors there are invisible to every metric reported.
- The comparison is against a *single* baseline (Ctrl-World). Dreamer-v4 and DreamDojo are discussed but not benchmarked.

## Why it matters [analyst's view]

The durable contribution is the **desiderata framing**, not the architecture. "Fidelity, long-horizon consistency, efficiency — and no prior WM has all three" is a genuinely useful lens, and the specific observation that **JEPA-style latents can't evaluate pixel-consuming policies** is the kind of constraint that reorganises a design space. It explains why this line has converged on frozen pretrained VAE encoders: you need a latent space that is both learnable-dynamics-friendly *and* invertible to pixels, and pretrained generative autoencoders are the only thing currently offering both. That's a real argument against the representation-only branch of world modelling for this particular use case — and it cuts against the direction [[papers/baldassarre-2025-dino-world-models]] and the JEPA line take, which is worth holding in tension rather than resolving.

The second thing worth extracting is that **latent reward heads are a systems optimisation, not just a modelling choice**. Replacing "decode to pixels → query a VLM judge" with "score the latent directly" is what makes best-of-N planning tractable at all; the paper's ~20× end-to-end speedup is substantially about *not decoding*. Any imagination-based planning loop should probably be audited for the same waste.

For the vault this slots into an increasingly crowded and directly comparable set. [[papers/lu-2026-driftworld]] attacks the same efficiency bottleneck from the opposite end — abolishing iterative denoising entirely rather than distilling it — and reports 0.95+ policy-evaluation correlation to WEAVER's 0.870, though on different tasks, so the comparison is suggestive rather than decisive. [[papers/shang-2025-roboscape]] reports r = 0.953 as a policy evaluator using physics-informed auxiliary losses (depth + keypoint consistency), which is precisely WEAVER's self-identified limitation #2. **Three papers now converging on "world model as policy evaluator" with correlation as the headline metric is a genuine trend, and the numbers are not yet comparable across them** — that's a synthesis worth writing once a common benchmark exists. [[papers/jiang-2026-robottt]] is the policy-side counterpart (context scaling for VLAs) and shares the real-hardware bimanual framing.

## Open questions / things to verify

- How much of the ρ = 0.870 survives evaluating a policy *family the WM wasn't finetuned on*? That's the actual claim the evaluation use-case needs, and it isn't tested.
- Is the +38% policy improvement partially a distillation-of-filtering effect rather than a world-model effect? An advantage-filtered *real*-data baseline exists (and gets within 4%), which is the right control — but a filter-only-no-WM ablation on synthetic-free data would isolate it better.
- Would physics priors (RoboScape-style depth/keypoint losses) close the Pour Beans gap? The authors name this themselves.
- Rectified-flow distillation is deferred to Appendix A2.3 — worth reading before re-implementing, since it's the step that unlocks planning latency.
- **Ctrl-World** [12], **Dreamer-v4** [16], **Diffusion Forcing** [7], **RoboMeter** [25], **π0.5** [21], and **SPRINT blocks** [30] are all load-bearing and all `_needs note_`. Ctrl-World especially — it's the sole baseline and the source of the memory/multi-view design.

## Connections

- Builds on: Ctrl-World (multi-view + memory architecture), Dreamer-v4 (latent reward/critic heads), Diffusion Forcing, flow matching / rectified flow, Stable Diffusion 3 VAE — `_needs note_` for all.
- Topic MOCs: [[topics/world-models]], [[topics/robotics]], [[topics/flow-matching]]
- Related in vault:
  - [[papers/lu-2026-driftworld]] — the other 2026 attack on world-model inference latency; single-pass drifting vs WEAVER's distilled few-step flow. Closest methodological rival.
  - [[papers/shang-2025-roboscape]] — physics-informed embodied WM, also validated as a policy evaluator (r = 0.953); supplies exactly the physics priors WEAVER lists as future work.
  - [[papers/jiang-2026-robottt]] — the policy side of the same real-hardware manipulation problem.
  - [[papers/gao-2025-adaworld]] — action-conditioned world modelling with learned latent actions.
  - [[papers/mazzaglia-2024-genrl]] — training policies purely in imagination, the downstream loop WEAVER's improvement experiments instantiate on real hardware.
  - [[papers/zhu-2026-sana-wm]] — same efficiency-frontier framing for world models, from the architecture side.
- Author index: [[authors/arnav-kumar-jain]]

## Selected quotes

> "JEPA-style WMs have latent states that may not be decodable into the images required to evaluate arbitrary visuomotor robot policies." — §1

> "Consistency across WM generations requires the WM to understand both what changes and what stays the same across an interaction." — §3.1

> "This indicates that our synthetic data is of such a high quality that it unlocks similar policy improvement to costly real world data." — §5.2.2
