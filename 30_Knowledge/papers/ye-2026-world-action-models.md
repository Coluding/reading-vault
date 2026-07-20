---
type: paper
title: "World Action Models are Zero-shot Policies"
authors: ["Seonghyeon Ye", "Yunhao Ge", "Kaiyuan Zheng", "Shenyuan Gao", "Sihyun Yu", "George Kurian", "Suneel Indupuru", "You Liang Tan", "Chuning Zhu", "Jiannan Xiang", "Ayaan Malik", "Kyungmin Lee", "William Liang", "Nadun Ranawaka", "Jiasheng Gu", "Yinzhen Xu", "Guanzhi Wang", "Fengyuan Hu", "Avnish Narayan", "Johan Bjorck", "Jing Wang", "Gwanghyun Kim", "Dantong Niu", "Ruijie Zheng", "Yuqi Xie", "Jimmy Wu", "Qi Wang", "Ryan Julian", "Danfei Xu", "Yilun Du", "Yevgen Chebotar", "Scott Reed", "Jan Kautz", "Yuke Zhu", "Linxi Fan", "Joel Jang"]
year: 2026
venue: "arXiv preprint"
url: https://arxiv.org/abs/2602.15922
rw_id: 01kvz9r9028y71g5b3kn4jg7yy
topics: [world-models, policy-learning, robotics]
priority: high
read_state: queued
relevance: ""
added: 2026-07-08
last_updated: 2026-07-14
---

# World Action Models are Zero-shot Policies

## TL;DR

The paper introduces **DreamZero**, a 14B **World Action Model (WAM)** built on a pretrained image-to-video diffusion backbone (Wan2.1-I2V-14B) that jointly predicts future video frames *and* robot actions, reframing action learning as inverse dynamics on top of a predicted visual future rather than direct state→action imitation. The core trick: because a video-diffusion prior already encodes physical dynamics from web-scale video, the robot policy only has to learn "what actions produce this predicted future," which lets it learn from **diverse, non-repetitive** robot data instead of many repeated demonstrations per task. This yields **>2× improvement** in zero-shot task/environment generalization over state-of-the-art VLAs (GR00T N1.6, π0.5) in real-robot experiments — e.g. 62.2% vs 27.4% average task progress on seen tasks in unseen environments. Through an autoregressive chunk-wise architecture plus a suite of system, implementation, and model-level optimizations (culminating in **DreamZero-Flash**), they get a **38× inference speedup** enabling real-time closed-loop control at **7Hz** for a 14B video-diffusion model. They further show cross-embodiment transfer from **video-only** data (>42% relative gain on unseen tasks from 10–20 min of another robot's or a human's video) and **few-shot embodiment adaptation** to a new robot from just 30 minutes of play data.

## Context & motivation

State-of-the-art **Vision-Language-Action (VLA)** models (GR00T, π0/π0.5, RT-2, Gemini Robotics) extend pretrained VLMs to output motor actions. They inherit strong *semantic* generalization — they know *what* to do ("move coke can to Taylor Swift") — but are pretrained on **static image-text** data, so they lack representations of *how* actions execute with spatial/dynamical precision. They fail on genuinely new motions ("untie the shoelace") unless that skill appeared in the robot training data, and achieving environment generalization has meant collecting teleop data across hundreds of environments.

DreamZero's thesis: **video is a dense representation of how the world evolves.** A model that predicts future video learns from *every consecutive frame pair*, and a video-diffusion backbone pretrained on internet video already encodes physical dynamics. The authors term this family **World Action Models (WAMs)** — models that predict future world state *and* action in an aligned way — and argue WAMs occupy a sweet spot between end-to-end VLAs (seamless gradient flow) and world-model planning (dense predictive supervision), while avoiding the "learn dynamics from scratch in latent space" cost of JEPA/Dreamer-style latent world models. The name is deliberately WAM not VAM (Video Action Model): video is just one world-modeling target; future WAMs could align actions to tactile/force/latent predictions.

## Method

### Problem formulation

DreamZero jointly predicts video $\mathbf{o}_{l:l+H}$ and actions $\mathbf{a}_{l:l+H}$ conditioned on a language instruction $\mathbf{c}$, proprioceptive state $\mathbf{q}_l$, and observation history $\mathbf{o}_{0:l}$, where $H$ is a fixed horizon and $l$ a random trajectory index. The key observation is that joint video+action prediction *decomposes* into autoregressive video prediction times an inverse-dynamics model (IDM):

$$\underbrace{\pi_0(\mathbf{o}_{l:l+H}, \mathbf{a}_{l:l+H} \mid \mathbf{o}_{0:l}, \mathbf{c}, \mathbf{q}_l)}_{\text{DreamZero}} = \underbrace{\pi_0(\mathbf{o}_{l:l+H} \mid \mathbf{o}_{0:l}, \mathbf{c}, \mathbf{q}_l)}_{\text{video prediction}} \cdot \underbrace{\pi_0(\mathbf{a}_{l:l+H} \mid \mathbf{o}_{0:l+H}, \mathbf{q}_l)}_{\text{IDM}} \tag{1}$$

Rather than training two separate models (a video predictor and an IDM), they train a **single end-to-end model** on the joint objective — the claim being that end-to-end coupling gives tighter video-action alignment, and since the video-prediction half is already solved by pretraining, the model mainly needs to (a) adapt video prediction to robot embodiment footage and (b) extract corresponding actions.

### Core idea

Video prediction serves as an **implicit visual planner** that guides action generation; "improving robotic capabilities reduces to improving video generation." Learning an implicit IDM from predicted futures is hypothesized to be more sample-efficient than direct policy learning because it reuses the pretrained model's physics understanding.

### Architecture / algorithm

Built on **Wan2.1-I2V-14B-480P** as backbone. Minimal new parameters are added — **state encoder, action encoder, action decoder** — to preserve the video model's generalization. Multi-view robot data is handled by concatenating all camera views into a single frame (no backbone change). Three inputs: visual context (encoded via a VAE), language (text encoder), proprioceptive state (state encoder); processed by an **autoregressive DiT** with **flow matching**; separate decoders emit future video frames and actions.

Autoregression is applied **only to the video modality** (not actions, to avoid closed-loop action error propagation) and brings three advantages: (1) KV-cache → faster inference, (2) visual-history conditioning for the next generation, (3) it sidesteps the modality-alignment problems of bidirectional diffusion, which needs fixed-length sequences and thus video subsampling that distorts native FPS and harms video-action alignment. Video is generated **chunk-wise**: each chunk has a fixed number of latent frames matched to the action horizon, allowing training on variable-length videos (analogous to LLM variable-length token training).

**Training objective (flow matching, teacher-forced).** For chunk index $k$ and denoising timestep $t_k \in [0,1]$, noisy video latent $\mathbf{z}_{t_k}^k$ and noisy normalized actions $\mathbf{a}_{t_k}^k$ are linear interpolations between clean vectors and Gaussian noise:

$$\mathbf{z}_{t_k}^k = t_k \mathbf{z}_1^k + (1 - t_k)\mathbf{z}_0^k, \qquad \mathbf{a}_{t_k}^k = t_k \mathbf{a}_1^k + (1 - t_k)\mathbf{a}_0^k \tag{2}$$

where $\mathbf{z}_0, \mathbf{a}_0 \sim \mathcal{N}(0, I)$ are noise, and $\mathbf{z}_1$ (clean video latent) / $\mathbf{a}_1$ (normalized action) are the targets. All frames in a chunk share one timestep $t_k$; different chunks get independent timesteps. Clean context from previous chunks is $C_k = \{(\mathbf{z}_1^j, \mathbf{a}_1^j)\}_{j=1}^{k-1}$. The model $\mathbf{u}_\theta$ predicts the **joint velocity** for both modalities:

$$\mathcal{L}(\theta) = \mathbb{E}_{\mathbf{z},\mathbf{a},\{t_k\}}\left[\frac{1}{K}\sum_{k=1}^K w(t_k)\left\| \mathbf{u}_\theta([\mathbf{z}_{t_k}^k, \mathbf{a}_{t_k}^k]; C_k, \mathbf{c}, \mathbf{q}_k, t_k) - \mathbf{v}^k \right\|^2\right] \tag{3}$$

where $w(t_k) > 0$ is a weighting function, and the velocity target $\mathbf{v}^k := [\mathbf{z}_1^k, \mathbf{a}_1^k] - [\mathbf{z}_0^k, \mathbf{a}_0^k]$. **Teacher forcing:** the noisy current chunk is denoised conditioned on the *clean* previous chunks. Trajectory-level updates with attention masking let the current noisy chunk attend to clean past context. Notably, DreamZero **shares the denoising timestep between video and action** (unlike several prior WAMs) for faster early-training convergence.

### Derivations / why it works

The load-bearing derivation is Eq. (1): joint distribution = video prediction × IDM. This is what justifies initializing from a video-diffusion model — the video-prediction factor is (approximately) already learned at web scale, so finetuning concentrates capacity on the robot-specific IDM factor. The flow-matching formulation (Eqs. 2–3) is standard rectified-flow/velocity prediction, extended to a **joint** $[\mathbf{z}, \mathbf{a}]$ vector so both modalities are denoised under one velocity field, which is what enforces cross-modal alignment. A key closed-loop insight: because at test time ground-truth observations replace generated frames in the KV cache after each execution, the autoregressive video error-accumulation problem is eliminated — an advantage unique to the embodied closed-loop setting versus pure video generation.

### Training procedure

Backbone Wan2.1-I2V-14B-480P. Train **100K steps, global batch 128** for both AgiBot (~500 h teleop) and DROID (Franka). Update **all DiT blocks + state/action encoder + action decoder**; **freeze** text encoder, image encoder, VAE. Filter idle actions; default action representation = **relative joint positions**. LoRA was tried but gave suboptimal results. Post-training on downstream tasks runs 50K steps per task, again all params except frozen text/image/VAE encoders.

### Inference / sampling (real-time execution)

A naive single-GPU implementation needs ~5.7 s per action chunk (bottlenecks: 16 diffusion steps for smooth actions, 14B DiT cost, sequential execution blocking motion). Fixes, staged:

- **Asynchronous closed-loop execution:** the motion controller keeps executing the most recent action chunk while inference runs concurrently on the latest observation. This turns "inference must finish before the robot moves" into "inference must finish before the current chunk expires." They deploy at action horizon 48 @ 30Hz control (1.6 s/chunk) for bimanual robots, targeting <~200 ms latency.
- **CFG Parallelism:** the two classifier-free-guidance passes (conditional/unconditional) are split across two GPUs → 47% lower per-step latency.
- **DiT Caching:** exploits directional consistency of flow-matching velocities — when cosine similarity between successive velocities exceeds a threshold, reuse cached velocities, cutting effective DiT steps from 16→4.
- **Implementation-level:** torch.compile + CUDA Graphs; NVFP4 weight/activation quantization on Blackwell (QKV/Softmax kept FP8, non-linear ops FP16); cuDNN attention backend; scheduler ops migrated to GPU.
- **DreamZero-Flash (model-level):** decouples video and action noise schedules during training. Standard DreamZero samples a shared $t \sim U(0,1)$; but few-step inference needs to predict *clean actions from still-noisy video*. Flash biases the video timestep toward high noise via $t_\text{video} = 1 - \beta$, $\beta \sim \text{Beta}(\alpha,\beta)$ with $\alpha > \beta$ (they use $\text{Beta}(7,1)$, giving $\mathbb{E}[t_\text{video}] = 0.125$, mostly noisy), while action timesteps stay uniform. This matches the single-step inference regime and lets them go **4→1 diffusion step**, cutting latency ~350ms→~150ms. **Action chunk smoothing:** upsample chunk 2×, apply Savitzky-Golay filter, downsample.

Cumulative speedups (Table 1): system+implementation give ~9× on H100 / ~16× on GB200; adding Flash reaches **38×** on GB200 (5.7s → ~150ms). All optimizations except DiT caching and quantization are mathematically equivalent to baseline with no measurable degradation.

## Experimental setup

- **Embodiments:** AgiBot G1 (mobile bimanual) and Franka single-arm; separate pretraining per embodiment. Cross-embodiment uses YAM robot + human egocentric data.
- **Data:** ~500 h AgiBot teleop across **22 environments** (homes, restaurants, supermarkets, coffee shops, offices), 7.2K episodes, ~4.4 min/episode, ~42 subtasks/episode — deliberately diverse and long-horizon. Franka uses **DROID** (public, heterogeneous).
- **Baselines:** GR00T N1.6 and π0.5, each in two flavors — *from-scratch* (VLM weights only, apples-to-apples) and *from-pretrained* (official checkpoints pretrained on thousands of hours of cross-embodiment robot data). Both retrained on identical data as DreamZero; compute budget matched (batch size, gradient steps).
- **Metric:** average **task progress** (partial completion, 0–1), plus success rate. AgiBot: 10 seen + 10 unseen tasks, 8 rollouts × 4 robots (80 rollouts/checkpoint). DROID: 20 seen + 20 unseen tasks. Default eval is **unseen environment, unseen objects** (eval site is a different geographic location than training) — so every benchmark is OOD by construction. Task granularity = motion + object type (folding a black shirt = "seen"; folding socks = "unseen" because the motion differs).

## Key results

- **Q1 (learning from diverse data, seen tasks / unseen env):** DreamZero **62.2%** avg task progress vs best pretrained VLA **27.4%** (>2×), while from-scratch VLAs are ~0%. Similar win on DROID-Franka. Notably, most DreamZero *failures stem from video-generation errors*, not action extraction — the policy faithfully executes whatever the video predicts, implying video-backbone improvements translate directly to policy gains.
- **Q2 (zero-shot unseen tasks):** DreamZero **39.5%** vs pretrained VLA **16.3%** (from-scratch VLAs <1%) on 10 held-out tasks (untie shoelaces, ironing, painting, shaking hands). Strong on "Remove Hat from Mannequin" (85.7%), "Shake Hands" (59.2%). On DROID: 49% progress / 22.5% success vs GR00T 31%/12.5% and π0.5 33%/7.5%. Pretrained VLAs tend to reach-and-grasp regardless of instruction (overfit to pick-and-place).
- **Q3 (post-training):** matches or beats VLAs across shirt folding / table bussing / fruit packing (significantly better on fruit packing); environment generalization is *retained* after task-specific fine-tuning, outperforming SOTA VLAs by ~10% avg progress.
- **Q4 (cross-embodiment, video-only):** co-training 10K steps on a 1:1 mix with 72 video-only trajectories of 9 unseen tasks. Robot→robot (YAM→AgiBot, 20 min) lifts **38.3% → 55.4%**; human→robot (12 min egocentric) **38.3% → 54.3%**. Uses *no action labels* for the transfer data.
- **Q5 (few-shot embodiment adaptation):** DreamZero-AgiBot adapts to the entirely new YAM robot from **55 trajectories / ~30 min of play data**, retaining strong language following and generalizing to unseen objects (pumpkins, teddy bears, cup noodles).
- **Q6 (Flash):** on table bussing, plain DreamZero drops 83%→52% going 4→1 step, while **DreamZero-Flash holds 74%** at 1 step (only 9% below the 4-step baseline) at ~2× speed.

## Ablations

(All ablation models: 50K steps, batch 32, evaluated on PnP-Easy; Table 4.)

- **Data diversity:** 500h *diverse* vs 500h *repetitive* (70 tasks with many repeats): **33% → 50%**. Rationale — since video prediction is inherited from pretraining, the bottleneck is the IDM, which needs diverse state-action correspondences that repetitive data lacks.
- **Model scale:** 14B vs 5B: **50% vs 21%**; the 5B model hallucinates visually and those errors propagate to actions. Scaling VLAs to 8B/32B still gives **0%** on diverse data — capacity alone doesn't fix VLAs' diverse-data difficulty.
- **Architecture (AR vs bidirectional):** task progress similar (50% both), but **AR produces much smoother motions** (backprop through full action sequences → temporal consistency) and AR inference is **3–4× faster** via KV caching.

## Limitations

Paper's own: (1) **inference cost** — 7Hz on 2× GB200 vs VLAs at >20Hz on consumer GPUs; edge deployment awaits smaller strong video backbones. (2) **Short-horizon memory** — visual memory is only ~6 s; long-horizon needs a System-2 planner or much longer context; memory-dependent tasks were not evaluated/post-trained. (3) **High-precision tasks** — inherits behavior-cloning limits on sub-cm precision (key insertion, fine assembly); the breadth-first data strategy underrepresents dense precise demos. (4) No multi-embodiment joint training yet; per-embodiment pretraining only. (5) No established scaling laws for WAMs.

Honest reader's view [analyst's view]: results are **task-progress** (partial-completion) scores, which are more forgiving than binary success — the DROID success rates (22.5%) are still low in absolute terms. Cross-embodiment gains are from very small in-lab video sets (12–20 min) and "moderate" absolute success. The from-scratch VLA baselines hitting ~0% is a strong claim that hinges on the fairness of the matched-compute setup. Comparisons are largely to the authors' own/NVIDIA-adjacent models (GR00T).

## Why it matters [analyst's view]

This is a flagship demonstration that a **video-diffusion prior is a better substrate for a generalist robot policy than a VLM prior**, at least for motion/skill generalization — reframing "collect more robot demos" as "improve the video model." The decomposition in Eq. (1) is the conceptual crux: it turns the robotics data problem into a (comparatively data-rich) video problem plus a smaller IDM-learning problem, which is *why* diverse non-repetitive data and video-only cross-embodiment transfer work. The engineering contribution (38× speedup, DreamZero-Flash's decoupled noise schedules) is what makes a 14B video-diffusion model actually deployable at 7Hz — this is the practical unlock, and Flash's train/test noise-mismatch fix is a reusable trick for any few-step diffusion-conditioned prediction. It sharpens the "world model" taxonomy: WAMs (pixel-space, pretrained) vs latent world models (V-JEPA 2, Dreamer) that model $p(s_{t+1}\mid s_t, a_t)$ and need test-time planning/search. If the failure mode really is "video errors, not action errors," then the whole field's progress on video generation becomes a free tailwind for robotics.

## Open questions / things to verify

- Does the "failures are video errors" claim hold under rigorous attribution, or is action extraction also silently failing on hard contact-rich tasks?
- How far does video-only human→robot transfer scale? The paper hypothesizes web-scale egocentric human video could give skills without action labels — untested at scale.
- Scaling laws for WAMs (model × data × compute) — explicitly left open; would determine whether this beats VLAs at fixed compute.
- Binary success rates (vs task progress) across the full benchmark would clarify absolute reliability.
- How does the shared-timestep choice (vs decoupled, as in Flash) trade off against alignment for the *main* model, beyond convergence speed?

## Release status

*(Not addressed in the originally fetched paper content; checked live on 2026-07-14 via the project page, which is outside the arXiv text so treat as supplementary, not part of the paper's own claims.)*

- **Code**: released — GitHub repo linked from the project page (`github.com/dreamzero0/dreamzero`).
- **Dataset**: no full downloadable dataset dump found. The project page links an "Eval Gallery" (100+ zero-shot rollout videos) and a "30-min play data" gallery for the YAM cross-embodiment experiments (`dreamzero0.github.io/yam_gallery/`), but that page is a filterable *visualizer* (credited "adapted from Jimmy Wu"), not a data-download portal — no raw trajectories/actions confirmed downloadable there.
- Training data (AgiBot ~500h teleop, DROID) are pre-existing datasets, not something DreamZero itself is releasing.
- _needs verification_: whether the GitHub repo itself bundles or links a downloadable data release (not checked — only the project page and its galleries were fetched).

## Connections

- Topic MOCs: [[topics/world-models]], [[topics/policy-learning]], [[topics/robotics]]
- Author indices: [[authors/seonghyeon-ye]], [[authors/joel-jang]], [[authors/linxi-fan]], [[authors/yuke-zhu]], [[authors/jan-kautz]], [[authors/yilun-du]]
- Related: [[papers/jiang-2024-dexmimicgen]] (data-generation for manipulation), [[papers/porcher-2026-flowwm]] (flow-based world model)
- Contrasts with latent world models (V-JEPA 2, Dreamer) and VLAs (GR00T N1.6, π0.5)

## Selected quotes

> "By jointly modeling video and action, DreamZero learns diverse skills effectively from heterogeneous robot data without relying on repetitive demonstrations." — Abstract

> "This formulation not only means that improving robotic capabilities reduces to improving video generation, but also enables three capabilities that elude current VLAs." — §2.2, "Why WAMs"

> "Most DreamZero failures stem from video generation errors rather than action prediction—the policy faithfully executes whatever trajectory the video predicts." — §5.1, Q1
