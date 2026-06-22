---
type: moc
last_updated: 2026-06-22
status: living

---

# Dashboard — Recent

Most recent items added to the vault, regardless of category. Refreshed by
`/process-inbox`.

## This week

_(Triaged 2026-06-22 — the flow-matching / OT / generative cluster, plus robotics, attention, reliability, and geometry. Source-saved 06-10 to 06-15.)_

### Papers
- [[papers/malnick-2026-designing-ot-flows]] — *(06-10)* **Optimal Transport Flow Matching by Design**: stop *solving* OT — design a prior (low-frequency image projections) whose identity coupling to the data is empirically OT-optimal; >2× straighter trajectories and better few-step FID (+20% vs OT-FM, +38% vs IFM at 1 step on FFHQ) with no flow-model changes; composes with MeanFlow.
- [[papers/cai-2026-mode-mean-seeking]] — *(06-10)* Decoupled Diffusion Transformer for **fast long video**: a mean-seeking flow-matching head (coherence from scarce long-video data) + a mode-seeking distribution-matching head distilled from a frozen short-video teacher; FM head discarded at inference for ~30 s minute-scale video.
- [[papers/karcini-2026-robots-beyond-vla]] — *(06-10)* Position paper: generalist robotics is mis-framed as policy scaling; the real bottleneck is **grounding** abundant unstructured behavioural data. Four missing pillars — physical data engine, task-preserving retargeting, physics-grounded world models, self-improving deployment loops. VLAs are one layer, not the stack.
- [[papers/litman-2026-attention-priors]] — *(06-10)* Recasts attention as **entropic optimal transport** with an implicit uniform prior; positional encodings are heuristic approximations of an EOT-derived $\log\pi$ term; introduces a principled learnable prior incl. a dedicated key-only attention-sink bias.
- [[papers/galil-2025-sign-bit-flips]] — *(06-10)* **Deep Neural Lesion**: data-free, optimization-free attack that destroys DNNs by flipping a few weight *sign bits*; works across vision, detection, and reasoning LLMs. (reliability/hardware-faults cluster.)

### Blogs
- [[blogs/dieleman-diffusion-integral]] — *(06-10)* Sander Dieleman's unifying tutorial on **flow maps** (predict any point on a noise→data path from any other) via three consistency rules (compositionality / Lagrangian / Eulerian); the map of the 2024–26 few-step-sampling literature.
- [[blogs/jiha-autoregression-vs-diffusion]] — *(06-10)* AR vs diffusion as two solutions to **one optimal-transport problem**: AR = Knothe–Rosenblatt rearrangement (triangular, exact, ordering-biased); diffusion/flow = learned unconstrained Brenier map.
- [[blogs/flow-based-llms-intro]] — *(06-10)* Floor Eijkelboom's intro to **flow-based language models**: softmax + cross-entropy is exactly the Variational-Flow-Matching objective on the simplex; flows distill into one/two-step Categorical Flow Maps (discrete diffusion can't).
- [[blogs/accelerated-diffusion-tutorial]] — *(06-11)* CVPR 2026 "FastGen" tutorial: faster sampling, training efficient samplers, and distillation for real-time image/video generation.
- [[blogs/nicolas-riemannian-manifolds]] — *(06-15)* Foundations of Riemannian geometry for Geometric Deep Learning (manifolds, tangent spaces, geodesics, exp/log maps, metric, connection, curvature, Lie groups); part 1 of a series, payoff is computing on data manifolds with `Geomstats`.

## Last week

### Papers
- [[papers/ding-2024-diffusion-world-model]] — *(06-03)* Diffusion World Model: predicts a whole future trajectory in one diffusion pass (conditioned on state/action/return-to-go), killing the compounding error of recursive one-step rollouts; +44% over one-step models on D4RL, 4.6× faster than Decision Diffuser, closes the model-based-vs-model-free gap.
- [[papers/lee-2026-looped-diffusion-lm]] — *(06-03, looped-models cluster)* **LoopMDM**: first looped transformer for masked diffusion LMs; selectively loops early-middle layers for up to 3.3× fewer training FLOPs, +8.5 GSM8K, with loop count as an inference-time compute knob.
- [[papers/deng-2026-lt2-looped]] — *(06-03, looped-models cluster)* **LT2**: replaces softmax attention in looped transformers with subquadratic (linear/sparse) mixers — "looping turns compute into context" (rank-$T$ memory / $O(Tw)$ receptive field); hybrid beats the full-attention loop at ~5× decode throughput; converts pretrained Ouro → Ouro-Hybrid-1.4B.
- [[papers/chen-2026-training-free-looped]] — *(06-03, looped-models cluster)* loops a frozen checkpoint's mid-block at inference (no training) via the forward-Euler/ODE view; Runge–Kutta integration, +2.64 pp MMLU-Pro on Qwen3-4B; layer-mode needed for MoE.
- [[papers/bartosh-2026-dual-rate-diffusion]] — *(05-31)* heavy context encoder (every $K$-th step) + light denoiser (every step) → 2–4× cheaper diffusion inference at equal/better ImageNet FID; composes with Moment Matching Distillation (student beats teacher).
- [[papers/knight-2026-riemannian-low-rank]] — *(06-03)* Riemannian gradient descent over ten geometries for rank-factored attention params; honest **negative result** — does not conclusively beat AdamW at small scale; nice framing of AdamW/Muon as steepest descent under $\ell_\infty$/spectral norms.

### Blogs
- [[blogs/clark-surfing-uncertainty-precis]] — *(06-04, predictive-processing cluster)* Andy Clark's own précis of *Surfing Uncertainty*: the brain as a hierarchical prediction machine; prediction-as-bootstrap of world knowledge; action as active inference. The cognitive-science taproot of self-supervised prediction.
- [[blogs/alexander-surfing-uncertainty-review]] — *(06-04, predictive-processing cluster)* Scott Alexander's SSC explainer: two precision-weighted streams reconciled by Bayes at every layer; "controlled hallucination," surprisal minimization, autism (over-precise priors) vs schizophrenia (weak priors). The ML-legible on-ramp.
- [[blogs/buskell-surfing-uncertainty-review]] — *(06-04, predictive-processing cluster)* Buskell's BSPS scholarly review: PP as a "mid-level organizational sketch"; flags an ambiguity in what "moving up the hierarchy" means (scope vs gist vs hyperprior). The honest counterweight.
- [[blogs/lampinen-continual-learning-problems]] — *(06-03, Andrew Lampinen / Infinite Faculty)* argues scale + pretraining largely dissolve catastrophic interference & loss of plasticity; the real frontier of continual learning is positive transfer / cumulative learning, and dissolving the in-context-vs-parametric boundary.

## Older

### Late May – early June (exploration / open-endedness, JEPA & world-models, GPU cluster)

#### Papers
- [[papers/clune-2019-ai-gas]] — *(manual citation chase from Jiang 2022)* "AI-Generating Algorithms" with three pillars: meta-learning architectures, meta-learning learning algorithms, generating environments. Pre-LLM articulation of the "data/environment is the bottleneck" thesis.
- [[papers/schmidhuber-2013-powerplay]] — *(manual citation chase from Jiang 2022)* algorithmic framework for self-invention of training problems via Kolmogorov-complexity-ordered search over (new-task, solver-modification) pairs.
- [[papers/lehman-2011-novelty-search]] — *(manual citation chase from Jiang 2022)* foundational novelty-search paper: abandoning explicit objectives and selecting purely for behavioural novelty can outperform objective-based search on deceptive problems.
- [[papers/jiang-2022-rethinking-exploration]] — position paper (Meta/UCL/Cohere, late 2022): exploration is the bottleneck for general intelligence; proposes the outer/inner-loop generalized-exploration framework unifying SL and RL.
- [[papers/pao-huang-2026-flux-matching]] — generalizes score matching to the full Fokker–Planck-consistent family of generative vector fields; turns the vector field into a design choice; faster mixing + interpretable fields + structured dynamics at 2–4× DSM training cost.
- [[papers/wang-2025-hierarchical-reasoning-model]] — foundational HRM (Sapient Intelligence): two recurrent modules at slow/fast timescales, DEQ-style 1-step gradient; 27M params, 1000 examples solve Sudoku-Extreme / Maze-Hard / ARC-AGI without pretraining or CoT.
- [[papers/baek-2026-gram]] — extends HRM into probabilistic multi-trajectory recursion via stochastic latent transitions + amortized variational inference; introduces width-based inference-time scaling; 99.7% N-Queens 8×8 vs HRM 78.7%.
- [[papers/yang-2026-replaid-continuous-diffusion]] — closes the AR-vs-continuous-diffusion compute gap from 64× to 20× via protocol alignment; SOTA 22.1 PPL on OpenWebText among continuous DLMs; shows ELBO-variance noise schedule = linear cross-entropy.
- [[papers/guo-2022-byol-explore]] — BYOL-style self-predictive loss as joint world-model + intrinsic reward; solves DM-HARD-8 without demos.
- [[papers/khetarpal-2024-byol-ac]] — ODE theory for action-conditional BYOL representations; unifies BYOL-Π / BYOL-AC / BYOL-VAR.
- [[papers/joseph-2026-physics-video-world-models]] — interpretability of physics in video encoders; identifies the Physics Emergence Zone and a circular population code for motion direction.
- [[papers/higuera-2026-visuo-tactile-world-models]] — multi-task world model fusing vision + tactile; claims tactile grounding fixes physics-violation failure modes and yields up to 35% higher zero-shot real-robot success.
- [[papers/tong-2026-beyond-language-modeling]] — controlled from-scratch native-multimodal pretraining (Transfusion + MoE); finds a scaling asymmetry (vision more data-hungry than language) and world-modeling capabilities emerge from generic training.
- [[papers/huang-2026-semantic-tube-prediction]] — generalises JEPA to language via the "Geodesic Hypothesis"; claims 16× LLM data-efficiency over Chinchilla on NL-RX-SYNTH.
- [[papers/maes-2026-leworldmodel]] — first stable end-to-end JEPA from pixels; two-term loss (prediction MSE + SIGReg) replaces EMA / stop-gradient / multi-term-loss orthodoxy; plans up to 48× faster than DINO-WM.

#### Blogs
- [[blogs/shing-diffusionblocks]] — *(Sakana AI, 05-27, ICLR 2026 summary)* train networks one block at a time by casting block-wise dynamics as diffusion denoising; ~1/B training memory at performance comparable to end-to-end across 5 architectures (ViT/DiT/Masked-Diffusion/AR/Recurrent-depth). Bonus: replaces K-step BPTT with a single forward pass for looped Transformers. _Headline numbers in the paper, not yet verified._
- [[blogs/blainsmith-just-use-go]] — *(05-27, off the AI/math axis — CS/backend polemic)* "use boring Go": stdlib-as-framework, cheap goroutines, single static binary deploy. Low durable value; kept for the audit trail + as a boring-monolith reference.
- [[blogs/bansal-kv-cache]] — *(GPU/inference cluster, 05-25)* KV-cache management as the binding constraint in LLM serving; six-era evolution (contiguous → PagedAttention → heterogeneous → distributed → unified) mirroring OS memory management. Llama-3-70B at 8K × 32 reqs = 81.9 GB of KV cache alone.
- [[blogs/bierling-coalesced-matmul]] — *(05-25, author = vault owner)* naive vs coalesced CUDA matmul; a one-line thread-index remap → coalesced global loads → ~10.5× (1433 ms → 136 ms on an A100). Stays memory-bound; tiling next.
- [[blogs/bierling-prefix-scan]] — *(05-25, author = vault owner)* parallel prefix scan: shared-memory Hillis–Steele ($O(\log N)$ depth) + divide-and-conquer recursion over block sums for arbitrary sizes.
- [[blogs/pytorch-torch-amp]] — *(05-25)* `torch.amp` reference: `autocast` (per-op dtype) + `GradScaler` (fp16 loss scaling); wrap forward only, cast back to fp32 on exit, bf16-pretrained models overflow in fp16.
- [[blogs/aiwithmaha-cuda-concepts]] — *(05-25)* beginner glossary of 20 CUDA concepts (kernel/grid/block/thread/warp, global vs shared memory, occupancy, streams, tensor cores, cuDNN). Shared-vocabulary entry point for the GPU cluster.
- [[blogs/biswas-rlm-deep-dive]] — practitioner deep-dive on Recursive Language Models: pass-by-reference agentic scaffold (LLM operates inside a Python REPL with `llm_query` for subagents); 60M-character Lex Fridman transcript in 4 minutes for $0.20.

---

## How this dashboard works

This file is rewritten at the end of every `/process-inbox` run. It shows:

- **This week**: items added in the last 7 days, grouped by category.
- **Last week**: items added 7–14 days ago.
- **Older**: rolled-up by month.

For deeper queries see the other dashboards in `30_Knowledge/_MOCs/`:

- `dashboard-deep-read.md` — items marked `priority: high` and `read_state: queued`
- `dashboard-orphans.md` — notes with no inbound links (potentially mis-filed)
