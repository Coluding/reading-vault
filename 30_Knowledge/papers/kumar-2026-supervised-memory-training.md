---
type: paper
title: "Pretraining Recurrent Networks without Recurrence"
authors: ["Akarsh Kumar", "Phillip Isola"]
year: 2026
venue: "arXiv preprint"
url: https://arxiv.org/abs/2606.06479
rw_id: 01kvq3sxbk3wzx86rdew9hcc4w
topics: [recurrence, sequence-models, optimization]
priority: high
read_state: queued
relevance: ""
added: 2026-07-08
last_updated: 2026-07-08
---

## TL;DR

The paper introduces **Supervised Memory Training (SMT)**, a way to train nonlinear RNNs *without* backpropagation through time (BPTT). Instead of unrolling the RNN and propagating gradients across the whole sequence, SMT first trains a bidirectional Transformer encoder to compress the past into a memory state $m_t$ (a "predictive state" that retains only what's needed to predict the future), and then reduces RNN training to plain supervised learning on one-step memory transitions $(m_t, x_{t+1}) \to m_{t+1}$. The key trick: by timestamping observations, the past becomes a *set* rather than a *sequence*, so the optimal memory is a permutation-invariant function computable in parallel over time. This gives a stable **O(1)** gradient path between any two tokens (vs BPTT's O(T)), fully time-parallel training, and fixed-memory inference. On synthetic credit-assignment tasks, character-level language modeling (TinyStories), and pixel-sequence modeling (MNIST/Sketchy), SMT (with a DAgger-style finetune, "DMT") outperforms BPTT — most starkly on long-range tasks where BPTT's recency bias fails. It is positioned as a *pretraining* method that needs lightweight post-training to exceed the teacher's expressivity.

## Context & motivation

RNNs store a fixed-size compressed memory of the past that becomes useful only later; the central training difficulty is **credit assignment** across the delay between writing and using a memory. The standard solution, BPTT, unrolls the recurrence and backpropagates gradients across the full computation graph. This has two well-known flaws: (1) it is sequential in time (a for-loop), so it doesn't parallelize on modern hardware; and (2) it produces high-variance gradients that vanish or explode across up to O(T) steps, causing recency bias (when they vanish) or chaotic instability (when they explode). The paper asks: *is recurrent credit propagation avoidable?*

Transformers sidestepped both problems by being time-parallel and having an O(1) credit path, but they don't compress the past — they store the entire token history and attend to all of it, so memory grows with sequence length (prohibitive for lifetime-scale sequences; sliding windows lose old information). Linear-attention RNNs get time-parallel training and fixed memory, but their linear transition function fundamentally limits expressivity and fails on state-tracking tasks. SMT aims to combine all four desiderata at once: **time-parallel training, O(1) long-range credit, fixed-memory inference, and full nonlinear expressivity.**

## Method

### Problem formulation
Causal conditional sequence modeling: given input $x = [x_0,\dots,x_T]$ and output $y = [y_0,\dots,y_T]$, learn $\prod_{t=0}^{T} p_\theta(y_t \mid x_{\le t})$, where each $y_t$ depends only on $x_0,\dots,x_t$ (autoregressive modeling is the special case $x_t = y_{t-1}$). An RNN models this with a fixed-size latent state updated by the transition function $f_\theta$:
$$m_{t+1} = f_\theta(m_t, x_{t+1}) \tag{1}$$
with readout $p_\theta(y_t \mid x_{\le t}) = \mathrm{softmax}(g_\theta(m_t))$. Here $m_t$ is the *memory*: it should remember important past information and forget the rest. Under BPTT the unrolled graph $m_t = f_\theta(\dots f_\theta(f_\theta(m_\emptyset, x_0), x_1)\dots, x_t)$ (with $m_\emptyset = \mathbf{0}$) means gradients travel up to O(T) steps, and vanish/explode depending on the singular values of the Jacobian of $f_\theta$.

### Core idea
Decouple *what to remember* (learning the memory representation — a **non-sequential** problem) from *how to update memory* (learning the dynamics — sequential, but supervisable one step at a time). If an oracle $Q$ gave the optimal memory $m^*_t = Q(x^{\text{ctx}}_t)$ at every timestep, RNN training would collapse to supervised regression on transitions $(m^*_t, x_{t+1}) \to m^*_{t+1}$. The insight is that $Q$ need not be recurrent: by augmenting each observation with its timestamp, the context $[x_0,\dots,x_t]$ can be losslessly viewed as the **set** $\{(x_0,0),(x_1,1),\dots,(x_t,t)\}$, so the optimal memory is a *permutation-invariant* function estimable by a parallel-over-time model (a Transformer).

### Architecture / algorithm
Three networks: a nonlinear RNN $f_\theta$; a bidirectional Transformer encoder $E_\phi$; a causal Transformer decoder $D_\psi$. For each timestep $t$, decompose into past and future:
$$\mathbf{x}^{\text{ctx}}_t = [x_0,\dots,x_t], \quad \mathbf{x}^{\text{fut}}_t = [x_{t+1},\dots,x_T], \quad \mathbf{y}^{\text{fut}}_t = [y_t,\dots,y_T]$$
The encoder maps context to memory, $m_t = E_\phi(\mathbf{x}^{\text{ctx}}_t)$. The decoder predicts the future outputs from that memory plus teacher-forced future inputs:
$$p_{\phi,\psi}(\mathbf{y}^{\text{fut}}_t \mid \mathbf{x}^{\text{ctx}}_t, \mathbf{x}^{\text{fut}}_t) = \prod_{\tau=t}^{T} p_\psi(y_\tau \mid m_t, \mathbf{x}_{t+1:\tau}) = D_\psi(m_t, \mathbf{x}^{\text{fut}}_t)$$
This future-prediction objective *operationalizes the predictive state*: $m_t$ is trained to retain exactly the past information needed to predict the future. Three losses:

**Decoding loss** (shapes the memory representation), with CE the sequence-level cross-entropy:
$$\mathcal{L}^{\text{dec}}_t = \mathrm{CE}\!\left(\mathbf{y}^{\text{fut}}_t,\, p_{\phi,\psi}(\mathbf{y}^{\text{fut}}_t \mid \mathbf{x}^{\text{ctx}}_t, \mathbf{x}^{\text{fut}}_t)\right) \tag{2}$$

**Dynamics loss** (trains the RNN *and* forces $m_t$ to be Markovian, i.e. $m_{t+1}$ predictable from $(m_t, x_{t+1})$ alone), with $\hat m_{t+1} = f_\theta(m_t, x_{t+1})$:
$$\mathcal{L}^{\text{dyn}}_t = \mathrm{MSE}(\hat m_{t+1}, m_{t+1}) \tag{3}$$

**Uniformity loss** (prevents memory-space collapse):
$$\mathcal{L}^{\text{unif}} = \log \mathbb{E}_{t_a, t_b \sim [0,\dots,T]} \exp\!\left(-2\|m_{t_a} - m_{t_b}\|_2^2\right) \tag{4}$$

Full objective, a weighted sum:
$$\mathcal{L}^{\text{smt}} = \lambda_{\text{dec}}\,\mathbb{E}_t[\mathcal{L}^{\text{dec}}_t] + \lambda_{\text{dyn}}\,\mathbb{E}_t[\mathcal{L}^{\text{dyn}}_t] + \lambda_{\text{unif}}\,\mathcal{L}^{\text{unif}} \tag{5}$$

In principle $E_\phi, D_\psi$ can be trained with only $\mathcal{L}^{\text{dec}}$ and $f_\theta$ separately with only $\mathcal{L}^{\text{dyn}}$ (proof in their Appendix F), but joint training of all three is better in practice because it explicitly shapes $m_t$ to be Markovian and yields extra temporal-credit benefits. To save compute, the expectation over $t$ is Monte-Carlo estimated by sampling a **single** random timestep per sequence, giving a training memory footprint of **O(M + T)** vs BPTT's O(MT), where M is memory size.

**DAgger Memory Training (DMT).** After SMT, the RNN has low one-step error on encoder-provided memories, but at test time it is unrolled on its *own* predicted memories, so errors accumulate and the RNN trajectory $[\hat m_0,\dots,\hat m_T]$ **drifts** from the encoder trajectory $[m_0,\dots,m_T]$ (drift $\delta_t = \mathrm{MSE}(\hat m_t, m_t)$). DMT is an on-policy imitation-learning finetune: compute the encoder trajectory, unroll the RNN, then train on labels $(\hat m_t, x_{t+1}) \to m_{t+1}$ instead of $(m_t, x_{t+1}) \to m_{t+1}$:
$$\mathcal{L}^{\text{dmt}} = \mathbb{E}_t[\mathrm{MSE}(\hat m_t, m_t)] \tag{6}$$
DMT freezes encoder+decoder, trains only the RNN with a small LR. It unrolls the RNN (and can optionally propagate gradients through time) but is *not* standard BPTT because the long-range credit was already assigned in the encoder labels — it is a lightweight drift-correction phase, not the primary learner. DMT is not time-parallel (but the authors note it could be parallelized via DEER).

SMT is behavior cloning on encoder memories (off-policy IL); DMT is on-policy IL over the RNN's own induced memory distribution.

### Derivations / why it works
The load-bearing argument is about **where credit flows**. In BPTT, the gradient $\|\partial \mathcal{L}/\partial m_t\|$ must traverse the recurrent chain and vanishes/explodes with $t$. In SMT the credit path between any two tokens is O(1) and independent of $t$, because the encoder assigns long-range credit in parallel via attention, and the RNN only ever learns a one-step map — it is never unrolled, so there is no long backward chain. A subtle ablation finding (see below): even when the future length $T_f = 1$ (no direct credit path to early memories through the decoder), *joint* SMT still solves T-length credit-assignment tasks. The only possible mechanism is **credit amortization across optimization steps**: each gradient step passes information from $m_t$ back to $m_{t-1}$ through $f_\theta$, so T gradient steps propagate T steps back in the sequence — meaning a T-length task needs at least T gradient steps. The authors liken this to value bootstrapping in RL.

### Training procedure
Nonlinear RNN backbones tested: Transformer-based, MLP-based, and GRU. Datasets: character-level TinyStories (language); raster-scan pixel-sequence modeling of sparse MNIST and Sketchy images ("Attneave's task", framed after perceptual psychology). Context length truncated to $T_c$, future to $T_f$. Best hyperparameters from the ablation: $\lambda_{\text{dyn}} = 0.1$, $\lambda_{\text{unif}} = 0.001$. Runs capped at one day on a single H200 GPU. (Full architecture/optimizer details deferred to their Appendix B.)

### Inference / sampling
At inference the RNN runs autoregressively with O(1) memory and O(1) compute per generated token (vs Transformer's O(T)). It uses only its own predicted memories $\hat m_{t+1} = f_\theta(\hat m_t, x_{t+1})$ — the encoder/decoder are discarded. This is the payoff for training an RNN rather than just keeping the Transformer teacher.

## Experimental setup

- **Baselines:** "BPTT RNN" (standard training); "SMT Encoder*" (the teacher Transformer with the same memory bottleneck, used as a reference upper bound); "SMT→DMT RNN" (the full method).
- **Datasets:** TinyStories (char LM); sparse MNIST and Sketchy (pixel-sequence modeling); five synthetic probes.
- **Synthetic probes** (each sweeping a difficulty axis): Retrieval (gradient stability), String Copy (memory capacity), Stack Operations (state tracking), Keys-Values (associative recall), Modular Arithmetic (in-context learning).
- **Metrics:** validation loss/task accuracy, plus *sequential FLOPs* (serial-compute proxy) and *data (tokens processed)* as efficiency axes.

## Key results

- On **all five synthetic tasks across all settings**, SMT→DMT beats BPTT. BPTT degrades as sequences lengthen even on easy tasks (e.g. retrieval); SMT is roughly agnostic to sequence length and solves the hard credit-assignment probes — the one exception where it still struggles is associative recall.
- **Attneave pixel modeling:** BPTT (even with a GRU) fails to capture the long-range structure; SMT→DMT with a *non-gated* RNN captures stroke structure and produces recognizable MNIST/Sketchy samples — direct evidence SMT has no recency bias.
- **Efficiency:** SMT Encoder and SMT→DMT are substantially more sequential-compute efficient than BPTT (Transformer and MLP backbones). Data efficiency is comparable on TinyStories but *significantly better* on MNIST, consistent with the longer-range memory demands of pixels vs text.
- **Scaling:** smooth, predictable improvements with larger context length, larger memory state, and more parameters; the RNN tracks the teacher more closely at larger scale. The encoder follows a standard power-law; the RNN has a differently-shaped but smooth curve.
- **Compression as a scaling axis:** for a fixed target loss, more compute buys *more compression* (smaller memory state) — proposed as a novel property to scale. (Transformers do no compression, which may explain their training efficiency.)
- **Length generalization:** an SMT→DMT RNN underperforms its Transformer teacher on in-distribution lengths but *generalizes far better* to longer-than-training sequences on synthetic state tracking — the RNN's finite-state inductive bias beats the Transformer's "growing lookup table."

## Ablations

- **Predictive-state / detached-RNN:** sweeping $T_f$ shows the task is solvable when $T_f$ is large enough (O(1) credit path everywhere) *or* under joint training even at $T_f = 1$ — revealing the credit-amortization mechanism above.
- **λ coefficients:** best RNNs at $\lambda_{\text{dyn}} = 0.1$, $\lambda_{\text{unif}} = 0.001$; setting $\lambda_{\text{unif}} = 0$ preserves RNN performance but collapses the memory space.
- **Drift / DMT:** DMT reduces rollout drift and improves RNN performance across SMT hyperparameter settings; DMT-discovered RNNs may start with higher drift but plateau at a lower *equilibrium* drift, which is only partially predicted by one-step drift.
- **Gradient analysis:** BPTT's $\|\partial \mathcal{L}/\partial m_t\|$ vanishes/explodes with $t$ and depends on init; SMT's is flat in $t$ and init-agnostic.
- **Memory-space geometry (2D visualization):** retrieval induces a few collapsed FSM-like states; string copy induces a tree-like geometry matching the space of all strings — showing memory forms task-appropriate temporal abstractions.

## Limitations

- **GRU backbone fails** under SMT→DMT: GRU induces memory-space collapse during SMT, degrading rollout.
- **Bounded by the teacher's expressivity.** The Transformer teacher is provably limited (low-circuit-depth class), so an SMT RNN inherits that ceiling; BPTT finetuning may be required to exceed it. (Author-acknowledged.)
- **Not for reasoning as-is:** intermediate steps aren't supervised, so SMT alone isn't suited to multi-step reasoning without post-training.
- **DMT is not time-parallel** — it reintroduces a sequential phase (though lightweight, and potentially parallelizable via DEER).
- Only a single $m_t$ per sequence is trained; training all memories gave no gain at their scale but may matter at larger scale.

## Why it matters [analyst's view]

This is a genuinely different answer to the oldest problem in recurrent nets: instead of making BPTT's gradients better-behaved (orthogonal parameterizations, gating, gradient clipping), it **removes BPTT entirely** by manufacturing supervised targets for the memory. The reframing of "the past as a timestamped *set*, not a sequence" is the crux and is elegant — it converts a serial credit-assignment problem into a parallel representation-learning problem plus a trivial one-step regression. If it scales, it could revive nonlinear RNNs as a practical class for unbounded-horizon / lifelong-learning settings where Transformers' growing KV cache is fatal, precisely the regime linear-attention models can't fully serve because of expressivity limits. The "compression as a scaling axis" idea is a provocative reframe of scaling laws worth watching. The obvious caveat is the teacher ceiling — SMT is explicitly a *pretraining* method, and the whole pitch rests on post-training being able to push past the teacher, which they argue by analogy to Transformers but don't fully demonstrate at scale here.

## Open questions / things to verify

- Does the teacher-expressivity ceiling actually lift with BPTT/other post-training, and by how much? This is the make-or-break claim and is asserted, not shown at scale.
- Why does SMT still struggle on **associative recall** specifically, when it solves state tracking and in-context learning?
- Does the compression-scaling law hold at LLM scale, or is it a small-model artifact?
- Can the drift equilibrium be predicted/minimized *during* SMT (one-step) rather than requiring the separate DMT phase?
- How does this compare head-to-head against modern linear-attention/SSM models on real language benchmarks (only the teacher-bottlenecked reference is compared here)?
- Relationship to concurrent NextLat (Teoh et al.): equivalent under a particular hyperparameter setting but differs in the T=1 emphasis — worth reading side by side.

## Connections

- Topic MOCs: [[topics/recurrence]], [[topics/sequence-models]], [[topics/optimization]]
- Author indices: [[authors/akarsh-kumar]], [[authors/phillip-isola]]
- Related: predictive state representations (PSRs); linear-attention / SSM models (Mamba-family); parallelizing nonlinear RNNs via Newton/DEER (Lim et al.); NextLat (Teoh et al., concurrent). No existing vault paper notes obviously linked yet.

## Selected quotes

> "By decoupling *what to remember* from *how to update memory*, SMT enables time-parallel RNN training with a stable O(1) length gradient path between any two tokens—without ever unrolling the RNN." — Abstract

> "Our key insight is that, by augmenting each observation with its timestamp, the past can instead be losslessly represented as a *set* of timestamped events, rather than a sequence." — §1 Introduction

> "This implies that solving T sequence length credit assignment task when $T_f = 1$, requires at least T *gradient optimization steps*. This credit amortization phenomenon is reminiscent of value bootstrapping in RL." — §3.6 Ablations
