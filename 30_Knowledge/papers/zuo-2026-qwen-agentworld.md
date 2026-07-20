---
type: paper
title: "Qwen-AgentWorld: Language World Models for General Agents"
authors: ["Yuxin Zuo", "Zikai Xiao", "Li Sheng", "Fei Huang", "Jianhong Tu", "Yuxuan Liu", "Tianyi Tang", "Xiaomeng Hu", "Yang Su", "Qingfeng Lan", "et al."]
year: 2026
venue: "arXiv 2026 (Qwen Team technical report)"
url: https://arxiv.org/abs/2606.24597
rw_id: 01kx67vpmtebqq8yh2mn7x4jrv
topics: [world-models, language-models, reinforcement-learning, reasoning]
priority: high
read_state: queued
relevance: ""
added: 2026-07-14
last_updated: 2026-07-16
---

# Qwen-AgentWorld: Language World Models for General Agents

## TL;DR

Qwen-AgentWorld is the first *native* language world model (LWM): a text
generator trained from pre-training onward to predict the next environment
observation an agent would receive, given the interaction history and the
agent's current action, across **seven agent domains** (MCP, Search, Terminal,
SWE, Android, Web, OS) unified under a single textual trajectory schema. Two MoE
models are released — **Qwen-AgentWorld-35B-A3B** and **397B-A17B** — trained by
a three-stage recipe summarized as *"CPT injects, SFT activates, RL sharpens"*:
continual pre-training injects state-transition dynamics and professional world
knowledge, SFT activates explicit next-state-prediction chain-of-thought, and RL
with a hybrid rubric-and-rule reward sharpens simulation fidelity. On the new
**AgentWorldBench** (2,170 samples, ground-truth observations from real
executions of frontier agents), the 397B model scores **58.71** overall,
narrowly beating GPT-5.4 (58.25) and every other frontier model; LWM training
lifts the 35B base by **+8.66 points** (47.73 → 56.39). Beyond fidelity, the
paper shows two payoffs: as a *decoupled* simulator it enables Sim RL that
**matches or beats real-environment RL** (e.g. WideSearch F1-item +16.29 at 35B;
50.3% vs 45.6% for a live search engine), and as a *unified* agent foundation
model an LWM-RL warm-up transfers to downstream agentic benchmarks (+8.96 avg
over 7 benchmarks, including +11.3 on out-of-domain Claw-Eval).

## Context & motivation

A world model predicts environment dynamics from current state + action and is
argued to be a prerequisite for general intelligence — the paper leans on
Richens et al. (2025), who prove that *any* agent that generalizes across a
sufficiently broad task range must have learned a world model. The authors'
framing: the agent–environment loop has two halves, a **policy** (states →
actions) and a **world model** ((states, actions) → next states), yet LLM-agent
research has focused almost entirely on the policy side. Language environments
in which LLM agents operate lack a general-purpose world model; that is "a
crucial missing piece."

The stated purpose is **not cost reduction** but "a complementary axis for
pushing the frontier." A learned LWM gives two things real environments cannot:
(i) **scalability** — turn-level simulation of arbitrary environments without
sandboxes / GUI VMs, including high-value domains where real execution is
infeasible (irreversible ops, proprietary systems); and (ii) **controllability**
— natural-language instructions that inject targeted perturbations (partial
results, injected errors, paginated responses) to expose agent weaknesses that
rarely arise in real environments. It also motivates the *unified* view: an agent
that can predict feedback before committing can in principle do no worse than one
that cannot, internalizing next-state prediction as a future-oriented analogue of
"reflection." Prior LLM-as-world-model work is largely post-hoc fine-tuning of a
general LLM (WebDreamer, WMA, Simia, RLVR-World, etc.); Qian et al. (2026)'s
negative result — agents fail to invoke world models as planning tools — is cited
as the reason to make environment modeling the *training objective from CPT
onward* rather than bolting it on.

## Method

### Problem formulation

A language world model $f_\theta$ is a conditional text generator that predicts
the next observation. Let $c$ be the system prompt, $o_t$ the environment
observation at turn $t$, and $a_t$ the agent's action at turn $t$. The model
produces

$$\hat{o}_{t+1} = f_\theta(c,\, o_{\leq t},\, a_{\leq t}), \tag{1}$$

where the conditioning context is $c$ plus the full interaction history and the
current action, and the training target is the ground-truth observation
$o_{t+1}$. Domains are either **stateless** (e.g. Search — state carried
implicitly in the dialogue) or **stateful** (Terminal, OS — explicit internal
state that evolves per action); both are handled by the same observation
sequence. The **action** is the agent's output at a turn (tool call / shell
command); the **observation** is the environment's feedback (tool response /
command output). For the three GUI domains (Android, Web, OS), observations are
**textual accessibility trees / UI view hierarchies**, not pixels — Android next
states are even predicted as renderable HTML.

### Core idea

Train one model to *be the environment* for seven domains at once, under a shared
trajectory schema, with next-state prediction as the explicit objective across
all three training stages — so a single world-modeling objective simultaneously
exercises reasoning, factual knowledge, instruction-following, and long-context
handling.

### Architecture / algorithm

**Unified environment trajectory schema.** Every domain and stage uses:

$$\texttt{system\_prompt} := \texttt{task\_desc} \oplus \texttt{action\_space} \oplus \texttt{initial\_state} \oplus \texttt{demos} \oplus \texttt{sim\_instruction}$$
$$\texttt{turn}_t := (\texttt{action}_t, \texttt{observation}_t), \qquad \texttt{trajectory} := \texttt{system\_prompt} \oplus [\texttt{turn}_1, \dots, \texttt{turn}_T]$$

The five system-prompt components: **task description** (act as world model for
this domain), **action space** (available tools/ops + calling convention),
**initial state** (starting config: packages, filesystem, UI state — optional but
strongly constrains the expected observation), **demonstrations** (optional
few-shot (action, observation) examples), and **simulation instruction**
(optional controllable conditions, e.g. "hide the answer from web_search
responses"). For most domains action-space and demos are *static*; MCP and SWE
need *per-trajectory* action spaces (tools differ per server / repo). Initial
state and simulation instruction are *dynamic* when present.

**CPT objective (Stage 1).** Standard next-token prediction, with multi-turn
trajectories framed as world modeling so the LM objective maps directly onto

$$p(o_{t+1} \mid o_{\leq t},\, a_{\leq t}).$$

Each trajectory is expanded into turn-level prediction samples (any turn $t$ can
be a target: preceding turns + current state/action form the input, $o_t$ the
target), so supervision arrives at every turn. Specialized-domain knowledge
corpora enter as single-turn data under the same loss.

**Turn-level information-theoretic loss masking.** Many tool-use turns are
boilerplate (APIs echoing inputs); their gradients are noisy but the turns can't
be deleted (later turns depend on them as context). For each (action,
observation) pair, from word sets $W_{\text{act}}, W_{\text{obs}}$ (lowercased,
deduped tokens), four statistics are computed:

- Overlap $\text{OL} = |W_{\text{act}} \cap W_{\text{obs}}| / |W_{\text{act}}|$ — how much action vocabulary the observation echoes;
- Novelty $\text{Nov} = |W_{\text{obs}} \setminus W_{\text{act}}| / |W_{\text{obs}}|$ — fraction of genuinely new info;
- Jaccard $\text{Jac} = |W_{\text{act}} \cap W_{\text{obs}}| / |W_{\text{act}} \cup W_{\text{obs}}|$ — symmetric similarity;
- Length ratio $R = |\text{obs}| / |\text{act}|$ (character level).

Turns are sorted into 7 categories with keep-ratios: retrieval / expansion /
action → 100%, transform → 50%, boilerplate → 10%, echo → 5%, other → 100%.
Masked turns are dropped from the loss but kept as context — decoupling "learning
the next state" from "learning the next token." Classification is statistical, not
by tool name, so it's tool-agnostic and needs no annotation.

**RL objective (Stage 3).** The RL algorithm is **GSPO** (Zheng et al., 2025).
The reward combines two signals:

- **Five-dimensional rubric (LLM judge):** each predicted observation scored 1–5
  on Format, Factuality, Consistency, Realism, Quality; total reward $=$ mean
  $\times 5 \in [5, 25]$; invalid judge output → reward 0. Domain-tailored judge
  prompts with content-type classification reduce false negatives from
  irreproducible details (timestamps, PIDs).
- **Rule-based verifier:** a data subset carries executable verifier code giving
  binary $0/1$, scaled to $[0, 25]$ to match the rubric range; serves as an
  objective anchor against reward hacking.

The two are combined at **9:1 (rubric:rule)**. Strict tag extraction ensures only
the predicted observation (not the thinking block) reaches the judge, preventing
self-praise from inflating scores. A key RL property is **extreme prompt–output
asymmetry**: the prompt is the full history (tens of thousands of tokens, capped
at **128k**) while the output is a single observation (hundreds–few-thousand
tokens), so per-sample compute is dominated by prompt processing.

### Derivations / why it works

_No formal derivation; empirical paper._ The theoretical backing is cited, not
re-derived: Richens et al. (2025) (general agents must contain a world model) and
the LeCun et al. (2022) world-model–actor architecture motivate the *unified*
paradigm. The cross-domain generalization experiment (§below) provides the
empirical "why it works": RL on Terminal alone transfers to MCP/SWE/Search,
arguing that RL reinforces *generalizable* world knowledge (how environments
respond, how errors propagate, how state composes) rather than domain-specific
output formats.

### Training procedure

Three **strictly disjoint** data pools. **CPT** draws from dedicated agent
infrastructure (containerized sandboxes, MCP servers, persistent
terminal/Android/browser/desktop-OS sessions on Ubuntu/macOS/Android VMs), open
interaction traces (cleaned by a multi-agent fetch→denoise→segment→align→score
pipeline), and **specialized-domain world-knowledge corpora** (industrial
control, cybersecurity, law, medicine, finance, current affairs) that ground the
model where trajectories alone can't (a compliance platform needs legal
knowledge; a hospital system needs medical knowledge). The report cites **>10M
environment interaction trajectories** across the 7 domains.

**SFT and RL** use only internally accumulated trajectories. Data statistics
(Table 2): SFT total **7,094** trajectories, RL train **92,308**, avg 19,443
tokens / 13.4 turns. Filtering drops <2-turn sequences, tool-calls outside the
declared action space (MCP/SWE), GUI trajectories broken by env failures
(missing state files, CAPTCHAs, HTTP errors); plus **retry-cycle skipping**
(skip garbage→error→retry pairs while preserving state) and **no-change-turn
filtering** for GUI (drop turns where pre/post state is unchanged, else the model
learns to copy state).

- **Stage 1 CPT:** non-thinking trajectories, next-token loss + turn masking.
- **Stage 2 SFT:** activates explicit thinking. Uses a **256k-token context
  window**. Prompt-template diversification (each sample gets one of 10 template
  variants) + **rejection sampling**: 3 rollouts per query from a general-purpose
  reasoning model, judge-scored and pairwise-compared, best kept if above
  threshold. Retention **69.2%** (10,250 → 7,094 queries; Table 4).
- **Stage 3 RL:** GSPO with the 9:1 hybrid reward; prompt capped at 128k.

**System-prompt templates via AutoResearch.** Rather than hand-craft prompts,
prompt optimization is posed as an automated-research loop (propose → evaluate on
held-out real trajectories via a judge → refine), 10 iterations × 12 parallel
runs seeded with distinct style directives, yielding 12 variants **v0–v11**
(~30-line minimal to ~1,100-line spec-style), human-audited. Pools draw disjoint
subsets: **RL uses v0, CPT uses v1, SFT samples v2–v11**.

**Training stability** — three failure modes fixed: (1) **Reward collapse from
multi-turn expansion** — shared long prefixes across expanded samples collapse
reward variance (related to the "Echo Trap"); fixed by expanding **exactly one
turn per trajectory** in the RL pool. (2) **Reward shaping** — two alternatives
fail: *Reference-Reward* (pairwise A/B vs initial checkpoint against ground truth)
converges slowly (sparse/noisy); *Turing-Test Reward* (is this observation from a
real env?) barely converges (false-negative rate too high when prediction ≈
ground truth). The 5-dim rubric + rule anchor converges stably. (3) **Reward
hacking via self-praise** — policy inserts affirmations ("operation completed
successfully…") to inflate scores; mitigated by the rule anchor, content-type
classification (deterministic content judged by exact match → self-praise earns
0), and strict tag separation.

### Inference / sampling

At test time the LWM is queried per turn: given system prompt + history + current
action, it emits a chain-of-thought then the predicted observation. Sampling uses
**temperature 0.6 with thinking enabled**. As a **simulator** (App I), the policy
agent and LWM are separate models; the LWM rolls out environment responses for up
to 50 interaction turns of multi-turn Sim RL, with controllable simulation
instructions shaping turn-level behavior (e.g. "tease information without fully
answering," forcing extra web_extractor calls). As a **unified agent** (App II),
the same model both selects actions and predicts states, performing "mental
simulation" of environment responses inside the thinking trace before executing.

## Experimental setup

- **AgentWorldBench:** 2,170 evaluation samples across the 7 domains, built from
  real-environment executions of **5 frontier agents on 9 established
  benchmarks** (Terminal-Bench 1.0 & 2.0, OSWorld-Verified, Tool Decathlon,
  in-house SWE, etc.). Text-domain trajectories mostly **Claude Opus 4.6** (an
  early subset Sonnet 4.5); GUI adds 3 Qwen-family models. Four construction
  principles: widely-used queries, frontier-agent trajectories, real
  observations, and out-of-distribution (train/bench partitioned at data-source
  level). Text domains = 72.4% of the benchmark; GUI domains 9.2% each. Avg
  context ranges 12.9k (Terminal) to 59.3k tokens (MCP, full tool schema in
  prompt). Turn sampling: first + last + 3 intermediate turns per trajectory.
- **Evaluation protocol:** reference-grounded LLM-judge scoring on 5 dimensions
  (Format, Factuality, Consistency, Realism, Quality), 1–5 → normalized 0–100.
  Judge sees ground-truth alongside prediction (converts open-ended judgment into
  factual comparison). Three-way content split — deterministic (exact match),
  pre-existing (format+plausibility), runtime metadata (format+range) — avoids
  penalizing irreproducible PIDs/timestamps. **Judge = GPT-5.2** (chosen for
  highest Turing-test accuracy; Gemini 3 Flash most lenient, GPT-5.2 most
  stringent; cross-judge Spearman ρ = 0.92–0.99).
- **Baselines (14):** frontier proprietary (Claude Opus 4.8 / 4.6, Sonnet 4.6,
  GPT-5.4, Gemini 3.1 Pro), open-weight (DeepSeek-V4-Pro, Kimi K2.6, GLM-5.1,
  MiniMax-M2.7), and Qwen family without LWM training (Qwen3.5 / 3.6 checkpoints,
  which isolate the training effect since they share architecture).
- Open-weight models served with SGLang; proprietary via official APIs.

## Key results

- **Main (Table 5, 0–100 rubric mean):** Qwen-AgentWorld-**397B-A17B = 58.71**
  overall, best of all models, beating GPT-5.4 (58.25). On text domains it leads
  (58.07 vs GPT-5.4's 56.84), strongest on **Terminal (57.73 vs 53.69)** and
  **SWE (68.49 vs 66.29)**. On GUI it ranks 5th (59.69); Claude Opus 4.6/4.8 lead
  (~61) — attributed to multimodal pre-training that text-only modeling misses.
- **Effect of LWM training:** 397B base 54.74 → 58.71; **35B: 47.73 → 56.39
  (+8.66)**, lifting the 35B model above Claude Sonnet 4.6 (56.04). Qwen3.6
  checkpoints without LWM training (Plus 50.81, Max-Preview 52.42) sit well below,
  ruling out "just general capability."
- **Hardest domain:** Search — best score ~37.82, roughly half of SWE/MCP —
  because web content constantly evolves. Factuality is the weakest of the five
  dimensions throughout (and shows the largest RL gain, +11.3% relative).
- **Cross-domain generalization:** RL Stage-3 on **Terminal alone** → Terminal
  +14.2 (32.8→47.0) within 100 steps, and held-out domains transfer *without any
  domain signal*: SWE +11.5, Search +11.8, MCP +5.0 — gains appearing within the
  first 10 RL steps.
- **App I — decoupled simulator:** Sim RL on 4k synthesized (out-of-training)
  OpenClaw environments → Claw-Eval +4.3 (65.4→69.7), QwenClawBench +7.1
  (47.9→55.0). *Controllable* Sim RL → Tool Decathlon +3.7, **MCPMark +12.3**,
  **WideSearch F1-item +16.29** (34.02→50.31 at 35B; +3.87 at 397B where base is
  already 70.11). Controllable Sim RL **beats Real RL** against a live search
  engine (F1-item **50.3% vs 45.6%**) and shapes behavior: Sim-RL agents *increase*
  web_extractor calls (2.5→4.0) while Real-RL agents *decrease* them (2.5→1.5),
  because simulated snippets deliberately withhold answers. Fictional-world Sim RL
  (fully invented, self-consistent search worlds) transfers to real WideSearch.
- **App II — unified agent foundation model:** a **single-turn, non-agentic**
  LWM-RL warm-up (predict next state, no tool calls) transfers to **multi-turn,
  tool-calling** tasks with no further fine-tuning: **+8.96 avg over 7
  benchmarks**. In-domain: WideSearch F1-item +12.79, Terminal-Bench 2.0 +6.30,
  SWE-Bench Verified +3.4, SWE-Bench Pro +5.2. Out-of-domain (absent from LWM
  training entirely): **Claw-Eval +11.3, QwenClawBench +9.7, BFCL v4 +9.0**.
  Mechanism evidence: on Terminal-Bench 2.0 the warm-up raises the agent's
  in-trace environment-prediction accuracy from **69.9% → 78.3% (+8.4)**;
  qualitatively it "predicts before acting" (the Postfix `mailman` case study:
  correctly predicting recipient validation precedes transport routing lets it fix
  `local_recipient_maps` instead of oscillating on `transport_maps`).

## Ablations

- **Reward design ablation:** the 5-dim rubric + rule anchor is compared against
  Reference-Reward and Turing-Test rewards; only the rubric+rule combo converges
  stably (§training stability above).
- **Reward mixing ratio:** 9:1 rubric:rule chosen to balance multi-dimensional
  feedback against strict binary correctness.
- **Multi-turn expansion:** expanding many samples per trajectory collapses RL;
  one-turn-per-trajectory in the RL pool fixes it.
- **Simulator quality (App I):** using Qwen3.6-Plus as simulator yields only
  marginal Sim-RL gains vs large gains from the dedicated Qwen-AgentWorld-397B —
  the training pipeline, not just scale, produces useful simulators.
- **Controllability necessity:** uncontrolled Sim RL gives negligible gain (Tool
  Decathlon even drops 32.4→31.5); controllability is a *prerequisite*, not just a
  magnitude booster, in MCP/Search.
- **Loss-masking categories:** 7 statistical categories with tuned keep-ratios
  (Table 3) implement the information-theoretic masking.
- **Micro-fidelity from RL (§7.2):** RL improves low-salience details — Search URL
  identifiers become more realistic across steps; Terminal byte-arithmetic (`wc
  -c` → exact 53 via character enumeration incl. `\n`); MCP schema consistency
  across 9 sequential Notion API calls.

## Limitations

- **GUI weakness (paper-acknowledged):** text-only world modeling trails
  multimodal frontier models on GUI domains (ranks 5th); future work proposes
  fusing screenshots with text state.
- **Factuality** is the persistent weak dimension, especially Search — simulating
  evolving web content and maintaining factual consistency over long retrieval
  chains remains hard for all models.
- **State is the bottleneck** (their own takeaway): Sim RL fidelity depends on a
  sufficiently detailed initial state; without it, downstream gains vanish.
- _[analyst's view]_ Evaluation leans on LLM judges even with reference grounding;
  self-praise reward hacking was observed and only partially mitigated. Many
  baselines are 2026-dated proprietary models evaluated via API, so exact scores
  are not reproducible. The 397B model beats GPT-5.4 by only **0.46** overall —
  the headline "outperforms frontier models" is narrow at the top end, and much of
  the value is in the *applications*, not the leaderboard margin.

## Why it matters [analyst's view]

This is the clearest large-scale demonstration that **language** world models
(not video/latent ones) can be trained as *foundation models* and pay off in two
orthogonal ways. The *decoupled* result — controllable Sim RL matching or beating
a live search engine — is the strongest evidence yet that a learned text
simulator can be a better RL environment than reality, because it can inject
adversarial conditions on demand. The *unified* result reframes world-modeling as
a cheap, transferable **agent pre-training objective**: a single-turn prediction
warm-up with no tool calls raises multi-turn agentic performance across unseen
domains, which is a genuinely surprising transfer story and connects to the
"predict-before-acting" thesis. For the vault, this sits at the intersection of
world models and agentic RL: it's the language-domain analogue of the
world-model-for-RL line ([[papers/jiang-2025-world4rl]]) and the
world-*action*-model paradigm ([[papers/ye-2026-world-action-models]]), and its
"CPT injects, SFT activates, RL sharpens" staging is a reusable recipe template.

## Open questions / things to verify

- How much of the App-II transfer is world-modeling per se vs. simply more
  reasoning-heavy RL on in-distribution-style data? The concurrent ECHO result
  (auxiliary observation-prediction loss ~doubles Terminal-Bench 2.0) suggests the
  effect is real, but a controlled comparison would help.
- Does the 9:1 rubric:rule ratio and the v0/v1/v2–v11 template split generalize,
  or are they tuned to these 7 domains?
- Reward hacking via self-praise was observed — how robust are the mitigations
  under longer RL? The rule verifier only covers a subset of data.
- Fictional-world Sim RL for Search is intriguing but the sim-to-real transfer
  mechanism (why invented facts teach real search skill) is asserted more than
  explained.
- GUI gap: would a multimodal extension close it, or is text-state modeling
  fundamentally lossy for pixel-driven UIs?

## Connections

- Builds on / relates to: [[papers/jiang-2025-world4rl]] (world models for RL),
  [[papers/ye-2026-world-action-models]] (world-action-model paradigm, cited),
  [[papers/gao-2025-adaworld]] (action-conditioned world modeling),
  [[papers/ding-2024-world-models-survey]] (survey context).
- Contrasts with: video/latent world models (DreamerV3/4, Genie 3, Cosmos,
  V-JEPA 2) — Qwen-AgentWorld predicts *structured text* observations instead of
  pixels; and post-hoc LLM-simulator work (WebDreamer, WMA, RLVR-World, Simia) —
  here environment modeling is the objective from CPT onward.
- Topic MOCs: [[topics/world-models]], [[topics/language-models]],
  [[topics/reinforcement-learning]], [[topics/reasoning]]
- Author indices: [[authors/yuxin-zuo]], [[authors/fei-huang]],
  [[authors/jianhong-tu]], [[authors/bowen-yu]], [[authors/jingren-zhou]]

## Selected quotes (optional)

> "any agent capable of generalizing across a sufficiently broad range of tasks
> must have learned a world model, establishing world models not merely as useful
> but as necessary for general-purpose agents." — §1 (paraphrasing Richens et
> al., 2025)

> "Not for Cost Reduction, but as a Complementary Axis for Pushing the Frontier"
> — §1 header

> "CPT injects, SFT activates, RL sharpens" — §3, the three-stage principle

> "an agent capable of predicting environment feedback prior to committing to an
> action can in principle perform no worse than its counterpart lacking such
> capacity." — §1
