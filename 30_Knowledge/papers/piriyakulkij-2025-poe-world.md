---
type: paper
title: "PoE-World: Compositional World Modeling with Products of Programmatic Experts"
authors: ["Wasu Top Piriyakulkij", "Yichao Liang", "Hao Tang", "Adrian Weller", "Marta Kryven", "Kevin Ellis"]
year: 2025
venue: NeurIPS 2025
url: https://arxiv.org/abs/2505.10819
rw_id: 01ky5renddzya0p0m6cb4f6s9y
topics: [world-models, reinforcement-learning, language-models]
priority: high
read_state: queued
relevance: ""
added: 2026-07-23
last_updated: 2026-07-23
---

# PoE-World: Compositional World Modeling with Products of Programmatic Experts

## TL;DR

PoE-World represents a world model not as a neural net and not as one monolithic program, but as an **exponentially-weighted product of hundreds of small LLM-synthesized Python programs**, each encoding a single causal law ("if the player touches a skull, the player dies"). Each deterministic program is interpreted as a probability distribution over the next symbolic (object-centric) observation; a gradient-fit weight per expert lets bad programs be softly downweighted and pruned, and a separately learned set of hard constraints rules out physically impossible scenes. From a demonstration of under 1000 frames (<1 min of gameplay, never even reaching positive reward), PoE-World + a TAMP-style hierarchical planner is the only method tested that gets positive score on Montezuma's Revenge (100 ± 0.00 vs 0.00 for PPO even at 20M environment steps), and it generalizes zero-shot to rearranged "Alt" versions of both Pong and Montezuma's Revenge with no new demonstrations. The learned models run to 4000+ lines of code vs. WorldCoder's <100, and roughly double WorldCoder's next-observation prediction accuracy on held-out random frames (0.31 vs 0.10 on MR; 0.43 vs 0.12 on MR-Alt). The authors claim this is the first symbolic world model learned for environments of this complexity.

## Context & motivation

Neural world models (Dreamer et al.) are flexible but data-hungry and generalize poorly — the paper notes Diamond (diffusion world model) still imagines players walking through walls or teleporting after ~100 hours of observations. Programmatic world models (WorldCoder, CodeWorldModels), where an LLM writes a Python transition function, are far more sample-efficient and extrapolate systematically, but they search for **one large program describing everything**, a discrete combinatorial search that has not scaled beyond text and gridworld games. PoE-World's contribution is to break that search apart: learn hundreds of small programs, each a context-specific "expert", and probabilistically aggregate them. The framing is explicitly inspired by the cognitive-science view of mind as a community of interacting experts (Fodor, Minsky's Society of Mind, Dennett, Marcus) merged with the learning-as-program-synthesis paradigm (Lake et al., Ellis et al.). Scope caveats the paper states upfront: it models fine-grained pixel-level *movement* but not visual appearance (symbolic observations come from an object detector), it does not attempt exploration (it learns from a given demonstration), and it does not try to solve planning itself — planning is just how the model is evaluated.

## Method

### Problem formulation

A sequential decision problem is $(O, A, P, R)$: observation space, action space, dynamics $P = p_{env}(o_{t+1}\mid o_{1:t}, a_{1:t})$, reward. Note the **full-history conditioning** — this is formally equivalent to a POMDP but deliberately avoids a compressed Markov latent state (see below). Given trajectories $D = \{\tau_i\}_{i=1}^n$, $\tau = (o_{1:T+1}, a_{1:T})$, the goal is a model $\hat{P} = p_{model}(o_{t+1}\mid o_{1:t}, a_{1:t})$ found by empirical risk minimization:

$$p_{model}^{*} = \underset{p_{model}}{\arg\min} \sum_{(o_{1:T+1}, a_{1:T}) \in D} \sum_{t=1}^{T} \ell(p_{model}; o_{1:t+1}, a_{1:t})$$

with $\ell$ the negative log-likelihood $-\log p_{model}(o_{t+1}\mid o_{1:t}, a_{1:t})$. The learned model is then used by a model-based agent via lookahead planning or as an RL training simulator.

### Core idea

Decompose the world program into **hundreds of small programs, each one causal law**, combined as a weighted product of experts — modular (each expert only "has opinions" about its own aspect of the world) and learnable (no search for a single monolithic program).

### Architecture / algorithm

**The product-of-experts representation.** The world model is

$$p_{\theta}(o_{t+1}\mid o_{1:t}, a_{1:t}) \propto \prod_{i} p_{i}^{expert}(o_{t+1}\mid o_{1:t}, a_{1:t})^{\theta_{i}}$$

where each $p_i^{expert}$ is a small program interpreted as a distribution and $\theta_i \geq 0$ is a learned scalar weight (exponent). Why a *product*: each expert constrains only the features it speaks about and is uniform (uninformative) elsewhere, so multiplying lets many partial opinions compose into one sharp joint prediction — e.g. one expert encodes "touching a skull kills the player", another "action LEFT on a platform sets x-velocity to −2"; neither mentions the other's feature. The exponent $\theta_i$ controls how much each expert's opinion counts, which is how buggy synthesized programs get neutralized without deleting code.

**Factored states make the normalizer tractable.** Observations are object-centric: a list of objects, each with bounding box, x/y velocities, and visibility, stored as separate features. Assuming each feature $f$ of $o_{t+1}$ is conditionally independent given the full history, the intractable global partition function splits per-feature:

$$p_{\theta}(o_{t+1}\mid o_{1:t}, a_{1:t}) = \prod_{f} \frac{1}{Z_{f}} \prod_{i} p_{i}^{expert}(o_{t+1}^{f}\mid o_{1:t}, a_{1:t})^{\theta_{i}}, \qquad Z_{f} = \sum_{o^{f}} \prod_{i} p_{i}^{expert}(o^{f}\mid o_{1:t}, a_{1:t})^{\theta_{i}}$$

where $o^f$ ranges over values of a single object attribute — a small discrete sum, so each $Z_f$ is computable exactly. In the appendix this is refined further: each expert only touches a single attribute of a single object *type*, so the product also factorizes over object types (eq. 6), $p_\theta \propto \prod_{\text{obj-type}} \prod_i p_i^{\text{obj-type\_expert}}(\cdot)^{\theta_i}$.

**Why full history instead of a POMDP latent.** Learning a global latent state would entangle all experts: every expert conditions on the latent, so adding a new latent variable (e.g. "how long the player has been falling") would change the input/output space of *every* program and force global joint rewrites. Conditioning on raw history lets independent mechanisms be learned independently — this is the load-bearing design decision behind the modularity claim.

**Interpreting deterministic Python as distributions.** The LLM is asked to write plain deterministic Python (not probabilistic programs — plain Python is far more prevalent in LLM training data, and the authors found it much more effective). Each expert is a function `alter_{obj_type}_objects(obj_list, action) -> obj_list` that conditionally sets object attributes using hand-built helper classes (`Obj` with `touches()`, `ObjList`, and marker classes `RandomValues`/`SeqValues` that flag which attributes the program set). The program-to-distribution interpreter puts a single-peak distribution (plus noise, to keep alternative values at non-zero probability) on every attribute the program set, and a **uniform distribution on every attribute it didn't touch** — so an expert whose if-condition fails contributes nothing (uniform), which is exactly what makes the product composition clean.

**Hard constraints.** The soft product over-approximates possible futures (e.g. a fuzzy "falls downward" expert can predict the player sinking into the ground). To sharpen predictions, a collection of hard constraints $c_j : O \to \{0,1\}$ is also synthesized (same LLM pipeline) and multiplied in as an indicator:

$$p_{\theta}(o_{t+1}\mid o_{1:t}, a_{1:t}) \propto \prod_{i} p_{i}^{expert}(o_{t+1}\mid o_{1:t}, a_{1:t})^{\theta_{i}} \cdot \mathbb{1}\Big[\bigvee_{j} c_{j}(o_{t+1})\Big]$$

A **disjunction** (any one constraint satisfied suffices) rather than conjunction, because video-game physics is peculiar — a player climbing a ladder legitimately overlaps the platform; the authors say a conjunction would be the better choice in the real world. The MR constraints learned are alignment rules like "player's feet must align with the top of the platform" and "player's body must align with the center of the ladder". Constraints that contradict observations or explain <1% of observations are pruned.

**Multi-timestep experts.** A horizon-$H$ expert is reduced to next-step form by assuming its per-step predictions independent: $p^{expert}(o_{t+1:t+H}\mid o_{1:t}, a_{1:t}) = \prod_{k=1}^{H} p^{expert}(o_{t+k}\mid o_{1:t}, a_{1:t})$ (these use the `SeqValues` marker, e.g. a jump arc `[-6, -7, -4, 0, 2, 6, 9]` of y-velocities).

### Derivations / why it works

_No formal theory; the paper's checklist states it includes no theoretical results._ The justificatory logic is the factorization chain above: per-feature conditional independence → per-feature partition functions → tractable exact normalization; full-history conditioning → no shared latent → independent expert learning; uniform-off-support program semantics → products compose without experts fighting outside their domain.

### Training procedure (the learning loop)

1. **Synthesize experts** from observed trajectories via LLM (gpt-4o-2024-08-06, disk-cached responses, ~$20 of OpenAI credit per run). Transitions are processed in batches of 10. Ten LLM-based synthesis modules each target a different aspect (passive motion, interaction effects, object creation/deletion, ...). Template per module: turn a transition batch into a text representation (object list + actions + per-object-type attribute changes) → prompt for **natural-language causal explanations** ("player objects that touch a ladder set y-velocity to −4") → prompt to compile each explanation into a program.
2. **Fit weights** by maximum likelihood: $\theta^* = \arg\max_{\theta} \sum_{(o_{1:T+1},a_{1:T}) \in D} \sum_{t=1}^{T} \log p_{\theta}(o_{t+1}\mid o_{1:t}, a_{1:t})$ — eq. (1) instantiated with NLL loss. Optimizer: **L-BFGS with strong Wolfe line search, lr = 1, 4 epochs, no mini-batching, plus L1 regularization (weight 1)** on $\theta$; L-BFGS beat Adam and SGD because data and parameter counts are small. Weight fitting ignores the hard constraints so it stays gradient-friendly.
3. **Prune** experts with $\theta_i < \delta = 0.01$.
4. **Repeat** whenever new trajectories arrive — the agent acts, collects data, and "debugs" its model online.

Compute: world modeling on 4 CPUs / 64 GB, ~8 hours per run (including LLM latency).

### Inference / planning

Two uses. (a) **Simulator for RL**: pretrain PPO inside the world model, then fine-tune in real Atari. (b) **Lookahead planning**: pick $a^{*}_{t:t+H} = \arg\max_{a_{t:t+H}} \mathbb{E}_{p_{\theta}(o_{t+1:t+H}\mid o_{1:t}, a_{1:t+H})}\big[\sum_{k=0}^{H-1} R(o_{t+k+1}; \cdot)\big]$ with the model rolled out autoregressively.

Because MR needs 100+ actions before the first reward (naive search space $8^{100}$ over 8 buttons), they build a **TAMP-inspired hierarchical planner**: (1) learn an abstract graph whose nodes are abstract states defined by *object contact* and whose edges exist iff a low-level planner finds a traversal in simulation; (2) BFS in the graph for a shortest path to the goal (touch the key) — the high-level plan; (3) execute each subgoal with an online low-level planning agent; (4) on a "false" edge, update the world model, delete the edge, replan. Low-level planners: a modified MCTS (Manhattan-distance-to-goal heuristic instead of rollouts; **max** instead of expected value in backup, for optimism; exploration constant 1 for MR / 10 for Pong, ×10 every 1000 iterations) and a greedy search over "sticky-action" chunks (repeated primitive actions of length 1, 4, 8 → action space $3n$). MR tries MCTS for 4000 iterations then falls back to greedy; the other three environments use greedy only. The MR agent replans only 40% of the time when the plan breaks; because of this stochasticity, they run the planner 3× per initial world model and treat that as part of training.

## Experimental setup

- **Domains**: Atari Pong and Montezuma's Revenge (ALE, frameskip 3), plus compositional-extrapolation variants **Pong-Alt** (3 balls, 3 enemies — three synced layered Pongs) and **MR-Alt** (stacked/flipped lower sections of room 1, Kangaroo-like map, jump-over-skulls) — no demonstrations given for the Alt versions; world models transfer zero-shot from the base-game demos.
- **Observations**: OCAtari parses frames into object lists (category, bounding box, velocities) by reverse-engineering RAM; the authors had to manually patch OCAtari per game, which is why the full Atari suite is infeasible.
- **Demonstrations**: <1000 frames per base game, *unsuccessful* gameplay (the MR demo never earns positive reward); PoE-World/WorldCoder agents then train ≤3k environment steps before evaluation.
- **Baselines**: PPO (stable-baselines3, MlpPolicy, frame-stack 4; for Alt games fine-tuned from a 20M-step base-game model), ReAct-style LLM-as-agent, WorldCoder (single-program world model, given the same planner as PoE-World for fairness).
- **Metrics**: game score (Pong: point differential at 21; MR: positive iff key collected), next-observation and per-attribute prediction accuracy on train demo and 1000 random test frames; error bars over 5 seeds.

## Key results

- **Agent scores (Table 1)**: PoE-World + Planner is best in the low-data regime on all four environments — Pong −12.33 ± 0.88, Pong-Alt −13.67 ± 0.67, **MR 100 ± 0.00, MR-Alt 100 ± 0.00** — while PPO@100k, ReAct, and WorldCoder+Planner all score ≤ −17 on Pong variants and 0.00 on both MR variants. PPO needs >1M steps to pass PoE-World on Pong and **never** goes positive on MR even at 20M steps; only PoE-World gets positive MR reward, base or Alt.
- **Prediction accuracy (Tables 2–3)**: next-observation test accuracy 0.31 vs WorldCoder's 0.10 (MR), 0.43 vs 0.12 (MR-Alt); train 0.75 vs 0.36 (MR), 0.51 vs 0.33 (Pong). Per-attribute test accuracy 0.76 vs 0.58 (MR), 0.83 vs 0.65 (MR-Alt). Exception: Pong test frames from a random agent are uninformative (random play rarely hits the ball), so all methods look similar there.
- **World-model-as-simulator (Fig. 6)**: PPO pretrained inside the PoE-World model beats random after 200k real steps vs 1M for vanilla PPO; asymptotically both converge — pretraining is a warm start, not a ceiling raiser.
- **Qualitative**: PoE-World MR models are 4000+ lines vs WorldCoder's <100; WorldCoder's model lets the player "fly" and hallucinates nonexistent bullet-firing, with no granular way to downweight buggy parts — exactly what per-expert weights provide.

## Ablations

**Hard constraints (Table 4, MR only — Pong scenes are almost all physically possible anyway)**: removing them drops planning success from 5/9 to 2/9 (MR) and 4/9 to 2/9 (MR-Alt), while next-observation prediction accuracy is statistically unchanged (0.31 vs 0.30; 0.43 vs 0.43). Authors' interpretation: constraints don't turn bad predictions good — they do "damage control," refining poor predictions just enough to be usable for long-horizon planning. A nice dissociation between one-step prediction metrics and downstream planning utility.

## Limitations

**Paper's own**: no learning from pixels (symbolic object detections assumed); exploration, decision-making, and reward learning are all out of scope; OCAtari needs per-game manual patches, blocking full-Atari evaluation; the planner is stochastic and unoptimized (planning left to future work); the model over-approximates possible futures by design.

**Analyst-flagged**: Evaluation is two games (plus their Alt variants) — chosen partly because they're where OCAtari could be fixed, so selection effects are possible. Pong scores remain negative (−12.33), i.e. the agent still loses; the win is sample efficiency, not mastery. The helper-class layer (`Obj.touches`, side/percent touch parameters, `RandomValues`/`SeqValues`) embeds substantial hand-engineered inductive bias that the "learning from a demonstration" framing understates. Weight optimization ignoring hard constraints means the fitted $\theta$ optimizes a different distribution than the one deployed. Per-feature and per-timestep independence assumptions are strong; the paper does not quantify what they cost. ~8h wall-clock and $20 LLM budget per run is modest in dollars but slow as an inner loop.

## Why it matters [analyst's view]

This is the strongest evidence yet in the vault for the "world models as code" thesis: the trick that makes program synthesis scale is *not* better search over one big program but a probabilistic composition scheme (weighted PoE + soft-to-hard constraint split) that makes hundreds of independently-synthesized, individually-buggy programs jointly usable. The weight-then-prune mechanism is effectively a differentiable code-review layer over LLM output. The full-history-instead-of-latent-state argument is a genuinely interesting design principle: modularity of the *learner* dictates the probabilistic formulation, the reverse of the usual order. It sits directly in the "structured/causal simulator interface" line the vault tracks via [[papers/chen-2026-actionable-simulators]] (which cites PoE-World as an exemplar of products of programmatic experts), and is a concrete instance of the compositional world-model agenda in [[papers/momennejad-2026-compositional-open-ended]] — here compositionality is literal (experts recombine over novel object arrangements, giving zero-shot Alt-level transfer). Contrast with the neural end of the vault's world-model spectrum ([[papers/ding-2024-world-models-survey]], [[papers/ding-2024-diffusion-world-model]]): PoE-World trades pixel-level generality for four-orders-of-magnitude data efficiency and an interpretable, editable model. The closing speculation — that a program-structured world model turns exploration into software testing — is a genuinely novel framing worth watching.

## Open questions / things to verify

- How gracefully does the per-feature independence assumption degrade in environments with tightly coupled dynamics (e.g. multi-object collisions)?
- The disjunctive hard-constraint semantics is admittedly a video-game hack; what does the conjunctive real-world version look like, and does weight fitting need to account for constraints then?
- Could the abstract-graph planner's "false edge → debug model" loop be made the primary learning signal, closing the exploration gap the paper punts on?
- WorldCoder comparison uses PoE-World's planner and re-tailored prompts — worth checking the original WorldCoder paper's own numbers for calibration.
- How far does zero-shot transfer stretch beyond recombinations of *seen* object types (the paper's own "entity/relational composition" boundary, citing Sehgal et al.)?
- References to chase: WorldCoder (Tang et al., NeurIPS 2024), CodeWorldModels (Dainese et al., 2024), Schema Networks (Kansky et al., 2017), Theory-based RL (Tsividis et al., 2021), REx code repair (Tang et al., 2024).

## Connections

- Builds on: WorldCoder (Tang et al. 2024 — same lab, shared author Hao Tang; the monolithic-program predecessor) — _needs note_
- Cited as exemplar by: [[papers/chen-2026-actionable-simulators]] (products of programmatic experts, structured/causal world-model interfaces)
- Thematic kin: [[papers/momennejad-2026-compositional-open-ended]] (compositional world models for open-ended settings)
- Contrasts with: [[papers/ding-2024-diffusion-world-model]] (neural/diffusion end of the world-model spectrum); surveyed landscape in [[papers/ding-2024-world-models-survey]]
- Topic MOCs: [[topics/world-models]], [[topics/reinforcement-learning]], [[topics/language-models]]
- Author indices: [[authors/kevin-ellis]] (if/when created — Ellis lab is the program-synthesis-for-world-models nexus)

## Selected quotes

> "Algorithmically, our key idea is to decompose the problem of learning a world program into learning hundreds of small programs." — §1

> "Learning global latent variables would entangle all the experts, because every expert would condition on the latent state... The history formulation allows independent learning of independent mechanisms." — §3.1

> "We hypothesize that hard constraints do not necessarily turn bad predictions into good ones. Instead, they perform damage control—they refine poor predictions just enough to make them usable for long-horizon planning." — §4

> "Synthesizing a single monolithic program is, in our view, intractable not just for the real world, but even for Atari." — §6

> "A program-structured world model exposes an interpretable interface for describing beliefs about how the world works, and efficiently exploring the world is analogous to testing the program that encodes the world model." — §6
