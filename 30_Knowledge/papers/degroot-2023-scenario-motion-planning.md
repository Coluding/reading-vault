---
type: paper
title: "Scenario-Based Motion Planning with Bounded Probability of Collision"
authors: ["Oscar de Groot", "Laura Ferranti", "Dariu Gavrila", "Javier Alonso-Mora"]
year: 2023
venue: arXiv preprint (cs.RO), TU Delft
url: https://arxiv.org/abs/2307.01070
rw_id: 01kyhvhqm306x7q3avwpfbam9j
topics: [robotics, optimization]
priority: medium
read_state: queued
relevance: ""
added: 2026-07-27
last_updated: 2026-07-27
---

# Safe Horizon MPC: Scenario-Based Motion Planning with Bounded Probability of Collision

> **Note on provenance**: this arrived in the inbox as a title-less bullet (`"2307.01070"`, no author field). Identified from the fetched PDF — de Groot, Ferranti, Gavrila & Alonso-Mora (TU Delft), arXiv:2307.01070v1 [cs.RO], 3 July 2023.

## TL;DR

Robots planning among humans use **chance constraints** to bound collision risk, but essentially all prior work constrains the *marginal* probability — the independent collision probability at each timestep and for each obstacle — rather than the actual probability that the *planned trajectory* collides. That marginalisation ignores two things: the correlation over time (once you've collided, subsequent collisions are irrelevant, so summing per-step risks double-counts) and the joint effect of multiple obstacles. The result is conservative, sometimes infeasible, freezing robots. **Safe Horizon MPC (SH-MPC)** instead writes a *single* chance constraint over the whole horizon and all obstacles, and makes it tractable via **nonconvex scenario optimization**: sample $S$ *scenarios*, each a complete trajectory for *all* obstacles over the horizon, and enforce collision avoidance against every one. Probabilistic safety then follows from the sample size, which is computed **offline** from the desired risk $\epsilon$, confidence $\beta$, and a support limit $\bar{n}$ — the paper's key move, since the standard scenario approach requires re-solving the program $S$ times to certify a solution and is therefore useless in real time. The method is **distribution-agnostic** (it only samples), handles nonlinear dynamics, and runs at 20 Hz on a mobile robot and 10 Hz on a simulated self-driving car in Carla. For $\epsilon=0.05$, $\beta=0.01$: $S=1237$ samples.

## Context & motivation

Deploying mobile robots in human environments requires reasoning about human motion, which is non-deterministic, multi-modal (multiple plausible paths), and impossible to query directly. Modern perception returns *probabilistic* predictions, and the planner has to consume them.

The standard formulation allows constraint violation with probability at most $\epsilon$. But evaluating a chance constraint exactly is intractable, so prior work approximates — typically by assuming Gaussian uncertainty, and typically by imposing the constraint **per timestep and per obstacle**. The paper's central critique, and it is a good one:

> "Marginal constraints only assess the risk correctly for the first time step and a single obstacle. The risk of the remainder is under- or overestimated."

Two failure modes follow. First, using Boole's inequality $\mathbb{P}(\cup_k A_k) \le \sum_k \mathbb{P}(A_k)$ to bound trajectory risk by the sum of per-step risks gets looser as the horizon grows — cited prior work shows the implied trajectory collision probability tends to $\infty$ (additive form) or $1$ (multiplicative form) as the number of evaluations increases, *regardless of the real risk*. Second, overestimating risk inflates the unsafe region, which "can cause the planning problem to become infeasible and may cause the robot to freeze" — the well-known freezing-robot problem, here given a precise cause.

The correct object is the joint probability. If $A_k$ is the event of no collision at step $k$, the exact trajectory safety is $\mathbb{P}(A) = \prod_k \mathbb{P}(A_k \mid A_{0:k-1})$ — each step's risk conditional on having survived to reach it. Prior work either drops the conditioning (independence assumption) or, in the closest related approach, keeps it by truncating the in-collision part of the distribution at each step — but that is **limited to Gaussians**.

## Method

### Problem formulation

The robot body is modelled as $n_d$ discs, each obstacle as a single disc of radius $r$. With $\boldsymbol{p}_k^d$ the position of robot disc $d$ at stage $k$ and $\boldsymbol{\delta}_{k,j}$ the position of obstacle $j$ at stage $k$, the chance-constrained problem is:

$$\min_{\boldsymbol{u}\in\mathbb{U},\,\boldsymbol{x}\in\mathbb{X}}\ \sum_{k=0}^N J(\boldsymbol{x}_k,\boldsymbol{u}_k) \tag{1a}$$
$$\text{s.t.}\quad \boldsymbol{x}_0 = \boldsymbol{x}_{\text{init}},\qquad \boldsymbol{x}_{k+1} = f(\boldsymbol{x}_k,\boldsymbol{u}_k) \tag{1b,1c}$$
$$\mathbb{P}\left[\bigwedge_{k=1}^{N}\bigwedge_{d=0}^{n_d}\bigwedge_{j=0}^{M}\big(\|\boldsymbol{p}_k^d-\boldsymbol{\delta}_{k,j}\|_2\geq r\big)\right]\geq 1-\epsilon \tag{1d}$$

The whole contribution is in **(1d)**: a *single* probability statement over a conjunction across all stages $k$, all robot discs $d$, and all obstacles $j$ — not $N \times n_d \times M$ separate statements. $\epsilon$ is the maximum collision probability of the entire planned trajectory. (When $\mathbb{P}$ comes from a learned predictor, guarantees are with respect to that *estimate* $\hat{\mathbb{P}}$ — stated explicitly and worth remembering.)

### Core idea: from chance constraint to scenario program

(1d) can't be evaluated directly, but it *can* be enforced when $\boldsymbol{\delta}$ is deterministic. So sample: draw $S$ **scenarios** $\boldsymbol{\delta}^{(i)}$, where each scenario is a full trajectory for *all* $M$ obstacles over the whole horizon (dimension $\mathbb{R}^{MN}$), and require the plan to be collision-free against every one. This turns a stochastic program into a deterministic one with $S$ hard constraints, and it is **distribution-agnostic** — nothing anywhere requires a density, a CDF, or Gaussianity, only the ability to sample. Multi-modality is free.

The two hard questions are then: *how many samples do you need*, and *which samples actually shape the solution*?

### The sample-size bound

Nonconvex scenario optimization gives a bound on the risk $V(\boldsymbol{\theta}^*)$ of a scenario solution in terms of its **support** $n$ — informally, the number of constraints whose removal would change the solution:

$$\mathbb{P}^{\mathcal{S}}\big[V(\boldsymbol{\theta}^*) > \epsilon(n)\big] \le \sum_{n=0}^{S-1}\binom{S}{n}\big[1-\epsilon(n)\big]^{S-n} = \beta \tag{7}$$

Read this as: with confidence $1-\beta$ over the draw of the multi-sample, the solution's true risk does not exceed $\epsilon(n)$. The function $\epsilon(n)$ distributes the total risk budget across possible support values, and the paper uses the even split:

$$\epsilon(n) = \begin{cases} 1, & n > \bar{n}\\[4pt] 1 - \left(\dfrac{\beta}{S\binom{S}{n}}\right)^{\frac{1}{S-n}}, & n \le \bar{n}\end{cases} \tag{8}$$

Higher support ⇒ the solution is pinned by more constraints ⇒ weaker generalisation ⇒ higher risk. The $n > \bar{n}$ case is set to 1 (no guarantee) beyond the chosen support limit.

**Theorem 1** is the practical enabler: since $\epsilon(n)$ is monotonically increasing in $n$, if you pick a support limit $\bar{n}$ and choose $S$ such that $\epsilon(\bar{n}) \le \epsilon$, then $\epsilon(n) \le \epsilon$ for *all* $n \le \bar{n}$. So **the sample size can be computed offline, before deployment**, and online you only need to check that the realised support hasn't exceeded $\bar{n}$.

### Making support estimation real-time

This is the second obstacle the paper removes. In the standard framework, finding a support subsample means re-solving the program $S$ times (once per removed scenario) — hopeless online. **Lemma 1** and **Theorem 2** instead let the support be estimated from the **active constraints** aggregated across the convex iterations of the nonlinear solver, requiring no extra optimisations. This rests on two assumptions: each iteration is convex, and each iteration's solution satisfies all constraints. The support estimate is then the union of active sets across iterations $l$.

### Scenario removal

An extension that samples *more* scenarios initially so the most restrictive ones can be discarded before/during optimisation. Three benefits, all practical: it makes the approach applicable to **unbounded distributions** (Gaussians have no worst-case sample, so without removal the program can be arbitrarily conservative or infeasible); it **reduces variance** of the solution across consecutive control iterations, which matters a lot for a 20 Hz controller that would otherwise jitter; and it lets the planner "ignore extremely unlikely road-user trajectories that lead to too conservative motion plans." The cost is a larger required $S$, since removed scenarios count toward the support. SH-MPC's removal strategy is to drop exactly the active and infeasible constraints:

$$\mathcal{R}(\boldsymbol{\theta},\boldsymbol{\omega}) = \bigcup_{j=0}^{l}\big(\boldsymbol{\omega}_{\text{active}}^{j}\cup\boldsymbol{\omega}_{\text{infeasible}}^{j}\big) \tag{24}$$

These are already of support and already computed, so removal is free — a neat reuse.

### Making it fast: linearisation and free-space polytopes

Disc constraints are nonconvex and produce high support. Two reductions:

1. **Linearise** each collision region about the previously planned trajectory $\hat{\boldsymbol{p}}$, giving a halfspace per scenario per stage per disc:
   $$\mathcal{H}(\hat{\boldsymbol{p}}_k,\boldsymbol{\delta}_k)=\{\boldsymbol{p}_k \mid \boldsymbol{A}^T\boldsymbol{p}_k \le b\},\quad \boldsymbol{A}=\frac{\boldsymbol{\delta}_k-\hat{\boldsymbol{p}}_k}{\|\boldsymbol{\delta}_k-\hat{\boldsymbol{p}}_k\|},\quad b=\boldsymbol{A}^T\boldsymbol{\delta}_k-r \tag{26,27}$$
   $\boldsymbol{A}$ is the unit vector from the planned position toward the obstacle and $b$ places the halfspace boundary at distance $r$ from the obstacle centre. Crucially the linearised region **contains** the original collision region, so the probabilistic guarantee is preserved (conservative in the right direction), and it's applied locally per stage and per disc so the approximation stays tight.
2. **Reduce to a free-space polytope.** Because the halfspaces heavily overlap, the thousands of scenario constraints for a given $(k,d)$ collapse to a small polytope — 20 constraints in the experiments — computed *before* optimisation by a recursive search taking **under 100 µs** in 2D. This is what makes $S \approx 1237$ scenarios tractable at 20 Hz.

### Closed-loop algorithm

Offline: compute $S$ from $(\epsilon, \beta, \bar{n})$ — a Jupyter notebook ships with the paper. Online, each control cycle: sample the predicted distribution → repeatedly solve the SP, aggregating the support estimate from active constraints → if the support estimate exceeds $\bar{n}$ and a previously safe solution exists, return that one; if not, **slow the robot down** → apply the first input. A projection step (orthogonal to the direction of motion) keeps the previous plan feasible for linearisation, with a Douglas–Rachford feasibility program as fallback.

## Experimental setup

- **Robot sim**: horizon $N=20$ at $dt=0.2$ s (4.0 s lookahead), 20 Hz control (50 ms budget), Intel i9 @ 2.4 GHz. Pedestrian radius 0.3 m, robot 0.325 m. 4 and 8 pedestrians. All methods share solver, cost function, and weights — only the collision constraints differ.
- **Risk settings**: SH-MPC at $\epsilon = 0.05$, $\beta = 0.01$ ⇒ $S = 1237$.
- **Baselines**: (a) **"Gaussian"** — marginal chance constraints evaluated in closed form via the inverse error function, $\boldsymbol{a}^T(\boldsymbol{p}_k-\boldsymbol{\delta}_{k,j}) - r \ge \text{erf}^{-1}(1-2\epsilon_k)\sqrt{2\boldsymbol{a}^T\Sigma\boldsymbol{a}}$; strictly Gaussian-only but *tight* per step, so a strong baseline. Run at $\epsilon_k = 0.0025$ (= 5% over $N=20$) and also at $\epsilon_k = 0.0003125$, the setting that matches SH-MPC's actual guarantee. (b) **S-MPCC** — sampling on the marginals, conservatively assuming all 20 polygon constraints are of support, which at $\epsilon_k=0.0025$ demands **$S = 75{,}946$** samples.
- **Distributions**: unimodal Gaussian random-walk pedestrians ($\sigma_{wx}=\sigma_{wy}=0.3$); and a **21-mode Gaussian Mixture** from a Markov chain over crossing behaviours.
- **Validation**: actual collision probability computed *offline* by Monte Carlo, counting the fraction of samples where robot and obstacle discs overlap at *any* stage. 100 experiments per configuration.
- **Vehicle**: simulated self-driving car in Carla at 10 Hz, GMM crossing pedestrians, no robot–pedestrian interaction modelled.

## Key results

- **4 pedestrians, Gaussian**: the marginal baselines are conservative — max observed trajectory CP **0.0069** — while SH-MPC attains a higher but still-safe **0.0113**, both well under the 0.05 bound. Performance is roughly a wash here; the paper attributes this to receding-horizon replanning masking the difference, and notes marginal methods are genuinely accurate over the short term, hence less conservative near obstacles.
- **8 pedestrians** — where the argument bites: SH-MPC's overall CP is essentially unchanged (it reasons jointly), while **the baselines' risk more than doubles** with the extra obstacles. Compared against the Gaussian baseline tightened to $\epsilon_k=0.0003125$ (matched guarantee), SH-MPC "is able to move through this environment significantly faster."
- **Multi-modal (21-mode GMM), 8 pedestrians**: SH-MPC "outperforms the baselines on almost all metrics." Decisively, the Gaussian method's **computation times become excessive** because it must handle every mode, while **SH-MPC's are unaffected** — sampling doesn't care how many modes there are. Higher-risk methods actually collide here, and the mechanism is diagnosed precisely: when a pedestrian starts crossing toward the robot the prediction changes between control iterations, the optimisation becomes infeasible, and the robot freezes.
- **Empirical CP per support value** tracks below the theoretical bound for each support (Fig. 13) — the guarantee holds and is not vacuous.
- **Scenario removal**: cost decreases in both value and variance for any non-zero removal size, with diminishing returns; risk is "consistently less conservative with removal."
- **Sensitivity**: lower specified $\epsilon$ ⇒ longer task duration and more conservative behaviour, as designed.
- **Autonomous vehicle in Carla**: **88 ms average, 135 ms maximum** computation at 10 Hz control.

## Ablations

- **Removal size sweep** — the main ablation; see above.
- **$\epsilon$ and horizon-length $N$ sweeps** over 10 experiments, reporting empirical CP and split runtimes (optimisation vs polytope construction).
- **Baseline stratification by guarantee** — running the Gaussian baseline at both its "natural" $\epsilon_k$ and at the value that actually matches SH-MPC's trajectory-level guarantee is the right control, and is what makes the crowded-environment speed claim credible rather than an apples-to-oranges win.

## Limitations

**Paper's own:** (1) There remains a gap between the guaranteed and obtained risk — reducible by assuming some distributional knowledge, running scenario programs in parallel, or analysing risk in continuous time. (2) **No interaction during planning**: the probability measure $\mathbb{P}$ cannot depend on the optimisation variables in scenario optimization, so the planner cannot model pedestrians reacting to the robot (though the *joint* distribution can capture obstacle–obstacle interaction as given). (3) Guarantees are relative to the **assumed uncertainty model**; if the human-motion predictor is wrong, the guarantee is wrong. They note the prediction model could be replaced by recorded samples for guarantees w.r.t. true motion. (4) For extreme risk specifications ($\epsilon \le 5\cdot10^{-?}$) the sample size makes computation excessive; sample pruning or GPU parallelisation (most of the work is per-sample linear algebra) would help.

**An honest reader would add:**
- All evaluation is in simulation with *known* pedestrian distributions, deliberately chosen "to evaluate the performance of the planner in isolation (i.e., without prediction errors)." That is methodologically clean, but it means the headline safety property has never been tested against a real predictor's errors — which is exactly where deployed systems fail.
- The support-estimation-from-active-constraints shortcut rests on Assumptions 2 and 3 (convex iterates, feasible iterates). The paper itself notes that termination may not ensure $n \le \bar{n}$ in practice — and the fallback when the support limit is exceeded is to reuse a stale plan or slow down, which is a real behavioural degradation, not a no-op.
- The 4-pedestrian result shows essentially no performance gain. The method's value is specifically in crowded and multi-modal settings; the abstract's general "less conservative than state-of-the-art" is a touch broader than the evidence.

## Why it matters [analyst's view]

This is a **classical robotics/control paper and an outlier for this vault** — no learning anywhere, 2023, formal guarantees rather than benchmarks. It earns its place by supplying the thing the learned-world-model line conspicuously lacks: a precise account of what "safe under uncertainty" *means* over a trajectory, and machinery that delivers it in real time.

The transferable insight is the **marginal-vs-joint critique**, which generalises well beyond motion planning. Bounding a sequence-level failure probability by summing per-step probabilities is a mistake with a specific signature — the bound degrades as the horizon grows, unboundedly, *independent of the true risk* — and that pattern recurs anywhere per-step guarantees are chained: safety filters on learned policies, per-token verification in generation, per-step constraint satisfaction in planning. The correct object is almost always the conditional product $\prod_k \mathbb{P}(A_k \mid A_{0:k-1})$, and the reason people don't compute it is tractability, which is precisely what scenario optimization buys with sampling.

The second reusable idea is **"push the statistical work offline."** The entire contribution that makes this real-time is Theorem 1 plus active-set support estimation: pick $\bar{n}$ in advance, derive $S$ from the bound before deployment, and online just monitor whether the support limit was breached. Turning an expensive certification procedure into a cheap runtime *check* against a precomputed budget is a pattern worth stealing.

Read against [[papers/jain-2026-weaver]] — triaged the same day — the contrast is instructive and, I think, the most useful thing this note can record. WEAVER does test-time planning by sampling $B=4$ action chunks from a policy, imagining each with a learned world model, and picking the best by predicted advantage. SH-MPC samples $S=1237$ scenarios of the *environment's* uncertainty and optimises against all of them with a formal risk bound. Both are "sample possible futures and choose"; one has calibrated guarantees against an assumed distribution and no learned dynamics, the other has learned dynamics and no guarantees at all. Neither has both, and the gap between them — probabilistic safety guarantees over *learned* world models — is a genuinely open and under-served problem. It's also the natural bridge from this paper to the rest of the vault.

Finally: the freezing-robot diagnosis (over-conservative risk ⇒ infeasible program ⇒ frozen robot) is a clean, mechanical explanation of a failure everyone in robotics has seen, and worth remembering as a concrete instance of "conservatism is not free safety."

## Open questions / things to verify

- How badly does the guarantee degrade under a *realistic* learned predictor rather than a known distribution? The paper is explicit that guarantees are w.r.t. $\hat{\mathbb{P}}$, and this is the whole ballgame for deployment.
- Could the scenario formulation wrap a **learned** world model — sample obstacle futures from a generative predictor and inherit the sample-complexity bound? The bound only needs i.i.d. samples, so this seems formally available; the catch is that the guarantee then holds w.r.t. the model's distribution, not reality.
- The "no interaction during planning" restriction is fundamental to scenario optimization ($\mathbb{P}$ can't depend on decision variables). Game-theoretic or iterated-best-response formulations would be the obvious escape; how much of the guarantee survives?
- GPU parallelisation is flagged by the authors as untapped — most of the per-sample work is embarrassingly parallel linear algebra, and this would directly relax the low-$\epsilon$ compute wall.
- **Nonconvex scenario optimization** [11] is the theoretical backbone (Theorem 1 there underwrites Eq. 7) and **S-MPCC** is the sampling baseline — both `_needs note_` if this line is pursued.

## Connections

- Builds on: nonconvex scenario optimization [11] and its scenario-removal extension; nonlinear MPC. `_needs note_`.
- Topic MOCs: [[topics/robotics]], [[topics/optimization]]
- Related in vault:
  - [[papers/jain-2026-weaver]] — the learned-world-model counterpart to test-time planning under uncertainty: sampled futures and best-of-N selection, but no risk guarantee. The complementary halves of the same problem.
  - [[papers/shang-2025-roboscape]] — physics-informed embodied world model; shares the concern that predicted futures must be *physically* trustworthy before they can be planned against.
  - [[papers/jiang-2026-robottt]] — real-robot policy learning; the learned-control end of the spectrum this paper sits at the opposite end of.
- Author index: [[authors/oscar-de-groot]]

## Selected quotes

> "Existing methods do not consider the actual probability of collision for the planned trajectory, but rather its marginalization, that is, the independent collision probabilities for each planning step and/or dynamic obstacle, resulting in conservative trajectories." — Abstract

> "The time correlation exists because the first collision in a trajectory renders it unsafe, such that all subsequent collisions can be ignored." — §I

> "Marginal constraints only assess the risk correctly for the first time step and a single obstacle. The risk of the remainder is under- or overestimated. Overestimation of the risk and the associated unsafe space along the time horizon can cause the planning problem to become infeasible and may cause the robot to freeze." — §II-B

> "The guarantees provided in this paper rely on an accurate model of the uncertainty, which may be challenging to obtain, for example in the case of human motion prediction." — §X
