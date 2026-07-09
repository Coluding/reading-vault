---
type: paper
title: "Revisiting the Platonic Representation Hypothesis: An Aristotelian View"
authors: ["Fabian Gröger", "Shuo Wen", "Maria Brbić"]
year: 2026
venue: "arXiv preprint"
url: https://arxiv.org/abs/2602.14486
rw_id: 01kvq3nzh73ffyhnqx96v2561f
topics: [representation-learning, multimodal, model-similarity]
priority: high
read_state: queued
relevance: ""
added: 2026-07-08
last_updated: 2026-07-08
---

> **Sourcing note:** the Readwise `html_content` cached under this rw-id was cross-wired to a *different* paper (a self-supervised video paper, "TDV / Temporal Difference in Vision"), so this note was written from the paper's own arXiv HTML (`arxiv.org/html/2602.14486`, ICML 2026 camera-ready, v2 dated 2026-06-25) rather than the Readwise export. Equations are transcribed from that rendering; exact correlation/threshold numbers should be double-checked against the PDF before quoting.

## TL;DR

This paper argues that the **Platonic Representation Hypothesis (PRH)** — the claim that neural nets trained on different data/modalities converge to one shared statistical model of reality — is largely an artifact of *uncalibrated* similarity metrics. The authors show that simply increasing a model's **width or depth systematically inflates representational-similarity scores** even between independent representations, because standard metrics (CKA, SVCCA, Procrustes, mutual-kNN, CKNNA…) have non-zero null baselines that scale with dimension $d/n$ and with the number of layer pairs searched. They introduce a **permutation-based null-calibration framework** that subtracts each metric's chance level, then re-run the classic convergence analysis. After calibration, **global spectral convergence largely vanishes** (linear-CKA correlation with model capability drops from ~0.86 to ~0.45) while **local neighborhood agreement survives** (mutual-kNN stays ~0.85) — but local *distances* do not. From this they propose the weaker **Aristotelian Representation Hypothesis**: models converge on shared *local neighborhood relations* ("who is near whom"), not on a common global geometry.

## Context & motivation

The Platonic Representation Hypothesis (Huh et al., 2024) claimed that as models scale, their representation spaces align across modalities — vision and language models placing concepts in geometrically similar arrangements — hinting at a shared underlying "reality." That claim rests entirely on **representational-similarity metrics** computed between frozen encoders. The authors' concern is methodological: these metrics were designed to compare fixed representations, and their *expected value under independence is not zero*. If that null baseline grows with model size, then any observed "larger models are more aligned" trend could be measuring capacity, not convergence. The paper's contribution is (i) to characterize two confounders (width, depth) analytically, (ii) to give a finite-sample-valid calibration that removes them, and (iii) to re-examine which parts of the PRH survive the correction — landing on a narrower, empirically robust "Aristotelian" restatement.

## Method

### Problem formulation
Given paired representations $\mathbf{X}\in\mathbb{R}^{n\times d_x}$ and $\mathbf{Y}\in\mathbb{R}^{n\times d_y}$ (the same $n$ samples encoded by two models/modalities, rows aligned), a similarity function $s(\mathbf{X},\mathbf{Y})$ returns an alignment score. The question: is a high $s$ evidence of genuine convergence, or is it what you'd expect by chance given $n,d$, and the number of layer pairs compared?

### Core idea
Every similarity metric has a **null distribution** — the scores it produces when the two representations are independent. That null is *not* zero and, crucially, **shifts with $d/n$ (width) and with the size of the layer-pair search space $M$ (depth)**. Calibrate each observed score against its own permutation-null, and only the *excess above chance* counts as convergence.

### Architecture / algorithm — the two confounders

**Width confounder.** Spectral metrics build the normalized cross-covariance
$$\tilde{\mathbf{C}} = \tfrac{1}{n-1}\,\mathbf{X}_c^\top \mathbf{Y}_c \in \mathbb{R}^{d_x\times d_y},$$
where $\mathbf{X}_c,\mathbf{Y}_c$ are column-centered. Under independence (Proposition 4.1) its expected squared Frobenius norm is
$$\mathbb{E}\big[\|\tilde{\mathbf{C}}\|_F^2\big] = \frac{d_x\, d_y}{n-1},$$
i.e. an $O(d/n)$ floor: higher-dimensional models look more similar *purely from finite-sample noise accumulating over more dimensions*. This is what drives CKA/Procrustes upward with scale.

**Depth confounder.** Layer-wise studies report the *maximum* alignment over all $M = L_A\cdot L_B$ layer pairs. Taking a max over $M$ noisy comparisons inflates the expected value roughly as
$$\mathbb{E}[T_{\max}] \le \mu + C\,\sigma\sqrt{\log M},$$
so deeper models (larger $M$) win under the null through selection alone.

**Why neighborhood metrics are less confounded.** For mutual-$k$NN (Proposition 4.2) the null baseline is
$$\mathbb{E}[\mathrm{mKNN}(\mathbf{X},\mathbf{Y})] = \frac{k}{n-1},$$
an $O(k/n)$ floor. Since $k$ is small and fixed across experiments (unlike $d$), neighborhood overlap is far less contaminated by width — which is exactly why it survives calibration below.

### Null-calibration framework — the derivation

For row-aligned representations, the null $H_0$ ("no relationship beyond marginal statistics") is operationalized by **permuting the sample correspondence** of $\mathbf{Y}$. Draw $K$ permutations $\pi_k \sim \mathrm{Unif}(\Pi_n)$ and compute
$$s^{(k)} = s(\mathbf{X}, \pi_k(\mathbf{Y})),\quad k=1,\dots,K,$$
with observed $s_{\text{obs}} = s(\mathbf{X},\mathbf{Y})$. Sort the combined multiset $\{s_{\text{obs}}, s^{(1)},\dots,s^{(K)}\}$ into order statistics $s_{(1)}\le\dots\le s_{(K+1)}$ and set the right-tail critical value at level $\alpha$:
$$\tau_\alpha := s_{(\lceil (1-\alpha)(K+1)\rceil)}.$$
The empirical right-tail $p$-value is
$$p = \frac{1 + \#\{k : s^{(k)} \ge s_{\text{obs}}\}}{K+1}.$$
For a bounded metric with maximum $s_{\max}$, the **calibrated score** rescales the excess above the null threshold into $[0,1]$:
$$s_{\text{cal}} = \max\!\Big[\frac{s_{\text{obs}} - \tau_\alpha}{s_{\max} - \tau_\alpha},\, 0\Big].$$
So $s_{\text{cal}}=0$ when the observation is at or below chance and $s_{\text{cal}}=1$ at perfect alignment.

**Aggregation-aware version (for layer-wise matrices).** With per-layer scores $S_{\ell,\ell'} = s(\mathbf{X}^{(A)}_\ell, \mathbf{Y}^{(B)}_{\ell'})$ forming $\mathbf{S}\in\mathbb{R}^{L_A\times L_B}$, the *same* permutation $\pi_k$ is applied to **all** layers of model $B$:
$$S^{(k)}_{\ell,\ell'} := s(\mathbf{X}^{(A)}_\ell, \pi_k(\mathbf{Y}^{(B)}_{\ell'})),$$
then an aggregate $T^{(k)} := T(\mathbf{S}^{(k)})$ (e.g. $T=\max$) is formed per permutation. Its calibrated value uses $\tau_\alpha^{\text{agg}} := T_{(\lceil(1-\alpha)(K+1)\rceil)}$. Keeping the permutation consistent across layers means the null distribution of the aggregate directly absorbs the **selection inflation** from taking a max over $M$ pairs — the depth confounder is calibrated away by construction.

### Derivations / why it works
The type-I validity is a standard permutation-test result (Corollary 5.1): the permutation $p$-value is **super-uniform** under $H_0$, $\mathbb{P}(p\le\alpha)\le\alpha$ for all $\alpha$, giving finite-sample (not just asymptotic) control. The confounder characterizations (Props 4.1–4.2) are the load-bearing math: they show *analytically* that the null baseline of spectral metrics is $O(d/n)$ while neighborhood metrics are $O(k/n)$, predicting a priori which metrics will and won't survive calibration — a prediction the experiments then confirm.

### Training procedure
_Not applicable — this is an analysis/measurement paper on frozen pretrained encoders; no model training._

### Inference / sampling
_Not applicable (not a generative method)._

## Experimental setup

- **Primary (image–text):** 1,024 image–text pairs from WIT (Wikipedia Image Text). Vision families: ImageNet-21K supervised, MAE, DINOv2, CLIP, CLIP-finetuned (multiple scales each); language families: BLOOM, OpenLLaMA, LLaMA (multiple sizes). ~204 model pairs spanning $d/n\in[0.19, 8]$.
- **Secondary (video–language):** VideoMAE (small/base/large/huge, Kinetics-finetuned) vs the same three LM families.
- **Synthetic validation:** controlled sweeps over $n\in\{128,\dots,4096\}$, $d\in\{128,\dots,2048\}$ for width; independent synthetic models with $L\in\{2,4,\dots,128\}$ layers for depth; tested under Gaussian and heavy-tailed (Student-$t$, Laplace) noise.
- **Metrics compared:** spectral (CKA linear/RBF, SVCCA, RV, Procrustes), neighborhood (mutual-kNN, cycle-kNN, CKNNA), geometric (RSA).

## Key results

*Summarized — check exact figures against the PDF.*
- **Global spectral metrics lose the scaling trend after calibration.** Correlation of the metric with model capability: linear-CKA ~0.86 → ~0.45; Procrustes ~0.89 → ~0.39.
- **Local neighborhood metrics keep it:** mutual-kNN ~0.85 → ~0.85; CKNNA ~0.87 → ~0.87.
- **Synthetic:** raw CKA drifts upward with $d/n$; calibrated CKA collapses to ~0 across all $(n,d)$ and noise types. Raw max-CKA rises with depth; calibrated stays ~0.
- **Type-I control:** rejection rate stays at/below $\alpha=0.05$ across configurations; power rises quickly with true signal.
- **Local distances do not survive:** distance-sensitive metrics show no post-calibration alignment — models agree on neighbor *identity*, not on absolute distances.

## Ablations

The width and depth synthetic experiments *are* the ablations: they isolate each confounder and show the calibration neutralizes it while preserving power on genuine signal. The metric-family split (spectral vs neighborhood vs distance) functions as an ablation over *what kind* of structure survives — the paper's central empirical dissection.

## Limitations

- The whole analysis is on **frozen encoders and a fixed pairing** ($n=1{,}024$ WIT pairs); convergence claims are only as good as that probe set's coverage.
- Calibration tells you a score exceeds chance, not *how much* real-world meaning the excess carries; "significant but small" alignment is still possible.
- [analyst's view] The Aristotelian claim rests on neighborhood metrics surviving — but those have the *smallest* null baseline, so they also had the most room to survive; the paper's own framework implies neighborhood convergence is the hardest signal to falsely inflate, which is reassuring but worth stress-testing on larger $n$.

## Why it matters [analyst's view]

This is a corrective to a much-cited hypothesis, and its real contribution is methodological hygiene for the whole "representational alignment" literature: **any** claim of the form "bigger models are more similar" needs a null baseline before it means anything, and this paper hands the field a clean, finite-sample-valid recipe. The narrower Aristotelian claim — convergence on *relational* structure, not absolute geometry — is both more defensible and more interesting: it says cross-modal models agree on a nearest-neighbor graph over concepts, which is precisely the structure that matters for retrieval, alignment, and stitching representations together, while *not* committing to a shared metric embedding. It connects to interpretability work on universal features and to any project that assumes vision and language encoders can be linearly aligned.

## Open questions / things to verify

- Exact post-calibration correlation numbers (0.45, 0.39, 0.85, 0.87) — transcribed from the arXiv HTML rendering via an extraction pass; verify against the PDF/tables.
- Does neighborhood convergence keep strengthening with scale, or plateau? The trend "persists" but its slope after calibration matters for whether PRH-style scaling arguments hold in the weaker form.
- How sensitive is the conclusion to the probe set (WIT, $n=1024$)? Larger, more diverse $n$ would tighten the null and could change which metrics survive.
- The permutation test costs $K$ recomputations per pair — practicality at very large $n$ / many models.

## Connections

- Topic MOCs: [[topics/representation-learning]], [[topics/multimodal]], [[topics/model-similarity]]
- Author indices: [[authors/fabian-groger]], [[authors/shuo-wen]], [[authors/maria-brbic]]
- Related: builds on / rebuts the Platonic Representation Hypothesis (Huh et al., 2024); relevant to any note on CKA / representational similarity and cross-modal alignment.

## Selected quotes (optional)

_Verbatim quotes not preserved — note written from a summarizing extraction of the arXiv HTML rather than the raw text; re-open the PDF for exact quotable passages._
