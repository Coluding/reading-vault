---
type: paper
title: "Separating Representation from Reconstruction Enables Scalable Text Encoders"
authors: ["Megi Dervishi", "Mathurin Videau", "Yann LeCun"]
year: 2026
venue: arXiv preprint
url: https://arxiv.org/abs/2607.04011
rw_id: 01kyhvf0a0be516j8rz6khyp51
topics: [language-models, representation-learning, self-supervised-learning, scaling-laws]
priority: high
read_state: queued
relevance: ""
added: 2026-07-27
last_updated: 2026-07-27
---

# CrossBERT: Separating Representation from Reconstruction Enables Scalable Text Encoders

## TL;DR

Decoders scaled; encoders have barely changed since BERT. This paper re-evaluates them under **frozen-backbone probing** rather than full finetuning, and surfaces a genuinely counterintuitive result: as BERT models scale, their frozen features get **worse** — the two largest configurations (>500M params) probe below the *smallest* model in the sweep, a 10× smaller model — even though perplexity keeps improving. The diagnosis is architectural: BERT's "flat" design couples representation learning to the token-reconstruction loss, so scaling buys ever-better reconstruction at the cost of versatile abstraction. **CrossBERT** fixes this by splitting the model in two — a heavy encoder that only ever sees *visible* tokens, and a shallow **cross-attention-only** predictor that queries the encoder's output to reconstruct the masked ones. Removing self-attention among masked tokens forces the predictor to act as a pure readout interface. The split also unlocks masking ratios ≥50% (BERT degrades past 20–40%) and a **Complementary Masking Strategy** that collects gradients on *every* token, giving ~1.5–2× throughput and 2× sample efficiency. Result: monotonic scaling, 99% of full-finetuning MTEB performance retained when frozen (53.6 vs 53.9) where BERT loses 10+ points (53.9 → 42.1), **retrieval 40.7 vs 19.2**, and at large scale CrossBERT's *frozen* features beat *fully finetuned* baselines.

## Context & motivation

Encoders remain critical for data curation, RAG, and recommendation, yet "encoder architectures have remained largely unchallenged since BERT," with most gains coming from more data and elaborate post-training pipelines that use the pretrained model merely as initialisation. The paper's methodological complaint is that this hides the problem: "heavy reliance on downstream finetuning does not show the shortcomings of the pre-trained backbone, hindering its development." If you always finetune everything, you can't tell whether performance came from pretraining or from the finetuning.

So they change the measurement. Freeze the backbone; fit only probes. Under that lens a clear pathology appears — and it is not a scaling *slowdown* but a scaling *reversal* on frozen features while unfrozen performance keeps improving.

The mechanistic story starts from the standard MLM objective. Given a corrupted sequence $\tilde{X}$ where positions $\mathcal{M}$ have been replaced by `<MASK>`:

$$\mathcal{L}_{\text{MLM}} = -\sum_{m \in \mathcal{M}} \log P(x_m \mid \tilde{X}) \tag{1}$$

This "only measures the token reconstruction ability of the model but not the actual quality of its representations." Because BERT is flat — one stack doing both jobs — the representations stay "overly *grounded* in the local signal required to predict missing tokens instead of being versatile high-level abstractions." Supporting evidence that this is over-specialisation rather than mere capacity limits: probing BERT at *earlier* layers yields **better** frozen performance than the final output, i.e. the last layers actively specialise away from useful abstraction.

## Method

### Core idea

Give reconstruction its own module so the encoder never has to serve it directly. Partition the sequence: the **encoder** processes only the visible tokens; the **predictor** receives `<MASK>` placeholders and must recover the masked content *solely by cross-attending to the encoder's representations*. Because the predictor has no self-attention between masked positions and never sees masked token content, it cannot solve the task internally — it can only succeed if the encoder's embeddings already contain the needed information. The reconstruction pressure is thereby converted from "shape your features for token prediction" into "make your features *queryable*."

### Architecture

- **Encoder**: the heavy backbone (28 layers, hidden dim 768 in the single-scale runs). Takes unmasked tokens **and their positional indices**; masked tokens are *dropped*, not just replaced — which is where the compute saving comes from.
- **Predictor**: a few transformer blocks, roughly **one quarter of the encoder's depth** (6 layers to the encoder's 28), sharing the hidden dimension. Takes `<MASK>` placeholders + positions, **cross-attends only** to encoder outputs, trained with cross-entropy on the true tokens.
- **RoPE** encodes position throughout, so both modules know *which* positions are missing even though the encoder never sees their content.
- Predictor shape ablation (appendix): aspect ratio (width vs depth) doesn't matter much; more predictor capacity helps with diminishing returns but costs training speed — so it is kept deliberately small.

The design is explicitly contrasted with **T5**, which is also bipartite but whose decoder is heavy, autoregressive, and uses both self- and cross-attention. CrossBERT "adopts asymmetry strictly for representation learning" — the predictor exists to *offload the reconstruction burden*, not to generate text.

### Four consequences of the split

1. **Transferability.** The predictor learns to extract information from encoder representations alone, so it "functions as a learned pooling mechanism" and can be re-used as a warm-started adapter for downstream finetuning — a free bridge between frozen features and the target task.
2. **Robustness to high masking.** BERT degrades past 20–40% masking. CrossBERT going 20% → 50% costs a **negligible −0.7% GLUE**. The interpretation offered is that the bipartite architecture "insulates representation learning from the difficulty of the reconstruction task" — the encoder's job (encode what's visible) doesn't get harder just because more is hidden.
3. **Complementary Masking Strategy (CMS).** Augment every batch with its *inverse* mask, creating two complementary views, and process both in parallel under an attention mask that prevents leakage between them. Every token is now both encoded (in one view) and predicted (in the other), so gradients come from **all** tokens rather than the masked fraction — the standard MLM sample-inefficiency fixed. This only works because CrossBERT tolerates ≥50% masking: for BERT at 20–40%, the inverse view lands at 60–80%, outside its working range.
4. **Computational efficiency of CMS.** For a flat BERT, CMS would require two full forward-backward passes. CrossBERT's encoder drops masked tokens, so CMS costs just one forward-backward pass of encoder + predictor over the full sequence. Cost interpolates linearly between encoder and predictor as masking rises.

### Evaluation machinery

Deliberately cheap so it can be run throughout training:

- **Linear probing** via closed-form Ridge Regression on mean-pooled frozen features, $\mathcal{L}_{\text{Ridge}}(\mathbf{W}) = \|\mathbf{Y} - \mathbf{X}\mathbf{W}\|_F^2 + \lambda\|\mathbf{W}\|_F^2$, solved directly on GPU with `torch.linalg.solve` across a log-spaced $\lambda$ sweep (small datasets are very $\lambda$-sensitive). No iterative optimisation, one forward pass over the training set.
- **kNN probing** as a non-parametric complement measuring "the intrinsic geometry of the representation space" — majority vote over $k$ nearest neighbours, sweeping $k \in \{1,3,10,30\}$ and L2/cosine.
- Cost: full linear + kNN probing on GLUE for a 250M model takes **<5 minutes on one H100**, dominated by the forward pass.
- **Contrastive probing** for MTEB: a lightweight cross-attention pooler is trained on MS-MARCO (500k queries, hard negatives, one epoch, batch 512) with the standard InfoNCE-style loss, in two configurations — *frozen* backbone (pooler only) and *unfrozen* (everything). For CrossBERT the pooler is warm-started from the predictor.

## Experimental setup

- **Models**: BERT, Electra (RTD objective, flat backbone), and CrossBERT — same corpus, tokenizer, and encoder backbone. Published ModernBERT / NeoBERT / OptiBERT numbers reported for context.
- **Corpus**: ~4T tokens. Masking 20% for BERT, **50% for CrossBERT**, all CrossBERT runs using CMS.
- **Single-scale runs**: 50B tokens at a data-to-model ratio of **35:1**, deliberately above the compute-optimal ~15:1 to rule out undertraining as an explanation.
- **Scaling sweep**: ratio fixed at **20:1** (still above compute-optimal), 8 BERT and 9 CrossBERT models. Batch size and learning rate set by fitting power laws over 50M–700M sweeps and extrapolating.
- Engineering note: `torch.compile` needs static graphs, which conflicts with random MLM masking; resolved by fixing the masked-token *count* per GPU and permuting the mask with `torch.randperm` — static shapes without losing randomness.
- **Benchmarks**: GLUE (full finetune + linear/kNN probes) and MTEB(eng, v1 and v2), frozen and unfrozen.

## Key results

- **The headline pathology**: BERT's frozen performance "suffers a brutal drop as model size increases"; the two largest (>500M) configurations fall **below the smallest model in the sweep**, which is 10× smaller. Since the data-to-model ratio is held above compute-optimal, undertraining is excluded. CrossBERT scales **monotonically** on every metric.
- **Frozen vs unfrozen gap**: BERT needs full finetuning to close a ~15-point deficit on MTEB(eng, v2); CrossBERT's frozen features sit at worst ~1 point below its own fully-finetuned optimum. On MTEB(eng, v1), BERT drops 53.9 → 42.1 going frozen while CrossBERT retains **99%** (→ 53.6).
- **Retrieval is the sharpest split**: CrossBERT **40.7 vs BERT 19.2**. Across scales, standard MLM "effectively flatlines near zero," which the authors read as "a structural inability to learn dense retrieval without supervision," while CrossBERT's retrieval scales linearly with compute.
- **Linear separability**: +6.4 points over BERT on the GLUE linear probe.
- **Frozen beats unfrozen**: at larger scales CrossBERT's frozen adaptation begins to outperform *unfrozen* baselines — rendering heavy finetuning "unnecessary, and eventually inferior, to a lightweight adaptation of the frozen features."
- **Efficiency**: 207k vs 123k tokens/sec, a **1.68× wall-clock speedup** on H100. ~**100× less total compute** than ModernBERT/NeoBERT while remaining competitive; ~40% fewer FLOPs than similar-scale OptiBERT with better GLUE and MTEB(eng, v1).
- **Full finetuning is not sacrificed**: GLUE 86.6 vs the baseline's 86.4; MTEB(eng, v1) full-contrastive **54.5**, above NeoBERT (51.3) and ModernBERT (46.9), both of which used far more finetuning data.
- **Training stability**: validation loss tracks the scaling law smoothly at every budget with no divergence, whereas larger BERT runs "suffer from significant instability and loss spikes."

## Ablations

- **The Electra control is the most informative experiment.** Electra (replaced-token detection on a *flat* backbone) is *better* than both on token-level probes — 87.8 GLUE full finetune, 76.9 GLUE linear probe — but **collapses on sentence embeddings**: 49.0 MTEB unfrozen, dropping to **22.4 frozen with Retrieval exactly 0.0**. Two conclusions the paper draws, both worth keeping: (a) swapping the *objective* on a flat backbone doesn't produce versatile frozen features — it's the *bipartite separation* that does; and (b) **GLUE-style classification probes alone cannot diagnose representation quality**, since MTEB exposes failures GLUE hides entirely.
- **Masking ratio sweep**: 20% → 50% costs −0.7% GLUE for CrossBERT.
- **CMS**: doesn't damage baseline performance and halves the data needed to reach equivalent performance.
- **Layer-wise probing of BERT**: earlier layers probe better than the final output — the over-specialisation evidence.
- **Predictor shape**: aspect ratio insignificant; capacity has diminishing returns against training cost.

## Limitations

**Paper's own:** no theoretical account of *why* coupling representation and reconstruction degrades frozen features as models grow — the mechanism is demonstrated, not explained. All models are ≤ a few billion parameters; trends are monotonic across that range but qualitatively different behaviour at decoder-scale can't be formally excluded. And the Electra collapse (strong token probes, dead sentence embeddings) is reported without a mechanistic explanation.

**An honest reader would add:**
- The whole argument rests on frozen probing being the *right* proxy for representation quality. That's a reasonable methodological stance and the paper argues it well, but it is a stance: if the deployment path is full finetuning, BERT's frozen collapse may simply not matter, and the headline GLUE numbers (86.6 vs 86.4) are a tie.
- The comparisons to ModernBERT/NeoBERT/OptiBERT are against *published* numbers under different training corpora and budgets, so the "100× less compute" claim mixes controlled and uncontrolled comparisons. The BERT/Electra baselines are the controlled ones and should carry the weight.
- CrossBERT's advantage is partly confounded with masking ratio and CMS, which BERT structurally *cannot* use. That's a fair architectural win, but it means "bipartite separation causes better representations" and "training on 2× effective tokens causes better representations" aren't fully disentangled in the headline numbers.
- The predictor-as-warm-started-adapter gives CrossBERT a better starting point for contrastive probing than the baseline gets. Whether the frozen-MTEB gap survives an equally well-initialised BERT pooler isn't shown in the main text.

## Why it matters [analyst's view]

The transferable finding is not CrossBERT the architecture but **the measurement change that revealed the problem**. "Evaluate the frozen backbone, not the finetuned system" flipped a metric from monotonically-improving to monotonically-*degrading* — and nothing about the loss curve or perplexity would have told you. That is a general warning about proxy metrics in self-supervised pretraining: an objective can keep improving while the thing you actually wanted gets worse, and the standard evaluation protocol can be structurally incapable of noticing. The Electra result sharpens it further: GLUE said Electra was the best model in the paper; MTEB said its embedding space was degenerate (Retrieval = 0.0).

The mechanism also rhymes strongly with a pattern the vault keeps encountering: **put the task-specific pressure in a detachable head so the representation doesn't absorb it.** [[blogs/bayat-sigreg-first-principles]] and the JEPA line reach for the same structural move from the collapse direction, and [[papers/knobel-2026-structured-dynamics]] — triaged the same day — uses a *deliberately weak* predictor cross-attending to a frozen backbone for exactly the reason CrossBERT gives: a predictor that can't solve the task internally is forced to make the encoder's representation carry the information. LeCun is a co-author on this and adjacent to much of that line, and the family resemblance to the JEPA argument (predict in a space that isn't pixel/token reconstruction) is not accidental. CrossBERT is arguably the text-encoder instantiation of "reconstruction is the wrong thing to make your representation responsible for."

For the vault this is also the second 2026 paper (after [[papers/bayat-2026-tapered-language-models]]) arguing that a piece of the standard transformer recipe was never actually load-bearing and is now costing accuracy at scale. The "frozen features beat fully finetuned baselines at large scale" claim, if it holds, is the practically consequential one: it would make the entire contrastive-finetuning industry around embedding models mostly a workaround for a pretraining defect.

## Open questions / things to verify

- Does the frozen-BERT degradation reproduce on *other* flat encoders and corpora, or is it partly an artifact of this training setup? The claim is strong enough to deserve independent replication.
- Would a BERT with an equally well-initialised attention pooler close the frozen-MTEB gap? That's the cleanest confound to eliminate.
- Can the masking-ratio/CMS advantage be separated from the architectural one — e.g. CrossBERT trained at 20% masking without CMS?
- The paper's own suggested next steps: the bipartite design is **objective-agnostic**, so MLM could be swapped for a pretext task better suited to representation learning; and a systematic comparison against T5-style pretraining (predictor replaced by an autoregressive decoder) would isolate what the cross-attention-only constraint buys.
- Why does RTD produce strong token probes and dead sentence embeddings? Explicitly out of scope here, and genuinely interesting.
- **Electra**, **ModernBERT**, **NeoBERT**, **OptiBERT**, **T5**, and **CAPI** (whose kNN probing code is reused) are all `_needs note_`.

## Connections

- Builds on: BERT (MLM), Electra (RTD), T5 (bipartite but generative), and "recent architectural insights in the vision community" — the masked-autoencoder / JEPA line the paper takes its asymmetry from. `_needs note_`.
- Topic MOCs: [[topics/language-models]], [[topics/representation-learning]], [[topics/self-supervised-learning]], [[topics/scaling-laws]]
- Related in vault:
  - [[papers/knobel-2026-structured-dynamics]] — same structural trick in vision: a shallow predictor cross-attending to a frozen backbone, weak *by design* so the representation must carry the information.
  - [[blogs/bayat-sigreg-first-principles]] — the other side of "don't let the objective degrade the representation," via an explicit distributional regulariser rather than architectural separation.
  - [[papers/bayat-2026-tapered-language-models]] — companion critique of an unexamined default in the standard transformer recipe.
  - [[papers/baldassarre-2025-dino-world-models]] — frozen-encoder-plus-light-predictor as a design pattern, LeCun co-authored.
- Author index: [[authors/megi-dervishi]], [[authors/yann-lecun]]

## Selected quotes

> "The misalignment originates in BERT's flat design, which couples representation learning to the token reconstruction loss." — Abstract

> "By removing self-attention between masked tokens, we force the predictor to act strictly as a 'readout' interface that must satisfy its objective solely by querying the encoder's embeddings." — §3

> "Neither swapping the objective on a flat backbone (Electra) nor keeping MLM on a flat backbone (BERT) yields versatile frozen representations; the bipartite separation of representation from reconstruction is what produces them. Second, GLUE-style classification probes alone cannot diagnose representation quality, since MTEB exposes failures that GLUE hides." — §6

> "It produces representations that are naturally richer, rendering the heavy process of unfrozen finetuning unnecessary, and eventually inferior, to a lightweight adaptation of the frozen features." — §6
