---
type: moc
topic: looped-transformers
last_updated: 2026-07-08
---

# Looped Transformers

Architectures that gain *effective depth* by applying a shared block (or stack of
blocks) **repeatedly** before decoding, converting compute into depth at a fixed
unique-parameter count. Originating as **Universal Transformers** and revived by
**Ouro**, the family is now being pushed along three distinct fronts: (1)
**train-from-scratch** designs that decide *where* and *how much* to loop, (2)
**convert-pretrained / training-free** wrappers that retrofit looping onto frozen
checkpoints, and (3) **efficiency** work that removes the quadratic-attention tax
looping otherwise compounds. A recurring lever across all three is **inference-time
compute scaling** — vary the loop count $S$ (or $T$, $K$) at test time. This topic
is the architectural sibling of [[topics/recursive-reasoning]] (latent recursion).

## Train-from-scratch (where/how much to loop)

- [[papers/lee-2026-looped-diffusion-lm]] — **LoopMDM**: first looped transformer for *masked diffusion LMs*. Selectively loops a few **early-middle** layers; matches MDM quality with up to **3.3× fewer training FLOPs**, +8.5 on GSM8K, and beats deeper non-looped MDMs. Stochastic-$S$ training → arbitrary-$S$ inference (generalizes past $S_{\max}$). Mechanism: looping increases mask-to-mask attention interaction.

## Convert-pretrained / training-free

- [[papers/chen-2026-training-free-looped]] — loops a contiguous mid-block of a **frozen** checkpoint at inference, no training/params/architecture change. Key idea: a pre-norm layer is one **forward-Euler step** on a residual ODE, so looping = finer integration (Runge–Kutta). +2.64 pp MMLU-Pro on Qwen3-4B; **layer-mode** needed for MoE (block-mode thrashes expert routing).

## Efficiency (removing the quadratic-attention tax)

- [[papers/deng-2026-lt2-looped]] — **LT2**: replaces softmax attention in looped transformers with **subquadratic** mixers (linear: GDN/KDA; sparse: DSA). "Looping turns compute into context" — loop × DPLR linear attn → **rank-$T$** memory update; loop × window-$w$ sparse → **$O(Tw)$** receptive field. Hybrid (Full+GDN) beats the full-attention loop by +2.1 pts at ~5× decode throughput; converts pretrained Ouro into **Ouro-Hybrid-1.4B**.

## Recent
- [[papers/movahedi-2026-fixed-point-reasoners]] — FPRM (Fixed-Point Reasoning Model) is a non-hierarchical Looped Transformer for latent reasoning that (a) makes deep looping trainable by switching fr

## Related topics

- [[topics/recursive-reasoning]] — latent-recursion architectures (HRM, GRAM) and symbolic-recursion scaffolds; same "repeated computation" theme
- [[topics/generative-models]] — LoopMDM sits in the diffusion/MDM line
- [[topics/efficient-attention]] — LT2's subquadratic mixers
- [[topics/inference-time-scaling]] — varying loop count as a test-time compute knob
