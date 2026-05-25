---
type: blog
title: "Automatic Mixed Precision package - torch.amp"
author: "PyTorch (pytorch.org docs)"
url: https://docs.pytorch.org/docs/2.12/amp.html
rw_id: 01ksfhxe9nzrewa9mrx95kcqy9
topics: [mixed-precision, gpu-optimization]
priority: medium
read_state: queued
added: 2026-05-25
last_updated: 2026-05-25
---

# Automatic Mixed Precision package — torch.amp

**TL;DR** — Reference page for PyTorch's `torch.amp`, the API that runs a model in *mixed precision*: some ops in `float32`, others in a lower-precision float (`float16` on CUDA, `bfloat16` on CPU). Two orthogonal pieces do the work: `torch.autocast` (a context manager / decorator that auto-selects the dtype per op) and `torch.amp.GradScaler` (loss scaling to stop small `float16` gradients from underflowing to zero). The canonical fp16 training recipe uses both together; CPU `bfloat16` uses autocast alone. This is documentation, not an argument — treat the note as a cheat-sheet for the contract and the gotchas.

## Context

This is the official `torch.amp` API doc (PyTorch 2.12, page dated 2025-06-12), saved as a reference. AMP exists because the speed-sensitive ops in deep nets — matmuls and convolutions — run much faster in low precision (and route to tensor cores), while reductions and many loss/normalization ops need `float32`'s dynamic range to stay numerically stable. AMP's job is to match each op to the appropriate dtype automatically rather than forcing the whole model to `.half()`.

Note: the namespaced forms `torch.cuda.amp.autocast(...)` and `torch.cpu.amp.autocast(...)` (and the matching `GradScaler`s) are **deprecated** — the current spelling is `torch.amp.autocast("cuda", ...)` / `torch.amp.autocast("cpu", ...)`.

## Core argument

Two modular components, used together for fp16 or separately as needed:

**1. `autocast` — per-op dtype selection.** Wrap *only the forward pass(es) and loss computation* in the context manager; the backward pass runs outside it (backward ops inherit the dtype autocast chose for the corresponding forward op). Key rules:

- Don't call `.half()` / `.bfloat16()` on your model or inputs — autocast handles casting per op.
- Tensors produced inside an autocast region may be `float16`; after leaving the region, cast them back to `float32` before mixing with `float32` tensors, or you'll hit type-mismatch errors. (If the tensor is already `float32`, the cast is a no-op.)
- `autocast(enabled=False)` subregions can nest inside enabled regions to force a piece to run in a chosen dtype.
- autocast state is **thread-local** — relevant for `DataParallel` / `DistributedDataParallel` with multiple GPUs per process, where the context manager must be invoked inside the worker thread.
- Default dtype: `torch.float16` for CUDA, `torch.bfloat16` for CPU.

**2. `GradScaler` — loss scaling.** If a forward op had `float16` inputs, its backward produces `float16` gradients; small-magnitude gradients can't be represented in `float16` and flush to zero ("underflow"), losing the update. The fix:

$$\text{scaled\_loss} = s \cdot \mathcal{L}, \qquad g_{\text{scaled}} = s \cdot \nabla_\theta \mathcal{L}$$

Multiply the loss by a scale factor $s$, backprop the scaled loss (gradients come back scaled up, away from the underflow region), then **unscale before the optimizer step** so $s$ doesn't interfere with the learning rate. `GradScaler` defaults: `init_scale=65536.0`, `growth_factor=2.0`, `backoff_factor=0.5`, `growth_interval=2000`. CPU `bfloat16` training/inference needs autocast only — no `GradScaler`.

## Notable details

- **Op eligibility (the autocast cookbook).** Ops that autocast to **`float16`/low-precision**: `matmul`, `mm`, `bmm`, `addmm`, `conv1d/2d/3d`, `conv_transpose*`, `linear`, `prelu`, `scaled_dot_product_attention`, `_native_multi_head_attention`, RNN/LSTM/GRU cells. Ops forced to **`float32`** for stability: `log`, `log_softmax`, `softmax`, `softmin`, `exp`, `pow`, `cross_entropy`, `nll_loss`, `mse_loss`, `l1_loss`, `layer_norm`, `group_norm`, `norm`, `normalize`, `kl_div`, most loss functions, and a long list of `linalg_*` / `fft_*` ops. "Widest-input-type" ops (`bilinear`, `cross`, `grid_sample`, `index_put`, `scatter_add`, `tensordot`) promote all inputs to `float32` if any input is `float32`.
- **Only out-of-place ops are eligible.** `a.addmm(b, c)` autocasts; `a.addmm_(b, c)` and `a.addmm(b, c, out=d)` do not. Ops called with explicit `dtype=...` are not eligible. `float64` and non-float ops never autocast.
- **`binary_cross_entropy` / `BCELoss` raise an error in autocast regions** — their backward can produce gradients not representable in `float16`. Use `binary_cross_entropy_with_logits` / `BCEWithLogitsLoss`, which are safe.
- **fp16 doesn't suit every model.** Most `bfloat16`-pretrained models can't operate in fp16's range (max `65504`) and will *overflow* rather than underflow; `GradScaler` may then drive the scale below 1 (it does *not* guarantee scale ≥ 1). NaNs in loss/grads under AMP → check fp16 compatibility first.
- **Custom autograd functions** get `torch.amp.custom_fwd` / `custom_bwd` decorators to keep forward/backward autocast state consistent.

## Why it matters [analyst's view]

This is the load-bearing reference for the *practical* side of the GPU-efficiency cluster forming in the vault. The conceptual "why" — tensor cores eat low precision, that's where the FLOPS are — is exactly the point [[blogs/aiwithmaha-cuda-concepts]] makes at concept #19 (tensor cores) and #14 (memory-transfer bottleneck). This page is the "how": the precise op tables, the autocast-wrap-forward-only discipline, and the GradScaler underflow story. The most actionable single fact for future-me: **wrap only forward + loss in autocast, never the backward, and unscale before `optimizer.step()`** — getting either wrong is a silent correctness/perf bug. The `bf16-pretrained-model-in-fp16-overflows` warning is the kind of thing that explains a mysterious NaN at 2am.

## Connections

- Topic MOCs: [[topics/gpu-optimization]], [[topics/mixed-precision]]
- Related blogs: [[blogs/aiwithmaha-cuda-concepts]] (tensor cores / mixed precision motivation), [[blogs/bierling-coalesced-matmul]] (the matmul that AMP routes to low precision)
- Author index: [[authors/pytorch]]

## Selected quotes

> "Mixed precision tries to match each op to its appropriate datatype." — torch.amp overview

> "autocast should wrap only the forward pass(es) of your network, including the loss computation(s). Backward passes under autocast are not recommended." — autocast usage

> "most bf16-pretrained models cannot operate in the fp16 numerical range of max 65504 and will cause gradients to overflow instead of underflow ... our GradScaler does NOT [guarantee scale > 1] to maintain performance." — GradScaler note
