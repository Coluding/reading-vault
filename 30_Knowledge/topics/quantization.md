---
type: moc
topic: quantization
last_updated: 2026-06-10
---

# Quantization & Numeric Formats

Low-precision and floating-point representations of neural network weights (IEEE-754 FP32, INT8, etc.) and their implications for efficiency and vulnerability. New cluster in the vault.

## Foundational
- _none yet_

## Recent
- [[papers/galil-2025-sign-bit-flips]] — Exploits the IEEE-754 FP32 bit layout: sign-bit flip negates a weight ($\theta\mapsto-\theta$), exponent flips rescale magnitude; evaluates attack on INT8-quantized ImageNet models vs prior bit-flip work.

## Related topics
- [[topics/adversarial-robustness]]
- [[topics/hardware-faults]]
