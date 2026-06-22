---
type: moc
topic: hardware-faults
last_updated: 2026-06-10
---

# Hardware Faults & Bit-Flip Attacks

Hardware-level mechanisms for corrupting stored model parameters — Rowhammer, DMA from untrusted peripherals, firmware/SSD-controller exploits, rootkits, GPU cache tampering, voltage/clock glitching — and the ML attacks built on them. New cluster in the vault.

## Foundational
- _none yet_

## Recent
- [[papers/galil-2025-sign-bit-flips]] — Threat model is Rowhammer-style faults that flip a few bits at fixed offsets; sign bit (FP32 MSB) is chosen because hardware attacks reliably flip the same offset across addresses. Defense: selective ECC/replication on the ~0.001–1% critical sign bits.

## Related topics
- [[topics/adversarial-robustness]]
- [[topics/quantization]]
