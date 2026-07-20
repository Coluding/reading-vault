---
type: blog
title: "GDL Book (Geometric Deep Learning textbook)"
author: "Petar Veličković"
url: https://geometricdeeplearning.com/book/
rw_id: 01kxn33fsn06fek8sevega2swc
topics: [geometric-deep-learning]
priority: medium
read_state: skimmed
added: 2026-07-16
last_updated: 2026-07-16
---

> **Source note:** the fetched page is the **landing / table-of-contents page** for the Geometric Deep Learning textbook, not a chapter. It announces the book and lists its structure; there is no technical content to summarize yet. This note is therefore a **pointer/tracking note**, not a deep read — revisit as draft chapters are released. _(Update 2026-07-20: the vault holds three notes tagged `geometric-deep-learning`, so [[topics/geometric-deep-learning]] now exists.)_

## TL;DR

The **Geometric Deep Learning (GDL) textbook**, credited here to **Petar Veličković** (per the saved metadata) and forthcoming from **MIT Press in 2026**, is being released chapter-by-chapter as free online drafts at a cadence of roughly **one chapter every 2–3 weeks**; once posted, chapters stay free online. It is organized into three parts — **I. Geometric Foundations of Deep Learning**, **II. Learning on Geometric Domains**, **III. Geometric Deep Learning at the Bleeding Edge** — and its material already backs Master's-level courses at **Oxford and Cambridge**. The page positions GDL as a unifying, symmetry-first lens on deep learning (opening with Weyl on symmetry). It is a resource to follow, not yet a body of content to mine.

## Context

The page frames the book as an entry point "to help students and practitioners enter the field of geometric deep learning." It is licensed **CC-BY-NC-ND**, with copyright licensed exclusively to The MIT Press (final public version in 2026). For citation in the interim it directs readers to cite the **"proto-book"** (BibTeX provided on the page). The authors solicit corrections and exercise suggestions by email (gdl-book@googlegroups.com), crediting contributors in the Acknowledgement — i.e. it is an actively maintained, community-refined draft.

## Core argument

There is no technical argument on this page — it is front matter. The one substantive content signal is the **mathematical prerequisites** the authors flag: GDL "does not strongly assume any particular prior mathematical preparation, yet its foundations rest on" **Vector Calculus, Differential Geometry, Topology, Functional Analysis, Spectral Theory, and Graph Theory**. For readers wanting more grounding they recommend **"Mathematical Foundations of Geometric Deep Learning" (Borde and Bronstein, 2025)** as a companion. The three-part structure implies the pedagogical arc: establish the geometric foundations (symmetry, invariance/equivariance) → specialize to geometric domains (graphs, grids, groups, manifolds — implied, not enumerated on this page) → survey the research frontier (Part III listed as "TBD").

## Notable details

- **Release model:** individual draft chapters posted ~every 2–3 weeks; remain free after posting; final book via MIT Press in 2026.
- **Prereq math stack (explicit):** Vector Calculus, Differential Geometry, Topology, Functional Analysis, Spectral Theory, Graph Theory.
- **Companion text:** *Mathematical Foundations of Geometric Deep Learning*, Borde and Bronstein (2025).
- **Adoption:** used for Master's courses at Oxford and Cambridge; lecture slides to be shared alongside chapters where relevant.
- **License:** CC-BY-NC-ND; cite the proto-book in the interim.
- Contents listed so far: Acknowledgement, Preface, Notation, Part I, Part II, Part III (TBD).

## Why it matters [analyst's view]

Geometric Deep Learning — the program of deriving architectures from **symmetry and invariance principles** (the "Erlangen programme for ML") — is the theoretical backbone under a lot of what the vault already tracks: equivariant/graph models, and, more loosely, the representation-geometry themes in the autoencoder cluster ([[papers/liu-2026-geometric-autoencoder]] literally borrows the "geometric" framing for latent spaces). A free, authoritative textbook with a steady release cadence is a high-value **reference to track**: as chapters drop, individual ones (especially Part III, "bleeding edge") may warrant their own deep notes. For now this is a bookmark with structure. If a second GDL item lands, promote `geometric-deep-learning` to a real topic MOC and wire this in as the foundational reference.

## Connections

- Related (loosely — "geometry of latent spaces"): [[papers/liu-2026-geometric-autoencoder]]
- Author index: [[authors/petar-velickovic]]
- Topic (no MOC yet — singleton): geometric-deep-learning

## Selected quotes

> "Symmetry, as wide or as narrow as you may define its meaning, is one idea by which man through the ages has tried to comprehend and create order, beauty, and perfection." — Hermann Weyl, *Symmetry* (epigraph on the page)

> "The GDL Book does not strongly assume any particular prior mathematical preparation, yet its foundations rest on several key areas of mathematics – Vector Calculus, Differential Geometry, Topology, Functional Analysis, Spectral Theory and Graph Theory."
