---
type: blog
title: "Riemannian Manifolds: Foundational Concepts"
author: "Patrick R. Nicolas"
url: https://patricknicolas.substack.com/p/riemannian-manifolds-1-foundation
rw_id: 01kv54bbxrzgyw1v86vj0kg53n
topics: [riemannian-geometry, differential-geometry, manifold-learning, geometric-deep-learning]
priority: medium
read_state: queued
added: 2026-06-15
last_updated: 2026-06-17
---

# Riemannian Manifolds: Foundational Concepts

## TL;DR

This is the first installment of a series introducing smooth and Riemannian manifolds as the mathematical backbone of Geometric Deep Learning. It motivates why Euclidean geometry is inadequate for real-world data (which lies on curved, topologically non-trivial manifolds), then builds up the core machinery: smooth manifolds, tangent spaces, vector/tensor fields and differential forms, geodesics, the exponential/logarithm maps, the Riemannian metric, the Levi-Civita connection and Christoffel symbols, Riemann and Ricci curvature, intrinsic vs. extrinsic geometry (first/second fundamental forms), and Lie groups/algebras (GL, SPD, SO, SE). The intended payoff is being able to use the Python `Geomstats` library to compute on data manifolds. A promised second part will apply these concepts to the hypersphere.

## Context

The post is written by Patrick Nicolas (a 30-year software engineering veteran, author of *Scala for Machine Learning* and *Geometric Learning in Python*) and targets data scientists/engineers with a basic ML background, ideally some prior exposure to differential geometry. It positions itself within a broader Geometric Deep Learning series and explicitly recommends readers first see his "Introduction to Geometric Deep Learning." It deliberately scopes itself to *smooth* manifolds and continuous differential geometry, flagging that discrete structures like graphs or simplicial complexes need different tools (Ollivier-Ricci curvature, Gauss curvature, Euler characteristics) covered elsewhere.

## Core argument (concepts and math, in the order the post builds them)

**Why Euclidean geometry falls short.** Real-world data distributions lie on highly curved manifolds; forcing a curved manifold into a flat latent space introduces distortion. Embedding a hierarchical tree into Euclidean space with low distortion needs high dimensionality (curse of dimensionality). Euclidean space cannot natively represent topological features (holes, voids, loops, closed surfaces) without tearing, and it treats all directions uniformly, pulling interpolations toward the mean.

**Smooth manifold basics.** An $n$-dimensional manifold is a topological space where every point has a neighborhood homeomorphic to an open subset of $\mathbb{R}^n$. A *smooth* (differential) manifold adds a local differential structure permitting vector fields and tensors, hence a global tangent space. A *Riemannian* manifold is a smooth manifold equipped with a metric tensor for measuring distances and angles. The *tangent space* at a point is the set of tangent vectors there (a line tangent to a circle, a plane tangent to a surface); tangent vectors act as directional derivatives. A *geodesic* is the shortest arc between two points on a Riemannian manifold.

**Vectors, covectors, tensors, forms.** A vector is "an arrow"; a covector is its dual — an object that measures vectors, best visualized as evenly spaced parallel hyperplanes (a contour map). Vectors are *contravariant* (components scale inversely with the coordinate system); covectors are *covariant* (components scale with it). A vector field assigns a vector to each point; a tensor field generalizes this by assigning a tensor; a differential 1-form (covector field) assigns a covector to each point.

Given a metric $g$, a covector $\omega$ is the linear map $\omega : v \mapsto g(v,\cdot)$. A vector field is written $v = v^i \partial_{x^i} = \sum_{i=1}^n v^i \frac{\partial}{\partial x^i}$ and a 1-form $\alpha = \alpha_i\, dx^i$, with the duality relation $dx^i\!\left(\frac{\partial}{\partial x^j}\right) = \delta^i_j$ (1 if $i=j$, else 0). For a differentiable $f:\mathbb{R}^n\to\mathbb{R}$, the directional derivative of $f$ along $v$ at $x$ is $\nabla_v f(x) = v^i \frac{\partial f}{\partial x^i}(x)$. The gradient expressed intrinsically via the metric is $(\nabla f)^i = g^{ij}\sum_j \frac{\partial f}{\partial x^j}$. Differential $k$-forms extend 1-forms to higher tensor fields (e.g. a 2-form measures a 2-vector field).

**Geodesic & exponential/logarithm maps.** The *exponential map* sends a tangent vector to a point on the manifold; the *logarithm map* is its inverse, assigning a tangent vector to a manifold point. The classic illustration is geodesic (great-circle flight route) vs. straight-line Euclidean distance. The geodesic satisfies $\nabla_{\dot\gamma}\dot\gamma = 0$, which in coordinates becomes $\frac{d^2x^k}{dt^2} + \Gamma^k_{ij}\frac{dx^i}{dt}\frac{dx^j}{dt} = 0$. The geodesic length of a curve $f:[a,b]\to M$ is $L(f) = \int_a^b \sqrt{g_{f(t)}\!\left(\frac{df}{dx}(t),\frac{df}{dx}(t)\right)}\, dt$. For a tangent vector $v$ at $p$ there is a unique geodesic $G_v$ with $G_v(0)=p$, $\dot G_v(0)=v$, and $\exp_p(v) = G_v(1)$.

**Riemannian metric & Levi-Civita connection.** The metric $g$ is a smoothly varying, positive-definite inner product on each tangent space, giving lengths, angles, and volumes. Because tangent spaces at different points are "tilted" differently, vectors in distinct tangent spaces can't be compared directly; the *Levi-Civita connection* fixes this with a consistent rule for directional differentiation and parallel transport. It is the unique connection that is both *metric-compatible* (lengths/angles preserved along a path) and *torsion-free* (geodesics are the straightest paths): $\nabla_u V - \nabla_v U = [U,V]$. A *Killing field* $X$ preserves the metric, i.e. $\mathcal{L}_X g = 0$, equivalently $\nabla_u X_v = -\nabla_v X_u$.

The *Christoffel symbols* $\Gamma^k_{ij}$ encode how basis vectors tilt/stretch as you move across curved space (index $i$ = direction of motion, $j$ = basis vector tracked, $k$ = component of change). They are computed intrinsically from the metric:
$$\Gamma^k_{ij} = \tfrac{1}{2} g^{kl}\left(\partial_{x^i} g_{jl} + \partial_{x^j} g_{il} - \partial_{x^l} g_{ij}\right).$$
They vanish in Cartesian coordinates on flat space. Worked example — a 2-sphere of radius $r$ with coordinates $\theta, \phi$: $ds^2 = r^2(d\theta^2 + \sin^2\theta\, d\phi^2)$, so $g_{ij} = \mathrm{diag}(r^2,\; r^2\sin^2\theta)$.

**Curvature.** The *covariant derivative* corrects the naive partial derivative by adding the grid distortion of curved space: $(\nabla_U V)^k = \frac{\partial V^k}{\partial u^i}U^i + \Gamma^k_{ij}U^i V^j$. *Riemann curvature* measures the failure of the manifold to be flat — intuitively, parallel-transporting a vector around a closed loop on a sphere rotates it (the post's equator→pole→equator example rotates a vector by 90°): $R(U,V)W = \nabla_U\nabla_V W - \nabla_V\nabla_U W - \nabla_{[U,V]}W$, with the third term capturing non-commutativity. Intrinsically, $R^\rho_{\sigma\mu\nu} = \partial_{u^\mu}\Gamma^\rho_{\nu\sigma} - \partial_{u^\nu}\Gamma^\rho_{\mu\sigma} + \Gamma^\rho_{\mu\lambda}\Gamma^\lambda_{\nu\sigma} - \Gamma^\rho_{\nu\lambda}\Gamma^\lambda_{\mu\sigma}$. The *Ricci tensor* averages the Riemann tensor over directions, compressing four indices to two; it measures how the volume of a small sphere changes (the core of Einstein's field equations). Positive Ricci curvature = converging geodesics (sphere); negative = diverging geodesics (hyperbolic).

**Intrinsic vs. extrinsic geometry.** Measurements can be taken intrinsically (*first fundamental form* — the surface's own metric, giving lengths/angles/areas using only the surface, coordinate-independent) or extrinsically (*second fundamental form* — relative to the ambient Euclidean space, using embedding, extrinsic coordinates, and normal vectors).

**Lie groups & algebras.** A Lie group is both a group and a smooth manifold, satisfying closure, associativity, identity, invertibility, with smooth multiplication and inversion. A Lie algebra is its tangent space, a vector space with a Lie bracket encoding infinitesimal symmetries. The post defines: **GL(n)** invertible $n\times n$ matrices (e.g. $GL(2,\mathbb{R}) = \{[a,b;c,d] : ad-bc \neq 0\}$); **SPD(n)** symmetric ($S^T=S$) positive-definite ($x^TSx>0,\ \forall x\neq 0$) matrices; **SO(n)** rotations ($R R^{-1} = I$, $\det R = 1$); and **SE(3)** the special Euclidean group of rigid motions, $g = \{[R, t; 0, 1] \in \mathbb{R}^{4\times4} : R\in SO(3), t\in\mathbb{R}^3\}$ combining a rotation and translation.

## Notable details

- The series uses the **Geomstats** Python library for hands-on computation on manifolds.
- Practical caveat: computing the full curvature tensor is computationally intensive and noise-sensitive; the post recommends *intrinsic* computation as more robust on small or noisy datasets.
- Applications named throughout: motion planning in robotics, object recognition/detection in computer vision, equivariant and tensor-field neural networks using SO(3) for 3D shapes and molecular modeling, rotation-based data augmentation, and physics-informed neural networks (projecting PDE boundary conditions onto surface manifolds).
- It flags that real datasets often lie on *stratified spaces* (unions of manifolds of differing dimension), which classical manifold-learning theory doesn't fully handle.

## Why it matters [analyst's view]

This is a reference/cheat-sheet style consolidation rather than a novel argument — its value is precisely that it lays the differential-geometry definitions side by side with their intrinsic (metric-based) formulas, which is exactly the bridge needed to read Geometric Deep Learning and manifold-optimization papers. The matrix Lie group section (SPD, SO(n), SE(3)) is the most directly useful for ML practitioners, since those are the groups that show up in equivariant networks, pose estimation, and optimization over covariance matrices. It pairs naturally with vault work on Riemannian optimization and low-rank structure on manifolds; reading this first would make a paper like [[papers/knight-2026-riemannian-low-rank]] more tractable since both lean on tangent spaces, the exponential/log maps, and the metric. Worth queuing the promised hypersphere follow-up to see the abstractions made concrete.

## Connections

- Topic MOCs: [[topics/riemannian-geometry]], [[topics/differential-geometry]], [[topics/manifold-learning]], [[topics/geometric-deep-learning]]
- Related papers: [[papers/knight-2026-riemannian-low-rank]] — applies Riemannian/manifold structure (tangent spaces, metric) to low-rank methods
- Author index: [[authors/patrick-nicolas]]

## Selected quotes (optional)

> "Manifolds possess a local Euclidean structure known as the tangent space, where linear algebra, tensor computations, calculus, and standard machine learning algorithms remain applicable."

> "If the Christoffel symbols tell you how your coordinate grid bends, Riemann curvature tells you how the actual space bends, independently of whatever coordinates ... curvature measures the failure of a manifold to be flat."

> "Intuitively, a Lie algebra is a tangent space of a Lie group."
