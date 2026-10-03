# The Γ→∞ spectral-index problem: formulation

Notation follows Kirk, Guthmann, Gallant & Achterberg 2000 (KGGA, ApJ 542, 235) and
Keshet & Waxman 2005 (KW, PRL 94, 111102).

## Physical setup

Test particles with Lorentz factor ≫ Γ_shock, isotropic diffusion in direction angle
(D_μμ ∝ 1−μ², D constant on each side), steady planar shock, flow along +z.
In each fluid's rest frame (angle cosine μ, momentum p) but with z measured in the shock frame:

    Γ (u + μ) ∂_z f = ∂_μ [ (1−μ²) D ∂_μ f ]                                     (1)

Upstream z<0 (speed u_- = β_u), downstream z>0 (speed u_+ = β_d). Boundary conditions:
f → 0 as z → −∞; f bounded as z → +∞. Continuity of the invariant f across z = 0 with
the Lorentz boost u_rel = (u_- − u_+)/(1 − u_- u_+):

    p_+ = Γ_rel p_- (1 + u_rel μ_-),   μ_+ = (μ_- + u_rel)/(1 + u_rel μ_-).

The spectrum is a power law f ∝ p^{-s}; s is the eigenvalue of the matching problem.

## Separated solutions

`((1−μ²) Q')' = Λ (u+μ) Q`, Q regular at ±1, gives modes `Q(μ) e^{Λ z/Γ}` (D absorbed in z).
Downstream allowed: Λ ≤ 0 (constant mode Λ=0 included). Upstream allowed: Λ > 0.

## The limit u_- → 1 at fixed β = u_+

Let ε = 1 − u_- and y = (1+μ_-)/ε. Upstream, to leading order in ε,

    (y−1) ∂_Z g = ∂_y (2y ∂_y g),

whose decaying modes (regular at y=0, bounded as y→∞) are exactly

    Q_n(y) = e^{−(2n+1) y} L_n((4n+2) y),   Λ_n = 2(2n+1)²,   n = 0,1,2,…           (2)

(Laguerre polynomials; this reproduces the KGGA/Kirk–Schneider 1989 asymptotic eigenfunctions).
With κ = (1+β)/(1−β) one has 1 − u_rel = κε + O(ε²), and

    μ_+ = (y − κ)/(y + κ),     p_+/p_- = √ε (y + κ)/√κ · (1+O(ε)).

So the downstream-frame distribution at the shock is

    F(μ_+) = const · (y + κ)^s g(y) = const' · (1 − μ_+)^{−s} g.                    (3)

Note that μ_+ = −β ⇔ y = 1: the turning points (particles moving along the shock front) of the two
sides coincide. For β = 1/3 (ultra-relativistic downstream equation of state) κ = 2.

## Petrov–Galerkin condition (method A, `src/ur3d.py`)

F lies in the closed span of the downstream Λ≤0 modes iff it is orthogonal, in the indefinite inner product
⟨a,b⟩ = ∫(β+μ) a b dμ, to every downstream Λ>0 mode Q_j^+ (full-range orthogonality and completeness of the
eigenfunctions of this forward–backward problem; Beals 1981). Expanding g = Σ_{n<N} a_n Q_n and testing
against Q_j^+, j = 1..N:

    S_jn(s) = ∫_{−1}^{1} (β+μ) (y(μ)+κ)^s Q_n(y(μ)) Q_j^+(μ) dμ,      det S(s) = 0.        (4)

This is KGGA eq. (21)–(22) in the Γ→∞ limit. (KGGA print the factor as (y+2)^{−s} together with weight (y−1); with the
Jacobian dμ = 2κ dy/(y+κ)² and β+μ = (1+β)(y−1)/(y+κ), the correct y-space integrand is
(y−1)(y+κ)^{s−3} Q_n Q_j^+. The finite-Γ solver `src/finite_gamma.py`, which uses the exact boost and no limit,
converges to (4) as 1/Γ², confirming the sign of the exponent.)

Implementation: downstream Λ>0 eigenpairs from the minimal solution of the three-term recurrence in the
orthonormal Legendre basis (`k(k+1)c_k + Λ(β c_k + b_{k−1}c_{k−1} + b_k c_{k+1}) = 0`, b_k = (k+1)/√((2k+1)(2k+3))),
seeded in double precision from the reduced symmetric tridiagonal problem and refined by secant iteration on the
regularity condition β + b_0 c_1/c_0 = 0, all in Arb ball arithmetic. Quadrature: Gauss–Legendre on μ ∈ [−1,−β]
and dyadic Gauss–Legendre panels in y on [1, 2^9]. Newton on s with d log det S/ds = tr(S^{−1} S').

## Unified form (method B, `src/dord_hp.py`)

Substituting y = κ(1+μ)/(1−μ) into the upstream operator: ∂_y(2y∂_y) = (1−μ)²/(2κ) · ∂_μ(1−μ²)∂_μ and
y − 1 = (κ+1)(β+μ)/(1−μ). Hence, in the downstream angle variable, both sides obey the same operator with
different weights:

    downstream   ((1−μ²) Q')' = Λ (β+μ) Q,            allowed Λ ≤ 0
    upstream     ((1−μ²) Q')' = Λ (β+μ)(1−μ)^{−3} Q,  allowed Λ > 0
    glued by     F = (1−μ)^{−s} g.

Method B discretizes both on the same n Gauss–Legendre nodes (the operator is exact on polynomials); the
allowed discrete spaces have dimensions n_> and n_< (nodes on either side of −β) summing to n, giving a square
determinant det[V_d | diag((1−μ_k)^{−s}) V_u] = 0. It shares no code path with method A beyond Gauss nodes.

## Why convergence is algebraic

The exact shock-surface distribution is not analytic at the common turning point μ = −β (y = 1): the
equation is forward–backward parabolic there. Locally both sides reduce to x ∂_ζ f = ∂_x² f after rescaling
(the interface is invisible at leading order), so the singular terms are generated only at sub-leading order.
Empirically method A converges as s_N − s_∞ ≈ a N^{−4}(1 + b/N + …).
