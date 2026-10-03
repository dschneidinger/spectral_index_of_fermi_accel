# The spectral index of Fermi acceleration at ultra-relativistic shocks

25 digits, all dimensions, arbitrary scattering, physical shocks at all speeds, and the breakdown of the
Keshet–Waxman spectrum–anisotropy identity. The paper is [`paper/main.pdf`](paper/main.pdf).

**Problem.** Test particles undergoing isotropic diffusion in direction angle, accelerated at a planar
ultra-relativistic (Γ_u → ∞) shock, form a power law f ∝ p^(−s). The index s is the canonical benchmark of
relativistic shock acceleration theory: it underlies the p ≈ 2.2 electron spectra inferred for GRB afterglows,
and it is the number PIC studies of Weibel-mediated shocks compare against. For 25 years it has been known to only
2–3 digits: 4.23 ± 0.01 (eigenfunctions, Kirk et al. 2000), 4.22 ± 0.01 (Monte Carlo, Achterberg et al. 2001),
4.227 ± 0.001 (finite-difference relaxation, Nagar & Keshet 2020). An analytic value, 38/9 = 4.2222
(Keshet & Waxman 2005), rests on an "exact" spectrum–anisotropy identity plus a truncation.

## Results

| quantity | value | previous best |
|---|---|---|
| s∞, 3D (β_d = 1/3) | **4.226978816696084932553273** (±~1e-26) | 4.227 ± 0.001 |
| s∞, 2D (β_d = 1/2), relevant to 2D PIC | **3.2985080563907181433382986**; energy index s_E = s − 1 = 2.29850805639… | ≈ 2.30 (s_E) |
| ds∞/dβ_d at β_d → 0 (3D; s → 3 there) | **2.87946794932587**, a₂ = 1.7493935015, a₃ = 1.17662 | KW formula gives 3 |
| finite-Γ correction (3D, β_d = 1/3 fixed) | s(Γ) ≈ s∞ + 0.990/Γ² + 1.09/Γ⁴ | — |
| return probability P_ret, 3D | **0.3789847** (independent MC: 0.3767 ± 0.0008, see finding 4) | 0.437 ± 0.002 (MC, Achterberg et al. 2001) |
| return probability P_ret, 2D | 0.3339262 | — |
| grazing anisotropy F′/F at μ = −β_d, 3D | **0.85950 ± 0.00002** | KW identity requires 0.835117 |
| grazing anisotropy F′/F at μ = −β_d, 2D | **0.70965 ± 0.00001** | KW-type identity requires 0.665318 |
| full curve s∞(β_d), 0 < β_d ≤ 0.85 | `results/scan_beta_3d.json` (~18 digits per point) | KW formula, error up to 5.5e-3 |

### Findings

1. **The universal index, to 25 digits.** s∞ = 4.226978816696084932553273. This settles the spread among the
   literature values: 38/9 is excluded by 4.8×10⁻³, which is ~10²² times the uncertainty. 4.227 ± 0.001 is confirmed.
2. **The Keshet–Waxman spectrum–anisotropy identity is not exact.** KW (PRL 94, 111102) derived
   s = (2r_d + β_u + β_d)/(β_u − β_d), with r_d the anisotropy of particles moving along the shock front. They
   presented it as exact for any isotropic diffusion. Re-deriving it under their own assumptions
   (`src/kw_identity.py` reproduces their formula exactly) and testing it on the solution shows violations of
   2.9% in 3D and 6.7% in 2D, confirmed by two independent discretizations. The mechanism: the shock-front
   distribution is not C² at grazing incidence μ = −β_d. Locally both sides reduce to the forward–backward
   equation x∂_z f = ∂_x² f. They differ only in a drift term, which jumps across the shock and excites a
   self-similar corner solution z^(2/3)Φ(x/z^(1/3)). The limit τ → 0 of the second-order coefficient a₂(τ),
   which KW match across the shock, then differs from the trace's one-sided second derivatives. The same corner
   explains the observed convergence s_N − s∞ ≈ aN⁻⁴ + bN⁻⁵ + cN⁻⁶ log N. The log term is detected by model
   selection: it cuts out-of-sample prediction error ~100×. The KW *formula* for s is nevertheless accurate to
   ≤ 5.5×10⁻³ over the whole β_d range, with an error that changes sign twice (Fig. 3).
3. **2D index.** s_p = 3.29850805639071814333830 (s_E = 2.2985…). This is the number 2D PIC spectra should be
   compared with, under the same idealized scattering model.
4. **Return probability.** The flux-ratio return probability is exact in this model (downstream scattering
   conserves |p| in the downstream frame): P_ret = 0.37898. The value 0.437 ± 0.002 from Achterberg et al. (2001)
   cannot be right for the model it simulates. An independent Monte Carlo (`src/mc_downstream.py`), injecting the
   computed entering distribution and following direction-vector Brownian motion, gives 0.3772 ± 0.0011 (dt = 4e-3) and
   0.3761 ± 0.0011 (dt = 2e-3). Its exit-angle flux histogram matches the eigenfunction prediction. This excludes
   0.437 by ~70σ. A residual 2.9σ shortfall relative to the exact 0.37898 remains unresolved at this statistics;
   the expected sign is low, because end-of-step crossing detection misses brief returns of grazing particles. Their spectral slope agrees
   with ours, so the error is likely in the bookkeeping of crossing cycles. Their Bell-type check
   s = 1 + ln(1/P)/ln⟨g⟩ is itself only approximate; the exact statement is P⟨g^(3−s)⟩ = 1.
5. **Closed form?** None found at the heights 25 digits allow. Using BootLoops' `pslq_gate` with in-basket
   planted controls: no algebraic relation of degree ≤ 3 with coefficients ≤ 10⁵, and no rational combination of
   {1, π, π², log 2, log 3} ≤ 10³ or of {1, √2, √3, ζ(3)} ≤ 10⁴. This is a weak exclusion, stated as such.

## Extensions (see paper Secs. VIII–X)

| result | value |
|---|---|
| any dimension d (β_d = 1/d) | upstream modes e^(−ay) L_n^((d−3)/2)(2ay), a = 2n + (d−1)/2; energy index s_E(d) tabulated for d = 2…1024 (`results/scan_d_*.json`) |
| large-d law (numerical) | s_E = 2 + 2/d − 7.1(1) d^(−3/2) + …; the limit is the Newtonian strong-shock value 2, confirmed to 3×10⁻⁸ |
| upstream scattering | drops out of the Γ→∞ limit entirely (only D_up(−1) enters, and it rescales z) |
| response kernel K(μ) = δs/δ ln D(μ) | smooth, ∫K = 0; no enhanced sensitivity at grazing (dipole \|c\| < 10⁻⁷); reproduces the exact a = 10 and a = 3 shifts to 0.5% and 0.1% (`results/kernel_*.json`) |
| KGGA tangled field, D ∝ (μ²+a²)^(−1/2), a = 0.1 | s∞ = 4.2126801838 (shift −0.0143; previous estimates −0.02 and +0.03) |
| strong shock into cold gas, Jüttner–Synge | s(Γβ) at all speeds; dips below 4 for Γβ < 0.9249, minimum s = 3.988970 at Γβ = 0.5541 (`results/js_curve.json`) |

## Method

Formulation in [`notes/FORMULATION.md`](notes/FORMULATION.md). In the limit Γ_u → ∞ the upstream decaying
eigenfunctions are exact Laguerre functions, Q_n(y) = e^(−(2n+1)y) L_n((4n+2)y) in 3D and
e^(−(4n+1)y/2) L_n^(−1/2)((4n+1)y) in 2D. The Lorentz map becomes μ_+ = (y−κ)/(y+κ). The problem then reduces to
one Legendre operator ((1−μ²)Q′)′ = Λ w Q, with weights w = β+μ downstream and (β+μ)(1−μ)⁻³ upstream, glued by
F = (1−μ)^(−s) g. s is the root of det S(s), where S is a Petrov–Galerkin matrix (KGGA's method in the limit;
`src/urnd.py`, `src/ur3d.py`):

- Arb ball arithmetic (python-flint, as BootLoops uses throughout), working precision 200 + 3N bits.
- Downstream eigenpairs come from the minimal solution of the Legendre three-term recurrence.
- s_N is computed for N ≤ 320 (3D) and N ≤ 160 (2D), then extrapolated.

### Validation

- **Discretization:** at N = 128, s_N is unchanged to 50 digits when the quadrature nodes, the y-panels, the
  Legendre truncation and the y cutoff are each increased by 1.5–2.5×. Precision was doubled at selected N.
- **Independent method B** (`src/dord_hp.py`, discrete ordinates on Gauss nodes, no shared code path) agrees to
  ~10⁻¹¹; its convergence is oscillatory.
- **Physics of the limit:** an independent finite-Γ solver (`src/finite_gamma.py`, full Lorentz map, no limit
  taken) converges to the limiting s_N as 1/Γ² (extrapolation agrees to 2×10⁻⁷).
- **Controls:**
  - It reproduces nonrelativistic DSA, s → 3r/(r−1) for r = 4 and r = 2.5.
  - s∞(β_d) → 3 as β_d → 0 (P_ret → 1).
  - The literature: 4.227 ± 0.001 (3D), s_E ≈ 2.30 (2D).
- **Extrapolation:** the model was chosen by out-of-sample prediction of the largest-N point
  (`src/model_select.py`). The quoted digits are stable across fit orders m = 6–8 (spread 7×10⁻²⁸) and agree with
  the pure-power model to 3×10⁻²⁶.

### Limits of what is claimed

- These are results for the standard idealized model: test particles, isotropic small-angle scattering in
  direction angle, an unmagnetized planar shock with Γ_u → ∞. Real Weibel-mediated shocks have anisotropic and
  energy-dependent scattering, and the index is known to depend on D(μ) (KGGA 2000; Nagar & Keshet 2020). This is
  a definitive solution of the benchmark problem, not a prediction for any particular astrophysical shock.
- The 25-digit value rests on a well-validated but non-rigorous extrapolation in N. The ball radii certify only
  each finite-N computation.

## Repository map

| path | contents |
|---|---|
| `paper/` | `main.tex`, `refs.bib`, compiled `main.pdf` |
| `notes/FORMULATION.md` | derivation of the Γ→∞ problem and the unified two-weight form |
| `src/` | solvers and scans (table below) |
| `analysis/` | scripts that reproduce every derived number quoted in the paper from `results/` |
| `results/` | raw solver output (JSON + run logs); `results/analysis/*.log` are the outputs of `analysis/`; `results/SUMMARY.json` collects the headline numbers |
| `figures/` | PNG/PDF figures, built by `src/make_figures.py` and `src/make_figures_ext.py` |

| solver | purpose |
|---|---|
| `src/urnd.py`, `src/ur3d.py` | Γ→∞ Petrov–Galerkin solver in Arb (2D/3D; `ur3d.py` is the original 3D version) |
| `src/urd.py` | same, any dimension d |
| `src/aniso.py` | same, arbitrary downstream scattering D(μ) |
| `src/dord.py`, `src/dord_hp.py` | independent discrete-ordinates solver (double / Arb) |
| `src/finite_gamma.py` | finite-Γ solver (exact Lorentz map, no limit), double precision |
| `src/proto_galerkin.py` | first double-precision prototype |
| `src/extrap.py`, `src/model_select.py` | extrapolation in N and model selection |
| `src/anisotropy.py`, `src/kw_identity.py` | grazing anisotropy; symbolic re-derivation of the KW identity |
| `src/return_prob.py`, `src/mc_downstream.py` | exact return probability; independent Monte Carlo |
| `src/scan_beta.py`, `src/scan_d.py`, `src/scan_aniso.py`, `src/kernel.py` | parameter scans and the response kernel |
| `src/js_curve.py`, `src/js_min.py` | Jüttner–Synge strong shocks at all speeds |
| `src/closed_form_scan.py` | PSLQ closed-form test (needs a BootLoops checkout via `BOOTLOOPS_ROOT`) |

## Setup and reproduction

Python 3.12 with the pinned packages in `requirements.txt` (`pip install -r requirements.txt`). Run everything from
the repository root. Single-threaded BLAS is recommended (`export OMP_NUM_THREADS=1`).

```
python src/urnd.py --dim 3 --N 16 32 64          # s_N in 3D (precision scales with N)
python src/model_select.py                        # extrapolation from results/ur3d_*.json
python src/anisotropy.py --dim 3 --N 16 32 64     # grazing anisotropy vs the KW identity
python src/return_prob.py --dim 3 --N 16 32 64    # exact return probability
python src/urd.py --d 5 --N 16 32                 # any dimension
python src/aniso.py --a 0.1 --N 8 16              # tangled-field scattering
python src/js_curve.py                            # physical strong shocks (writes results/js_curve.json)
for f in analysis/*.py; do python $f; done        # all derived numbers (outputs as in results/analysis/)
python src/make_figures.py && python src/make_figures_ext.py
```

Costs: s_64 takes about 6 s and s_320 about an hour on one core. The full 3D sequence to N = 320 was split over
several processes (`results/ur3d_beta1_3_*.json`).

## Provenance

The computations were built on the [BootLoops](https://www.bootloops.ai) 1.0 harness: Arb ball arithmetic via
python-flint, and its `pslq_gate` integer-relation protocol for the closed-form test. Only
`src/closed_form_scan.py` imports BootLoops code; everything else needs only the packages in `requirements.txt`.
The code and the first draft of the paper were written with the assistance of an AI model (Claude, Anthropic).

Literature sources (arXiv): astro-ph/0005222 (KGGA), astro-ph/0107530 (Achterberg et al.),
astro-ph/0408489 (Keshet & Waxman), astro-ph/0608182 (Keshet 2006), 1910.10030 (Nagar & Keshet 2020).

No license has been chosen yet; add one before making the repository public.
