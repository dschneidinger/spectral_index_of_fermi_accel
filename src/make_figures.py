"""Figures for project_2 (static PNGs in figures/)."""
import json, glob, sys, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mpmath as mp
sys.path.insert(0, os.path.dirname(__file__))
from extrap import load

mp.mp.dps = 40
BLUE, ORANGE, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#1a1a19", "#6b6a64", "#e4e3dc"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
                     "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 2})
os.makedirs("figures", exist_ok=True)
S3 = mp.mpf("4.226978816696084932553273")
S2 = mp.mpf("3.2985080563907181433382986")

# 1. literature vs this work
fig, ax = plt.subplots(figsize=(7, 3.2))
rows = [("Bednarz & Ostrowski 1998 (MC, ≈)", 4.2, 0), ("Kirk et al. 2000 (eigenfunctions)", 4.23, 0.01),
        ("Achterberg et al. 2001 (MC)", 4.22, 0.01), ("Keshet & Waxman 2005 (38/9, approx.)", 38 / 9, 0),
        ("Keshet 2006 (moments, N=6)", 4.24, 0), ("Nagar & Keshet 2020 (relaxation)", 4.227, 0.001),
        ("This work (25 digits)", float(S3), 0)]
for i, (lab, v, e) in enumerate(rows):
    c = ORANGE if lab.startswith("This") else BLUE
    ax.errorbar(v, i, xerr=e if e else None, fmt="o", color=c, ms=6, capsize=3, lw=1.5)
ax.axvline(float(S3), color=ORANGE, lw=1, ls="--")
ax.set_yticks(range(len(rows))); ax.set_yticklabels([r[0] for r in rows])
ax.set_xlim(4.15, 4.30); ax.set_xlabel(r"$s$ in $f\propto p^{-s}$  ($\Gamma\to\infty$, 3D, isotropic)")
ax.set_title("Universal index: literature vs. this work", loc="left", color=INK)
fig.tight_layout(); fig.savefig("figures/fig1_literature.png", dpi=170); fig.savefig("figures/fig1_literature.pdf"); plt.close(fig)

# 2. convergence
fig, ax = plt.subplots(figsize=(6, 4))
for pat, S, c, lab in [("results/ur3d_beta1_3_*.json", S3, BLUE, "3D (β_d=1/3)"), ("results/ur2d_beta1_2.json", S2, ORANGE, "2D (β_d=1/2)")]:
    D = load(pat)
    N = np.array([n for n in D if n >= 2]); e = np.array([float(D[n] - S) for n in N])
    ax.loglog(N, e, "o-", color=c, ms=4, label=lab)
Ng = np.array([8, 320]); ax.loglog(Ng, 1.5e-4 * Ng ** -4.0, ":", color=MUTED, lw=1.5, label="∝ N⁻⁴")
ax.set_xlabel("N (upstream trial modes = downstream test modes)"); ax.set_ylabel(r"$s_N - s_\infty$")
ax.set_title("Truncation error of the eigenfunction matching", loc="left", color=INK)
ax.legend(frameon=False); fig.tight_layout(); fig.savefig("figures/fig2_convergence.png", dpi=170); fig.savefig("figures/fig2_convergence.pdf"); plt.close(fig)

# 3. s_inf(beta_d) vs KW
R = json.load(open("results/scan_beta_3d.json"))
b = np.array(sorted(v["beta"] for v in R.values())); s = np.array([float(R[k]["s_inf"]) for k in sorted(R, key=lambda k: R[k]["beta"])])
kw = (3 - 2 * b ** 2 + b ** 3) / (1 - b)
fig, (a1, a2) = plt.subplots(2, 1, figsize=(6, 5.6), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
a1.plot(b, s, "o-", color=BLUE, ms=4, label="exact (this work)")
a1.plot(b, kw, "--", color=ORANGE, label="Keshet–Waxman 2005 formula")
a1.set_ylabel(r"$s_\infty(\beta_d)$"); a1.legend(frameon=False); a1.set_title("Γ_u→∞ index vs downstream speed", loc="left", color=INK)
a2.plot(b, kw - s, "o-", color=ORANGE, ms=4); a2.axhline(0, color=MUTED, lw=1)
a2.set_ylabel("KW − exact"); a2.set_xlabel("β_d (downstream speed in shock frame)")
fig.tight_layout(); fig.savefig("figures/fig3_beta_scan.png", dpi=170); fig.savefig("figures/fig3_beta_scan.pdf"); plt.close(fig)

# 4. shock-front angular distributions
fig, ax = plt.subplots(figsize=(6, 4))
for f, c, lab, beta in [("results/F_shock_3d.json", BLUE, "3D", 1 / 3), ("results/F_shock_2d.json", ORANGE, "2D", 1 / 2)]:
    T = json.load(open(f)); mu = np.array([r[0] for r in T["F_over_F0"]]); F = np.array([r[1] for r in T["F_over_F0"]])
    ax.plot(mu, F, color=c, label="%s, F(μ)/F(−β_d)" % lab)
    ax.axvline(-beta, color=c, lw=1, ls=":")
ax.set_xlabel("μ = cos θ in the downstream rest frame"); ax.set_ylabel("shock-front distribution (normalised)")
ax.set_title("Angular distribution at the shock (Γ_u→∞)", loc="left", color=INK); ax.legend(frameon=False)
fig.tight_layout(); fig.savefig("figures/fig4_angular.png", dpi=170); fig.savefig("figures/fig4_angular.pdf"); plt.close(fig)
print("ok")

# 5. grazing anisotropy vs KW identity
import re
fig, axs = plt.subplots(1, 2, figsize=(8.5, 3.6))
for ax, log, c, req, lab in [(axs[0], "results/aniso3d.log", BLUE, 3 * (float(S3) - 2) / 8, "3D"),
                             (axs[1], "results/aniso2d.log", ORANGE, 2 * float(S2) * (float(S2) - 1) / (3 * (2 * float(S2) + 1)), "2D")]:
    D = {}
    for line in open(log):
        m = re.search(r"N= *(\d+).*F.\/F=\[([0-9.]+)", line)
        if m: D[int(m.group(1))] = float(m.group(2))
    N = np.array(sorted(D)); v = np.array([D[n] for n in N])
    ax.plot(N ** (-2 / 3), v, "o-", color=c, ms=4, label="Galerkin, N = %d…%d" % (N[0], N[-1]))
    ax.axhline(req, color=MUTED, ls="--", lw=1.5, label="required by KW identity")
    ax.plot([0], [0.85950 if lab == "3D" else 0.70965], "*", color=c, ms=12, label="extrapolated, N→∞")
    ax.set_xlim(0, None); ax.set_xlabel(r"$N^{-2/3}$"); ax.set_title(lab, loc="left", color=INK)
    ax.legend(frameon=False, fontsize=8, loc="center right")
axs[0].set_ylabel(r"$F'(-\beta_d)/F(-\beta_d)$")
fig.tight_layout(); fig.savefig("figures/fig5_kw_identity.png", dpi=170); fig.savefig("figures/fig5_kw_identity.pdf"); plt.close(fig)
