"""Figures for the extensions: dimension scan, scattering-response kernel + tangled-field family, realistic shocks."""
import json, glob, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
BLUE, ORANGE, AQUA, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#1baf7a", "#1a1a19", "#6b6a64", "#e4e3dc"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
                     "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 2})
def save(fig, name):
    fig.tight_layout(); fig.savefig("figures/%s.png" % name, dpi=170); fig.savefig("figures/%s.pdf" % name); plt.close(fig)

# 6. dimension
R = {}
for f in glob.glob("results/scan_d_*.json"): R.update(json.load(open(f)))
d = np.array(sorted(v["d"] for v in R.values())); sE = np.array([float(R[k]["s_E"]) for k in sorted(R, key=lambda k: R[k]["d"])])
fig, (a1, a2) = plt.subplots(1, 2, figsize=(8.5, 3.6))
a1.semilogx(d, sE, "o-", color=BLUE, ms=4, label=r"exact, $\beta_d=1/d$")
dd = np.geomspace(8, 1100, 100); a1.semilogx(dd, 2 + 2 / dd - 7.1 / dd ** 1.5, "--", color=ORANGE, lw=1.5, label=r"$2+2/d-7.1\,d^{-3/2}$")
a1.axhline(2, color=MUTED, lw=1); a1.set_xlabel("spatial dimension d"); a1.set_ylabel(r"energy index $s_E=s-(d-1)$")
a1.set_title(r"$\Gamma_u\to\infty$ index vs dimension", loc="left", color=INK); a1.legend(frameon=False)
m = d >= 4
a2.loglog(d[m], sE[m] - 2, "o-", color=BLUE, ms=4, label=r"$s_E-2$")
a2.loglog(dd, 2 / dd, ":", color=MUTED, lw=1.5, label=r"$2/d$")
a2.set_xlabel("d"); a2.set_title("approach to the Newtonian value 2", loc="left", color=INK); a2.legend(frameon=False)
save(fig, "fig6_dimension")

# 7. kernel + tangled family
K = {}
for f in glob.glob("results/kernel_*.json"): K.update(json.load(open(f)))
mu = np.array(sorted(v["mu"] for v in K.values())); kv = np.array([float(K[k]["K"]) for k in sorted(K, key=lambda k: K[k]["mu"])])
fig, (a1, a2) = plt.subplots(1, 2, figsize=(8.5, 3.6))
a1.plot(mu, kv, "o-", color=BLUE, ms=3)
a1.axvline(-1 / 3, color=MUTED, ls=":", lw=1); a1.axhline(0, color=MUTED, lw=1)
a1.set_xlabel(r"$\mu$ (downstream frame)"); a1.set_ylabel(r"$K(\mu)=\delta s/\delta\ln D(\mu)$")
a1.set_title(r"response of $s_\infty$ to downstream scattering", loc="left", color=INK)
A = {}
for f in glob.glob("results/scan_aniso_*.json"): A.update(json.load(open(f)))
if A:
    av = np.array(sorted(v["a"] for v in A.values())); sv = np.array([float(A[k]["s_inf"]) for k in sorted(A, key=lambda k: A[k]["a"])])
    a2.semilogx(av, sv, "o-", color=BLUE, ms=4, label="exact")
    a2.axhline(4.226978816696085, color=MUTED, ls="--", lw=1.2, label="isotropic")
    a2.plot([0.1], [4.21], "s", color=ORANGE, ms=7, label="KGGA 2000 (≈)")
    a2.plot([0.1], [4.26], "^", color=AQUA, ms=7, label="KW-type approx.")
    a2.set_xlabel(r"$a$ in $D\propto(\mu^2+a^2)^{-1/2}$"); a2.set_ylabel(r"$s_\infty$")
    a2.set_title("tangled-field scattering", loc="left", color=INK); a2.legend(frameon=False, fontsize=8)
save(fig, "fig7_scattering")

# 8. realistic strong shocks (Juettner-Synge)
J = json.load(open("results/js_curve.json")); mn = json.load(open("results/js_min.json"))
g = np.array([r["Gu_bu"] for r in J]); s = np.array([r["s"] for r in J])
fig, ax = plt.subplots(figsize=(6, 4))
ax.semilogx(g, s, "o-", color=BLUE, ms=4, label="exact (Jüttner–Synge, strong shock)")
ax.axhline(4, color=MUTED, lw=1, ls=":"); ax.axhline(4.226978816696085, color=MUTED, lw=1.2, ls="--", label=r"$s_\infty$")
ax.plot([mn["Gu_bu"]], [mn["s_min"]], "*", color=ORANGE, ms=12, label=r"minimum $s=%.5f$" % mn["s_min"])
ax.set_xlabel(r"upstream four-velocity $\Gamma_u\beta_u$"); ax.set_ylabel("s")
ax.set_title("Index for a strong shock at any speed", loc="left", color=INK); ax.legend(frameon=False, loc="center right")
save(fig, "fig8_realistic")
print("ok")
