"""Finite-Gamma eigenfunction matching (Kirk-Guthmann-Gallant-Achterberg 2000 'new method'), double precision.

Independent of the Gamma->inf derivation: upstream modes are computed from the full Legendre pencil at
upstream speed u (no Laguerre limit), and the exact Lorentz transformation is used:
  mu_+ = (mu_- + ur)/(1 + ur mu_-),  p_+/p_- = Gr (1 + ur mu_-),  ur = (u - b)/(1 - u b).
  S_ij(s) = int dmu_+ (b + mu_+) (1 + ur mu_-)^s Q_i^-(mu_-) Q_j^+(mu_+),  det S = 0.
Quadrature is done in y = (1+mu_-)/(1-u) with panel [0,1] (oscillatory) and dyadic panels to y=2/(1-u).
"""
import numpy as np
from numpy.polynomial import legendre as Lg
from scipy.linalg import eigh_tridiagonal
from scipy.optimize import brentq
import sys


def positive_modes(beta, N, nleg):
    """Legendre-series coefficients (standard P_l normalisation) of the N lowest Lam>0 modes of
    ((1-mu^2)Q')' = Lam (beta+mu) Q."""
    l = np.arange(nleg, dtype=float)
    b = (l[:-1] + 1) / np.sqrt((2 * l[:-1] + 1) * (2 * l[:-1] + 3))
    d = np.full(nleg - 1, beta); d[0] = beta - b[0] ** 2 / beta
    kk = l[1:] * (l[1:] + 1); sq = 1 / np.sqrt(kk)
    dT = d * sq * sq
    eT = b[1:nleg - 1] * sq[:-1] * sq[1:]
    sig, V = eigh_tridiagonal(dT, eT)
    idx = np.where(sig < 0)[0]
    idx = idx[np.argsort(sig[idx])][:N]
    lam = -1 / sig[idx]
    C = np.zeros((nleg, N))
    C[1:, :] = V[:, idx] * sq[:, None]
    C[0, :] = -b[0] * C[1, :] / beta
    C = C * np.sqrt((2 * l + 1) / 2)[:, None]     # orthonormal -> standard Legendre
    return lam, C


def leg_eval(x, C):
    return np.stack([Lg.legval(x, C[:, j]) for j in range(C.shape[1])])


def build(u, b, N, nleg_up=None, nleg_dn=None, KA=None, KB=60):
    eps = 1 - u
    ur = (u - b) / (1 - u * b)
    nleg_dn = nleg_dn or (6 * N + 150)
    nleg_up = nleg_up or int(6 * N / np.sqrt(eps) + 200)
    KA = KA or (10 * N + 100)
    _, Cu = positive_modes(u, N, nleg_up)
    _, Cd = positive_modes(b, N, nleg_dn)
    # nodes in y
    xg, wg = Lg.leggauss(KA)
    ys = [(xg + 1) / 2]; ws = [wg / 2]
    xg, wg = Lg.leggauss(KB)
    lo = 1.0
    while lo < 2 / eps:
        hi = min(2 * lo, 2 / eps)
        ys.append((lo + hi) / 2 + (hi - lo) / 2 * xg); ws.append((hi - lo) / 2 * wg)
        lo = hi
    y = np.concatenate(ys); w = np.concatenate(ws)
    mum = -1 + eps * y                     # mu_-
    dmum = eps * w
    mup = (mum + ur) / (1 + ur * mum)
    dmup = (1 - ur ** 2) / (1 + ur * mum) ** 2 * dmum
    Qm = leg_eval(mum, Cu)
    Qp = leg_eval(mup, Cd)
    Qm /= np.abs(Qm).max(axis=1, keepdims=True)
    Qp /= np.abs(Qp).max(axis=1, keepdims=True)
    base = dmup * (b + mup)
    lg = np.log(1 + ur * mum)

    def S(s):
        return (Qp * (base * np.exp(s * lg))) @ Qm.T
    return S


def solve(u, b, N, lo=2.5, hi=8.0, **kw):
    S = build(u, b, N, **kw)
    f = lambda s: np.linalg.det(S(s) / np.linalg.norm(S(s), axis=1, keepdims=True))
    grid = np.linspace(lo, hi, 300)
    v = [f(s) for s in grid]
    for i in range(len(grid) - 1):
        if np.sign(v[i]) != np.sign(v[i + 1]):
            return brentq(f, grid[i], grid[i + 1], xtol=1e-15)
    return None


if __name__ == "__main__":
    b = 1 / 3
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    for G in [1.5, 2, 3, 5, 10, 20, 40, 80]:
        u = np.sqrt(1 - 1 / G ** 2)
        print("Gamma_u=%6.1f  u=%.10f  s_N=%.12f" % (G, u, solve(u, b, N)), flush=True)
