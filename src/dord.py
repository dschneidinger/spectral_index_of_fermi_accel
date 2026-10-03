"""Independent method: discrete ordinates (Gauss-node collocation) for the Gamma_u -> inf spectral index.

Unified form (notes/FORMULATION.md): in the downstream angle variable mu, both sides obey
    ((1-mu^2) Q')' = Lam w(mu) Q,
    downstream  w_d = (beta+mu)                allowed: Lam <= 0
    upstream    w_u = (beta+mu) (1-mu)^-3       allowed: Lam >  0   (Gamma->inf limit, y = kappa(1+mu)/(1-mu))
glued at the shock by F(mu) = (1-mu)^-s g(mu)  (constant factors dropped).
On n Gauss-Legendre nodes the operator is exact on polynomials (Legendre-diagonal), the weights are nodal.
The allowed discrete spaces have dimensions n_> (nodes mu_k > -beta) and n_< (nodes < -beta), summing to n,
and s solves det[ V_d | diag((1-mu_k)^-s) V_u ] = 0.
"""
import numpy as np
from numpy.polynomial import legendre as Lg
import mpmath as mp
import sys


def allowed_modes(mu, wq, wfun, sign, prec=None):
    """nodal values of allowed eigenvectors for weight wfun on Gauss nodes mu (weights wq)."""
    n = len(mu)
    l = np.arange(n)
    # orthonormal Legendre Vandermonde: V[k,l] = p_l(mu_k);  V^T diag(wq) V = I
    V = np.stack([Lg.legval(mu, np.eye(n)[i]) * np.sqrt((2 * i + 1) / 2) for i in range(n)], axis=1)
    M = V.T @ (wq * wfun(mu))[:, None] * 1.0
    M = (V.T * (wq * wfun(mu))) @ V
    K = l * (l + 1.0)
    # eliminate c0:  M c = sigma K c,  row 0 gives c0 = -M[0,1:] c' / M[0,0]
    Ms = M[1:, 1:] - np.outer(M[1:, 0], M[0, 1:]) / M[0, 0]
    sq = 1 / np.sqrt(K[1:])
    T = sq[:, None] * Ms * sq[None, :]
    T = (T + T.T) / 2
    sig, U = np.linalg.eigh(T)
    Cp = sq[:, None] * U
    C0 = -(M[0, 1:] @ Cp) / M[0, 0]
    C = np.vstack([C0, Cp])               # columns: eigenvectors, sigma = -1/Lam
    # Lam = -1/sigma ; Lam>0 <-> sigma<0
    if sign > 0:
        sel = sig < 0
        cols = C[:, sel]
    else:
        sel = sig > 0
        cols = np.hstack([np.eye(n)[:, :1], C[:, sel]])  # include the constant (Lam=0)
    vals = V @ cols
    return vals / np.abs(vals).max(axis=0, keepdims=True)


def solve(beta, n, lo=None, hi=None):
    mu, wq = Lg.leggauss(n)
    Vd = allowed_modes(mu, wq, lambda m: beta + m, -1)
    Vu = allowed_modes(mu, wq, lambda m: (beta + m) / (1 - m) ** 3, +1)
    assert Vd.shape[1] + Vu.shape[1] == n, (Vd.shape, Vu.shape)
    lg = np.log(1 - mu)

    def f(s):
        A = np.hstack([Vd, np.exp(-s * lg)[:, None] * Vu])
        A = A / np.linalg.norm(A, axis=0, keepdims=True)
        return np.linalg.slogdet(A)
    from scipy.optimize import brentq
    g = lambda s: f(s)[0] * np.exp(f(s)[1] - f(4.2)[1])
    grid = np.linspace(lo or 3.9, hi or 4.6, 141)
    v = [g(x) for x in grid]
    roots = [brentq(g, grid[i], grid[i + 1], xtol=1e-15) for i in range(len(grid) - 1) if np.sign(v[i]) != np.sign(v[i + 1])]
    return roots


if __name__ == "__main__":
    beta = 1 / 3
    for n in [8, 16, 24, 32, 48, 64, 96, 128, 160, 200, 256]:
        print(n, ["%.14f" % r for r in solve(beta, n)], flush=True)
