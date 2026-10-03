"""Double-precision prototype: KGGA-type eigenfunction matching in the Gamma->inf limit (3D).

Downstream (fluid frame, speed beta): ((1-mu^2) Q')' = Lam (beta+mu) Q, Q regular at +-1.
Upstream (Gamma->inf scaling y=(1+mu_-)/eps): Q_n(y) = exp(-(2n+1)y) L_n((4n+2)y), Lam_n>0.
Matching: F(mu_+) = (y+kappa)^s g(y),  mu_+ = (y-kappa)/(y+kappa),  kappa=(1+beta)/(1-beta).
Condition: F orthogonal (weight beta+mu) to all downstream modes with Lam>0.
S_jn(s) = int_0^inf (y-1) (y+kappa)^(s-3) Q_n(y) Q_j^+(mu(y)) dy.
"""
import numpy as np
from numpy.polynomial import legendre as L
from scipy.special import eval_laguerre
from scipy.linalg import eig
from scipy.optimize import brentq
import sys


def downstream_modes(beta, nleg=200):
    l = np.arange(nleg)
    K = np.diag(l * (l + 1.0))
    b = (l[:-1] + 1) / np.sqrt((2 * l[:-1] + 1) * (2 * l[:-1] + 3))
    A = beta * np.eye(nleg) + np.diag(b, 1) + np.diag(b, -1)
    w, V = eig(K, A)          # K c = w A c, w = -Lam
    w = w.real
    V = V.real
    lam = -w
    pos = np.where(lam > 1e-9)[0]
    pos = pos[np.argsort(lam[pos])]
    # convert orthonormal-Legendre coefficients to Legendre-series coefficients
    scale = np.sqrt((2 * l + 1) / 2.0)
    C = V[:, pos] * scale[:, None]
    return lam[pos], C


def build(beta, N, nleg=200, nq=400):
    kappa = (1 + beta) / (1 - beta)
    lam, C = downstream_modes(beta, nleg)
    # quadrature in mu on [-1,1] via Gauss-Legendre (integrand C-infinity)
    x, w = L.leggauss(nq)
    y = kappa * (1 + x) / (1 - x)
    jac = 2 * kappa / (1 - x) ** 2
    Qd = np.stack([L.legval(x, C[:, j]) for j in range(N)])     # (N, nq)
    Qu = np.stack([np.exp(-(2 * n + 1) * y) * eval_laguerre(n, (4 * n + 2) * y) for n in range(N)])
    base = w * jac * (y - 1)
    def S(s):
        wt = base * np.exp((s - 3) * np.log(y + kappa))
        return (Qd * wt) @ Qu.T
    return S, lam


def det_fn(S):
    def f(s):
        M = S(s)
        # row/col normalise
        M = M / np.linalg.norm(M, axis=1, keepdims=True)
        return np.linalg.det(M)
    return f


if __name__ == "__main__":
    beta = float(sys.argv[1]) if len(sys.argv) > 1 else 1 / 3
    for N in range(1, 16):
        S, lam = build(beta, N)
        f = det_fn(S)
        grid = np.linspace(3.0, 6.0, 301)
        vals = [f(s) for s in grid]
        roots = []
        for i in range(len(grid) - 1):
            if np.sign(vals[i]) != np.sign(vals[i + 1]):
                roots.append(brentq(f, grid[i], grid[i + 1], xtol=1e-14))
        print(N, ["%.10f" % r for r in roots[:3]], "lam1..3", lam[:3])
