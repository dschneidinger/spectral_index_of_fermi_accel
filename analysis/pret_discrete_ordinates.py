"""Paper Sec. VII: return probability and grazing anisotropy from the independent discrete-ordinates solution."""
import _common as C
import numpy as np
from numpy.polynomial import legendre as Lg
from dord import allowed_modes, solve
beta = 1 / 3
for n in [24, 32, 40, 48]:
    s = [r for r in solve(beta, n) if abs(r - 4.227) < 0.01][0]
    mu, wq = Lg.leggauss(n)
    Vd = allowed_modes(mu, wq, lambda m: beta + m, -1)
    Vu = allowed_modes(mu, wq, lambda m: (beta + m) / (1 - m) ** 3, +1)
    A = np.hstack([Vd, np.exp(-s * np.log(1 - mu))[:, None] * Vu])
    c = np.linalg.svd(A)[2][-1]
    coef = Lg.legfit(mu, Vd @ c[:Vd.shape[1]], n - 1)
    def J(a, b):
        x, w = Lg.leggauss(400); t = (a + b) / 2 + (b - a) / 2 * x
        return np.sum(w * (b - a) / 2 * (beta + t) * Lg.legval(t, coef))
    print("n=%d  s=%.12f  P_ret=%.8f  F'/F(-beta)=%.5f" % (n, s, -J(-1, -beta) / J(-beta, 1),
          Lg.legval(-beta, Lg.legder(coef)) / Lg.legval(-beta, coef)))
