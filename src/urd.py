"""Gamma_u -> infinity spectral index in arbitrary spatial dimension d (real d >= 2), isotropic diffusion on S^{d-1}.

Angular operator (axisymmetric): (1-mu^2)^{-(d-3)/2} d/dmu [(1-mu^2)^{(d-1)/2} d/dmu], eigenvalues -k(k+d-2) on
Gegenbauer polynomials C_k^{(lam)}, lam = (d-2)/2, orthogonal w.r.t. (1-mu^2)^{lam-1/2} dmu.
Downstream modes:  -k(k+2lam) c_k + ... :  kk_k c_k + Lam(beta c_k + b_{k-1} c_{k-1} + b_k c_{k+1}) = 0 (orthonormal basis),
    b_0 = 1/sqrt(2(lam+1)),  b_k = sqrt((k+1)(k+2lam) / (4 (k+lam)(k+lam+1))).
Upstream (Gamma->inf): 2y g'' + (d-1) g' = Lam (y-1) g  =>  Q_n(y) = exp(-a y) L_n^{((d-3)/2)}(2 a y), a = 2n + (d-1)/2.
Matching F(mu_+) = (y+kappa)^s g(y), mu_+ = (y-kappa)/(y+kappa). Measure (1-mu^2)^{(d-3)/2} dmu = sin^{d-2}(phi) dphi.
d = 3 and d = 2 reproduce urnd.py exactly (basis normalisations differ only by constants).
"""
import sys, time, json, argparse
import numpy as np
from scipy.linalg import eigh_tridiagonal
from flint import arb, arb_mat, fmpq, ctx


class GeoD:
    def __init__(self, d):
        self.d = d                          # fmpq or int
        self.da = arb(d)
        self.lam = (self.da - 2) / 2
        self.alpha = (self.da - 3) / 2
        self.df = float(fmpq(d).p) / float(fmpq(d).q)
        self.lamf = (self.df - 2) / 2

    def kk(self, k):
        return arb(k) * (k + 2 * self.lam)

    def b(self, k):
        lam = self.lam
        if k == 0:
            return 1 / (2 * (lam + 1)).sqrt()
        return ((arb(k + 1) * (k + 2 * lam)) / (4 * (k + lam) * (k + lam + 1))).sqrt()

    def kk_f(self, k):
        return k * (k + 2 * self.lamf)

    def b_f(self, k):
        lam = self.lamf
        k = np.asarray(k, dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            v = np.sqrt((k + 1) * (k + 2 * lam) / (4 * (k + lam) * (k + lam + 1)))
        return np.where(k == 0, 1 / np.sqrt(2 * (lam + 1)), v)

    def h0(self):
        lam = self.lam
        return arb.pi().sqrt() * (lam + arb(1) / 2).gamma() / (lam + 1).gamma()

    def basis_table(self, mus, Lmax):
        K = len(mus)
        p0 = 1 / self.h0().sqrt()
        rows = [[p0] * K, [p0 * x / self.b(0) for x in mus]]
        for l in range(1, Lmax - 1):
            bl, blm = self.b(l), self.b(l - 1)
            pl, plm = rows[-1], rows[-2]
            rows.append([((mus[k] * pl[k] - blm * plm[k]) / bl).mid() for k in range(K)])
        return arb_mat(rows[:Lmax])

    def upstream(self, n, y):
        a = 2 * n + (self.da - 1) / 2
        return (-a * y).exp() * ((2 * a) * y).laguerre_l(n, self.alpha)


def seeds_double(geo, beta, N, nleg):
    l = np.arange(nleg, dtype=float)
    b = geo.b_f(l[:-1])
    d = np.full(nleg - 1, beta); d[0] = beta - b[0] ** 2 / beta
    kk = np.array([geo.kk_f(k) for k in l[1:]])
    s = 1 / np.sqrt(kk)
    sig = eigh_tridiagonal(d * s * s, b[1:nleg - 1] * s[:-1] * s[1:], eigvals_only=True)
    neg = np.sort(sig[sig < 0])
    return (-1.0 / neg)[:N]


class Downstream:
    def __init__(self, geo, beta, Lmax):
        self.beta, self.Lmax = beta, Lmax
        self.b = [geo.b(l) for l in range(Lmax + 2)]
        self.kk = [geo.kk(l) for l in range(Lmax + 2)]

    def ratios(self, lam):
        b, kk, beta = self.b, self.kk, self.beta
        r = [arb(0)] * self.Lmax
        rk = arb(0)
        for k in range(self.Lmax - 1, 0, -1):
            rk = -lam * b[k - 1] / (kk[k] + lam * beta + lam * b[k] * rk)
            r[k - 1] = rk
        return r

    def h(self, lam):
        return self.beta + self.b[0] * self.ratios(lam)[0]

    def refine(self, lam0):
        tol = arb(2) ** (-(ctx.prec - 20))
        x0 = arb(lam0); x1 = arb(lam0) * (1 + arb(2) ** -30)
        f0, f1 = self.h(x0), self.h(x1)
        for it in range(80):
            x2 = (x1 - f1 * (x1 - x0) / (f1 - f0)).mid()
            if abs((x2 - x1) / x2) < tol:
                return arb(x2)
            x0, f0 = x1, f1
            x1, f1 = x2, self.h(x2)
        raise RuntimeError("eigenvalue refinement failed near %r" % lam0)

    def coeffs(self, lam):
        r = self.ratios(lam)
        c = [arb(1)]
        for k in range(self.Lmax - 1):
            c.append((c[-1] * r[k]).mid())
        return c


def gauss_legendre(n):
    xs, ws = [], []
    for k in range(n):
        x, w = arb.legendre_p_root(n, k, weight=True)
        xs.append(x); ws.append(w)
    return xs, ws


def build_nodes(geo, beta, KA, KB, Ymax_pow):
    kappa = (1 + beta) / (1 - beta)
    mus, wts = [], []
    xg, wg = gauss_legendre(KA)
    a, b = (-beta).acos(), arb.pi()
    for x, w in zip(xg, wg):
        ph = (a + b) / 2 + (b - a) / 2 * x
        mus.append(ph.cos()); wts.append((b - a) / 2 * w * ph.sin() ** (geo.da - 2))
    xg, wg = gauss_legendre(KB)
    # geometric panels from y=1: first width h ~ 1/d (upstream modes decay like exp(-(d-1)y/2))
    h = min(arb(1), arb(4) / (geo.da - 1))
    edges = [arb(1), 1 + h]
    while edges[-1] < 2 ** Ymax_pow:
        edges.append(1 + 2 * (edges[-1] - 1))
    for lo, hi in zip(edges[:-1], edges[1:]):
        for x, w in zip(xg, wg):
            y = (lo + hi) / 2 + (hi - lo) / 2 * x
            mu = (y - kappa) / (y + kappa)
            jac = 2 * kappa / (y + kappa) ** 2 * (4 * kappa * y / (y + kappa) ** 2) ** ((geo.da - 3) / 2)
            mus.append(mu); wts.append((hi - lo) / 2 * w * jac)
        lo = hi
    ys = [kappa * (1 + mu) / (1 - mu) for mu in mus]
    return mus, wts, ys, kappa


class Problem:
    def __init__(self, d, beta, N, KA=None, KB=None, Lmax=None, Ymax_pow=9, verbose=False):
        t0 = time.time()
        geo = self.geo = GeoD(d)
        self.beta = arb(beta)
        bf = float(beta.p) / float(beta.q)
        self.N = N
        KA = KA or (8 * N + 120); KB = KB or 80; Lmax = Lmax or (6 * N + 120)
        # large d: modes need more room (exponents grow with d)
        Ymax_pow = max(Ymax_pow, 9)
        mus, wts, ys, kappa = build_nodes(geo, self.beta, KA, KB, Ymax_pow)
        self.mus, self.wts, self.ys, self.kappa, self.KA = mus, wts, ys, kappa, KA
        K = len(mus)
        seeds = seeds_double(geo, bf, N, max(Lmax, 6 * N + 200))
        ds = Downstream(geo, self.beta, Lmax)
        self.lams = [ds.refine(l0) for l0 in seeds]
        C = arb_mat([ds.coeffs(lam) for lam in self.lams])
        Qd = C * geo.basis_table(mus, Lmax)
        self.Qd = []
        for j in range(N):
            row = [Qd[j, k].mid() for k in range(K)]
            sc = max(abs(float(v)) for v in row[:KA]) or 1.0
            self.Qd.append([v / sc for v in row])
        self.Qu = [[geo.upstream(n, y) for y in ys] for n in range(N)]
        self.QuT = arb_mat([[self.Qu[n][k] for n in range(N)] for k in range(K)])
        self.base = [w * (self.beta + mu) for w, mu in zip(wts, mus)]
        self.logy = [(y + kappa).log() for y in ys]
        self.K = K
        if verbose:
            print("  built d=%s N=%d K=%d in %.1fs" % (d, N, K, time.time() - t0), flush=True)

    def S_and_dS(self, s):
        s = arb(s)
        K, N = self.K, self.N
        w0 = [self.base[k] * (s * self.logy[k]).exp() for k in range(K)]
        w1 = [w0[k] * self.logy[k] for k in range(K)]
        A0 = arb_mat([[self.Qd[j][k] * w0[k] for k in range(K)] for j in range(N)])
        A1 = arb_mat([[self.Qd[j][k] * w1[k] for k in range(K)] for j in range(N)])
        return A0 * self.QuT, A1 * self.QuT

    def solve_s(self, s0, maxit=60):
        s = arb(s0).mid()
        tol = arb(10) ** (-int(ctx.prec * 0.30103 * 0.5))
        prev = None
        for it in range(maxit):
            S, dS = self.S_and_dS(s)
            X = S.mid().solve(dS.mid(), algorithm="approx")
            tr = sum((X[i, i] for i in range(self.N)), arb(0))
            step = (1 / tr).mid()
            if not step.is_finite():
                raise RuntimeError("non-finite Newton step")
            # damp large steps
            if abs(step) > 0.5:
                step = step / abs(step) * arb("0.5")
            s = (s - step).mid()
            if prev is not None and abs(step) > abs(prev) and abs(prev) < arb(10) ** -20:
                return s, abs(prev)
            if abs(step) < tol:
                return s, abs(step)
            prev = step
        raise RuntimeError("Newton did not converge")


def parse_q(x):
    if "/" in x:
        p, q = x.split("/"); return fmpq(int(p), int(q))
    return fmpq(int(x))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--d", default="3")
    ap.add_argument("--beta", default=None, help="default 1/d")
    ap.add_argument("--N", type=int, nargs="+", default=[8, 16])
    ap.add_argument("--s0", default=None)
    a = ap.parse_args()
    d = parse_q(a.d)
    beta = parse_q(a.beta) if a.beta else 1 / d
    s0 = a.s0
    for N in a.N:
        ctx.prec = int(200 + 2.6 * N)
        pr = Problem(d, beta, N)
        df = float(d.p) / float(d.q); bf = float(beta.p) / float(beta.q)
        guess = s0 or str(df + (df * 0.96) * bf / (1 - bf))
        s, st = pr.solve_s(guess)
        s0 = s
        print("d=%s beta=%s N=%3d s=%s" % (a.d, beta, N, s.str(40, radius=False)), flush=True)
