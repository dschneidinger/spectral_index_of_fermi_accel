"""Dimension-generic (3D / 2D) high-precision eigenfunction matching for the Gamma_u -> infinity spectral index.

3D: angular operator d/dmu (1-mu^2) d/dmu, basis orthonormal Legendre, kk = k(k+1), b_k = (k+1)/sqrt((2k+1)(2k+3)),
    measure dmu, upstream modes  exp(-(2n+1) y) L_n^{(0)}((4n+2) y).
2D: angular operator d^2/dphi^2 (mu = cos phi, even functions), basis e_0 = 1/sqrt(pi), e_k = sqrt(2/pi) cos(k phi),
    kk = k^2, b_0 = 1/sqrt(2), b_k = 1/2, measure dphi, upstream modes exp(-(4n+1) y/2) L_n^{(-1/2)}((4n+1) y)
    (even Hermite functions of t = sqrt(y), the Gamma->inf limit of the Mathieu-type upstream modes near phi = pi).
Common: mu_+ = (y - kappa)/(y + kappa), kappa = (1+beta)/(1-beta); F(mu_+) = (y + kappa)^s g(y);
    S_jn(s) = int (beta + mu_+) (y+kappa)^s g_n(y) Q_j^+(mu_+) d(measure),  det S(s) = 0.
"""
import sys, time, json, argparse
import numpy as np
from scipy.linalg import eigh_tridiagonal
from flint import arb, arb_mat, fmpq, ctx


class Geometry:
    def __init__(self, dim):
        self.dim = dim

    def kk(self, k):
        return arb(k * (k + 1)) if self.dim == 3 else arb(k * k)

    def b(self, k):
        if self.dim == 3:
            return arb(k + 1) / (arb(2 * k + 1) * arb(2 * k + 3)).sqrt()
        return 1 / arb(2).sqrt() if k == 0 else arb(1) / 2

    def kk_f(self, k):
        return k * (k + 1.0) if self.dim == 3 else k * k * 1.0

    def b_f(self, k):
        k = np.asarray(k, dtype=float)
        if self.dim == 3:
            return (k + 1) / np.sqrt((2 * k + 1) * (2 * k + 3))
        return np.where(k == 0, 1 / np.sqrt(2), 0.5)

    def basis_table(self, mus, Lmax):
        """rows l: e_l(mu_k)."""
        K = len(mus)
        if self.dim == 3:
            rows = [[arb(1) / arb(2).sqrt()] * K, [arb(3).sqrt() / arb(2).sqrt() * x for x in mus]]
            for l in range(1, Lmax - 1):
                bl, blm = self.b(l), self.b(l - 1)
                pl, plm = rows[-1], rows[-2]
                rows.append([((mus[k] * pl[k] - blm * plm[k]) / bl).mid() for k in range(K)])
            return arb_mat(rows[:Lmax])
        # 2D: Chebyshev T_l(mu) scaled
        T = [[arb(1)] * K, list(mus)]
        for l in range(1, Lmax - 1):
            T.append([(2 * mus[k] * T[-1][k] - T[-2][k]).mid() for k in range(K)])
        c0 = 1 / arb.pi().sqrt(); c1 = (2 / arb.pi()).sqrt()
        rows = [[c0 * v for v in T[0]]] + [[c1 * v for v in T[l]] for l in range(1, Lmax)]
        return arb_mat(rows)

    def upstream(self, n, y):
        if self.dim == 3:
            return (-(2 * n + 1) * y).exp() * ((4 * n + 2) * y).laguerre_l(n)
        return (-(4 * n + 1) * y / 2).exp() * ((4 * n + 1) * y).laguerre_l(n, arb(-1) / 2)


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
        self.geo, self.beta, self.Lmax = geo, beta, Lmax
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
    """returns mus, measure weights, ys, kappa. Panel A covers mu in [-1,-beta] (y in [0,1]);
    panels B cover y in [1, 2^Ymax_pow]."""
    kappa = (1 + beta) / (1 - beta)
    mus, wts = [], []
    xg, wg = gauss_legendre(KA)
    if geo.dim == 3:
        a, b = arb(-1), -beta
        for x, w in zip(xg, wg):
            mus.append((a + b) / 2 + (b - a) / 2 * x); wts.append((b - a) / 2 * w)
    else:
        a, b = (-beta).acos(), arb.pi()          # phi in [acos(-beta), pi]
        for x, w in zip(xg, wg):
            ph = (a + b) / 2 + (b - a) / 2 * x
            mus.append(ph.cos()); wts.append((b - a) / 2 * w)
    xg, wg = gauss_legendre(KB)
    lo = arb(1)
    for m in range(Ymax_pow):
        hi = lo * 2
        for x, w in zip(xg, wg):
            y = (lo + hi) / 2 + (hi - lo) / 2 * x
            mu = (y - kappa) / (y + kappa)
            if geo.dim == 3:
                jac = 2 * kappa / (y + kappa) ** 2
            else:
                jac = (kappa / y).sqrt() / (y + kappa)
            mus.append(mu); wts.append((hi - lo) / 2 * w * jac)
        lo = hi
    ys = [kappa * (1 + mu) / (1 - mu) for mu in mus]
    return mus, wts, ys, kappa


class Problem:
    def __init__(self, dim, beta, N, KA=None, KB=None, Lmax=None, Ymax_pow=9, verbose=True):
        t0 = time.time()
        geo = self.geo = Geometry(dim)
        self.beta = arb(beta)
        bf = float(beta.p) / float(beta.q)
        self.N = N
        KA = KA or (8 * N + 120)
        KB = KB or 80
        Lmax = Lmax or (6 * N + 120)
        mus, wts, ys, kappa = build_nodes(geo, self.beta, KA, KB, Ymax_pow)
        self.mus, self.wts, self.ys, self.kappa, self.KA = mus, wts, ys, kappa, KA
        K = len(mus)
        seeds = seeds_double(geo, bf, N, max(Lmax, 6 * N + 200))
        ds = Downstream(geo, self.beta, Lmax)
        self.lams = [ds.refine(l0) for l0 in seeds]
        self.ds = ds
        C = arb_mat([ds.coeffs(lam) for lam in self.lams])
        P = geo.basis_table(mus, Lmax)
        Qd = C * P
        self.Qd = []
        for j in range(N):
            row = [Qd[j, k].mid() for k in range(K)]
            sc = max(abs(float(v)) for v in row[:KA]) or 1.0
            self.Qd.append([v / sc for v in row])
        Qu = [[geo.upstream(n, y) for y in ys] for n in range(N)]
        self.Qu = Qu
        self.QuT = arb_mat([[Qu[n][k] for n in range(N)] for k in range(K)])
        self.base = [w * (self.beta + mu) for w, mu in zip(wts, mus)]
        self.logy = [(y + kappa).log() for y in ys]
        self.K = K
        if verbose:
            print("  built dim=%d N=%d K=%d Lmax=%d in %.1fs; Lam_1=%s" % (
                dim, N, K, Lmax, time.time() - t0, self.lams[0].str(12)), flush=True)

    def S_and_dS(self, s):
        s = arb(s)
        K, N = self.K, self.N
        w0 = [self.base[k] * (s * self.logy[k]).exp() for k in range(K)]
        w1 = [w0[k] * self.logy[k] for k in range(K)]
        A0 = arb_mat([[self.Qd[j][k] * w0[k] for k in range(K)] for j in range(N)])
        A1 = arb_mat([[self.Qd[j][k] * w1[k] for k in range(K)] for j in range(N)])
        return A0 * self.QuT, A1 * self.QuT

    def solve_s(self, s0, maxit=40, verbose=False):
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
            s = (s - step).mid()
            if verbose:
                print("    it %d s=%s step=%s" % (it, s.str(30), arb(step).str(5)), flush=True)
            if prev is not None and abs(step) > abs(prev) and abs(prev) < arb(10) ** -20:
                return s, abs(prev)
            if abs(step) < tol:
                return s, abs(step)
            prev = step
        raise RuntimeError("Newton did not converge")

    def null_vector(self, s):
        """coefficients a_n of the upstream expansion at the root (smallest singular direction, approx)."""
        S, _ = self.S_and_dS(s)
        M = S.mid()
        # inverse iteration on M
        N = self.N
        v = arb_mat([[arb(1)] for _ in range(N)])
        for _ in range(4):
            v = M.solve(v, algorithm="approx")
            nrm = max(abs(v[i, 0]) for i in range(N))
            v = arb_mat([[v[i, 0] / nrm] for i in range(N)])
        return [v[i, 0] for i in range(N)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dim", type=int, default=3)
    ap.add_argument("--beta", default=None)
    ap.add_argument("--N", type=int, nargs="+", default=[4, 8])
    ap.add_argument("--prec", type=int, default=200)
    ap.add_argument("--precN", type=float, default=2.6)
    ap.add_argument("--KA", type=int, default=None)
    ap.add_argument("--KB", type=int, default=None)
    ap.add_argument("--Lmax", type=int, default=None)
    ap.add_argument("--ymaxpow", type=int, default=9)
    ap.add_argument("--s0", default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    betastr = a.beta or ("1/3" if a.dim == 3 else "1/2")
    p, q = betastr.split("/") if "/" in betastr else (betastr, "1")
    beta = fmpq(int(p), int(q))
    s0 = a.s0 or ("4.2269788167" if a.dim == 3 else "3.6")
    res = []
    for N in a.N:
        t = time.time()
        ctx.prec = int(a.prec + a.precN * N)
        pr = Problem(a.dim, beta, N, KA=a.KA, KB=a.KB, Lmax=a.Lmax, Ymax_pow=a.ymaxpow)
        s, step = pr.solve_s(s0)
        print("N=%4d s=%s  |last step|=%s  (%.1fs)" % (N, s.str(50, radius=False), step.str(3, radius=False), time.time() - t), flush=True)
        res.append({"dim": a.dim, "N": N, "s": s.str(60, radius=False), "prec": ctx.prec, "beta": betastr,
                    "KA": a.KA, "KB": a.KB, "Lmax": a.Lmax, "step": step.str(3, radius=False)})
        s0 = s
        if a.out:
            json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
