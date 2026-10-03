"""High-precision eigenfunction matching for the Gamma_u -> infinity spectral index (3D, isotropic
diffusion in direction angle), arbitrary downstream speed beta = beta_d.

Problem (derived in notes/FORMULATION.md):
  downstream modes   ((1-mu^2) Q')' = Lam (beta+mu) Q  on [-1,1], regular at +-1; we need Lam>0 (forbidden
                     downstream) as TEST functions;
  upstream modes     Q_n(y) = exp(-(2n+1) y) L_n((4n+2) y), n = 0..N-1 (allowed upstream, Gamma->inf limit) as
                     TRIAL functions;
  map                mu = (y-kappa)/(y+kappa),  kappa = (1+beta)/(1-beta);   F(mu) = (y+kappa)^s g(y);
  condition          S_jn(s) = int_{-1}^{1} (beta+mu) (y+kappa)^s Q_n(y(mu)) Q_j^+(mu) dmu,  det S(s) = 0.

All arithmetic in Arb balls (python-flint); the eigenvalues are seeded in double precision and
refined by secant iteration on the continued-fraction regularity condition.
"""
import sys, time, json, argparse
import numpy as np
from scipy.linalg import eigh_tridiagonal
from flint import arb, arb_mat, fmpq, ctx


# ---------------------------------------------------------------- downstream eigenproblem
def b_coef(l):
    # mu * p_l = b_{l-1} p_{l-1} + b_l p_{l+1}, p orthonormal Legendre
    return arb(l + 1) / ((arb(2 * l + 1) * arb(2 * l + 3)).sqrt())


def seeds_double(beta, N, nleg):
    """Lam_j>0, j=1..N, from the reduced symmetric tridiagonal problem (double precision)."""
    l = np.arange(nleg, dtype=float)
    b = (l[:-1] + 1) / np.sqrt((2 * l[:-1] + 1) * (2 * l[:-1] + 3))
    # A c = sigma K c, eliminate c_0 = -b0 c1/beta
    d = np.full(nleg - 1, beta)
    d[0] = beta - b[0] ** 2 / beta
    e = b[1:].copy()
    kk = l[1:] * (l[1:] + 1)
    s = 1 / np.sqrt(kk)
    dT = d * s * s
    eT = e[:-1] * s[:-1] * s[1:] if len(e) == nleg - 1 else None
    eT = b[1:nleg - 1] * s[:-1] * s[1:]
    sig = eigh_tridiagonal(dT, eT, eigvals_only=True)
    neg = np.sort(sig[sig < 0])          # most negative first -> smallest Lam
    lam = -1.0 / neg
    return lam[:N]


class Downstream:
    def __init__(self, beta, Lmax):
        self.beta = beta
        self.Lmax = Lmax
        self.b = [b_coef(l) for l in range(Lmax + 2)]
        self.kk = [arb(l * (l + 1)) for l in range(Lmax + 2)]

    def ratios(self, lam):
        """minimal-solution ratios r_k = c_{k+1}/c_k, k = 0..Lmax-1 (backward recurrence)."""
        b, kk, beta = self.b, self.kk, self.beta
        r = [arb(0)] * self.Lmax
        rk = arb(0)
        for k in range(self.Lmax - 1, 0, -1):
            # row k: kk_k c_k + lam(beta c_k + b_{k-1} c_{k-1} + b_k c_{k+1}) = 0
            rk = -lam * b[k - 1] / (kk[k] + lam * beta + lam * b[k] * rk)
            r[k - 1] = rk
        return r

    def h(self, lam):
        r = self.ratios(lam)
        return self.beta + self.b[0] * r[0]

    def refine(self, lam0, tol_bits=None):
        tol = arb(2) ** (-(tol_bits or ctx.prec - 20))
        x0 = arb(lam0)
        x1 = arb(lam0) * (1 + arb(2) ** -30)
        f0, f1 = self.h(x0), self.h(x1)
        for it in range(60):
            x2 = (x1 - f1 * (x1 - x0) / (f1 - f0)).mid()
            if abs((x2 - x1) / x2).upper() < tol:
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


def legendre_table(xs, Lmax):
    """orthonormal Legendre p_l(x) for l<Lmax at nodes xs -> arb_mat (Lmax x K)."""
    K = len(xs)
    rows = []
    p0 = [arb(1) / arb(2).sqrt()] * K
    p1 = [arb(3).sqrt() / arb(2).sqrt() * x for x in xs]
    rows.append(p0); rows.append(p1)
    for l in range(1, Lmax - 1):
        # x p_l = b_{l-1} p_{l-1} + b_l p_{l+1}
        bl = b_coef(l); blm = b_coef(l - 1)
        pl, plm = rows[-1], rows[-2]
        rows.append([((xs[k] * pl[k] - blm * plm[k]) / bl).mid() for k in range(K)])
    return arb_mat(rows[:Lmax])


def gauss_legendre(n):
    xs, ws = [], []
    for k in range(n):
        x, w = arb.legendre_p_root(n, k, weight=True)
        xs.append(x); ws.append(w)
    return xs, ws


# ---------------------------------------------------------------- quadrature and matrices
def build_nodes(beta, KA, KB, Ymax_pow=9):
    """nodes in mu with weights (dmu measure). Panel A: mu in [-1,-beta]; panels B: y in [1,2^m]."""
    beta = arb(beta)
    kappa = (1 + beta) / (1 - beta)
    mus, wts = [], []
    xg, wg = gauss_legendre(KA)
    a, b = arb(-1), -beta
    for x, w in zip(xg, wg):
        mus.append((a + b) / 2 + (b - a) / 2 * x)
        wts.append((b - a) / 2 * w)
    xg, wg = gauss_legendre(KB)
    lo = arb(1)
    for m in range(Ymax_pow):
        hi = lo * 2
        for x, w in zip(xg, wg):
            y = (lo + hi) / 2 + (hi - lo) / 2 * x
            mu = (y - kappa) / (y + kappa)
            jac = 2 * kappa / (y + kappa) ** 2
            mus.append(mu); wts.append((hi - lo) / 2 * w * jac)
        lo = hi
    ys = [kappa * (1 + mu) / (1 - mu) for mu in mus]
    return mus, wts, ys, kappa


class Problem:
    def __init__(self, beta, N, KA=None, KB=None, Lmax=None, Ymax_pow=9, verbose=True):
        t0 = time.time()
        self.beta = arb(beta)
        bf = float(fmpq(beta).p) / float(fmpq(beta).q) if isinstance(beta, fmpq) else float(beta)
        self.N = N
        KA = KA or (8 * N + 120)
        KB = KB or 80
        Lmax = Lmax or (6 * N + 120)
        mus, wts, ys, kappa = build_nodes(self.beta, KA, KB, Ymax_pow)
        self.mus, self.wts, self.ys, self.kappa = mus, wts, ys, kappa
        K = len(mus)
        # downstream test functions
        seeds = seeds_double(bf, N, max(Lmax, 6 * N + 200))
        ds = Downstream(self.beta, Lmax)
        self.lams = [ds.refine(l0) for l0 in seeds]
        C = arb_mat([ds.coeffs(lam) for lam in self.lams])        # N x Lmax
        P = legendre_table(mus, Lmax)                               # Lmax x K
        Qd = C * P                                                  # N x K
        # normalise each test function by its value at mu=-1 side (sup over panel A nodes)
        Qd_rows = []
        for j in range(N):
            row = [Qd[j, k].mid() for k in range(K)]
            sc = max(abs(float(v)) for v in row[:KA]) or 1.0
            Qd_rows.append([v / sc for v in row])
        self.Qd = Qd_rows
        # upstream trial functions
        Qu = []
        for n in range(N):
            Qu.append([((-(2 * n + 1)) * y).exp() * ((4 * n + 2) * y).laguerre_l(n) for y in ys])
        self.QuT = arb_mat([[Qu[n][k] for n in range(N)] for k in range(K)])   # K x N
        self.base = [w * (self.beta + mu) for w, mu in zip(wts, mus)]
        self.logy = [(y + kappa).log() for y in ys]
        self.K = K
        if verbose:
            print("  built N=%d K=%d Lmax=%d in %.1fs; Lam_1=%s Lam_N=%s" % (
                N, K, Lmax, time.time() - t0, self.lams[0].str(12), self.lams[-1].str(12)), flush=True)

    def S_and_dS(self, s):
        s = arb(s)
        K, N = self.K, self.N
        w0 = [self.base[k] * (s * self.logy[k]).exp() for k in range(K)]
        w1 = [w0[k] * self.logy[k] for k in range(K)]
        A0 = arb_mat([[self.Qd[j][k] * w0[k] for k in range(K)] for j in range(N)])
        A1 = arb_mat([[self.Qd[j][k] * w1[k] for k in range(K)] for j in range(N)])
        return A0 * self.QuT, A1 * self.QuT

    def solve_s(self, s0, tol_digits=None, maxit=40, verbose=False):
        """Newton on det S(s) = 0 with step 1/tr(S^-1 S'), computed by an approximate (midpoint) solve.
        Raises if Newton does not converge; accuracy is validated by repeating at higher precision."""
        s = arb(s0).mid()
        tol = arb(10) ** (-(tol_digits or int(ctx.prec * 0.30103 * 0.5)))
        prev = None
        for it in range(maxit):
            S, dS = self.S_and_dS(s)
            X = S.mid().solve(dS.mid(), algorithm="approx")
            tr = sum((X[i, i] for i in range(self.N)), arb(0))
            step = (1 / tr).mid()
            if not step.is_finite():
                raise RuntimeError("non-finite Newton step at it=%d" % it)
            s = (s - step).mid()
            if verbose:
                print("    it %d s=%s step=%s" % (it, s.str(30), arb(step).str(5)), flush=True)
            if prev is not None and abs(step) > abs(prev) and abs(prev) < arb(10) ** -20:
                return s, abs(prev)          # precision floor reached
            if abs(step) < tol:
                return s, abs(step)
            prev = step
        raise RuntimeError("Newton did not converge (last step %s)" % abs(step).str(3))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--beta", default="1/3")
    ap.add_argument("--N", type=int, nargs="+", default=[4, 8])
    ap.add_argument("--prec", type=int, default=256)
    ap.add_argument("--precN", type=float, default=0.0, help="extra bits per unit N (trial basis loses ~0.73 digits/N)")
    ap.add_argument("--KA", type=int, default=None)
    ap.add_argument("--KB", type=int, default=None)
    ap.add_argument("--Lmax", type=int, default=None)
    ap.add_argument("--ymaxpow", type=int, default=9)
    ap.add_argument("--s0", default="4.2269788167")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    ctx.prec = a.prec
    p, q = a.beta.split("/") if "/" in a.beta else (a.beta, "1")
    beta = fmpq(int(p), int(q))
    res = []
    s0 = a.s0
    for N in a.N:
        t = time.time()
        ctx.prec = int(a.prec + a.precN * N)
        pr = Problem(beta, N, KA=a.KA, KB=a.KB, Lmax=a.Lmax, Ymax_pow=a.ymaxpow)
        s, step = pr.solve_s(s0)
        print("N=%4d s=%s  |last step|=%s  (%.1fs)" % (N, s.str(50), step.str(3), time.time() - t), flush=True)
        res.append({"N": N, "s": s.str(60, radius=False), "KA": a.KA, "KB": a.KB, "Lmax": a.Lmax,
                    "prec": ctx.prec, "beta": a.beta, "err_est": step.str(3, radius=False)})
        s0 = s
        if a.out:
            json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
