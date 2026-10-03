"""Gamma_u -> infinity spectral index (3D) for an ARBITRARY downstream angular diffusion function D(mu):
    downstream  d/dmu[(1-mu^2) D(mu) dQ/dmu] = Lam (beta + mu) Q.
Upstream scattering drops out of the limit entirely: with mu = -1 + eps*y, D_up(mu) = D_up(-1) + O(eps), and
D_up(-1) only rescales z, so the Laguerre trial functions of urd.py are exact for ANY upstream D_up (continuous and
positive at mu = -1).

Downstream eigenpairs: Galerkin in orthonormal Legendre p_0..p_{L-1}: stiffness Kst_kl = int (1-mu^2) D p_k' p_l',
mass A = beta*I + J (exact, tridiagonal). Row/column 0 of Kst vanish, so c_0 = -(b_0/beta) c_1 and the reduced pencil
A_s c' = sigma Kst' c' (sigma = -1/Lam) is solved by an approximate high-precision eigensolver (Arb, acb_mat.eig).
"""
import sys, time, argparse, json
sys.path.insert(0, __file__.rsplit("/", 1)[0])
from flint import arb, acb, arb_mat, acb_mat, fmpq, ctx
import urd


def legendre_and_deriv(xs, L):
    """orthonormal p_k(x) and p_k'(x), k<L, as lists of rows."""
    K = len(xs)
    P = [[arb(1)] * K, list(xs)]
    dP = [[arb(0)] * K, [arb(1)] * K]
    for k in range(1, L - 1):
        P.append([((2 * k + 1) * xs[i] * P[k][i] - k * P[k - 1][i]) / (k + 1) for i in range(K)])
        dP.append([dP[k - 1][i] + (2 * k + 1) * P[k][i] for i in range(K)])
    sc = [((2 * arb(k) + 1) / 2).sqrt() for k in range(L)]
    return [[P[k][i] * sc[k] for i in range(K)] for k in range(L)], [[dP[k][i] * sc[k] for i in range(K)] for k in range(L)]


class DownstreamGeneral:
    def __init__(self, beta, Dfun, L, Q=None):
        self.beta, self.L = arb(beta), L
        Q = Q or (2 * L + 200)
        xg, wg = urd.gauss_legendre(Q)
        P, dP = legendre_and_deriv(xg, L)
        wD = [wg[i] * (1 - xg[i] ** 2) * Dfun(xg[i]) for i in range(Q)]
        dPw = arb_mat([[dP[k][i] * wD[i] for i in range(Q)] for k in range(L)])
        dPm = arb_mat(dP)
        self.Kst = dPw * dPm.transpose()
        self.b = [urd.GeoD(fmpq(3)).b(k) for k in range(L + 1)]

    def positive_modes(self, N):
        L, beta, b = self.L, self.beta, self.b
        # reduced mass (Schur complement eliminating c_0): A' = A[1:,1:] - (1/beta) a a^T, a = (b_0, 0, ...)
        n = L - 1
        Ar = arb_mat(n, n)
        for i in range(n):
            Ar[i, i] = beta
            if i + 1 < n:
                Ar[i, i + 1] = b[i + 1]; Ar[i + 1, i] = b[i + 1]
        Ar[0, 0] = beta - b[0] ** 2 / beta
        Kr = arb_mat([[self.Kst[i, j] for j in range(1, L)] for i in range(1, L)])
        M = Kr.solve(Ar, algorithm="approx")             # Kr^{-1} Ar : eigenvalues sigma
        E, R = acb_mat(M).eig(right=True, algorithm="approx")
        idx = sorted(range(n), key=lambda i: E[i].real)  # most negative sigma first <-> smallest Lam > 0
        idx = [i for i in idx if E[i].real < 0][:N]
        modes = []
        for i in idx:
            cp = [R[k, i].real for k in range(n)]
            c0 = -b[0] * cp[0] / beta
            modes.append(([c0] + cp, -1 / E[i].real))
        return modes


class ProblemAniso(urd.Problem):
    def __init__(self, beta, N, Dfun, L=None, KA=None, KB=None, Ymax_pow=9):
        d = fmpq(3)
        geo = self.geo = urd.GeoD(d)
        self.beta = arb(beta); self.N = N
        KA = KA or (8 * N + 120); KB = KB or 80
        L = L or (6 * N + 160)
        mus, wts, ys, kappa = urd.build_nodes(geo, self.beta, KA, KB, Ymax_pow)
        self.mus, self.wts, self.ys, self.kappa, self.KA = mus, wts, ys, kappa, KA
        K = self.K = len(mus)
        ds = DownstreamGeneral(beta, Dfun, L)
        modes = ds.positive_modes(N)
        self.lams = [m[1] for m in modes]
        C = arb_mat([m[0] for m in modes])
        Qd = C * geo.basis_table(mus, L)
        self.Qd = []
        for j in range(N):
            row = [Qd[j, k].mid() for k in range(K)]
            sc = max(abs(float(v)) for v in row[:KA]) or 1.0
            self.Qd.append([v / sc for v in row])
        self.Qu = [[geo.upstream(n, y) for y in ys] for n in range(N)]
        self.QuT = arb_mat([[self.Qu[n][k] for n in range(N)] for k in range(K)])
        self.base = [w * (self.beta + mu) for w, mu in zip(wts, mus)]
        self.logy = [(y + kappa).log() for y in ys]


def D_tangled(a):
    a2 = arb(a) ** 2
    return lambda mu: 1 / (mu * mu + a2).sqrt()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", default="0.1", help="KGGA tangled field: D = (mu^2 + a^2)^-1/2 ; 'iso' for D=1")
    ap.add_argument("--N", type=int, nargs="+", default=[8, 16])
    ap.add_argument("--L", type=int, default=None)
    ap.add_argument("--prec", type=int, default=200)
    a = ap.parse_args()
    Dfun = (lambda mu: arb(1)) if a.a == "iso" else D_tangled(a.a)
    s0 = "4.2"
    for N in a.N:
        ctx.prec = int(a.prec + 2.6 * N)
        t = time.time()
        pr = ProblemAniso(fmpq(1, 3), N, Dfun, L=a.L)
        s, _ = pr.solve_s(s0); s0 = s
        print("a=%s N=%3d s=%s  (%.0fs)" % (a.a, N, s.str(30, radius=False), time.time() - t), flush=True)
