"""Shock-front distribution from the Galerkin null vector; test of the Keshet-Waxman (2005) identity
    s = (2 r_d + beta_u + beta_d)/(beta_u - beta_d),   r_d = gamma_d^-2 F'(-beta)/F(-beta),
in the limit beta_u -> 1, where F(mu) is the downstream-frame distribution at the shock.
Also tabulates F(mu) (downstream frame) for figures.
"""
import sys, json, argparse
sys.path.insert(0, __file__.rsplit("/", 1)[0])
from urnd import Problem, Geometry
from flint import arb, fmpq, ctx


def dQu(geo, n, y):
    """d/dy of the upstream mode."""
    if geo.dim == 3:
        om = arb(4 * n + 2); a = arb(2 * n + 1)
        L = (om * y).laguerre_l(n)
        dL = -(om * y).laguerre_l(n - 1, 1) if n > 0 else arb(0)
        return (-a * y).exp() * (-a * L + om * dL)
    om = arb(4 * n + 1); a = om / 2
    L = (om * y).laguerre_l(n, arb(-1) / 2)
    dL = -(om * y).laguerre_l(n - 1, arb(1) / 2) if n > 0 else arb(0)
    return (-a * y).exp() * (-a * L + om * dL)


def F_and_dF(pr, a, s, mu):
    """F(mu) = (y+kappa)^s g(y), and dF/dmu."""
    geo, kappa = pr.geo, pr.kappa
    y = kappa * (1 + mu) / (1 - mu)
    dydmu = 2 * kappa / (1 - mu) ** 2
    g = sum((a[n] * geo.upstream(n, y) for n in range(pr.N)), arb(0))
    dg = sum((a[n] * dQu(geo, n, y) for n in range(pr.N)), arb(0))
    pref = (y + kappa) ** s
    F = pref * g
    dF = (s * (y + kappa) ** (s - 1) * g + pref * dg) * dydmu
    return F, dF


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dim", type=int, default=3)
    ap.add_argument("--beta", default=None)
    ap.add_argument("--N", type=int, nargs="+", default=[16, 32])
    ap.add_argument("--prec", type=int, default=200)
    ap.add_argument("--table", default=None)
    a = ap.parse_args()
    bs = a.beta or ("1/3" if a.dim == 3 else "1/2")
    p, q = bs.split("/")
    beta = fmpq(int(p), int(q))
    s0 = "4.2269788167" if a.dim == 3 else "3.2985080564"
    for N in a.N:
        ctx.prec = int(a.prec + 2.6 * N)
        pr = Problem(a.dim, beta, N, verbose=False)
        s, _ = pr.solve_s(s0)
        av = pr.null_vector(s)
        b = arb(beta)
        F0, dF0 = F_and_dF(pr, av, s, -b)
        rd = (1 - b * b) * dF0 / F0
        s_kw = (2 * rd + 1 + b) / (1 - b)
        print("dim=%d N=%3d  s_N=%s  a1/a0=F'/F=%s  r_d=%s  KW-identity s=%s  diff=%s" % (
            a.dim, N, s.str(18), (dF0 / F0).str(15), rd.str(15), s_kw.str(18), (s_kw - s).str(4)), flush=True)
        if a.table and N == a.N[-1]:
            rows = []
            for k in range(-200, 201):
                mu = arb(k) / 200 * (1 - arb(10) ** -6)
                F, dF = F_and_dF(pr, av, s, mu)
                rows.append([float(mu), float(F / F0)])
            json.dump({"dim": a.dim, "beta": bs, "N": N, "s": s.str(30, radius=False), "F_over_F0": rows},
                      open(a.table, "w"))
