"""Return probability for particles entering the downstream region, Gamma_u -> inf limit.
Downstream scattering conserves |p| in the downstream frame, so at fixed p_+ the number fluxes through the shock,
    J_in  = int_{mu>-beta} (beta+mu) F dm,   J_out = int_{mu<-beta} |beta+mu| F dm   (dm = dmu in 3D, dphi in 2D),
give P_ret = J_out / J_in exactly; J_in - J_out = C is the flux advected to downstream infinity.
Also reports the effective energy gain per cycle G defined by Bell's relation s = d - ln P_ret / ln G (d = 3 or 2).
"""
import sys, argparse
sys.path.insert(0, __file__.rsplit("/", 1)[0])
from urnd import Problem
from flint import arb, fmpq, ctx

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dim", type=int, default=3)
    ap.add_argument("--N", type=int, nargs="+", default=[16, 32, 64])
    a = ap.parse_args()
    bs = "1/3" if a.dim == 3 else "1/2"
    beta = fmpq(*map(int, bs.split("/")))
    s0 = "4.2269788167" if a.dim == 3 else "3.2985080564"
    for N in a.N:
        ctx.prec = int(200 + 2.6 * N)
        pr = Problem(a.dim, beta, N, verbose=False)
        s, _ = pr.solve_s(s0)
        av = pr.null_vector(s)
        b = arb(beta)
        Jin, Jout = arb(0), arb(0)
        for k in range(pr.K):
            g = sum((av[n] * pr.Qu[n][k] for n in range(N)), arb(0))
            F = (pr.ys[k] + pr.kappa) ** s * g
            fl = (b + pr.mus[k]) * F * pr.wts[k]
            if k < pr.KA:
                Jout -= fl
            else:
                Jin += fl
        P = Jout / Jin
        G = (-(P.log()) / (s - a.dim)).exp()
        print("dim=%d N=%3d  s=%s  P_ret=%s  G_eff=%s" % (a.dim, N, s.str(15), P.str(15), G.str(12)), flush=True)
