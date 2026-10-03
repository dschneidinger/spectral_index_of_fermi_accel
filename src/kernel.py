"""Linear-response kernel of the Gamma->inf index to the downstream angular scattering profile (3D, beta_d = 1/3):
    delta s = int_{-1}^{1} K(mu) delta ln D(mu) dmu   (about isotropic D = 1).
Computed by central differences with Gaussian bumps phi_i(mu) = exp(-(mu-m_i)^2/(2 sig^2)):
    g_i = ds/deps for D = 1 + eps*phi_i,   K_sig(m_i) = g_i / int phi_i  (kernel smoothed on scale sig).
Exact constraint: int K = 0 (uniform rescaling of D leaves s unchanged)."""
import sys, json, argparse, time
sys.path.insert(0, __file__.rsplit("/", 1)[0])
from flint import arb, fmpq, ctx
from aniso import ProblemAniso

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--centers", type=float, nargs="+", required=True)
    ap.add_argument("--sig", type=float, default=0.03)
    ap.add_argument("--N", type=int, default=16)
    ap.add_argument("--L", type=int, default=260)
    ap.add_argument("--eps", default="1e-12")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    ctx.prec = 200
    eps = arb(a.eps); sig = arb(a.sig)
    try:
        res = json.load(open(a.out))
    except Exception:
        res = {}
    for m in a.centers:
        key = "%.6f" % m
        if key in res:
            continue
        t = time.time()
        mc = arb(m)
        vals = []
        for sgn in (1, -1):
            D = (lambda e: (lambda mu: 1 + e * (-(mu - mc) ** 2 / (2 * sig ** 2)).exp()))(sgn * eps)
            pr = ProblemAniso(fmpq(1, 3), a.N, D, L=a.L)
            s, _ = pr.solve_s("4.22697881")
            vals.append(s)
        g = (vals[0] - vals[1]) / (2 * eps)
        # int phi over [-1,1]
        from mpmath import mp, erf, sqrt, mpf
        mp.dps = 30
        sg = mpf(a.sig); I = sg * sqrt(mp.pi / 2) * (erf((1 - mpf(m)) / (sqrt(2) * sg)) + erf((1 + mpf(m)) / (sqrt(2) * sg)))
        res[key] = {"mu": m, "g": g.str(20, radius=False), "int_phi": str(I), "K": str(mpf(g.str(25, radius=False)) / I)}
        json.dump(res, open(a.out, "w"), indent=1)
        print("mu=%+.4f  ds/deps=%s  K=%s  (%.0fs)" % (m, g.str(12, radius=False), res[key]["K"][:14], time.time() - t), flush=True)
