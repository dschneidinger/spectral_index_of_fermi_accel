"""s_inf(d) for beta_d = 1/d (ultra-relativistic d-dimensional gas), Gamma_u -> inf, isotropic diffusion on S^{d-1}.
N-range scales as n0 = max(12, ceil(sqrt(d))); Richardson with the N^-4, N^-5, ... ansatz; error = spread of the
two highest fit orders. Output: energy index s_E = s - (d - 1)."""
import sys, json, argparse, time, math
sys.path.insert(0, __file__.rsplit("/", 1)[0])
from urd import Problem, parse_q
from flint import fmpq, ctx
import mpmath as mp
from scan_beta import extrapolate

mp.mp.dps = 50

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds", nargs="+", required=True)
    ap.add_argument("--mult", type=float, nargs="+", default=[1, 1.25, 1.5, 2, 2.5, 3, 4])
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    for ds in a.ds:
        try:
            res = json.load(open(a.out))
        except Exception:
            res = {}
        if ds in res:
            continue
        d = parse_q(ds); df = float(d.p) / float(d.q)
        beta = 1 / d
        n0 = max(12, math.ceil(math.sqrt(df)))
        Ns = sorted(set(int(round(n0 * m)) for m in a.mult))
        s0 = str(df + 1 + 1.0 / df)
        vals = []
        t = time.time()
        for N in Ns:
            ctx.prec = int(200 + 2.6 * N)
            pr = Problem(d, beta, N)
            s, _ = pr.solve_s(s0)
            s0 = s
            vals.append(mp.mpf(s.str(45, radius=False)))
        ex = extrapolate(Ns, vals)
        sE = ex[-1] - (mp.mpf(int(d.p)) / int(d.q) - 1)
        res[ds] = {"d": df, "Ns": Ns, "s_N": [mp.nstr(v, 40) for v in vals], "s_inf": mp.nstr(ex[-1], 30),
                   "s_E": mp.nstr(sE, 30), "err": mp.nstr(abs(ex[-1] - ex[-2]), 3),
                   "err_prev": mp.nstr(abs(ex[-2] - ex[-3]), 3)}
        json.dump(res, open(a.out, "w"), indent=1)
        print("d=%-6s s_E=%s  (+- %s)  Ns=%s  %.0fs" % (ds, mp.nstr(sE, 20), mp.nstr(abs(ex[-1] - ex[-2]), 2), Ns, time.time() - t), flush=True)
