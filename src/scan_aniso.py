"""s_inf for KGGA's tangled-field diffusion D = (mu^2 + a^2)^(-1/2) downstream (any D upstream), 3D, beta_d=1/3,
Gamma_u -> inf; Richardson in N with the N^-4 ansatz."""
import sys, json, argparse, time, math
sys.path.insert(0, __file__.rsplit("/", 1)[0])
from aniso import ProblemAniso, D_tangled
from flint import fmpq, ctx
import mpmath as mp
from scan_beta import extrapolate
mp.mp.dps = 50
ap = argparse.ArgumentParser(); ap.add_argument("--as_", nargs="+", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--Ns", type=int, nargs="+", default=[12, 16, 20, 24, 32, 40])
a = ap.parse_args()
for av in a.as_:
    try: res = json.load(open(a.out))
    except Exception: res = {}
    if av in res: continue
    vals = []; s0 = "4.21"; t = time.time()
    for N in a.Ns:
        ctx.prec = int(200 + 2.6 * N)
        L = 6 * N + 160 + int(5 / float(av))
        pr = ProblemAniso(fmpq(1, 3), N, D_tangled(av), L=L)
        s, _ = pr.solve_s(s0); s0 = s
        vals.append(mp.mpf(s.str(45, radius=False)))
    ex = extrapolate(a.Ns, vals)
    res[av] = {"a": float(av), "Ns": a.Ns, "s_N": [mp.nstr(v, 35) for v in vals], "s_inf": mp.nstr(ex[-1], 25),
               "err": mp.nstr(abs(ex[-1] - ex[-2]), 3)}
    json.dump(res, open(a.out, "w"), indent=1)
    print("a=%-6s s_inf=%s (+- %s) %.0fs" % (av, mp.nstr(ex[-1], 18), mp.nstr(abs(ex[-1] - ex[-2]), 2), time.time() - t), flush=True)
