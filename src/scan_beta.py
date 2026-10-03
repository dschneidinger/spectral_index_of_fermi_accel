"""s_inf(beta_d) in the Gamma_u -> infinity limit: Galerkin at N in Ns, Richardson-extrapolated with the
N^-4, N^-5, ... ansatz (validated at beta_d = 1/3). Reports the extrapolated value and the spread between the
two highest fit orders as an error estimate."""
import sys, json, argparse, time
sys.path.insert(0, __file__.rsplit("/", 1)[0])
from urnd import Problem
from flint import arb, fmpq, ctx
import mpmath as mp

mp.mp.dps = 50


def extrapolate(Ns, vals, p0=4):
    out = []
    for m in range(1, len(Ns)):
        sel = list(range(len(Ns) - m - 1, len(Ns)))
        A = mp.matrix(len(sel), m + 1); b = mp.matrix(len(sel), 1)
        for i, k in enumerate(sel):
            A[i, 0] = 1
            for j in range(m):
                A[i, j + 1] = mp.mpf(Ns[k]) ** (-(p0 + j))
            b[i] = vals[k]
        out.append(mp.lu_solve(A, b)[0])
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dim", type=int, default=3)
    ap.add_argument("--betas", nargs="+", required=True, help="rationals p/q")
    ap.add_argument("--Ns", type=int, nargs="+", default=[16, 20, 24, 32, 40, 48, 64])
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    try:
        res = json.load(open(a.out))
    except Exception:
        res = {}
    s0 = None
    for bs in a.betas:
        if bs in res:
            continue
        p, q = bs.split("/")
        beta = fmpq(int(p), int(q))
        bf = int(p) / int(q)
        # initial guess: KW-type formula (3D) or previous
        guess = (3 - 2 * bf**2 + bf**3) / (1 - bf) if a.dim == 3 else (2 + 2.6 * bf)
        vals = []
        t = time.time()
        for N in a.Ns:
            ctx.prec = int(160 + 2.6 * N)
            pr = Problem(a.dim, beta, N, verbose=False)
            s, _ = pr.solve_s(s0 if (s0 is not None and N > a.Ns[0]) else guess)
            s0 = s
            vals.append(mp.mpf(s.str(45, radius=False)))
        ex = extrapolate(a.Ns, vals)
        err = abs(ex[-1] - ex[-2])
        res[bs] = {"beta": bf, "s_inf": mp.nstr(ex[-1], 30), "err": mp.nstr(err, 3),
                   "s_N": {str(N): mp.nstr(v, 40) for N, v in zip(a.Ns, vals)}}
        print("beta=%-8s s_inf=%s  (+- %s)  %.0fs" % (bs, mp.nstr(ex[-1], 22), mp.nstr(err, 2), time.time() - t), flush=True)
        json.dump(res, open(a.out, "w"), indent=1)
        s0 = None
