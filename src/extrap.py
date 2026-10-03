"""Richardson-type extrapolation of s_N -> s_inf with ansatz s_N = s + sum_{k=0}^{m-1} a_k (N+c)^-(p0+k)."""
import json, sys, glob, mpmath as mp
mp.mp.dps = 80

def load(pattern="results/ur3d_beta1_3_*.json"):
    D = {}
    for f in glob.glob(pattern):
        for r in json.load(open(f)):
            D[r['N']] = mp.mpf(r['s'])
    return dict(sorted(D.items()))

def fit(D, Ns, m, p0=4, c=0, powers=None):
    """exact solve with len(Ns) = m+1 points."""
    powers = powers or [p0 + k for k in range(m)]
    A = mp.matrix(len(Ns), m + 1); b = mp.matrix(len(Ns), 1)
    for i, N in enumerate(Ns):
        A[i, 0] = 1
        for k, p in enumerate(powers):
            A[i, k + 1] = mp.mpf(N + c) ** (-p)
        b[i] = D[N]
    x = mp.lu_solve(A, b)
    return x[0]

if __name__ == "__main__":
    D = load(sys.argv[1] if len(sys.argv) > 1 else "results/ur3d_beta1_3_*.json")
    Ns = [N for N in D if N >= 8]
    print("available N:", Ns)
    for m in range(1, 12):
        if len(Ns) < m + 1: break
        sel = Ns[-(m + 1):]
        print("m=%2d using N=%s  s=%s" % (m, sel, mp.nstr(fit(D, sel, m), 30)))
