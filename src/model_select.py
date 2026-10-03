"""Model selection for the truncation-error expansion s_N = s + sum_k a_k phi_k(N).
For each candidate basis {phi_k}, fit (exactly) on m+1 consecutive points and predict the NEXT larger-N point;
the out-of-sample prediction error ranks the models. Then report s_inf from the best models."""
import sys, glob, json
import mpmath as mp
sys.path.insert(0, __file__.rsplit("/", 1)[0])
from extrap import load
mp.mp.dps = 80
T = mp.mpf(1) / 3

MODELS = {
    "int(4,5,6..)":       lambda m: [("p", 4 + k) for k in range(m)],
    "third(4,13/3,..)":   lambda m: [("p", 4 + k * T) for k in range(m)],
    "twothirds(4,14/3..)": lambda m: [("p", 4 + 2 * k * T) for k in range(m)],
    "int+log6":           lambda m: ([("p", 4), ("p", 5), ("l", 6)] + [("p", 6 + k) for k in range(m)])[:m],
    "int+log5":           lambda m: ([("p", 4), ("l", 5), ("p", 5)] + [("p", 6 + k) for k in range(m)])[:m],
    "int+log4":           lambda m: ([("p", 4), ("l", 4), ("p", 5)] + [("p", 6 + k) for k in range(m)])[:m],
}


def phi(kind, p, N):
    N = mp.mpf(N)
    return N ** (-p) if kind == "p" else N ** (-p) * mp.log(N)


def fit(D, sel, basis):
    A = mp.matrix(len(sel), len(basis) + 1); b = mp.matrix(len(sel), 1)
    for i, N in enumerate(sel):
        A[i, 0] = 1
        for j, (k, p) in enumerate(basis):
            A[i, j + 1] = phi(k, p, N)
        b[i] = D[N]
    return mp.lu_solve(A, b)


def predict(x, basis, N):
    return x[0] + sum(x[j + 1] * phi(k, p, N) for j, (k, p) in enumerate(basis))


if __name__ == "__main__":
    pat = sys.argv[1] if len(sys.argv) > 1 else "results/ur3d_beta1_3_*.json"
    D = load(pat)
    Ns = [N for N in D if N >= 24]
    print("N:", Ns)
    for name, mk in MODELS.items():
        print("== model", name)
        for m in (4, 5, 6, 7, 8):
            basis = mk(m)
            errs = []
            ests = []
            for end in range(m + 2, len(Ns)):
                sel = Ns[end - m - 1:end]
                x = fit(D, sel, basis)
                errs.append(abs(predict(x, basis, Ns[end]) - D[Ns[end]]))
                ests.append(x[0])
            # estimate from the last window (including largest N)
            xl = fit(D, Ns[-m - 1:], basis)
            print("  m=%d  last-3 prediction errors %s   s_inf(last window)=%s" % (
                m, [mp.nstr(e, 2) for e in errs[-3:]], mp.nstr(xl[0], 27)))
