"""Paper Sec. VIII: large-d expansion of the energy index s_E(d) (weighted least squares, d >= dmin)."""
import _common as C
import mpmath as mp
mp.mp.dps = 40
R = C.load_glob("results/scan_d_*.json")
pts = sorted((mp.mpf(v["d"]), mp.mpf(v["s_E"]), max(mp.mpf(v["err"]), mp.mpf("1e-17"))) for v in R.values())
def fit(sel, powers, c0=None):
    n = len(powers) + (1 if c0 is None else 0)
    A = mp.matrix(len(sel), n); y = mp.matrix(len(sel), 1)
    for i, (d, s, e) in enumerate(sel):
        cols = ([1] if c0 is None else []) + [d ** (-p) for p in powers]
        w = 1 / (e + mp.mpf("1e-14"))
        for j, c in enumerate(cols):
            A[i, j] = c * w
        y[i] = (s - (c0 or 0)) * w
    x = mp.qr_solve(A, y)[0]
    res = max(abs(sum(A[i, j] * x[j] for j in range(n)) - y[i]) for i in range(len(sel)))
    return x, res
h = mp.mpf(1) / 2
sel = [p for p in pts if p[0] >= 16]
for name, pw in [("1/d series (6 terms)", [1, 2, 3, 4, 5, 6]), ("half-integer powers from 1 (7)", [1 + h * k for k in range(7)]),
                 ("half-integer powers from 1/2 (6)", [h * k for k in range(1, 7)])]:
    x, res = fit(sel, pw)
    print("%-34s free c0=%s  coeffs=%s  max weighted resid %s" % (name, mp.nstr(x[0], 10), [mp.nstr(c, 8) for c in x[1:4]], mp.nstr(res, 2)))
print("half-integer series, varying dmin and order (c0 free | c0 = 2):")
for dmin in (16, 24, 32):
    for K in (6, 7, 8, 9):
        pw = [1 + h * k for k in range(K)]
        s_ = [p for p in pts if p[0] >= dmin]
        x, _ = fit(s_, pw); x2, _ = fit(s_, pw, c0=2)
        print("  dmin=%2d K=%d  c0=%s c1=%s c3/2=%s | c1=%s c3/2=%s" % (dmin, K, mp.nstr(x[0], 10), mp.nstr(x[1], 8),
              mp.nstr(x[2], 7), mp.nstr(x2[0], 10), mp.nstr(x2[1], 8)))
print("unweighted comparison (d >= 16, constant free), as quoted in the paper:")
def fit_unw(sel, powers):
    n = len(powers) + 1
    A = mp.matrix(len(sel), n); y = mp.matrix(len(sel), 1)
    for i, (d, s, e) in enumerate(sel):
        cols = [1] + [d ** (-p) for p in powers]
        for j, c in enumerate(cols):
            A[i, j] = c
        y[i] = s
    x = mp.qr_solve(A, y)[0]
    return x, max(abs(sum(A[i, j] * x[j] for j in range(n)) - y[i]) for i in range(len(sel)))
for name, pw in [("1/d series (6 terms)", [1, 2, 3, 4, 5, 6]), ("half-integer powers from 1 (7)", [1 + h * k for k in range(7)])]:
    x, res = fit_unw(sel, pw)
    print("  %-34s c0=%s  max residual %s" % (name, mp.nstr(x[0], 10), mp.nstr(res, 2)))
