"""Paper Sec. V.B: expansion s_inf = 3 + a1 b + a2 b^2 + a3 b^3 + ... about beta_d -> 0 (3D)."""
import json, _common as C
import mpmath as mp
mp.mp.dps = 40
R = json.load(open("results/scan_beta_3d.json"))
pts = sorted((mp.mpf(k.split("/")[0]) / mp.mpf(k.split("/")[1]), mp.mpf(v["s_inf"])) for k, v in R.items())
small = [p for p in pts if p[0] <= mp.mpf(1) / 20]
for deg in range(4, len(small) + 1):
    A = mp.matrix(deg, deg); y = mp.matrix(deg, 1)
    for i, (b, s) in enumerate(small[:deg]):
        for k in range(deg):
            A[i, k] = b ** (k + 1)
        y[i] = s - 3
    a = mp.lu_solve(A, y)
    print("points=%d  a1=%s  a2=%s  a3=%s" % (deg, mp.nstr(a[0], 16), mp.nstr(a[1], 12), mp.nstr(a[2], 8)))
