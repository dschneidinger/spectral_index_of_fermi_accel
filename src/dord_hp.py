"""High-precision discrete-ordinates method (independent of ur3d.py). See dord.py for the formulation.

  nodes: n Gauss-Legendre points mu_k on [-1,1]
  downstream: ((1-mu^2)Q')' = Lam (beta+mu) Q,            allowed Lam<=0  (n_> vectors incl. constant)
  upstream:   ((1-mu^2)Q')' = Lam (beta+mu)(1-mu)^-3 Q,   allowed Lam>0   (n_< vectors)
  det[ V_d | diag((1-mu_k)^-s) V_u ] = 0
"""
import sys, time, json, argparse
from flint import arb, acb, arb_mat, acb_mat, fmpq, ctx


def gauss(n):
    xs, ws = [], []
    for k in range(n):
        x, w = arb.legendre_p_root(n, k, weight=True)
        xs.append(x); ws.append(w)
    return xs[::-1], ws[::-1]


def vandermonde(xs, n):
    """V[k][l] = p_l(x_k), orthonormal Legendre."""
    rows = []
    for x in xs:
        p = [arb(1) / arb(2).sqrt(), arb(3).sqrt() / arb(2).sqrt() * x]
        for l in range(1, n - 1):
            bl = arb(l + 1) / (arb(2 * l + 1) * arb(2 * l + 3)).sqrt()
            blm = arb(l) / (arb(2 * l - 1) * arb(2 * l + 1)).sqrt()
            p.append((x * p[l] - blm * p[l - 1]) / bl)
        rows.append(p[:n])
    return rows


def allowed(xs, ws, V, wvals, sign):
    n = len(xs)
    Vm = arb_mat(V)
    Dw = [ws[k] * wvals[k] for k in range(n)]
    VtD = arb_mat([[V[k][l] * Dw[k] for k in range(n)] for l in range(n)])
    M = VtD * Vm
    m00 = M[0, 0]
    sq = [1 / arb(l * (l + 1)).sqrt() for l in range(1, n)]
    T = arb_mat(n - 1, n - 1)
    for i in range(1, n):
        for j in range(1, n):
            T[i - 1, j - 1] = (M[i, j] - M[i, 0] * M[0, j] / m00) * sq[i - 1] * sq[j - 1]
    E, R = acb_mat(T).eig(right=True, algorithm="approx")
    cols = []
    if sign < 0:
        e0 = [arb(0)] * n; e0[0] = arb(1); cols.append(e0)
    for idx, e in enumerate(E):
        sig = e.real
        if (sign > 0 and sig < 0) or (sign < 0 and sig > 0):
            cp = [R[i, idx].real * sq[i] for i in range(n - 1)]
            c0 = -sum((M[0, j + 1] * cp[j] for j in range(n - 1)), arb(0)) / m00
            cols.append([c0] + cp)
    # nodal values
    C = arb_mat([[cols[c][l] for c in range(len(cols))] for l in range(n)])
    vals = Vm * C
    out = []
    for c in range(len(cols)):
        col = [vals[k, c].mid() for k in range(n)]
        sc = max(abs(float(v)) for v in col)
        out.append([v / sc for v in col])
    return out


def solve(beta, n, s0, verbose=False):
    beta = arb(beta)
    xs, ws = gauss(n)
    V = vandermonde(xs, n)
    Vd = allowed(xs, ws, V, [beta + x for x in xs], -1)
    Vu = allowed(xs, ws, V, [(beta + x) / (1 - x) ** 3 for x in xs], +1)
    assert len(Vd) + len(Vu) == n, (len(Vd), len(Vu))
    lg = [(1 - x).log() for x in xs]

    def det(s):
        cols = Vd + [[(-s * lg[k]).exp() * c[k] for k in range(n)] for c in Vu]
        A = arb_mat([[cols[c][k] for c in range(n)] for k in range(n)])
        return A.det()
    # secant
    x0, x1 = arb(s0), arb(s0) + arb("1e-6")
    f0, f1 = det(x0), det(x1)
    for it in range(60):
        if (f1 - f0).contains(0):
            break
        x2 = (x1 - f1 * (x1 - x0) / (f1 - f0)).mid()
        if verbose:
            print("   ", it, x2.str(40))
        if abs(x2 - x1) < arb(10) ** (-(int(ctx.prec * 0.301) - 10)):
            x1 = x2
            break
        x0, f0, x1, f1 = x1, f1, x2, det(x2)
    return x1, abs(x1 - x0)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, nargs="+", default=[16, 32])
    ap.add_argument("--prec", type=int, default=300)
    ap.add_argument("--precn", type=float, default=0.0)
    ap.add_argument("--beta", default="1/3")
    ap.add_argument("--s0", default="4.2269788167")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    p, q = a.beta.split("/")
    res = []
    for n in a.n:
        ctx.prec = int(a.prec + a.precn * n)
        t = time.time()
        s, ds = solve(fmpq(int(p), int(q)), n, a.s0)
        print("n=%4d s=%s  (last secant step %s, %.1fs, prec %d)" % (n, s.str(45, radius=False), ds.str(3, radius=False), time.time() - t, ctx.prec), flush=True)
        res.append({"n": n, "s": s.str(80, radius=False), "prec": ctx.prec, "beta": a.beta})
        if a.out:
            json.dump(res, open(a.out, "w"), indent=1)
