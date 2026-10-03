"""Independent Monte Carlo of the downstream half-space (3D, isotropic direction-angle diffusion).

Particles enter the downstream region (downstream rest frame) with direction cosine mu drawn from the flux-weighted
entering distribution (beta+mu) F(mu), mu > -beta, where F is the shock-surface distribution from the eigenfunction
solution (or, as a control, an isotropic F). The direction vector performs Brownian motion on the unit sphere with
generator D * Laplace_S (so D_mumu = D (1-mu^2)); the distance from the shock obeys dzeta/dt = mu + beta.
A particle 'returns' when zeta crosses 0, 'escapes' at zeta = Zmax. Outputs P_ret and the exit-angle histogram,
which must reproduce F(mu) (beta+mu) on mu < -beta.
"""
import numpy as np, json, sys, argparse

def sample_entry(rng, n, mu_grid, F_grid, beta):
    m = mu_grid > -beta
    x = mu_grid[m]; w = (beta + x) * F_grid[m]
    c = np.concatenate([[0], np.cumsum(0.5 * (w[1:] + w[:-1]) * np.diff(x))]); c /= c[-1]
    return np.interp(rng.random(n), c, x)

def run(mu0, beta, dt, Zmax, rng, D=1.0, chunk=200000):
    n = len(mu0)
    ph = rng.random(n) * 2 * np.pi
    st = np.sqrt(1 - mu0 ** 2)
    v = np.stack([st * np.cos(ph), st * np.sin(ph), mu0], axis=1)   # z-axis = shock normal (downstream direction)
    zeta = np.zeros(n)
    alive = np.ones(n, bool)
    ret = np.zeros(n, bool); mu_exit = np.full(n, np.nan)
    sig = np.sqrt(2 * D * dt)
    # first step: move off the shock
    t = 0
    idx = np.arange(n)
    while idx.size:
        vv = v[idx]
        z_old = zeta[idx]
        z_new = z_old + (vv[:, 2] + beta) * dt
        # Brownian step on the sphere
        g = rng.standard_normal((idx.size, 3)) * sig
        g -= (g * vv).sum(1)[:, None] * vv
        vv = vv + g
        vv /= np.linalg.norm(vv, axis=1)[:, None]
        v[idx] = vv; zeta[idx] = z_new
        r = z_new < 0
        e = z_new > Zmax
        ret[idx[r]] = True
        mu_exit[idx[r]] = vv[r, 2]
        idx = idx[~(r | e)]
        t += 1
    return ret, mu_exit

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", required=True)
    ap.add_argument("--n", type=int, default=400000)
    ap.add_argument("--dts", type=float, nargs="+", default=[4e-3, 2e-3, 1e-3])
    ap.add_argument("--Zmax", type=float, default=12.0)
    ap.add_argument("--iso", action="store_true", help="control: isotropic F")
    a = ap.parse_args()
    T = json.load(open(a.table))
    beta = 1 / 3
    mu = np.array([r[0] for r in T["F_over_F0"]]); F = np.array([r[1] for r in T["F_over_F0"]])
    if a.iso:
        F = np.ones_like(F)
    rng = np.random.default_rng(12345)
    for dt in a.dts:
        mu0 = sample_entry(rng, a.n, mu, F, beta)
        ret, mx = run(mu0, beta, dt, a.Zmax, rng)
        P = ret.mean(); err = np.sqrt(P * (1 - P) / a.n)
        print("dt=%.1e  P_ret=%.5f +- %.5f" % (dt, P, err), flush=True)
        h, edges = np.histogram(mx[ret], bins=12, range=(-1, -beta))
        print("   exit-angle histogram (normalised flux):", np.round(h / h.sum(), 4).tolist(), flush=True)
