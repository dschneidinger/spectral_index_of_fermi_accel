"""Paper Sec. IX: checks on the scattering-response kernel K(mu):
  integral (must vanish), N=24 vs N=16, bound on a grazing dipole c*delta'(mu+beta), linear response vs exact
  tangled-field shifts."""
import json, _common as C
import numpy as np
from scipy.interpolate import CubicSpline
K = C.load_glob("results/kernel_[abc].json")
mu = np.array(sorted(v["mu"] for v in K.values())); k = np.array([float(K[x]["K"]) for x in sorted(K, key=lambda x: K[x]["mu"])])
print("%d kernel points; trapezoid integral over grid %.2e, end contributions %.2e" % (len(mu), np.trapezoid(k, mu), k[0] * (mu[0] + 1) + k[-1] * (1 - mu[-1])))
i, j = np.argmin(k), np.argmax(k)
print("min K = %.4f at mu = %.3f ; max K = %.4f at mu = %.3f" % (k[i], mu[i], k[j], mu[j]))
chk = json.load(open("results/kernel_N24_check.json")); c16 = json.load(open("results/kernel_N16_check.json"))
for key, v in sorted(chk.items(), key=lambda kv: kv[1]["mu"]):
    m = v["mu"]; n16 = k[np.argmin(abs(mu - m))] if np.min(abs(mu - m)) < 1e-3 else None
    k16 = c16[key]["K"][:12] if key in c16 else ("%.10f" % n16 if n16 is not None else "-")
    print("N=24 vs N=16 at mu=%+.4f:  K_24=%s  K_16=%s" % (m, v["K"][:12], k16))
sel = (mu > -0.48) & (mu < -0.18); m = mu[sel]; kk = k[sel]; b = 1 / 3; s = 0.03
t = ((-b - m) / s ** 2) * np.exp(-(-b - m) ** 2 / (2 * s * s)) / (np.sqrt(2 * np.pi) * s)
for deg in (3, 4, 5):
    A = np.column_stack([m ** q for q in range(deg + 1)] + [t])
    x = np.linalg.lstsq(A, kk, rcond=None)[0]; r = kk - A @ x
    cov = np.linalg.inv(A.T @ A) * np.sum(r ** 2) / (len(kk) - A.shape[1])
    print("dipole fit, background degree %d: c = %.2e +- %.2e" % (deg, x[-1], np.sqrt(cov[-1, -1])))
cs = CubicSpline(mu, k, extrapolate=True)
xg = np.linspace(-1, 1, 200001); kx = cs(xg); kx -= np.trapezoid(kx, xg) / 2
A = C.load_glob("results/scan_aniso_*.json"); s0 = float(C.S3)
for a in ["10", "3", "2", "1", "0.5"]:
    lin = np.trapezoid(kx * (-0.5 * np.log(xg ** 2 + float(a) ** 2)), xg); ex = float(A[a]["s_inf"]) - s0
    print("a=%-4s exact delta s = %.6e  linear response = %.6e  ratio %.5f" % (a, ex, lin, lin / ex))
