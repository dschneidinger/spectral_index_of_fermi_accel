"""Locate the minimum of s(Gamma_u beta_u) for the Juettner-Synge strong shock (trans-relativistic dip below 4)."""
import sys, json, numpy as np
sys.path.insert(0, __file__.rsplit("/", 1)[0])
from js_curve import jump, s_extrap
from scipy.optimize import minimize_scalar
def S(lt):
    bu, bd, R, Gr = jump(10 ** lt); return s_extrap(bu, bd)[0]
r = minimize_scalar(S, bracket=(-1.6, -1.25, -1.0), tol=1e-7)
bu, bd, R, Gr = jump(10 ** r.x); G = 1 / np.sqrt(1 - bu * bu)
res = {"theta": 10 ** r.x, "Gu_bu": G * bu, "beta_u": bu, "beta_d": bd, "R": R, "s_min": r.fun}
print(res); json.dump(res, open("results/js_min.json", "w"), indent=1)
