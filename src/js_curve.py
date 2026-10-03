"""Spectral index vs shock four-velocity for a strong shock into a cold gas with the Juettner-Synge downstream
equation of state (3D, isotropic direction-angle diffusion), via the finite-Gamma eigenfunction solver.

Jump conditions (cold upstream, w_u = 1, p_u = 0), downstream temperature theta = kT/mc^2, z = 1/theta:
  specific enthalpy  w = K3(z)/K2(z),   e/rho = w - theta,   p/rho = theta
  energy per particle in the downstream frame equals Gamma_rel:  Gamma_r = w - theta
  Taub adiabat:  w^2 - 1 = theta (R + w),  R = rho_d/rho_u      =>  R = (w^2 - 1 - theta w)/theta
  shock frame:   rho_u G_u b_u = rho_d G_d b_d,   G_r = G_u G_d (1 - b_u b_d)
"""
import sys, json
import numpy as np
from scipy.special import kve
from scipy.optimize import brentq
sys.path.insert(0, __file__.rsplit("/", 1)[0])
from finite_gamma import solve


def jump(theta):
    z = 1 / theta
    w = kve(3, z) / kve(2, z)
    Gr = w - theta
    R = (w * w - 1 - theta * w) / theta
    br = np.sqrt(1 - 1 / Gr ** 2)
    # solve for b_d: b_u = (b_d + br)/(1 + b_d br) (velocity addition), flux: G_u b_u = R G_d b_d
    def f(bd):
        bu = (bd + br) / (1 + bd * br)
        return bu / np.sqrt(1 - bu * bu) - R * bd / np.sqrt(1 - bd * bd)
    bd = brentq(f, 1e-12, br * 0.999999)
    bu = (bd + br) / (1 + bd * br)
    return bu, bd, R, Gr


def s_extrap(u, b, Ns=(8, 12, 16, 20)):
    v = [solve(u, b, N, lo=3.0, hi=6.0) for N in Ns]
    A = np.array([[1, N ** -4.0, N ** -5.0, N ** -6.0][:len(Ns)] for N in Ns])
    x = np.linalg.solve(A, np.array(v))
    return x[0], v


if __name__ == "__main__":
    out = []
    for th in np.geomspace(1e-3, 300, 22):
        bu, bd, R, Gr = jump(th)
        G = 1 / np.sqrt(1 - bu * bu)
        if G > 40:
            continue
        s, v = s_extrap(bu, bd)
        out.append({"theta": th, "Gu_bu": G * bu, "beta_u": bu, "beta_d": bd, "R": R, "Gamma_rel": Gr, "s": s,
                    "s_N": v})
        print("theta=%9.4g  G_u b_u=%9.4f  b_d=%.6f  R=%7.4f  s=%.10f  (s_20-s=%.1e)" % (th, G * bu, bd, R, s, v[-1] - s), flush=True)
        json.dump(out, open("results/js_curve.json", "w"), indent=1)
