"""Paper Sec. X: location of the minimum of s for Juettner-Synge strong shocks, tested against beta_d = 1/8
(Richardson-corrected symmetric parabola fits)."""
import _common as C
from js_curve import jump, s_extrap
from scipy.optimize import brentq
def theta_for_bd(bd):
    return brentq(lambda lt: jump(10 ** lt)[1] - bd, -3, 1, xtol=1e-14)
V = {}
for h in (2e-3, 1e-3, 5e-4):
    ss = []
    for bd in (0.125 - h, 0.125, 0.125 + h):
        bu, bdd, R, Gr = jump(10 ** theta_for_bd(bd)); ss.append(s_extrap(bu, bdd)[0])
    a = (ss[0] + ss[2] - 2 * ss[1]) / (2 * h * h); bb = (ss[2] - ss[0]) / (2 * h)
    V[h] = 0.125 - bb / (2 * a)
    print("h=%g  parabola vertex beta_d = %.10f  s(beta_d=1/8) = %.12f" % (h, V[h], ss[1]))
print("Richardson (h^2) vertex: %.10f  and  %.10f   (1/8 = 0.125)" % (V[1e-3] + (V[1e-3] - V[2e-3]) / 3, V[5e-4] + (V[5e-4] - V[1e-3]) / 3))
