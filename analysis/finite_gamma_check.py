"""Paper Sec. IV (validation of the Gamma->inf limit) and Sec. V.C (finite-Gamma correction):
  (1) s_8(Gamma) from the finite-Gamma solver -> limiting s_8 as 1/Gamma^2;
  (2) c2, c4 in s(Gamma) = s_inf + c2/Gamma^2 + c4/Gamma^4 at fixed beta_d = 1/3;
  (3) nonrelativistic DSA control s -> 3r/(r-1)."""
import json, _common as C
import numpy as np
from finite_gamma import solve
b = 1 / 3
ref = {int(r["N"]): float(r["s"]) for r in json.load(open("results/ur3d_beta1_3_small.json"))}
Gs = [10, 14, 20, 28, 40]
v = {G: solve(np.sqrt(1 - 1 / G ** 2), b, 8) for G in Gs}
for G in Gs:
    print("Gamma=%3d  s_8=%.12f  Gamma^2 (s_8 - s_8(inf)) = %.5f" % (G, v[G], G * G * (v[G] - ref[8])))
G4 = Gs[-3:]
A = np.array([[1, g ** -2.0, g ** -4.0] for g in G4]); x = np.linalg.solve(A, [v[g] for g in G4])
print("extrapolated s_8(inf) = %.10f  vs limit solver %.10f  (diff %.1e)" % (x[0], ref[8], x[0] - ref[8]))
out = [(G, solve(np.sqrt(1 - 1 / G ** 2), b, 12, lo=4.0, hi=5.0)) for G in (6, 8, 10, 14)]
A = np.array([[1, 1 / G ** 2] for G, _ in out]); y = np.array([G ** 2 * (s - ref[12]) for G, s in out])
c = np.linalg.lstsq(A, y, rcond=None)[0]
print("finite-Gamma correction (N=12): c2 = %.4f  c4 = %.3f" % tuple(c))
for u, r in [(0.1, 4), (0.03, 4), (0.01, 4), (0.03, 2.5)]:
    print("nonrelativistic control u=%.2f r=%.1f: s=%.8f  (3r/(r-1) = %.8f)" % (u, r, solve(u, u / r, 6, lo=3.2, hi=8.0), 3 * r / (r - 1)))
