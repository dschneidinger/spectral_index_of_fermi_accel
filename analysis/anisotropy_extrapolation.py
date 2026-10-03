"""Paper Sec. VI: grazing-incidence anisotropy F'(-beta)/F(-beta) extrapolated N -> inf (N^-2/3 ansatz family),
compared with the value required by the Keshet-Waxman identity (3D) and its 2D analogue."""
import _common as C
import mpmath as mp
mp.mp.dps = 30
T = mp.mpf(1) / 3
for dim, log, S, req in [(3, "results/aniso3d.log", C.S3, lambda s: 3 * (s - 2) / 8),
                         (2, "results/aniso2d.log", C.S2, lambda s: 2 * s * (s - 1) / (3 * (2 * s + 1)))]:
    D = {k: mp.mpf(v) for k, v in C.aniso_log(log).items()}
    Ns = sorted(D)
    print("== %dD  N = %s" % (dim, Ns))
    for pw in ([2 * T], [2 * T, 4 * T], [2 * T, 1, 4 * T], [2 * T, 4 * T, 2]):
        sel = Ns[-(len(pw) + 1):]
        A = mp.matrix(len(sel), len(pw) + 1); b = mp.matrix(len(sel), 1)
        for i, N in enumerate(sel):
            A[i, 0] = 1
            for j, p in enumerate(pw):
                A[i, j + 1] = mp.mpf(N) ** (-p)
            b[i] = D[N]
        print("  powers %-22s N=%s  limit F'/F = %s" % ([mp.nstr(p, 3) for p in pw], sel, mp.nstr(mp.lu_solve(A, b)[0], 8)))
    print("  required by KW-type identity: %s" % mp.nstr(req(mp.mpf(S)), 10))
