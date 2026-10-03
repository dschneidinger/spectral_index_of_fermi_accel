"""Closed-form test of the new constants with BootLoops' pslq_gate protocol (tools/lockpick/pslq_gate.py):
two-precision canonicalized PSLQ, capacity refusal, members_audit, planted positive controls inside each basket.
The available digits are modest (~25 for s3, ~25 for s2, ~15 for a1), so only low heights can be excluded;
the printed capacity says exactly which."""
import sys, math
import os
# Requires a checkout of BootLoops 1.0 (https://www.bootloops.ai) for its pslq_gate module.
_BL = os.environ.get("BOOTLOOPS_ROOT", os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(_BL, "tools", "lockpick"))
sys.path.insert(0, os.path.join(_BL, "tools"))
try:
    import pslq_gate  # noqa: F401
except ImportError:
    sys.exit("closed_form_scan.py needs BootLoops: set BOOTLOOPS_ROOT to a BootLoops 1.0 checkout")
import mpmath as mp
import pslq_gate as G

mp.mp.dps = 60
TARGETS = {
    "s3D": ("4.226978816696084932553273", 25),
    "s2D": ("3.2985080563907181433382986", 25),
}


def consts(dps):
    with mp.workdps(dps + 10):
        return {
            "1": mp.mpf(1), "pi": mp.pi, "pi2": mp.pi ** 2, "log2": mp.log(2), "log3": mp.log(3),
            "zeta3": mp.zeta(3), "sqrt2": mp.sqrt(2), "sqrt3": mp.sqrt(3), "euler": mp.euler,
        }


BASKETS = {
    "algebraic-deg3": None,                      # {1, t, t^2, t^3}
    "rational+pi,pi2,log2,log3": ["1", "pi", "pi2", "log2", "log3"],
    "rational+sqrt2,sqrt3,zeta3": ["1", "sqrt2", "sqrt3", "zeta3"],
}


def run(name, tstr, nd):
    dps_pair = (nd - 3, nd)
    out = []
    C = consts(nd)
    t = mp.mpf(tstr)
    for bname, members in BASKETS.items():
        if members is None:
            names = ["t", "t2", "t3"]
            vals = {"t": t, "t2": t ** 2, "t3": t ** 3}
            # relation among (1, t, t^2, t^3) with target 1
            target = "1"
            pool = {k: mp.nstr(v, nd + 5, strip_zeros=False) for k, v in vals.items()}
            tgt = mp.nstr(mp.mpf(1), nd + 5)
            tgt = "1." + "0" * (nd + 4)
        else:
            names = members
            pool = {k: mp.nstr(C[k], nd + 5, strip_zeros=False) for k in members}
            tgt = tstr + "0" * 5 if len(tstr.replace(".", "")) < nd + 5 else tstr
        n = len(names)
        # largest height the capacity rule admits with zero margin (digits budget only)
        hmax = 10 ** ((min(dps_pair) - 2) / (n + 1))
        H = int(10 ** math.floor(math.log10(hmax)))
        if H < 10:
            out.append((bname, "refused: capacity < 10"))
            continue
        # planted control inside the basket: 3 * m1 - 2 * m_last (coefficients << H) - must be found
        if members is not None:
            with mp.workdps(nd + 10):
                ctrl = 3 * mp.mpf(pool[names[0]]) - 2 * mp.mpf(pool[names[-1]]) + mp.mpf(pool[names[1]]) / 1
            try:
                vc = G.two_prec_stable(mp.nstr(ctrl, nd + 5, strip_zeros=False), names, pool, dps_pair=dps_pair, maxcoeff=H, margin=2)
            except Exception as e:
                vc = "ERR %s" % e
            if vc is None or isinstance(vc, str):
                out.append((bname, "CONTROL FAILED (%s) -> no verdict" % vc))
                continue
        try:
            v = G.two_prec_stable(tgt, names, pool, dps_pair=dps_pair, maxcoeff=H, margin=2)
        except G.DegeneratePoolError as e:
            out.append((bname, "degenerate pool %s" % e.relation)); continue
        except Exception as e:
            out.append((bname, "ERR %s" % e)); continue
        if v is None:
            out.append((bname, "NULL: no relation with all |coeff| <= %d (n=%d members, %d digits)" % (H, n, nd)))
        else:
            out.append((bname, "HIT %s  (needs held-out verification)" % (v,)))
    return out


if __name__ == "__main__":
    for name, (tstr, nd) in TARGETS.items():
        print("==", name, tstr)
        for b, r in run(name, tstr, nd):
            print("   %-32s %s" % (b, r))
