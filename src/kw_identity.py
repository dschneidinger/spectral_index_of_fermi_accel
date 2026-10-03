"""Re-derivation of the Keshet-Waxman (2005) spectrum-anisotropy identity under its own assumption
(trace of f is C^2 at grazing incidence, and the PDE relation a2 = -a1*k at mu=-beta holds for the trace),
for 3D (D d/dmu (1-mu^2) d/dmu) and 2D (d^2/dphi^2), isotropic constant D, in the limit beta_u -> 1.
Prints the predicted downstream anisotropy rho_d = F'(-beta_d)/F(-beta_d) as a function of s."""
import sympy as sp
s, x, eps, bd = sp.symbols('s x epsilon beta_d', positive=True)

def identity(dim):
    bu = 1 - eps
    br = (bu - bd) / (1 - bu * bd)
    gr = 1 / sp.sqrt(1 - br**2)
    mu_u = -bu + x
    mu_d = (mu_u + br) / (1 + br * mu_u)
    xd = sp.series(mu_d + bd, x, 0, 3).removeO()
    m1, m2 = xd.coeff(x, 1), xd.coeff(x, 2)
    fac = sp.series((1 + br * mu_u)**(-s), x, 0, 3).removeO()
    c0 = fac.coeff(x, 0); q1 = fac.coeff(x, 1) / c0; q2 = fac.coeff(x, 2) / c0
    g = lambda b: 1 / (1 - b**2)
    k = (lambda b: b * g(b)) if dim == 3 else (lambda b: b * g(b) / 2)
    rd, ru = sp.symbols('rho_d rho_u')
    e1 = sp.Eq(ru, m1 * rd + q1)
    e2 = sp.Eq(-ru * k(bu), m2 * rd - k(bd) * m1**2 * rd + m1 * q1 * rd + q2)
    sol = sp.solve([e1, e2], [rd, ru], dict=True)[0]
    rho = sp.simplify(sp.limit(sp.simplify(sol[rd]), eps, 0))
    return sp.simplify(rho)

for dim in (3, 2):
    r = identity(dim)
    print("dim", dim, " rho_d(s, beta_d) =", sp.factor(r))
    b = sp.Rational(1, 3) if dim == 3 else sp.Rational(1, 2)
    print("    at beta_d =", b, ":", sp.simplify(r.subs(bd, b)))
