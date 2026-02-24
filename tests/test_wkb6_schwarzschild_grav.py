import numpy as np
import pytest

from geometric_spectroscopy.qnm_wkb import qnm_wkb


def _tortoise_grid(r: np.ndarray, f: np.ndarray) -> np.ndarray:
    """Compute r_* = ∫ dr/f using midpoint trapezoid (deterministic)."""
    r = np.asarray(r, dtype=float)
    f = np.asarray(f, dtype=float)

    dr = np.diff(r)
    f_mid = 0.5 * (f[:-1] + f[1:])
    if np.any(f_mid <= 0) or np.any(~np.isfinite(f_mid)):
        raise ValueError("Non-positive or non-finite f on grid (cannot build tortoise coordinate)")

    drstar = dr / f_mid
    rstar = np.zeros_like(r)
    rstar[1:] = np.cumsum(drstar)

    # center for numerical stability
    return rstar - rstar[rstar.size // 2]


def _schwarzschild_regge_wheeler_potential(r: np.ndarray, f: np.ndarray, ell: int, M: float = 1.0) -> np.ndarray:
    """
    Axial gravitational perturbations (s=2): Regge–Wheeler potential
      V = f [ l(l+1)/r^2 - 6M/r^3 ].
    """
    r = np.asarray(r, dtype=float)
    f = np.asarray(f, dtype=float)
    ell = int(ell)
    return f * (ell * (ell + 1) / (r * r) - 6.0 * M / (r**3))


@pytest.mark.parametrize("ell,n", [(2, 0)])
def test_schwarzschild_grav_wkb6_pade33_sanity(ell: int, n: int):
    """
    Sanity benchmark (NOT strict Leaver precision).

    Reference value (tables / Leaver) for Schwarzschild gravitational (s=2),
    axial RW potential, l=2, n=0 (M=1):
      ω ≈ 0.373672 − 0.0889623 i

    We keep tolerances loose enough for WKB6 + Padé[3/3] and numerical derivative extraction,
    but tight enough to catch sign/scale/branch bugs.
    """
    M = 1.0

    r_min = 2.0 * M * 1.001
    r_max = 200.0 * M
    N_r = 24000
    r = np.linspace(r_min, r_max, N_r)

    f = 1.0 - 2.0 * M / r
    assert np.all(f > 0)

    rstar = _tortoise_grid(r, f)
    V = _schwarzschild_regge_wheeler_potential(r, f, ell=ell, M=M)

    res = qnm_wkb(rstar, V, n=n, order=6, ell=ell)
    w = complex(res.omega)

    assert np.isfinite(w.real)
    assert np.isfinite(w.imag)
    assert w.imag < 0.0

    w_ref = 0.373672 - 0.0889623j

    rel_re = abs((w.real - w_ref.real) / w_ref.real)
    rel_im = abs((w.imag - w_ref.imag) / w_ref.imag)

    assert rel_re < 0.08, f"Re(ω) too far: got {w.real}, ref {w_ref.real}, rel {rel_re}"
    assert rel_im < 0.12, f"Im(ω) too far: got {w.imag}, ref {w_ref.imag}, rel {rel_im}"