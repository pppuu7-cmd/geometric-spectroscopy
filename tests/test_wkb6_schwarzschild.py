import numpy as np
import pytest

from geometric_spectroscopy.qnm_wkb import qnm_wkb


def _tortoise_grid(r: np.ndarray, f: np.ndarray) -> np.ndarray:
    r = np.asarray(r, dtype=float)
    f = np.asarray(f, dtype=float)

    dr = np.diff(r)
    f_mid = 0.5 * (f[:-1] + f[1:])
    drstar = dr / f_mid

    rstar = np.zeros_like(r)
    rstar[1:] = np.cumsum(drstar)
    return rstar - rstar[rstar.size // 2]


def _schwarzschild_scalar_potential(r: np.ndarray, f: np.ndarray, ell: int, M: float = 1.0) -> np.ndarray:
    r = np.asarray(r, dtype=float)
    f = np.asarray(f, dtype=float)
    ell = int(ell)
    df_dr = np.gradient(f, r, edge_order=2)
    return f * (ell * (ell + 1) / (r * r) + df_dr / r)


@pytest.mark.parametrize("ell,n", [(2, 0)])
def test_schwarzschild_scalar_wkb6_pade33_sanity(ell: int, n: int):
    """
    Sanity benchmark (NOT a strict precision test).

    Reference (tables / Leaver):
      ω ≈ 0.483644 − 0.0967588 i  (M=1)
    We allow looser tolerances because:
      - this project uses WKB6 + Padé[3/3] (series only through ε^6),
      - derivative extraction is numerical.
    """
    M = 1.0
    r_min = 2.0 * M * 1.001
    r_max = 200.0 * M
    N_r = 24000
    r = np.linspace(r_min, r_max, N_r)

    f = 1.0 - 2.0 * M / r
    rstar = _tortoise_grid(r, f)
    V = _schwarzschild_scalar_potential(r, f, ell=ell, M=M)

    w = complex(qnm_wkb(rstar, V, n=n, order=6, ell=ell).omega)
    assert w.imag < 0.0

    w_ref = 0.483644 - 0.0967588j

    rel_re = abs((w.real - w_ref.real) / w_ref.real)
    rel_im = abs((w.imag - w_ref.imag) / w_ref.imag)

    # sanity-level thresholds
    assert rel_re < 0.06, f"Re(ω) too far: got {w.real}, ref {w_ref.real}, rel {rel_re}"
    assert rel_im < 0.10, f"Im(ω) too far: got {w.imag}, ref {w_ref.imag}, rel {rel_im}"