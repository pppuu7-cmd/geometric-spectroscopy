import numpy as np
import pytest

from geometric_spectroscopy.jacobian import build_jacobian, conditioning_raw


def _effective_rank(Jc: np.ndarray, tol: float = 1e-6) -> int:
    Jr = np.vstack([Jc.real, Jc.imag])
    s = np.linalg.svd(Jr, compute_uv=False)
    smax = float(s[0])
    return int(np.sum(s > tol * smax))


@pytest.mark.parametrize("model", ["hayward", "bardeen", "simpson_visser"])
def test_smoke_order6_pipeline(model: str):
    """
    Smoke-test for the full Standard-B chain on order=6:
      model -> V(r*) -> QNM (WKB6+Padé[3/3]) -> data-vector -> Jacobian -> SVD stats.

    We keep it small so it's CI-friendly, but still exercises:
      - multiple ells,
      - at least one overtone,
      - extras via ps_ells.
    """
    GRID = dict(r_min=2.2, r_max=50.0, N_r=2200, profile_width=3.0)

    L_max = 5
    ells = range(2, 7)      # 5 ells
    ns = (0, 1)             # 2 overtones -> 10 modes
    ps_ells = (2, 3)        # extras: 10
    # total Ndata = 20

    J1 = build_jacobian(
        model=model,
        alpha=0.05,
        L_max=L_max,
        ells=ells,
        ns=ns,
        order=6,
        delta=1e-3,
        grid_kwargs=GRID,
        ps_ells=ps_ells,
    )
    J2 = build_jacobian(
        model=model,
        alpha=0.05,
        L_max=L_max,
        ells=ells,
        ns=ns,
        order=6,
        delta=1e-3,
        grid_kwargs=GRID,
        ps_ells=ps_ells,
    )

    Nmodes = len(list(ells)) * len(ns)  # 5*2=10
    Nextra = 5 * len(ps_ells)           # 10
    assert J1.shape == (Nmodes + Nextra, L_max + 1)
    assert np.allclose(J1, J2, rtol=0, atol=0)  # exact determinism
    assert np.isfinite(J1.real).all()
    assert np.isfinite(J1.imag).all()

    smax, smin, kappa = conditioning_raw(J1)
    assert np.isfinite(smax)
    assert np.isfinite(smin)
    assert np.isfinite(kappa)
    assert smax > 0.0
    assert smin >= 0.0

    # guard against catastrophic regressions (e.g. NaN or huge ill-conditioning)
    assert kappa < 1e8

    # should remain full rank in standard configuration
    r_eff = _effective_rank(J1, tol=1e-6)
    assert r_eff == (L_max + 1)