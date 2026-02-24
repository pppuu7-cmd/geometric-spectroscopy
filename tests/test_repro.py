import numpy as np

from geometric_spectroscopy.jacobian import build_jacobian, conditioning


def test_jacobian_deterministic():
    """
    Determinism check: build_jacobian must be identical across runs
    for the same inputs (no randomness in physics chain).
    """
    GRID = dict(r_min=2.2, r_max=60.0, N_r=2000)

    J1 = build_jacobian(
        model="hayward",
        alpha=0.05,
        L_max=5,
        delta=1e-4,
        grid_kwargs=GRID,
        ps_ells=(),  # speed + legacy-style determinism check
        ells=range(2, 8),
        ns=(0, 1),
    )
    J2 = build_jacobian(
        model="hayward",
        alpha=0.05,
        L_max=5,
        delta=1e-4,
        grid_kwargs=GRID,
        ps_ells=(),
        ells=range(2, 8),
        ns=(0, 1),
    )

    assert J1.shape == (12, 6)
    assert np.allclose(J1, J2, rtol=0, atol=0)  # exact determinism


def test_conditioning_finite():
    GRID = dict(r_min=2.2, r_max=60.0, N_r=2000)

    J = build_jacobian(
        model="hayward",
        alpha=0.05,
        L_max=5,
        delta=1e-4,
        grid_kwargs=GRID,
        ps_ells=(),
        ells=range(2, 8),
        ns=(0, 1),
    )
    smax, smin, kappa = conditioning(J)
    assert np.isfinite(smax)
    assert np.isfinite(smin)
    assert np.isfinite(kappa)
    assert smax > 0
    assert smin >= 0