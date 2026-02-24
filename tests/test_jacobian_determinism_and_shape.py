import numpy as np

from geometric_spectroscopy.jacobian import build_jacobian


def test_jacobian_shape_and_determinism():
    GRID = dict(r_min=2.2, r_max=50.0, N_r=2000)

    J1 = build_jacobian(
        model="hayward",
        alpha=0.05,
        L_max=5,
        delta=1e-4,
        ells=range(2, 8),   # 6 ells
        ns=(0, 1),          # 2 overtones
        grid_kwargs=GRID,
        ps_ells=(),         # modes only
    )
    J2 = build_jacobian(
        model="hayward",
        alpha=0.05,
        L_max=5,
        delta=1e-4,
        ells=range(2, 8),
        ns=(0, 1),
        grid_kwargs=GRID,
        ps_ells=(),
    )

    assert J1.shape == (12, 6)
    assert np.allclose(J1, J2, rtol=0, atol=0)