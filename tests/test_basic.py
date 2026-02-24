import numpy as np

from geometric_spectroscopy.jacobian import build_jacobian


def test_jacobian_shape_modes_only():
    """
    Legacy-style check: Jacobian for modes only (no extras).
    We make this explicit by setting ps_ells=().
    """
    J = build_jacobian(
        model="hayward",
        alpha=0.05,
        L_max=5,
        ells=range(2, 8),   # 6 ells: 2..7
        ns=(0, 1),          # 2 overtones
        ps_ells=(),         # no extras -> only modes
    )
    assert J.shape == (12, 6)  # 6*2=12 data, 6 params


def test_jacobian_shape_with_extras_default():
    """
    Standard-B check: by default we include extras for ps_ells=(2,3).
    Nmodes = (2..12)=11 ells * (0,1,2)=3 -> 33
    Nextra = 5 * len(ps_ells)=10
    Total Ndata = 43
    """
    J = build_jacobian(model="hayward", alpha=0.05, L_max=5)
    assert J.shape == (43, 6)
    assert np.isfinite(J.real).all()
    assert np.isfinite(J.imag).all()