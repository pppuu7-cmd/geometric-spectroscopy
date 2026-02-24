from geometric_spectroscopy.jacobian import build_jacobian

def test_jacobian_stability():
    J1 = build_jacobian(
        model="hayward",
        alpha=0.05,
        L_max=5,
        delta=1e-3,
        ells=range(2, 13),
        ns=(0, 1, 2),
        ps_ells=(2, 3),
    )

    J2 = build_jacobian(
        model="hayward",
        alpha=0.05,
        L_max=5,
        delta=1e-3,
        ells=range(2, 13),
        ns=(0, 1, 2),
        ps_ells=(2, 3),
    )

    assert (J1 - J2).max() < 1e-10