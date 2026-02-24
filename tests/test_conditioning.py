from geometric_spectroscopy.jacobian import build_jacobian, conditioning_raw

def test_conditioning_finite():
    J = build_jacobian(
        model="hayward",
        alpha=0.05,
        L_max=5,
        delta=1e-3,
        ells=range(2, 13),
        ns=(0, 1, 2),
        ps_ells=(2, 3),
    )

    _, _, kappa = conditioning_raw(J)
    assert kappa < 1e7