import numpy as np
from geometric_spectroscopy.high_order_derivatives import peak_derivatives


def _gaussian_derivative_at_zero(k: int) -> float:
    if k % 2 == 1:
        return 0.0
    m = k // 2
    num = 1.0
    for t in range(2 * m, 0, -1):
        num *= t
    den = 1.0
    for t in range(m, 0, -1):
        den *= t
    return ((-1.0) ** m) * (num / den)


def test_peak_derivatives_gaussian_up_to_26():
    """
    Standard B contract:

      - k=0..6: validated accuracy (what WKB6 actually needs)
      - k>=7: only sanity (finite + finite MAD)
    """

    x = np.linspace(-3.0, 3.0, 12001)
    y = np.exp(-(x * x))

    pd = peak_derivatives(
        x,
        y,
        max_order=26,
        deg=26,
        windows=(220, 200, 180, 160, 140, 120, 110, 100, 90, 80, 70, 60),
        require_interior=400,
        k_sg=6,
    )

    assert abs(pd.x0) < 1e-3
    assert np.isfinite(pd.derivs).all()
    assert np.isfinite(pd.derivs_mad).all()

    # ---- Low orders: require real accuracy ----
    for k in range(0, 7):
        truth = _gaussian_derivative_at_zero(k)
        got = float(pd.derivs[k])
        mag = max(1.0, abs(truth))

        if k <= 4:
            rel = 1e-6
        elif k == 5:
            rel = 5e-5
        else:  # k=6
            rel = 5e-4

        assert abs(got - truth) <= rel * mag + 1e-12, \
            f"low-order k={k}: got {got}, truth {truth}, rel={rel}"

    # ---- High orders: only numerical sanity ----
    for k in range(7, 27):
        assert np.isfinite(pd.derivs[k]), f"k={k} derivative not finite"
        assert np.isfinite(pd.derivs_mad[k]), f"k={k} MAD not finite"