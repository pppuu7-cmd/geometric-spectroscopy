import numpy as np

from geometric_spectroscopy.high_order_derivatives import peak_derivatives


def test_peak_derivatives_contract_boundary_no_crash():
    """
    Contract (Standard B):
    peak_derivatives must not hard-fail just because the peak is close to a boundary.
    It should return finite arrays whenever a local window can be formed.
    """
    # Make a peak close to the left edge (but still enough points overall)
    x = np.linspace(0.0, 1.0, 1200)
    y = np.exp(-((x - 0.02) / 0.03) ** 2)  # peak near x=0

    pd = peak_derivatives(
        x,
        y,
        max_order=12,
        deg=12,
        windows=(120, 100, 80, 60),
        require_interior=500,  # intentionally "too strict" to ensure we don't hard-fail on this knob
        k_sg=6,
    )

    assert np.isfinite(pd.x0)
    assert np.isfinite(pd.derivs).all()
    assert np.isfinite(pd.derivs_mad[:7]).all()  # low orders have MAD=0


def test_peak_derivatives_contract_even_function_odd_zeroed():
    """
    Contract (Standard B):
    If the sampled profile is locally even about the detected peak (to near machine precision),
    we are allowed to enforce parity for low orders (<=6), i.e. odd derivatives == 0.
    """
    x = np.linspace(-3.0, 3.0, 12001)
    y = np.exp(-(x * x))  # perfectly even

    pd = peak_derivatives(
        x,
        y,
        max_order=10,
        deg=10,
        windows=(220, 200, 180, 160, 140, 120),
        require_interior=400,
        k_sg=6,
    )

    # Odd derivatives in low-order trusted band should be exactly zero (enforced).
    for k in (1, 3, 5):
        assert abs(float(pd.derivs[k])) == 0.0


def test_peak_derivatives_contract_locally_odd_not_forced_zero():
    """
    Contract (Standard B):
    For a profile that is NOT locally even about its peak, parity enforcement must NOT trigger.
    Then at least one low-order odd derivative (<=5) should be nonzero.

    Note:
    A shifted Gaussian is STILL even about its own peak, so it is not a valid "non-even" example.
    We break local evenness by multiplying by (1 + eps*(x-x0)).
    """
    x = np.linspace(-3.0, 3.0, 12001)

    x0 = 0.1
    eps = 0.02
    base = np.exp(-((x - x0) ** 2))
    y = base * (1.0 + eps * (x - x0))  # explicitly breaks evenness about x0

    pd = peak_derivatives(
        x,
        y,
        max_order=10,
        deg=10,
        windows=(220, 200, 180, 160, 140, 120),
        require_interior=400,
        k_sg=6,
    )

    assert np.isfinite(pd.derivs).all()
    assert np.isfinite(pd.derivs_mad).all()

    odd_vals = [abs(float(pd.derivs[k])) for k in (1, 3, 5)]
    assert max(odd_vals) > 1e-12