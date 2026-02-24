import numpy as np

from geometric_spectroscopy.jacobian import build_jacobian
from geometric_spectroscopy.monte_carlo import fisher_sigma_theta, monte_carlo_theta


def test_monte_carlo_and_fisher():
    J = build_jacobian(model="hayward", alpha=0.05, L_max=5)

    # Fisher
    sig = fisher_sigma_theta(J, sigma_omega=0.01, ridge=1e-6)
    assert sig.shape == (6,)
    assert np.all(np.isfinite(sig))

    # Monte Carlo
    mean, std = monte_carlo_theta(
        J,
        theta_true=np.zeros(6),
        sigma_omega=0.01,
        ridge=1e-6,
        realizations=100,
        seed=1,
    )

    assert mean.shape == (6,)
    assert std.shape == (6,)
    assert np.all(np.isfinite(mean))
    assert np.all(np.isfinite(std))
    assert np.all(std >= 0.0)