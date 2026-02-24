"""Monte Carlo and Fisher analysis for parameter estimation."""

import numpy as np
from .jacobian import realify_complex_residuals


def fisher_cov_theta(Jc: np.ndarray, sigma_omega: float, ridge: float = 0.0) -> np.ndarray:
    """
    Fisher covariance for Theta in y = Jr Theta + noise, noise ~ N(0, sigma_omega^2 I).

    F = (Jr^T Jr)/sigma^2
    cov = pinv(F + ridge_eff I)

    ridge:
      - 0.0 : none
      - <0  : automatic ridge based on trace(F)/M
      - >0  : fixed ridge
    """
    Jr = realify_complex_residuals(Jc)
    F = (Jr.T @ Jr) / (sigma_omega**2)
    M = F.shape[0]

    if ridge < 0:
        scale = float(np.trace(F)) / max(M, 1)
        ridge_eff = max(scale * 1e-10, 0.0)
    else:
        ridge_eff = float(ridge)

    A = F + ridge_eff * np.eye(M)
    cov = np.linalg.pinv(A)
    return cov


def fisher_sigma_theta(Jc: np.ndarray, sigma_omega: float, ridge: float = 0.0) -> np.ndarray:
    """Marginal 1-sigma uncertainties for each parameter."""
    cov = fisher_cov_theta(Jc, sigma_omega=sigma_omega, ridge=ridge)
    return np.sqrt(np.clip(np.diag(cov), 0.0, np.inf))


def monte_carlo_theta(
    Jc: np.ndarray,
    theta_true: np.ndarray,
    sigma_omega: float,
    realizations: int = 500,
    seed: int = 1,
    ridge: float = 0.0,
):
    """
    Monte Carlo estimator for full Theta.

    Uses:
      - ridge_eff = 0 : least squares via lstsq
      - ridge_eff !=0 : ridge via solve(J^T J + ridge I, J^T y)
    """
    rng = np.random.default_rng(seed)
    Jr = realify_complex_residuals(Jc)

    theta_true = np.asarray(theta_true, dtype=float)
    M = Jr.shape[1]
    if theta_true.shape != (M,):
        theta_true = theta_true[:M]  # Truncate if needed

    y0 = Jr @ theta_true

    if ridge < 0:
        scale = float(np.trace(Jr.T @ Jr)) / max(M, 1)
        ridge_eff = max(scale * 1e-10, 0.0)
    else:
        ridge_eff = float(ridge)

    est = np.zeros((realizations, M), dtype=float)
    for i in range(realizations):
        noise = rng.normal(0.0, sigma_omega, size=y0.shape)
        y = y0 + noise

        if ridge_eff == 0.0:
            theta_hat = np.linalg.lstsq(Jr, y, rcond=None)[0]
        else:
            A = Jr.T @ Jr + ridge_eff * np.eye(M)
            b = Jr.T @ y
            theta_hat = np.linalg.solve(A, b)

        est[i, :] = theta_hat

    return est.mean(axis=0), est.std(axis=0, ddof=1)