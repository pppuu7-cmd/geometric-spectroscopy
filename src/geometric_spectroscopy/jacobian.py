"""Jacobian construction and conditioning analysis."""

import numpy as np
from .resonances import compute_data_vector
from .potentials import build_potential_rstar


def build_jacobian(
    model: str,
    alpha: float,
    L_max: int,
    ells=range(2, 13),
    ns=(0, 1, 2),
    *,
    order: int = 3,
    delta: float = 1e-3,
    theta0=None,
    grid_kwargs=None,
    ps_ells=(2, 3),
) -> np.ndarray:
    """
    Finite-difference Jacobian for the full data vector:
      y = [QNM modes] + extras(ps_ells)

    Parameters
    ----------
    order : int
        WKB order selector. Supported: 3 or 6.
    """
    order = int(order)
    if order not in (3, 6):
        raise ValueError("order must be 3 or 6")

    if theta0 is None:
        theta0 = np.zeros(L_max + 1, dtype=float)
    else:
        theta0 = np.asarray(theta0, dtype=float)
        if theta0.shape != (L_max + 1,):
            raise ValueError(f"theta0 must have shape ({L_max+1},)")

    if grid_kwargs is None:
        grid_kwargs = {}

    base = compute_data_vector(
        model,
        alpha,
        theta0,
        L_max,
        ells,
        ns,
        order=order,
        grid_kwargs=grid_kwargs,
        ps_ells=ps_ells,
    )
    N = base.size
    M = L_max + 1
    J = np.zeros((N, M), dtype=complex)

    for L in range(M):
        th = theta0.copy()
        th[L] += float(delta)
        pert = compute_data_vector(
            model,
            alpha,
            th,
            L_max,
            ells,
            ns,
            order=order,
            grid_kwargs=grid_kwargs,
            ps_ells=ps_ells,
        )
        J[:, L] = (pert - base) / float(delta)

    return J


def realify_complex_residuals(Jc: np.ndarray) -> np.ndarray:
    """Stack real and imaginary parts for real-valued inference."""
    Jc = np.asarray(Jc, dtype=complex)
    return np.vstack([Jc.real, Jc.imag])


def _svd_stats(Jr: np.ndarray):
    """Compute SVD statistics for real matrix."""
    s = np.linalg.svd(Jr, compute_uv=False)
    smax = float(s[0])
    smin = float(s[-1])
    kappa = float(smax / smin) if smin > 0 else float("inf")
    return smax, smin, kappa


def _svd_singular_values(Jr: np.ndarray) -> np.ndarray:
    """Return singular values for a real matrix."""
    return np.linalg.svd(Jr, compute_uv=False)


def effective_rank(Jc: np.ndarray, eps: float = 1e-30) -> float:
    """
    Effective rank (entropy-based) of the realified Jacobian.

    r_eff = exp( - sum_i p_i log p_i ), where p_i = s_i / sum s.

    This is a smooth proxy for rank that is useful for stability monitoring.
    """
    Jr = realify_complex_residuals(Jc)
    s = _svd_singular_values(Jr).astype(float, copy=False)
    s = s[np.isfinite(s)]
    if s.size == 0:
        return float("nan")

    ssum = float(np.sum(s))
    if (not np.isfinite(ssum)) or ssum <= 0.0:
        return float("nan")

    p = s / max(ssum, eps)
    p = np.clip(p, eps, 1.0)
    H = float(-np.sum(p * np.log(p)))
    return float(np.exp(H))


def conditioning_raw(Jc: np.ndarray):
    """Conditioning in the raw parameterization."""
    Jr = realify_complex_residuals(Jc)
    return _svd_stats(Jr)


def conditioning(Jc: np.ndarray):
    """
    Backward-compatible alias.

    Older tests and scripts imported `conditioning` from this module.
    It now maps to the raw conditioning metric.
    """
    return conditioning_raw(Jc)


def conditioning_geom(Jc: np.ndarray, eps: float = 1e-30):
    """
    Scale-invariant conditioning using column normalization on Jr.
    Returns:
      (smax, smin, kappa), col_norms
    """
    Jr = realify_complex_residuals(Jc)
    col_norms = np.linalg.norm(Jr, axis=0)
    Dinv = np.diag(1.0 / np.maximum(col_norms, eps))
    Jrn = Jr @ Dinv
    return _svd_stats(Jrn), col_norms


def deformation_weights_phys(
    model: str,
    alpha: float,
    L_max: int,
    ells=range(2, 8),
    grid_kwargs=None,
):
    """
    Physical weights w_L = ∫ |Delta_L(r*)|^2 dr* averaged over ell in `ells`.
    """
    if grid_kwargs is None:
        grid_kwargs = {}
    theta0 = np.zeros(L_max + 1, dtype=float)

    W = []
    for ell in ells:
        _rstar, _V, weights = build_potential_rstar(
            model=model,
            ell=int(ell),
            alpha=float(alpha),
            theta=theta0,
            L_max=int(L_max),
            **grid_kwargs,
        )
        W.append(weights)

    W = np.mean(np.stack(W, axis=0), axis=0)
    return W


def conditioning_phys(Jc: np.ndarray, weights: np.ndarray, eps: float = 1e-30):
    """
    Conditioning in the physical deformation norm:
      ||Theta||_phys^2 = sum w_L theta_L^2
    => scale parameters by W^{-1/2}.
    """
    Jr = realify_complex_residuals(Jc)
    weights = np.asarray(weights, dtype=float)
    Winv_sqrt = np.diag(1.0 / np.sqrt(np.maximum(weights, eps)))
    Jrp = Jr @ Winv_sqrt
    return _svd_stats(Jrp)