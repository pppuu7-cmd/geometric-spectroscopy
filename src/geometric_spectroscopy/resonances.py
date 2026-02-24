"""Resonance computations for geometric spectroscopy."""

import numpy as np

from .potentials import build_potential_rstar
from .qnm_wkb import qnm_wkb


def _trapz(y: np.ndarray, x: np.ndarray) -> float:
    """Compatibility trapezoid rule without relying on np.trapz/np.trapezoid.

    Works on NumPy versions where np.trapz may be removed.
    """
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    if y.ndim != 1 or x.ndim != 1 or y.size != x.size:
        raise ValueError("_trapz expects 1D arrays of equal length")
    if y.size < 2:
        return 0.0
    dx = np.diff(x)
    return float(np.sum(0.5 * (y[:-1] + y[1:]) * dx))


def _quad_vertex(x0, y0, x1, y1, x2, y2):
    """Vertex of parabola through three points."""
    A = np.array(
        [[x0 * x0, x0, 1.0], [x1 * x1, x1, 1.0], [x2 * x2, x2, 1.0]],
        dtype=float,
    )
    b = np.array([y0, y1, y2], dtype=float)

    try:
        a, bb, c = np.linalg.solve(A, b)
        xv = -bb / (2.0 * a)
        yv = a * xv * xv + bb * xv + c
        ypp = 2.0 * a
        return float(xv), float(yv), float(ypp)
    except np.linalg.LinAlgError:
        # Fallback to a crude discrete curvature
        denom = (x1 - x0) * (x2 - x1)
        ypp = float((y2 - 2 * y1 + y0) / denom) if denom != 0 else -1.0
        return float(x1), float(y1), ypp


def peak_features_smooth(rstar, V):
    """Smooth peak extraction via quadratic fit around discrete max."""
    rstar = np.asarray(rstar, dtype=float)
    V = np.asarray(V, dtype=float)

    i0 = int(np.argmax(V))
    if i0 < 1 or i0 > len(V) - 2:
        return float(rstar[i0]), float(V[i0]), -1.0

    x0, x1, x2 = rstar[i0 - 1], rstar[i0], rstar[i0 + 1]
    y0, y1, y2 = V[i0 - 1], V[i0], V[i0 + 1]
    return _quad_vertex(x0, y0, x1, y1, x2, y2)


def global_proxies(rstar, V, rpk, R=12.0):
    """
    Stable nonlocal observables from a window around the peak.
      A = ∫ V dr*
      W = (1/A) ∫ (r*-rpk)^2 V dr*   (second moment / width proxy)
    """
    rstar = np.asarray(rstar, dtype=float)
    V = np.asarray(V, dtype=float)

    mask = np.abs(rstar - float(rpk)) <= float(R)
    xs = rstar[mask]
    Vs = V[mask]

    if xs.size < 5:
        return 0.0, 0.0

    A = _trapz(Vs, xs)
    if A == 0.0 or (not np.isfinite(A)):
        return 0.0, 0.0

    W = _trapz(((xs - rpk) ** 2) * Vs, xs) / A
    if not np.isfinite(W):
        W = 0.0

    return float(A), float(W)


def compute_qnms(
    model: str,
    alpha: float,
    theta: np.ndarray,
    L_max: int,
    ells,
    ns,
    *,
    order: int = 3,
    grid_kwargs=None,
):
    """Compute QNMs for given modes."""
    if grid_kwargs is None:
        grid_kwargs = {}

    order = int(order)
    if order not in (3, 6, 13):
        raise ValueError("order must be 3, 6, or 13")

    omegas = []
    for ell in ells:
        rstar, V, _weights = build_potential_rstar(
            model=model,
            ell=int(ell),
            alpha=float(alpha),
            theta=np.asarray(theta, dtype=float),
            L_max=int(L_max),
            **grid_kwargs,
        )
        for n in ns:
            w = qnm_wkb(rstar, V, n=int(n), order=order, ell=int(ell))
            omegas.append(complex(w))

    return np.asarray(omegas, dtype=complex)


def compute_data_vector(
    model: str,
    alpha: float,
    theta: np.ndarray,
    L_max: int,
    ells,
    ns,
    *,
    order: int = 3,
    grid_kwargs=None,
    ps_ells=(2, 3),
    Rwin=12.0,
):
    """
    Full data vector: [omega_{ell,n}] + extras for each ps_ell:
        [Omega_proxy, r*_peak, Vpp_peak, A_win, W_win]
    """
    if grid_kwargs is None:
        grid_kwargs = {}

    order = int(order)
    if order not in (3, 6, 13):
        raise ValueError("order must be 3, 6, or 13")

    y = list(
        compute_qnms(
            model,
            alpha,
            theta,
            L_max,
            ells,
            ns,
            order=order,
            grid_kwargs=grid_kwargs,
        )
    )

    for ps_ell in ps_ells:
        rstar, V, _weights = build_potential_rstar(
            model=model,
            ell=int(ps_ell),
            alpha=float(alpha),
            theta=np.asarray(theta, dtype=float),
            L_max=int(L_max),
            **grid_kwargs,
        )

        rpk, V0, Vpp = peak_features_smooth(rstar, V)
        A, W = global_proxies(rstar, V, rpk, R=float(Rwin))

        denom = float(ps_ell * (ps_ell + 1))
        Omega_proxy = float(np.sqrt(max(V0, 0.0) / denom)) if denom > 0 else 0.0

        y.append(complex(Omega_proxy, 0.0))
        y.append(complex(rpk, 0.0))
        y.append(complex(Vpp, 0.0))
        y.append(complex(A, 0.0))
        y.append(complex(W, 0.0))

    return np.asarray(y, dtype=complex)