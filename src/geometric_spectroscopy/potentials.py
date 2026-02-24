"""Potential construction for geometric spectroscopy.

Compatible with NumPy builds where np.trapz may be removed.

This module builds a 1D Schrödinger-like potential V(r_*) and perturbs it with
a compact deformation basis in r_*.
"""

from __future__ import annotations

from typing import Tuple
import numpy as np


def _trapz(y: np.ndarray, x: np.ndarray) -> float:
    """Compatibility trapezoid rule without relying on np.trapz/np.trapezoid."""
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    if y.ndim != 1 or x.ndim != 1 or y.size != x.size:
        raise ValueError("_trapz expects 1D arrays of equal length")
    if y.size < 2:
        return 0.0
    dx = np.diff(x)
    return float(np.sum(0.5 * (y[:-1] + y[1:]) * dx))


def metric_f(r: np.ndarray, *, model: str, alpha: float, M: float = 1.0) -> np.ndarray:
    """
    Metric function f(r) for regular black hole models.
    Returns f(r) where ds^2 = -f(r)dt^2 + f(r)^{-1}dr^2 + r^2 dΩ^2.
    """
    r = np.asarray(r, dtype=float)
    alpha = float(alpha)
    M = float(M)

    model = model.lower()

    if model == "hayward":
        # Hayward: f = 1 - 2Mr^2/(r^3 + 2Mα^2)
        return 1.0 - (2.0 * M * r * r) / (r**3 + 2.0 * M * alpha * alpha)

    elif model == "bardeen":
        # Bardeen: f = 1 - 2Mr^2/(r^2 + α^2)^(3/2)
        denom = (r**2 + alpha**2) ** 1.5
        f = 1.0 - (2.0 * M * r * r) / denom
        return np.maximum(f, 1e-12)

    elif model == "simpson_visser":
        # Simpson-Visser: f = 1 - 2M/√(r^2 + α^2)
        return 1.0 - 2.0 * M / np.sqrt(r**2 + alpha**2)

    else:
        raise ValueError(f"Unknown model: {model}")


def tortoise_grid(r: np.ndarray, f: np.ndarray) -> np.ndarray:
    """Compute r_* = ∫ dr / f(r) with cumulative trapezoid."""
    r = np.asarray(r, dtype=float)
    f = np.asarray(f, dtype=float)

    dr = np.diff(r)
    f_mid = 0.5 * (f[:-1] + f[1:])
    drstar = dr / f_mid

    rstar = np.zeros_like(r)
    rstar[1:] = np.cumsum(drstar)

    # Center around zero for numerical stability
    return rstar - rstar[len(rstar) // 2]


def scalar_single_barrier_potential(r: np.ndarray, f: np.ndarray, *, ell: int) -> np.ndarray:
    """Standard scalar-field barrier potential."""
    r = np.asarray(r, dtype=float)
    f = np.asarray(f, dtype=float)
    ell = int(ell)

    df_dr = np.gradient(f, r, edge_order=2)
    return f * (ell * (ell + 1) / (r * r) + df_dr / r)


def multipolar_profiles_phys(
    rstar: np.ndarray,
    *,
    L_max: int,
    center: float,
    width: float = 3.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build compact profiles in r_* (Gaussian-based)."""
    rstar = np.asarray(rstar, dtype=float)
    x = (rstar - float(center)) / float(width)

    env = np.exp(-0.5 * x**2)

    profiles = []
    weights = []

    for L in range(int(L_max) + 1):
        if L == 0:
            prof = env
        elif L == 1:
            prof = env * x
        elif L == 2:
            prof = env * (1.5 * x**2 - 0.5)
        elif L == 3:
            prof = env * (2.5 * x**3 - 1.5 * x)
        elif L == 4:
            prof = env * (4.375 * x**4 - 3.75 * x**2 + 0.375)
        elif L == 5:
            prof = env * (7.875 * x**5 - 8.75 * x**3 + 1.875 * x)
        else:
            prof = env * x**L

        # Normalize to unit max for stable finite differences
        m = np.max(np.abs(prof))
        if m > 0:
            prof = prof / m

        w = _trapz(prof**2, rstar)
        w = w if (w > 0 and np.isfinite(w)) else 1.0

        profiles.append(prof)
        weights.append(w)

    return np.asarray(profiles, dtype=float), np.asarray(weights, dtype=float)


def build_potential_rstar(
    model: str,
    ell: int,
    alpha: float,
    theta: np.ndarray,
    L_max: int,
    *,
    M: float = 1.0,
    r_min: float = 2.2,
    r_max: float = 60.0,
    N_r: int = 6000,
    profile_width: float = 3.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build potential V(r_*) on a tortoise grid.
    Returns: (rstar, V, weights) where weights are L2 norms of profiles.
    """
    theta = np.asarray(theta, dtype=float)
    if theta.shape != (int(L_max) + 1,):
        raise ValueError(f"theta must have shape ({int(L_max) + 1},)")

    r = np.linspace(float(r_min), float(r_max), int(N_r))
    f = metric_f(r, model=model, alpha=float(alpha), M=float(M))

    if np.any(~np.isfinite(f)) or np.any(f <= 0):
        raise ValueError(
            f"Invalid f(r) on grid for {model} with r_min={r_min}. "
            f"f min={np.min(f):.6g}, f max={np.max(f):.6g}"
        )

    rstar = tortoise_grid(r, f)
    V0 = scalar_single_barrier_potential(r, f, ell=int(ell))

    i0 = int(np.argmax(V0))
    rstar_peak = float(rstar[i0])

    profiles, weights = multipolar_profiles_phys(
        rstar,
        L_max=int(L_max),
        center=rstar_peak,
        width=float(profile_width),
    )

    dV = np.tensordot(theta, profiles, axes=(0, 0))
    V = V0 + dV

    return rstar, V, weights