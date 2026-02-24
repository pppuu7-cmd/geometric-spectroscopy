"""Second-order utilities (placeholder scaffold).

The Standard-B roadmap calls out a `second_order.py` module for second-variation
objects (e.g., second derivatives of observables w.r.t. deformation coefficients),
or for analyticity/"radius" diagnostics.

The current repository focuses on:
  V(r_*) -> ω_{ℓn} (WKB3) -> finite-difference Jacobian -> SVD/Fisher/MC.

This file provides a minimal, *deterministic* scaffold with explicit
NotImplementedError markers so that downstream code can import it safely and
so that future work has a clear home.

Nothing in the current demo pipeline depends on this module.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class SecondOrderResult:
    """Container for a second-order diagnostic."""
    value: float
    details: Optional[dict] = None


def finite_difference_hessian(
    f: Callable[[np.ndarray], np.ndarray],
    theta0: np.ndarray,
    *,
    delta: float = 1e-3,
) -> np.ndarray:
    """Compute a real Hessian of a real-valued function via finite differences.

    This is a generic utility (deterministic). It is *not* used by default,
    but it is useful for experiments and debugging.

    Parameters
    ----------
    f : callable
        f(theta) -> scalar (float) or 1-element array-like.
    theta0 : np.ndarray
        Expansion point.
    delta : float
        Finite-difference step.

    Returns
    -------
    np.ndarray
        Hessian matrix (shape MxM).
    """
    theta0 = np.asarray(theta0, dtype=float)
    M = int(theta0.size)
    delta = float(delta)
    if delta <= 0:
        raise ValueError("delta must be positive")

    def _as_scalar(v):
        v = np.asarray(v)
        if v.size != 1:
            raise ValueError("f(theta) must be scalar-valued for this helper")
        return float(v.reshape(-1)[0])

    f0 = _as_scalar(f(theta0))
    H = np.zeros((M, M), dtype=float)

    for i in range(M):
        ei = np.zeros(M, dtype=float)
        ei[i] = 1.0
        for j in range(i, M):
            ej = np.zeros(M, dtype=float)
            ej[j] = 1.0

            fp = _as_scalar(f(theta0 + delta * ei + delta * ej))
            fm = _as_scalar(f(theta0 + delta * ei - delta * ej))
            mp = _as_scalar(f(theta0 - delta * ei + delta * ej))
            mm = _as_scalar(f(theta0 - delta * ei - delta * ej))

            Hij = (fp - fm - mp + mm) / (4.0 * delta * delta)
            H[i, j] = Hij
            H[j, i] = Hij

    return H


def analyticity_radius_placeholder(*args, **kwargs) -> SecondOrderResult:
    """Placeholder for an analyticity-radius / second-order diagnostic."""
    raise NotImplementedError(
        "Analyticity/second-order diagnostics are not yet implemented in this repo. "
        "This placeholder exists to satisfy the Standard-B module layout."
    )