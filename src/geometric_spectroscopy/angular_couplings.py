"""Angular couplings for geometric spectroscopy.

In the "full method" pipeline, the *radial* potential V_ℓ(r_*) is perturbed by
a deformation basis labelled by L. If/when one lifts the toy 1D problem to a
more faithful angular decomposition, one needs deterministic angular selection
rules / coupling coefficients.

This module provides **deterministic** helpers based on Wigner-3j symbols.

Important
---------
SymPy is an OPTIONAL dependency here.
- The main demo pipeline does not require angular couplings.
- To avoid breaking minimal installs / CI, SymPy is imported lazily only when
  a Wigner symbol is actually requested.

All functions are pure/deterministic.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Tuple

import numpy as np


def _require_sympy_wigner_3j():
    """Import wigner_3j lazily to keep SymPy optional."""
    try:
        from sympy.physics.wigner import wigner_3j  # type: ignore
    except Exception as e:
        raise ImportError(
            "angular_couplings requires optional dependency 'sympy'.\n"
            "Install it via: pip install sympy\n"
            "This dependency is only needed if you call angular_couplings.* APIs."
        ) from e
    return wigner_3j


@lru_cache(maxsize=None)
def _w3j(j1: int, j2: int, j3: int, m1: int, m2: int, m3: int) -> float:
    """Cached Wigner 3j evaluated as a Python float."""
    wigner_3j = _require_sympy_wigner_3j()
    val = wigner_3j(int(j1), int(j2), int(j3), int(m1), int(m2), int(m3))
    return float(val.evalf())


def scalar_Ylm_coupling(ell: int, L: int, ellp: int) -> float:
    """Scalar spherical-harmonic coupling for m=0 sector.

    For axisymmetric (m=0) scalars, a standard coupling coefficient is:

        C_{ℓ L ℓ'} = sqrt((2ℓ+1)(2L+1)(2ℓ'+1)/(4π)) * (ℓ L ℓ'; 0 0 0)^2

    Returns
    -------
    float
        Deterministic coupling coefficient.
    """
    ell = int(ell)
    L = int(L)
    ellp = int(ellp)

    # Triangle selection rules are enforced by the 3j symbol returning 0.
    threej = _w3j(ell, L, ellp, 0, 0, 0)
    pref = np.sqrt((2 * ell + 1) * (2 * L + 1) * (2 * ellp + 1) / (4.0 * np.pi))
    return float(pref * (threej**2))


def coupling_matrix(ells: Tuple[int, ...], L_max: int) -> np.ndarray:
    """Build a simple coupling tensor for a set of multipoles.

    Returns an array C with shape (len(ells), L_max+1, len(ells)) such that
        C[i, L, j] = C_{ℓ_i, L, ℓ_j}

    Parameters
    ----------
    ells : tuple of int
        Multipoles to include.
    L_max : int
        Maximum deformation multipole.

    Returns
    -------
    np.ndarray
        Coupling tensor (float).
    """
    ells = tuple(int(e) for e in ells)
    L_max = int(L_max)

    C = np.zeros((len(ells), L_max + 1, len(ells)), dtype=float)
    for i, ell in enumerate(ells):
        for j, ellp in enumerate(ells):
            for L in range(L_max + 1):
                C[i, L, j] = scalar_Ylm_coupling(ell, L, ellp)
    return C