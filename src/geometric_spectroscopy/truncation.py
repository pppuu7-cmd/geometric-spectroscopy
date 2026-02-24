"""Truncation utilities for deformation bases.

In Standard-B, the deformation is represented by a finite set of coefficients
θ_L (L=0..L_max) multiplying compact profiles Δ_L(r_*). Any practical
implementation must understand what happens when the basis is truncated.

This module provides **deterministic** (and intentionally conservative)
bookkeeping for truncation error estimates in the *physical* deformation norm:

    ||δV||^2_phys = Σ_L w_L θ_L^2,
    where w_L = ∫ |Δ_L(r_*)|^2 dr_*.

The weights w_L are already produced by potentials.build_potential_rstar().
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np


def truncation_tail_norm(theta: np.ndarray, weights: np.ndarray, *, L_trunc: int) -> float:
    """Estimate the physical-norm size of the truncated tail.

    Parameters
    ----------
    theta : np.ndarray
        Deformation coefficients θ_L for L=0..L_max.
    weights : np.ndarray
        Physical weights w_L = ∫ |Δ_L|^2 dr_* for the *same* L range.
    L_trunc : int
        Keep modes up to L_trunc; estimate tail from L_trunc+1..L_max.

    Returns
    -------
    float
        sqrt( Σ_{L>L_trunc} w_L θ_L^2 )
    """
    theta = np.asarray(theta, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if theta.shape != weights.shape:
        raise ValueError("theta and weights must have the same shape")
    L_trunc = int(L_trunc)
    if L_trunc < -1 or L_trunc >= theta.size:
        raise ValueError("Invalid L_trunc")

    tail = theta[(L_trunc + 1) :]
    wtail = weights[(L_trunc + 1) :]
    val = float(np.sqrt(np.sum(np.maximum(wtail, 0.0) * (tail * tail))))
    return val


def choose_L_trunc_by_fraction(weights: np.ndarray, *, frac_keep: float = 0.99) -> int:
    """Choose a truncation L_trunc so that cumulative weight >= frac_keep.

    This is a heuristic, but deterministic and useful for quick experiments.

    Parameters
    ----------
    weights : np.ndarray
        Physical weights w_L (nonnegative).
    frac_keep : float
        Target fraction of total weight to keep.

    Returns
    -------
    int
        Smallest L_trunc such that Σ_{L<=L_trunc} w_L >= frac_keep Σ_{all} w_L.
    """
    weights = np.asarray(weights, dtype=float)
    frac_keep = float(frac_keep)
    if not (0.0 < frac_keep <= 1.0):
        raise ValueError("frac_keep must be in (0,1]")

    w = np.maximum(weights, 0.0)
    total = float(np.sum(w))
    if total == 0.0:
        return int(weights.size - 1)

    cum = np.cumsum(w)
    target = frac_keep * total
    idx = int(np.searchsorted(cum, target))
    return min(max(idx, 0), int(weights.size - 1))