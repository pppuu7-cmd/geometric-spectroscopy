"""
Padé approximants for a truncated power series with diagnostics.

We consider a formal series:
  f(eps) = sum_{k=0..K} c_k eps^k.

We build a Padé approximant [m/n]:
  P_m(eps) / Q_n(eps),
where:
  P_m(eps) = a0 + a1 eps + ... + a_m eps^m
  Q_n(eps) = 1 + b1 eps + ... + b_n eps^n

such that the Maclaurin expansion matches f(eps) up to order m+n:
  Q_n(eps) * f(eps) - P_m(eps) = O(eps^{m+n+1})

This module is intentionally standalone so it can be reused by WKB engines.

Diagnostics:
- Solving system conditioning (rough)
- Poles of Q_n (roots)
- Distance of nearest pole to eps=1 (or provided eval point)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class PadeResult:
    value: complex
    m: int
    n: int
    a: np.ndarray  # (m+1,) complex
    b: np.ndarray  # (n+1,) complex with b[0]=1
    poles: np.ndarray  # roots of Q_n
    eval_point: complex
    solve_method: str
    cond_est: float
    pole_distance: float
    stable: bool


def _poly_roots_desc(coeffs_desc: np.ndarray) -> np.ndarray:
    """Roots of polynomial given coefficients in descending powers."""
    coeffs_desc = np.asarray(coeffs_desc, dtype=complex)
    # Strip leading zeros if any
    i = 0
    while i < coeffs_desc.size and abs(coeffs_desc[i]) == 0:
        i += 1
    if i == coeffs_desc.size:
        return np.asarray([], dtype=complex)
    coeffs_desc = coeffs_desc[i:]
    if coeffs_desc.size <= 1:
        return np.asarray([], dtype=complex)
    return np.roots(coeffs_desc)


def pade_build(coeffs: List[complex], m: int, n: int) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    """
    Build Padé [m/n] coefficients (a, b) for a series coeffs c0..c_{m+n}.

    Returns:
      a: (m+1,) complex
      b: (n+1,) complex with b[0]=1
      stats: dict with keys: cond_est, solve_method
    """
    m = int(m)
    n = int(n)
    if m < 0 or n < 0:
        raise ValueError("m and n must be >= 0")
    if n == 0:
        # Pure polynomial
        if len(coeffs) < m + 1:
            raise ValueError("Need coefficients c0..c_m")
        a = np.asarray(coeffs[: m + 1], dtype=complex)
        b = np.zeros(1, dtype=complex)
        b[0] = 1.0 + 0j
        return a, b, {"cond_est": 1.0, "solve_method": "trivial"}

    need = m + n + 1
    if len(coeffs) < need:
        raise ValueError(f"Need coefficients c0..c_{m+n} (need {need}, got {len(coeffs)})")

    c = np.asarray(coeffs[:need], dtype=complex)

    # Solve for b1..bn from matching conditions for k = m+1..m+n:
    # c_k + sum_{j=1..n} b_j c_{k-j} = 0
    # => sum_{j=1..n} b_j c_{k-j} = -c_k
    # Matrix A has rows indexed by k=m+1..m+n and columns j=1..n:
    # A[row, j-1] = c_{k-j}
    A = np.zeros((n, n), dtype=complex)
    rhs = np.zeros((n,), dtype=complex)
    for row, k in enumerate(range(m + 1, m + n + 1)):
        rhs[row] = -c[k]
        for j in range(1, n + 1):
            A[row, j - 1] = c[k - j]

    # conditioning estimate (rough)
    try:
        cond_est = float(np.linalg.cond(A))
    except Exception:
        cond_est = float("inf")

    # Solve
    solve_method = "solve"
    try:
        b_tail = np.linalg.solve(A, rhs)
    except np.linalg.LinAlgError:
        solve_method = "lstsq"
        b_tail = np.linalg.lstsq(A, rhs, rcond=None)[0]

    b = np.zeros((n + 1,), dtype=complex)
    b[0] = 1.0 + 0j
    b[1:] = b_tail

    # Compute numerator a0..a_m:
    # a_i = sum_{j=0..min(i,n)} b_j c_{i-j} with b0=1
    a = np.zeros((m + 1,), dtype=complex)
    for i in range(m + 1):
        s = 0.0 + 0j
        jmax = min(i, n)
        for j in range(jmax + 1):
            s += b[j] * c[i - j]
        a[i] = s

    return a, b, {"cond_est": cond_est, "solve_method": solve_method}


def pade_eval(coeffs: List[complex], m: int, n: int, eps: complex = 1.0, *, pole_guard: float = 0.15) -> PadeResult:
    """
    Evaluate Padé [m/n] at eps with diagnostics.

    pole_guard:
      threshold for considering the approximant unstable if a pole is too close to eps.
      stable := min|pole-eps| >= pole_guard
    """
    a, b, stats = pade_build(coeffs, m=m, n=n)
    eps = complex(eps)

    # Evaluate numerator/denominator
    num = 0.0 + 0j
    for i in range(a.size - 1, -1, -1):
        num = num * eps + a[i]

    den = 0.0 + 0j
    for i in range(b.size - 1, -1, -1):
        den = den * eps + b[i]

    value = num / den

    # Poles (roots of Q_n)
    # Q_n(eps) = 1 + b1 eps + ... + bn eps^n
    # np.roots expects descending powers: [bn, ..., b1, b0]
    q_desc = np.asarray(b[::-1], dtype=complex)
    poles = _poly_roots_desc(q_desc)

    if poles.size > 0:
        pole_distance = float(np.min(np.abs(poles - eps)))
    else:
        pole_distance = float("inf")

    stable = bool(pole_distance >= float(pole_guard))

    return PadeResult(
        value=complex(value),
        m=int(m),
        n=int(n),
        a=a,
        b=b,
        poles=poles,
        eval_point=eps,
        solve_method=str(stats.get("solve_method", "")),
        cond_est=float(stats.get("cond_est", float("inf"))),
        pole_distance=pole_distance,
        stable=stable,
    )


def pade_ensemble(
    coeffs: List[complex],
    approximants: List[Tuple[int, int]],
    eps: complex = 1.0,
    *,
    pole_guard: float = 0.15,
) -> Dict[str, object]:
    """
    Evaluate multiple Padé approximants and return an ensemble summary:
      - results: list of PadeResult
      - mean_value: mean of stable approximants (if any)
      - std_value: std of stable approximants (if >=2)
      - stable_count / total_count
      - worst_pole_distance among stable ones
    """
    results: List[PadeResult] = []
    for (m, n) in approximants:
        results.append(pade_eval(coeffs, m=m, n=n, eps=eps, pole_guard=pole_guard))

    stable = [r for r in results if r.stable and np.isfinite(r.value.real) and np.isfinite(r.value.imag)]
    total = len(results)
    stable_count = len(stable)

    if stable_count == 0:
        return {
            "results": results,
            "mean_value": None,
            "std_value": None,
            "stable_count": 0,
            "total_count": total,
            "worst_pole_distance": None,
        }

    vals = np.asarray([r.value for r in stable], dtype=complex)
    mean_value = complex(np.mean(vals))
    std_value = complex(np.std(vals)) if stable_count >= 2 else 0.0 + 0j
    worst_pole_distance = float(np.min([r.pole_distance for r in stable]))

    return {
        "results": results,
        "mean_value": mean_value,
        "std_value": std_value,
        "stable_count": stable_count,
        "total_count": total,
        "worst_pole_distance": worst_pole_distance,
    }