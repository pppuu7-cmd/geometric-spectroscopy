"""
Stability / robustness layer (Standard B).

Goal
----
Provide a programmatic way to *quantify* stability of the full pipeline, not just
produce a single Jacobian once.

This module implements "error bars" and sanity monitors across:
  - finite-difference step (delta),
  - grid settings (N_r, r_max, profile_width, ...),
  - optional mode ranges (ells, ns, ps_ells).

This is *not* trying to beat high-precision QNM tables; instead it is designed to
make the inference chain scientifically reliable and regression-resistant.

Outputs
-------
We report spread of:
  - condition numbers (raw / geom / phys),
  - effective rank,
  - "damped" sanity check for QNMs (Im ω < 0 fraction).

All functions are deterministic and do not rely on randomness.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from .jacobian import (
    build_jacobian,
    conditioning_raw,
    conditioning_geom,
    conditioning_phys,
    deformation_weights_phys,
    effective_rank,
)
from .resonances import compute_qnms


@dataclass(frozen=True)
class StabilityPoint:
    """One evaluation point in a stability scan."""
    delta: float
    grid_kwargs: Dict[str, float]

    # Condition numbers (kappa) in different parameterizations
    kappa_raw: float
    kappa_geom: float
    kappa_phys: float

    # Rank proxy
    rank_eff: float

    # Sanity monitor
    damped_fraction: float
    max_imag_omega: float  # should be < 0 in typical cases


@dataclass(frozen=True)
class StabilitySummary:
    """Aggregated summary over StabilityPoint list."""
    n: int
    points: Tuple[StabilityPoint, ...]

    kappa_raw_med: float
    kappa_raw_rel_spread: float

    kappa_geom_med: float
    kappa_geom_rel_spread: float

    kappa_phys_med: float
    kappa_phys_rel_spread: float

    rank_eff_med: float
    rank_eff_rel_spread: float

    damped_fraction_min: float
    max_imag_omega_max: float


def _finite_or_nan(x: float) -> float:
    x = float(x)
    return x if np.isfinite(x) else float("nan")


def _rel_spread(values: Sequence[float], eps: float = 1e-30) -> float:
    """
    Relative spread: (p90 - p10) / median.
    Robust enough to ignore occasional outliers.
    """
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return float("nan")
    med = float(np.median(v))
    p10 = float(np.percentile(v, 10.0))
    p90 = float(np.percentile(v, 90.0))
    return float((p90 - p10) / max(abs(med), eps))


def _damped_monitor(
    model: str,
    alpha: float,
    theta0: np.ndarray,
    L_max: int,
    ells: Iterable[int],
    ns: Iterable[int],
    *,
    order: int,
    grid_kwargs: Dict,
) -> Tuple[float, float]:
    """
    Compute fraction of modes with Im ω < 0, and max(Im ω).
    """
    w = compute_qnms(
        model=model,
        alpha=alpha,
        theta=theta0,
        L_max=L_max,
        ells=ells,
        ns=ns,
        order=order,
        grid_kwargs=grid_kwargs,
    )
    im = np.imag(w)
    damped_fraction = float(np.mean(im < 0.0)) if im.size else 1.0
    max_imag = float(np.max(im)) if im.size else float("-inf")
    return damped_fraction, max_imag


def stability_point(
    *,
    model: str,
    alpha: float,
    L_max: int,
    ells=range(2, 8),
    ns=(0, 1),
    ps_ells=(2, 3),
    order: int = 6,
    delta: float = 1e-3,
    grid_kwargs: Optional[Dict] = None,
    theta0: Optional[np.ndarray] = None,
) -> StabilityPoint:
    """
    Single deterministic evaluation of stability metrics.

    This is the core primitive used by stability_scan().
    """
    if grid_kwargs is None:
        grid_kwargs = {}

    order = int(order)
    if order not in (3, 6):
        raise ValueError("order must be 3 or 6")

    if theta0 is None:
        theta0 = np.zeros(L_max + 1, dtype=float)
    else:
        theta0 = np.asarray(theta0, dtype=float)
        if theta0.shape != (L_max + 1,):
            raise ValueError(f"theta0 must have shape ({L_max+1},)")

    # Jacobian
    J = build_jacobian(
        model=model,
        alpha=alpha,
        L_max=L_max,
        ells=ells,
        ns=ns,
        order=order,
        delta=float(delta),
        theta0=theta0,
        grid_kwargs=grid_kwargs,
        ps_ells=ps_ells,
    )

    # Conditioning: raw, geom, phys
    _smax, _smin, k_raw = conditioning_raw(J)
    ( _g_smax, _g_smin, k_geom ), _col_norms = conditioning_geom(J)

    w_phys = deformation_weights_phys(
        model=model,
        alpha=alpha,
        L_max=L_max,
        ells=ells,
        grid_kwargs=grid_kwargs,
    )
    _p_smax, _p_smin, k_phys = conditioning_phys(J, w_phys)

    # Effective rank
    r_eff = effective_rank(J)

    # Damped monitor (modes only — extras do not affect damping)
    damped_fraction, max_imag = _damped_monitor(
        model=model,
        alpha=alpha,
        theta0=theta0,
        L_max=L_max,
        ells=ells,
        ns=ns,
        order=order,
        grid_kwargs=grid_kwargs,
    )

    return StabilityPoint(
        delta=float(delta),
        grid_kwargs=dict(grid_kwargs),
        kappa_raw=_finite_or_nan(k_raw),
        kappa_geom=_finite_or_nan(k_geom),
        kappa_phys=_finite_or_nan(k_phys),
        rank_eff=_finite_or_nan(r_eff),
        damped_fraction=_finite_or_nan(damped_fraction),
        max_imag_omega=_finite_or_nan(max_imag),
    )


def stability_scan(
    *,
    model: str,
    alpha: float,
    L_max: int,
    ells=range(2, 8),
    ns=(0, 1),
    ps_ells=(2, 3),
    order: int = 6,
    deltas: Sequence[float] = (1e-3, 5e-4, 2e-3),
    grids: Optional[Sequence[Dict]] = None,
    theta0: Optional[np.ndarray] = None,
) -> StabilitySummary:
    """
    Evaluate stability over a small grid of (delta, grid_kwargs).

    Default grids are intentionally small but cover typical sources of regressions.

    The function is deterministic and suitable for CI smoke checks
    (if the scan size is kept small).
    """
    if grids is None:
        grids = (
            dict(r_min=2.2, r_max=50.0, N_r=2000, profile_width=3.0),
            dict(r_min=2.2, r_max=50.0, N_r=2500, profile_width=3.0),
            dict(r_min=2.2, r_max=60.0, N_r=2000, profile_width=3.0),
        )

    pts: List[StabilityPoint] = []
    for g in grids:
        for d in deltas:
            pts.append(
                stability_point(
                    model=model,
                    alpha=alpha,
                    L_max=L_max,
                    ells=ells,
                    ns=ns,
                    ps_ells=ps_ells,
                    order=order,
                    delta=float(d),
                    grid_kwargs=dict(g),
                    theta0=theta0,
                )
            )

    k_raw = [p.kappa_raw for p in pts]
    k_geom = [p.kappa_geom for p in pts]
    k_phys = [p.kappa_phys for p in pts]
    r_eff = [p.rank_eff for p in pts]

    damped_min = float(np.min([p.damped_fraction for p in pts])) if pts else 1.0
    max_imag_max = float(np.max([p.max_imag_omega for p in pts])) if pts else float("-inf")

    def _med(v: Sequence[float]) -> float:
        vv = np.asarray(v, dtype=float)
        vv = vv[np.isfinite(vv)]
        return float(np.median(vv)) if vv.size else float("nan")

    return StabilitySummary(
        n=len(pts),
        points=tuple(pts),

        kappa_raw_med=_med(k_raw),
        kappa_raw_rel_spread=_rel_spread(k_raw),

        kappa_geom_med=_med(k_geom),
        kappa_geom_rel_spread=_rel_spread(k_geom),

        kappa_phys_med=_med(k_phys),
        kappa_phys_rel_spread=_rel_spread(k_phys),

        rank_eff_med=_med(r_eff),
        rank_eff_rel_spread=_rel_spread(r_eff),

        damped_fraction_min=_finite_or_nan(damped_min),
        max_imag_omega_max=_finite_or_nan(max_imag_max),
    )


def format_stability_summary(summary: StabilitySummary) -> str:
    """
    Human-readable summary (for logs / README snippets / CLI prints).
    """
    s = summary
    lines = [
        f"Stability scan: N={s.n}",
        f"  κ_raw  : median={s.kappa_raw_med:.3e}, rel_spread(p90-p10)/med={s.kappa_raw_rel_spread:.3e}",
        f"  κ_geom : median={s.kappa_geom_med:.3e}, rel_spread(p90-p10)/med={s.kappa_geom_rel_spread:.3e}",
        f"  κ_phys : median={s.kappa_phys_med:.3e}, rel_spread(p90-p10)/med={s.kappa_phys_rel_spread:.3e}",
        f"  rank_eff: median={s.rank_eff_med:.3f}, rel_spread={s.rank_eff_rel_spread:.3e}",
        f"  damped_fraction_min={s.damped_fraction_min:.3f}",
        f"  max(Im ω)_max={s.max_imag_omega_max:.3e}",
    ]
    return "\n".join(lines)


def assert_stability(
    summary: StabilitySummary,
    *,
    max_kappa_rel_spread: float = 0.30,
    min_damped_fraction: float = 1.0,
    require_max_imag_negative: bool = True,
) -> None:
    """
    Optional guardrail (can be used in CI or by users).

    We intentionally default to conservative thresholds:
      - κ spreads should not wildly jump for small delta/grid tweaks
      - all tested modes should remain damped
    """
    if not np.isfinite(summary.kappa_phys_rel_spread):
        raise ValueError("Stability: κ_phys_rel_spread is not finite")

    if summary.kappa_phys_rel_spread > float(max_kappa_rel_spread):
        raise ValueError(
            f"Stability regression: κ_phys_rel_spread={summary.kappa_phys_rel_spread:.3f} "
            f"> {max_kappa_rel_spread:.3f}"
        )

    if summary.damped_fraction_min + 1e-15 < float(min_damped_fraction):
        raise ValueError(
            f"Stability regression: damped_fraction_min={summary.damped_fraction_min:.3f} "
            f"< {min_damped_fraction:.3f}"
        )

    if require_max_imag_negative and (summary.max_imag_omega_max >= 0.0):
        raise ValueError(
            f"Stability regression: max(Im ω)_max={summary.max_imag_omega_max:.3e} (expected < 0)"
        )