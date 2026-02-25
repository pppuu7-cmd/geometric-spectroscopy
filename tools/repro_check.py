"""Reproducibility checks for the geometric spectroscopy code.

This script is meant to be *paper-aligned*:
- It runs the Standard-B Jacobian/QNM pipeline in a stable configuration.
- It reports (and does NOT crash) when single-barrier/WKB assumptions fail
  for a configuration, since such cases are explicitly out-of-scope in the manuscript.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple, Optional

import numpy as np

from geometric_spectroscopy.jacobian import (
    build_jacobian,
    conditioning_phys,
    deformation_weights_phys,
)


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def effective_rank(Jc: np.ndarray, tol: float = 1e-6) -> int:
    """Effective numerical rank for complex Jacobian via realification."""
    Jr = np.vstack([Jc.real, Jc.imag])
    s = np.linalg.svd(Jr, compute_uv=False)
    smax = float(s[0]) if len(s) else 0.0
    if smax <= 0:
        return 0
    return int(np.sum(s > float(tol) * smax))


@dataclass(frozen=True)
class Report:
    ok: bool
    message: str
    Ndata: int
    Nmodes: int
    rank_eff: int
    kappa_phys: float
    deterministic: Optional[bool] = None


def _nmodes(ells: Iterable[int], ns: Tuple[int, ...]) -> int:
    return len(list(ells)) * len(tuple(ns))


def report(
    *,
    model: str,
    alpha: float,
    L_max: int,
    ells: Iterable[int],
    ns: Tuple[int, ...],
    grid_kwargs: dict,
    ps_ells: Tuple[int, ...] = (2, 3),
    order: int = 6,
    delta: float = 1e-3,
) -> Report:
    """
    Build J and return paper-relevant diagnostics.

    If the WKB single-barrier assumptions fail (e.g. V''>=0 at the detected peak),
    we return ok=False with a clear message (no crash).
    """
    Nm = _nmodes(ells, ns)

    try:
        J = build_jacobian(
            model=model,
            alpha=alpha,
            L_max=L_max,
            ells=ells,
            ns=ns,
            order=int(order),
            delta=float(delta),
            grid_kwargs=grid_kwargs,
            ps_ells=ps_ells,
        )

        r_eff = effective_rank(J, tol=1e-6)

        weights = deformation_weights_phys(
            model=model,
            alpha=alpha,
            L_max=L_max,
            ells=ells,
            grid_kwargs=grid_kwargs,
        )
        _, _, kphys = conditioning_phys(J, weights)

        # Determinism check (exact equality expected)
        J2 = build_jacobian(
            model=model,
            alpha=alpha,
            L_max=L_max,
            ells=ells,
            ns=ns,
            order=int(order),
            delta=float(delta),
            grid_kwargs=grid_kwargs,
            ps_ells=ps_ells,
        )
        is_det = bool(np.allclose(J, J2, rtol=0.0, atol=0.0))

        return Report(
            ok=True,
            message="OK",
            Ndata=int(J.shape[0]),
            Nmodes=int(Nm),
            rank_eff=int(r_eff),
            kappa_phys=float(kphys),
            deterministic=is_det,
        )

    except Exception as e:
        # Paper-aligned behavior:
        # failures are *reported* (often indicate out-of-scope or guardrail violation),
        # but the script continues for other models/sets.
        return Report(
            ok=False,
            message=str(e),
            Ndata=0,
            Nmodes=int(Nm),
            rank_eff=0,
            kappa_phys=float("inf"),
            deterministic=None,
        )


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main() -> None:
    # Paper/README-aligned default configuration:
    # Standard-B: order=6 (WKB6), alpha=0.05, L_max=5, extras via ps_ells=(2,3).
    alpha = 0.05
    L_max = 5
    order = 6
    ps_ells = (2, 3)

    # Use a single stable grid across models (matches CI smoke-test spirit).
    # IMPORTANT: do NOT push r_min above the barrier region; otherwise WKB peak detection can fail.
    GRID = dict(r_min=2.2, r_max=60.0, N_r=4000, profile_width=3.0)

    print("[repro_check] Configuration:")
    print(f"  alpha={alpha}, L_max={L_max}, order={order}, ps_ells={ps_ells}")
    print(f"  GRID={GRID}")

    models = ["hayward", "bardeen", "simpson_visser"]

    for model in models:
        print(f"\n=== {model.upper()} ===")

        # Small set: ℓ=2..5, n=0,1  (CI-friendly)
        small = report(
            model=model,
            alpha=alpha,
            L_max=L_max,
            ells=range(2, 6),
            ns=(0, 1),
            grid_kwargs=GRID,
            ps_ells=ps_ells,
            order=order,
            delta=1e-3,
        )

        # Large set: ℓ=2..12, n=0,1,2  (paper-like “standard” data size)
        large = report(
            model=model,
            alpha=alpha,
            L_max=L_max,
            ells=range(2, 13),
            ns=(0, 1, 2),
            grid_kwargs=GRID,
            ps_ells=ps_ells,
            order=order,
            delta=1e-3,
        )

        def _fmt(tag: str, rep: Report) -> None:
            if rep.ok:
                print(
                    f"{tag}: Ndata={rep.Ndata}, Nmodes={rep.Nmodes}, "
                    f"rank_eff={rep.rank_eff}, kappa_phys={rep.kappa_phys:.6g}, "
                    f"deterministic={rep.deterministic}"
                )
            else:
                print(
                    f"{tag}: FAILED (out-of-scope/guardrail)\n"
                    f"  reason: {rep.message}"
                )

        _fmt("Small set", small)
        _fmt("Large set", large)


if __name__ == "__main__":
    main()