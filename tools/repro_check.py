"""Reproducibility checks for the geometric spectroscopy code."""

import numpy as np
from geometric_spectroscopy.jacobian import (
    build_jacobian,
    conditioning_phys,
    deformation_weights_phys,
)


def effective_rank(Jc, tol=1e-6):
    """Calculate effective rank."""
    Jr = np.vstack([Jc.real, Jc.imag])
    s = np.linalg.svd(Jr, compute_uv=False)
    smax = s[0]
    return int(np.sum(s > tol * smax))


def report(model, L_max, ells, ns, grid, delta=1e-3, ps_ells=(2,3)):
    """Build J and return statistics."""
    try:
        J = build_jacobian(
            model=model,
            alpha=0.05,
            L_max=L_max,
            ells=ells,
            ns=ns,
            delta=delta,
            grid_kwargs=grid,
            ps_ells=ps_ells,
        )
        r = effective_rank(J)
        weights = deformation_weights_phys(
            model=model, 
            alpha=0.05, 
            L_max=L_max, 
            ells=ells, 
            grid_kwargs=grid
        )
        _, _, kphys = conditioning_phys(J, weights)
        Nmodes = len(list(ells)) * len(ns)
        return J.shape[0], Nmodes, r, kphys
    except Exception as e:
        print(f"Error in report for {model}: {e}")
        return 0, 0, 0, float('inf')


def main():
    """Run reproducibility checks for all models."""
    GRID_CONFIGS = {
        "hayward": dict(r_min=2.2, r_max=60.0, N_r=4000, profile_width=3.0),
        "bardeen": dict(r_min=4.5, r_max=60.0, N_r=4000, profile_width=3.0),
        "simpson_visser": dict(r_min=3.5, r_max=60.0, N_r=4000, profile_width=3.0),
    }
    
    L_max = 5

    for model in ["hayward", "bardeen", "simpson_visser"]:
        print(f"\n=== {model.upper()} ===")
        grid = GRID_CONFIGS[model]
        
        # Small set: ℓ=2..5, n=0,1
        small = report(model, L_max, ells=range(2, 6), ns=(0, 1), grid=grid)
        # Large set: ℓ=2..12, n=0,1,2
        large = report(model, L_max, ells=range(2, 13), ns=(0, 1, 2), grid=grid)

        print("Small set: Ndata, Nmodes, rank, kappa_phys =", small)
        print("Large set: Ndata, Nmodes, rank, kappa_phys =", large)

        # Check for determinism: two runs should be identical
        J1 = build_jacobian(
            model=model, 
            alpha=0.05, 
            L_max=5, 
            ells=range(2,6), 
            ns=(0,1), 
            grid_kwargs=grid
        )
        J2 = build_jacobian(
            model=model, 
            alpha=0.05, 
            L_max=5, 
            ells=range(2,6), 
            ns=(0,1), 
            grid_kwargs=grid
        )
        is_det = np.allclose(J1, J2, atol=0.0, rtol=0.0)
        print(f"Jacobian deterministic: {is_det}")


if __name__ == "__main__":
    main()