"""Main demo script for all three metrics.

This script is also used by an integration test (pytest) which can override
grid sizes and mode ranges via environment variables to keep CI fast.
"""

import os
import csv
import numpy as np

from geometric_spectroscopy.jacobian import (
    build_jacobian,
    conditioning_raw,
    conditioning_geom,
    conditioning_phys,
    deformation_weights_phys,
)
from geometric_spectroscopy.monte_carlo import fisher_sigma_theta, monte_carlo_theta


def _env_int(name: str, default: int) -> int:
    v = os.getenv(name, "").strip()
    if not v:
        return default
    try:
        return int(v)
    except Exception:
        return default


def _env_float(name: str, default: float) -> float:
    v = os.getenv(name, "").strip()
    if not v:
        return default
    try:
        return float(v)
    except Exception:
        return default


# Switch WKB order here (3 or 6). Final version: 6th-order WKB + Padé[3/3]
WKB_ORDER = _env_int("GS_WKB_ORDER", 6)

# CI/smoke knobs (defaults are your normal “real” demo values)
DEFAULT_N_R = _env_int("GS_N_R", 6000)
DEFAULT_R_MAX = _env_float("GS_R_MAX", 60.0)
DEFAULT_PROFILE_WIDTH = _env_float("GS_PROFILE_WIDTH", 3.0)
MC_REALIZATIONS = _env_int("GS_MC_REALIZATIONS", 300)

# Mode-range knobs
# By default: ells=2..12 and ns=(0,1,2)
ELL_MAX = _env_int("GS_ELL_MAX", 12)  # inclusive
NS_MAX = _env_int("GS_NS_MAX", 2)     # inclusive

# Grid configurations for different models
GRID_CONFIGS = {
    "hayward": dict(r_min=2.2, r_max=DEFAULT_R_MAX, N_r=DEFAULT_N_R, profile_width=DEFAULT_PROFILE_WIDTH),
    "bardeen": dict(r_min=2.2, r_max=DEFAULT_R_MAX, N_r=DEFAULT_N_R, profile_width=DEFAULT_PROFILE_WIDTH),
    "simpson_visser": dict(r_min=2.2, r_max=DEFAULT_R_MAX, N_r=DEFAULT_N_R, profile_width=DEFAULT_PROFILE_WIDTH),
}

RESULTS_DIR = os.getenv("GS_RESULTS_DIR", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# Configuration
PS_ELLS = (2, 3)
ELLS = range(2, max(2, ELL_MAX) + 1)  # 2..ELL_MAX
NS = tuple(range(0, max(0, NS_MAX) + 1))  # 0..NS_MAX
L_MAX = 5
ALPHA_TRUE = 0.05


def effective_rank(Jc, tol=1e-6):
    """Calculate effective rank based on singular values."""
    Jr = np.vstack([Jc.real, Jc.imag])
    s = np.linalg.svd(Jr, compute_uv=False)
    smax = s[0]
    return int(np.sum(s > tol * smax)), s


def run_model(model: str):
    """Run complete analysis for a single model."""
    print(f"\n=== {model.upper()} ===")

    # Use model-specific grid parameters
    grid_kwargs = GRID_CONFIGS[model]

    # Build the Jacobian
    J = build_jacobian(
        model=model,
        alpha=ALPHA_TRUE,
        L_max=L_MAX,
        ells=ELLS,
        ns=NS,
        order=WKB_ORDER,
        delta=1e-3,
        grid_kwargs=grid_kwargs,
        ps_ells=PS_ELLS,
    )

    # Conditioning analyses
    smax_raw, smin_raw, kappa_raw = conditioning_raw(J)
    (smax_geom, smin_geom, kappa_geom), _col_norms = conditioning_geom(J)
    weights = deformation_weights_phys(
        model=model,
        alpha=ALPHA_TRUE,
        L_max=L_MAX,
        ells=ELLS,
        grid_kwargs=grid_kwargs,
    )
    smax_phys, smin_phys, kappa_phys = conditioning_phys(J, weights)

    rank_eff, singular_vals = effective_rank(J)

    # Print summary
    Nmodes = len(list(ELLS)) * len(NS)
    Nextra = 5 * len(PS_ELLS)
    print(f"Nmodes={Nmodes} + extras={Nextra} (ps_ells={PS_ELLS}) => Ndata={J.shape[0]}")
    print(f"WKB order   = {WKB_ORDER}")
    print(f"kappa_raw  = {kappa_raw:.3e}")
    print(f"kappa_geom = {kappa_geom:.3e}")
    print(f"kappa_phys = {kappa_phys:.3e}")
    print(f"Effective rank = {rank_eff} / {L_MAX+1}")

    # Fisher and Monte Carlo for full Theta vector
    theta_true = np.zeros(L_MAX + 1)
    theta_true[0] = ALPHA_TRUE

    sigma_omega = 0.01
    ridge = -1.0  # auto ridge
    sig_fisher = fisher_sigma_theta(J, sigma_omega=sigma_omega, ridge=ridge)
    mc_mean, mc_std = monte_carlo_theta(
        J,
        theta_true=theta_true,
        sigma_omega=sigma_omega,
        realizations=MC_REALIZATIONS,
        seed=1,
        ridge=ridge,
    )

    # Save results to CSV
    out_path = os.path.join(RESULTS_DIR, f"{model}_summary.csv")
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", model])
        w.writerow(["alpha_true", ALPHA_TRUE])
        w.writerow(["L_max", L_MAX])
        w.writerow(["delta", 1e-3])
        w.writerow(["wkb_order", WKB_ORDER])
        w.writerow(["sigma_omega", sigma_omega])
        w.writerow(["ells", f"{min(ELLS)}..{max(ELLS)}"])
        w.writerow(["ns", ",".join(str(x) for x in NS)])
        w.writerow(["ps_ells", ",".join(str(x) for x in PS_ELLS)])
        w.writerow(["N_r", grid_kwargs["N_r"]])
        w.writerow(["r_max", grid_kwargs["r_max"]])
        w.writerow(["profile_width", grid_kwargs.get("profile_width", "")])
        w.writerow(["mc_realizations", MC_REALIZATIONS])
        w.writerow([])

        w.writerow(["kappa_raw", kappa_raw])
        w.writerow(["kappa_geom", kappa_geom])
        w.writerow(["kappa_phys", kappa_phys])
        w.writerow(["effective_rank", rank_eff])
        w.writerow([])

        w.writerow(["singular_values"])
        w.writerow(list(singular_vals))
        w.writerow([])

        w.writerow(["param", "theta_true", "fisher_sigma", "mc_mean", "mc_std"])
        for i in range(L_MAX + 1):
            w.writerow([i, theta_true[i], sig_fisher[i], mc_mean[i], mc_std[i]])

    print(f"Saved: {out_path}")


def main():
    for model in ("hayward", "bardeen", "simpson_visser"):
        run_model(model)


if __name__ == "__main__":
    main()