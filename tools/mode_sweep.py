import os
import csv
import numpy as np

from geometric_spectroscopy.jacobian import (
    build_jacobian,
    conditioning_phys,
    deformation_weights_phys,
)

GRID = dict(r_min=2.2, r_max=60.0, N_r=5000, N_rstar=5000, profile_width=3.0)
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)


def sweep_modes(model="hayward", L_max=5):

    out = []
    for lmax in range(6, 14):   # ℓ=2..lmax
        ells = range(2, lmax)
        ns = (0, 1, 2)

        J = build_jacobian(
            model=model,
            alpha=0.05,
            L_max=L_max,
            ells=ells,
            ns=ns,
            delta=1e-4,
            grid_kwargs=GRID,
        )

        weights = deformation_weights_phys(model=model, alpha=0.05, L_max=L_max, ells=ells, grid_kwargs=GRID)
        _, _, kappa_phys = conditioning_phys(J, weights)

        Nmodes = len(ells) * len(ns)
        out.append((Nmodes, kappa_phys))
        print(f"Nmodes={Nmodes}, kappa_phys={kappa_phys:.3e}")

    csv_path = os.path.join(RESULTS_DIR, f"{model}_mode_sweep.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Nmodes", "kappa_phys"])
        for row in out:
            w.writerow(row)

    print(f"Saved: {csv_path}")


if __name__ == "__main__":
    sweep_modes("hayward")