import os
import csv
import numpy as np

from geometric_spectroscopy.jacobian import (
    build_jacobian,
    conditioning_phys,
    deformation_weights_phys,
)

# NOTE:
# build_potential_rstar() принимает N_r (число точек сетки по r).
# Параметра N_rstar в текущем API нет, поэтому не передаем его в grid_kwargs.
GRID = dict(r_min=2.2, r_max=60.0, N_r=5000, profile_width=3.0)

RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)


def sweep_modes(model: str = "hayward", L_max: int = 5) -> None:
    """
    Sweep over number of modes (by increasing max ell) and record kappa_phys.

    This is an exploratory tool (not part of the main paper reproduction pipeline).
    Output: results/<model>_mode_sweep.csv
    """
    print(f"[mode_sweep] model={model}, L_max={L_max}")
    print(f"[mode_sweep] GRID={GRID}")

    out = []
    for lmax in range(6, 14):  # use ell = 2..(lmax-1)
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

        weights = deformation_weights_phys(
            model=model,
            alpha=0.05,
            L_max=L_max,
            ells=ells,
            grid_kwargs=GRID,
        )
        _, _, kappa_phys = conditioning_phys(J, weights)

        Nmodes = len(tuple(ells)) * len(ns)
        out.append((Nmodes, float(kappa_phys)))
        print(f"Nmodes={Nmodes}, kappa_phys={kappa_phys:.3e}")

    csv_path = os.path.join(RESULTS_DIR, f"{model}_mode_sweep.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Nmodes", "kappa_phys"])
        w.writerows(out)

    print(f"Saved: {csv_path}")


if __name__ == "__main__":
    sweep_modes("hayward")