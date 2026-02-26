# jacobian_L_scan_dense.py
# ------------------------------------------------------------
# Dense scan over L_max to study conditioning growth.
# Produces (ALWAYS in repo_root/artifacts/order_stability/):
#   jacobian_vs_L_dense.csv
#   cond_growth_fit.csv
#   cond_vs_L_fit_log10.png  (if matplotlib installed)
#   cond_vs_L_fit_ln.png     (if matplotlib installed)
#
# Key change vs previous version:
#   Output directory is anchored at the REPO ROOT (not CWD),
#   so running from tools/ or anywhere writes to the same place
#   as tikhonov_scan.py (repo_root/artifacts/order_stability).
# ------------------------------------------------------------

from __future__ import annotations

import csv
import os
import time
from pathlib import Path

import numpy as np

# matplotlib optional
try:
    import matplotlib.pyplot as plt
    HAVE_PLT = True
except Exception:
    HAVE_PLT = False


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def write_csv(path: str, rows: list[dict]) -> None:
    """Write list of dicts to CSV, safely handling heterogeneous keys."""
    if not rows:
        print("No rows.")
        return

    # Union of all keys, stable order of first appearance.
    fieldnames: list[str] = []
    seen = set()
    for r in rows:
        for k in r.keys():
            if k not in seen:
                seen.add(k)
                fieldnames.append(k)

    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    print(f"[ok] wrote {path}")


def import_mod(path: str):
    __import__(path)
    import sys
    return sys.modules[path]


def compute_svd(J: np.ndarray):
    s = np.linalg.svd(J, compute_uv=False)
    smax = float(s[0])
    smin = float(s[-1])
    cond = float(smax / smin) if smin > 0 else float("inf")
    rank = int(np.sum(s > 1e-12 * smax))
    return rank, smax, smin, cond


def linear_fit(x, y):
    """
    Fit y = m x + b via least squares.
    Returns (m, b, SSE, R2).
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    A = np.vstack([x, np.ones_like(x)]).T
    m, b = np.linalg.lstsq(A, y, rcond=None)[0]
    yhat = m * x + b
    resid = y - yhat
    sse = float(np.sum(resid ** 2))
    sst = float(np.sum((y - np.mean(y)) ** 2)) if len(y) > 1 else 0.0
    r2 = 1.0 - sse / sst if sst > 0 else float("nan")
    return float(m), float(b), sse, r2


def find_repo_root(start: Path) -> Path:
    """
    Find repo root by walking upwards until we find a folder
    that contains 'geometric_spectroscopy' package directory.
    Falls back to parent of this script if not found.
    """
    for p in [start] + list(start.parents):
        if (p / "geometric_spectroscopy").is_dir():
            return p
    # Fallback: script directory's parent (reasonable default)
    return start.parent


def main():
    pot = import_mod("geometric_spectroscopy.potentials")
    qnm = import_mod("geometric_spectroscopy.qnm_wkb")

    # ------------------------
    # CONFIG
    # ------------------------
    model = "hayward"
    alpha = 0.0

    # Dense L scan. Adjust upper bound if you want more points.
    L_values = list(range(4, 26, 2))  # 4,6,...,24

    ells = [2, 3, 4, 5]
    ns = [0, 1, 2]

    # Potential grid settings
    N_r = 6000
    r_min = 2.2
    r_max = 60.0

    # Finite difference step(s). Add more values for stability check.
    deltas = [1e-3]

    # ------------------------
    # OUTPUT DIR (anchored at repo root)
    # ------------------------
    script_dir = Path(__file__).resolve().parent
    repo_root = find_repo_root(script_dir)
    out_dir = repo_root / "artifacts" / "order_stability"
    ensure_dir(str(out_dir))

    dense_rows: list[dict] = []

    # ------------------------
    # COMPUTE
    # ------------------------
    for delta in deltas:
        for L_max in L_values:
            t0 = time.time()
            theta0 = np.zeros(L_max + 1)

            def data_vector(theta):
                out = []
                for ell in ells:
                    rstar, V, _ = pot.build_potential_rstar(
                        model=model,
                        ell=ell,
                        alpha=alpha,
                        theta=theta,
                        L_max=L_max,
                        N_r=N_r,
                        r_min=r_min,
                        r_max=r_max,
                    )
                    for n in ns:
                        res = qnm.qnm_wkb6(rstar, V, ell=ell, n=n, pade=True)
                        out.append(complex(res.omega))
                out = np.array(out)
                return np.concatenate([out.real, out.imag])

            y0 = data_vector(theta0)
            J = np.zeros((len(y0), len(theta0)))

            for j in range(len(theta0)):
                th = theta0.copy()
                th[j] += delta
                y1 = data_vector(th)
                J[:, j] = (y1 - y0) / delta

            rank, smax, smin, cond = compute_svd(J)
            dt = time.time() - t0

            dense_rows.append({
                "model": model,
                "alpha": alpha,
                "pade": 1,
                "delta_fd": float(delta),
                "L_max": int(L_max),
                "M_params": int(L_max + 1),
                "N_data": int(len(y0) // 2),
                "rank": int(rank),
                "sigma_max": float(smax),
                "sigma_min": float(smin),
                "cond": float(cond),
                "runtime_sec": float(dt),
            })

            print(f"[L={L_max:>2}] cond={cond:.3e}  smin={smin:.3e}  rank={rank}  dt={dt:.1f}s")

    dense_csv = out_dir / "jacobian_vs_L_dense.csv"
    write_csv(str(dense_csv), dense_rows)

    # ------------------------
    # FITS (use first delta by default)
    # ------------------------
    delta0 = float(deltas[0])
    rows0 = [r for r in dense_rows if float(r["delta_fd"]) == delta0]

    rows0_sorted = sorted(rows0, key=lambda r: r["L_max"])
    L = np.array([r["L_max"] for r in rows0_sorted], dtype=float)
    cond = np.array([r["cond"] for r in rows0_sorted], dtype=float)

    ln_cond = np.log(cond)
    alpha_hat, b_hat, sse_ln, r2_ln = linear_fit(L, ln_cond)

    log10_cond = np.log10(cond)
    a_hat, c_hat, sse_log10, r2_log10 = linear_fit(L, log10_cond)

    alpha_equiv = a_hat * np.log(10.0)
    diff = alpha_hat - alpha_equiv

    fit_rows = [
        {
            "delta_fd": delta0,
            "fit_form": "ln(cond) = alpha*L + b",
            "alpha": alpha_hat,
            "b": b_hat,
            "SSE": sse_ln,
            "R2": r2_ln,
        },
        {
            "delta_fd": delta0,
            "fit_form": "log10(cond) = a*L + c",
            "a": a_hat,
            "c": c_hat,
            "SSE": sse_log10,
            "R2": r2_log10,
            "alpha_equiv": alpha_equiv,
        },
        {
            "delta_fd": delta0,
            "fit_form": "consistency alpha - a*ln10",
            "alpha_minus_a_ln10": float(diff),
        },
    ]

    fit_csv = out_dir / "cond_growth_fit.csv"
    write_csv(str(fit_csv), fit_rows)

    print("\n=== FIT RESULTS (delta_fd = {}) ===".format(delta0))
    print("exp-fit:   ln(cond)=alpha*L+b     alpha = {:.6f},  b={:.6f},  R2={:.4f}".format(alpha_hat, b_hat, r2_ln))
    print("10-fit:    log10(cond)=a*L+c      a     = {:.6f},  c={:.6f},  R2={:.4f}".format(a_hat, c_hat, r2_log10))
    print("check:     alpha ?= a*ln10        a*ln10= {:.6f},  diff={:.3e}".format(alpha_equiv, diff))

    # ------------------------
    # PLOTS (optional)
    # ------------------------
    if HAVE_PLT:
        # log10(cond) vs L
        plt.figure()
        plt.plot(L, log10_cond, marker="o", label=r"data: $\log_{10}(\kappa)$")
        plt.plot(L, a_hat * L + c_hat, label=rf"fit: $a={a_hat:.3f}$")
        plt.xlabel(r"$L_{\max}$")
        plt.ylabel(r"$\log_{10}(\mathrm{cond})$")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(str(out_dir / "cond_vs_L_fit_log10.png"))
        plt.close()

        # ln(cond) vs L
        plt.figure()
        plt.plot(L, ln_cond, marker="o", label=r"data: $\ln(\kappa)$")
        plt.plot(L, alpha_hat * L + b_hat, label=rf"fit: $\alpha={alpha_hat:.3f}$")
        plt.xlabel(r"$L_{\max}$")
        plt.ylabel(r"$\ln(\mathrm{cond})$")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(str(out_dir / "cond_vs_L_fit_ln.png"))
        plt.close()

        print(f"[ok] plots saved into {out_dir}/ (cond_vs_L_fit_log10.png, cond_vs_L_fit_ln.png)")
    else:
        print("[info] matplotlib not installed -> skipping plots")


if __name__ == "__main__":
    main()