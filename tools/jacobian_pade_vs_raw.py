import csv
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def write_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        print(f"[warn] no rows for {path}")
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"[ok] wrote {path} ({len(rows)} rows)")


def import_mod(path: str):
    __import__(path)
    import sys

    return sys.modules[path]


@dataclass
class Case:
    name: str
    model: str
    alpha: float


def default_case() -> Case:
    # Schwarzschild benchmark via Hayward(alpha=0)
    return Case(name="schwarzschild_via_hayward", model="hayward", alpha=0.0)


def qnms_vector(
    *,
    model: str,
    alpha: float,
    theta: np.ndarray,
    L_max: int,
    ells: List[int],
    ns: List[int],
    pade: bool,
    N_r: int = 6000,
    r_min: float = 2.2,
    r_max: float = 60.0,
) -> np.ndarray:
    """
    Returns complex vector stacking omega(ell,n) for all ell in ells and n in ns
    using WKB6 with pade flag.
    """
    pot = import_mod("geometric_spectroscopy.potentials")
    qnm = import_mod("geometric_spectroscopy.qnm_wkb")

    out: List[complex] = []
    for ell in ells:
        rstar, V, _w = pot.build_potential_rstar(
            model=model,
            ell=int(ell),
            alpha=float(alpha),
            theta=theta,
            L_max=int(L_max),
            N_r=int(N_r),
            r_min=float(r_min),
            r_max=float(r_max),
        )
        for n in ns:
            res = qnm.qnm_wkb6(
                rstar,
                V,
                ell=int(ell),
                n=int(n),
                pade=bool(pade),
                return_diagnostics=False,
            )
            out.append(complex(res.omega))
    return np.array(out, dtype=np.complex128)


def realify_complex_vector(v: np.ndarray) -> np.ndarray:
    """Maps C^m -> R^{2m} by concatenating Re and Im."""
    return np.concatenate([v.real, v.imag]).astype(np.float64)


def fd_jacobian(
    *,
    model: str,
    alpha: float,
    theta0: np.ndarray,
    L_max: int,
    ells: List[int],
    ns: List[int],
    pade: bool,
    delta: float = 1e-3,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Finite-difference Jacobian of the data vector with respect to theta coefficients.
    Returns:
      y0 (complex m-vector),
      J (real 2m x d matrix)
    """
    y0 = qnms_vector(
        model=model,
        alpha=alpha,
        theta=theta0,
        L_max=L_max,
        ells=ells,
        ns=ns,
        pade=pade,
    )
    y0r = realify_complex_vector(y0)

    d = len(theta0)
    m2 = len(y0r)
    J = np.zeros((m2, d), dtype=np.float64)

    for j in range(d):
        th = theta0.copy()
        th[j] += float(delta)
        y1 = qnms_vector(
            model=model,
            alpha=alpha,
            theta=th,
            L_max=L_max,
            ells=ells,
            ns=ns,
            pade=pade,
        )
        y1r = realify_complex_vector(y1)
        J[:, j] = (y1r - y0r) / float(delta)

    return y0, J


def svd_stats(J: np.ndarray) -> Dict[str, Any]:
    s = np.linalg.svd(J, compute_uv=False)
    smax = float(s[0])
    smin = float(s[-1])
    cond = float(smax / smin) if smin > 0 else float("inf")
    rank = int(np.sum(s > 1e-12 * smax))
    return {
        "rank": rank,
        "sigma_max": smax,
        "sigma_min": smin,
        "cond": cond,
    }


def main():
    case = default_case()

    out_dir = os.path.join("artifacts", "order_stability")
    ensure_dir(out_dir)

    # Settings: keep this modest so it runs fast.
    # You can increase later for the paper.
    L_max = 6
    # theta[0] is unused in your conventions (often), but we keep full vector length.
    theta0 = np.zeros(L_max + 1, dtype=float)

    ells = [2, 3, 4, 5]
    ns = [0, 1, 2]
    delta = 1e-3

    rows: List[Dict[str, Any]] = []

    for pade in [False, True]:
        print(f"[run] FD Jacobian: pade={pade}")
        y0, J = fd_jacobian(
            model=case.model,
            alpha=case.alpha,
            theta0=theta0,
            L_max=L_max,
            ells=ells,
            ns=ns,
            pade=pade,
            delta=delta,
        )
        stats = svd_stats(J)

        # Also store a simple “scale” measure of y0 for context
        y_abs_mean = float(np.mean(np.abs(y0)))
        y_abs_max = float(np.max(np.abs(y0)))

        row = {
            "case": case.name,
            "model": case.model,
            "alpha": case.alpha,
            "pade": int(pade),
            "L_max": L_max,
            "ells": ",".join(map(str, ells)),
            "ns": ",".join(map(str, ns)),
            "delta": delta,
            "y_abs_mean": y_abs_mean,
            "y_abs_max": y_abs_max,
            **stats,
        }
        rows.append(row)

    out_csv = os.path.join(out_dir, "jacobian_pade_vs_raw.csv")
    write_csv(out_csv, rows)

    print("\n[done] wrote:")
    print(f"  - {out_csv}")
    print("\nInterpretation:")
    print("  Compare sigma_min and cond for pade=0 vs pade=1.")
    print("  Larger sigma_min and smaller cond => more stable inverse diagnostics.")


if __name__ == "__main__":
    main()