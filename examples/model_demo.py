#!/usr/bin/env python3
"""
Geometric Spectroscopy — Model Demo (real computation)

Canonical demo entry point.

It performs:
- Jacobian build for a chosen model and alpha
- conditioning diagnostics (raw / geom / phys)
- effective rank
- Fisher prediction for theta
- Monte Carlo reconstruction for theta (optional)
- saves a compact results bundle (CSV + JSON) into results/

Usage:
  python examples/model_demo.py --model hayward --alpha 0.05 --Lmax 5 --save
  python examples/model_demo.py --model bardeen --mc 0
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any

import numpy as np

from geometric_spectroscopy.jacobian import (
    build_jacobian,
    conditioning_raw,
    conditioning_geom,
    conditioning_phys,
    deformation_weights_phys,
    effective_rank,
)
from geometric_spectroscopy.monte_carlo import fisher_sigma_theta, monte_carlo_theta


def _now_stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _ensure_results_dir(path: str = "results") -> str:
    os.makedirs(path, exist_ok=True)
    return path


def _to_float(x: Any) -> float:
    """Convert scalar-like / array-like object to float robustly."""
    if isinstance(x, (float, int, np.floating, np.integer)):
        return float(x)

    arr = np.asarray(x)
    if arr.ndim == 0:
        return float(arr)
    if arr.size == 0:
        return float("nan")
    return float(arr.reshape(-1)[-1])


def _kappa_from_conditioning(ret: Any) -> float:
    """
    conditioning_* may return:
      - kappa (float-like), or
      - (smax, smin, kappa) where kappa can be scalar or array-like.
    Normalize to scalar kappa.
    """
    if isinstance(ret, (tuple, list)):
        if len(ret) == 0:
            return float("nan")
        return _to_float(ret[-1])
    return _to_float(ret)


@dataclass(frozen=True)
class DemoResult:
    model: str
    alpha_true: float
    L_max: int
    M: int
    N: int
    sigma_omega: float
    ridge: float
    mc: int
    seed: int

    kappa_raw: float
    kappa_geom: float
    kappa_phys: float
    rank_eff: float

    fisher_sigma_alpha: float
    fisher_sigma_theta: list[float]

    mc_mean_alpha: float | None
    mc_std_alpha: float | None
    mc_mean_theta: list[float] | None
    mc_std_theta: list[float] | None

    timestamp: str


def _print_summary(r: DemoResult) -> None:
    print("=" * 72)
    print(f"Geometric Spectroscopy Demo — {r.model.upper()}")
    print("=" * 72)
    print(f"timestamp     = {r.timestamp}")
    print(f"L_max         = {r.L_max}  (M={r.M} parameters)")
    print(f"alpha_true    = {r.alpha_true:g}")
    print(f"sigma_omega   = {r.sigma_omega:g}   (complex iid noise level)")
    print(f"ridge         = {r.ridge:g}")
    print(f"MC samples    = {r.mc}   (seed={r.seed})")
    print()
    print("Jacobian diagnostics")
    print(f"  shape       = {r.N} x {r.M}")
    print(f"  rank_eff    = {r.rank_eff:.3g} / {min(r.N, r.M)}")
    print(f"  kappa_raw   = {r.kappa_raw:.3e}")
    print(f"  kappa_geom  = {r.kappa_geom:.3e}")
    print(f"  kappa_phys  = {r.kappa_phys:.3e}")
    print()
    print("Alpha reconstruction (theta[0])")
    print(f"  Fisher sigma(alpha) = {r.fisher_sigma_alpha:.3g}")
    if r.mc_mean_alpha is not None and r.mc_std_alpha is not None:
        print(f"  MonteCarlo alpha    = {r.mc_mean_alpha:.6g} ± {r.mc_std_alpha:.3g}")
        print(f"  Bias                = {r.mc_mean_alpha - r.alpha_true:+.3g}")
    else:
        print("  MonteCarlo          = (skipped)")
    print("=" * 72)


def _save_bundle(r: DemoResult, out_prefix: str) -> tuple[str, str]:
    results_dir = _ensure_results_dir("results")
    csv_path = os.path.join(results_dir, f"{out_prefix}.csv")
    json_path = os.path.join(results_dir, f"{out_prefix}.json")

    d = asdict(r)

    csv_fields = [
        "timestamp",
        "model",
        "alpha_true",
        "L_max",
        "N",
        "M",
        "sigma_omega",
        "ridge",
        "mc",
        "seed",
        "kappa_raw",
        "kappa_geom",
        "kappa_phys",
        "rank_eff",
        "fisher_sigma_alpha",
        "mc_mean_alpha",
        "mc_std_alpha",
    ]

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=csv_fields)
        w.writeheader()
        w.writerow({k: d.get(k) for k in csv_fields})

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(d, f, indent=2, ensure_ascii=False)

    return csv_path, json_path


def run_demo(
    model: str,
    alpha_true: float,
    L_max: int,
    sigma_omega: float,
    ridge: float,
    mc: int,
    seed: int,
) -> DemoResult:
    J = build_jacobian(model=model, alpha=alpha_true, L_max=L_max)

    k_raw = _kappa_from_conditioning(conditioning_raw(J))
    k_geom = _kappa_from_conditioning(conditioning_geom(J))

    # IMPORTANT: in your API, phys-conditioning needs model+alpha+L_max to build weights
    weights = deformation_weights_phys(model=model, alpha=alpha_true, L_max=L_max)
    k_phys = _kappa_from_conditioning(conditioning_phys(J, weights))

    r_eff = float(_to_float(effective_rank(J)))

    N, M = int(J.shape[0]), int(J.shape[1])

    theta_true = np.zeros(M, dtype=float)
    theta_true[0] = float(alpha_true)

    sig = fisher_sigma_theta(J, sigma_omega=sigma_omega, ridge=ridge)
    sig_list = [float(x) for x in np.asarray(sig).ravel()]

    mc_mean_alpha: float | None = None
    mc_std_alpha: float | None = None
    mc_mean_theta: list[float] | None = None
    mc_std_theta: list[float] | None = None

    if mc and mc > 0:
        mean, std = monte_carlo_theta(
            J,
            theta_true=theta_true,
            sigma_omega=sigma_omega,
            ridge=ridge,
            realizations=int(mc),
            seed=int(seed),
        )
        mean = np.asarray(mean).ravel()
        std = np.asarray(std).ravel()

        mc_mean_alpha = float(mean[0])
        mc_std_alpha = float(std[0])
        mc_mean_theta = [float(x) for x in mean]
        mc_std_theta = [float(x) for x in std]

    return DemoResult(
        model=str(model),
        alpha_true=float(alpha_true),
        L_max=int(L_max),
        M=int(M),
        N=int(N),
        sigma_omega=float(sigma_omega),
        ridge=float(ridge),
        mc=int(mc),
        seed=int(seed),
        kappa_raw=float(k_raw),
        kappa_geom=float(k_geom),
        kappa_phys=float(k_phys),
        rank_eff=float(r_eff),
        fisher_sigma_alpha=float(sig_list[0]),
        fisher_sigma_theta=sig_list,
        mc_mean_alpha=mc_mean_alpha,
        mc_std_alpha=mc_std_alpha,
        mc_mean_theta=mc_mean_theta,
        mc_std_theta=mc_std_theta,
        timestamp=_now_stamp(),
    )


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Canonical demo for any supported model.")
    ap.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["hayward", "bardeen", "simpson_visser"],
        help="Model name.",
    )
    ap.add_argument("--alpha", type=float, default=0.05, help="True alpha used to build J.")
    ap.add_argument("--Lmax", type=int, default=5, help="Maximum deformation multipole L_max.")
    ap.add_argument("--sigma-omega", type=float, default=1e-2, help="Complex iid noise level.")
    ap.add_argument("--ridge", type=float, default=1e-6, help="Ridge regularization for Fisher/MC.")
    ap.add_argument("--mc", type=int, default=500, help="Monte Carlo realizations (0 to skip).")
    ap.add_argument("--seed", type=int, default=1, help="Random seed for Monte Carlo.")
    ap.add_argument("--save", action="store_true", help="Save CSV+JSON bundle into results/.")
    args = ap.parse_args(argv)

    r = run_demo(
        model=args.model,
        alpha_true=args.alpha,
        L_max=args.Lmax,
        sigma_omega=args.sigma_omega,
        ridge=args.ridge,
        mc=args.mc,
        seed=args.seed,
    )

    _print_summary(r)

    if args.save:
        prefix = f"{args.model}_demo_{r.timestamp}"
        csv_path, json_path = _save_bundle(r, prefix)
        print("Saved:")
        print(f"  {csv_path}")
        print(f"  {json_path}")


if __name__ == "__main__":
    main()