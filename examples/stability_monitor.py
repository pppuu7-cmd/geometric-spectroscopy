"""
Stability monitor (Standard B).

Goal
----
Run a small sweep over discretization / FD settings and report robustness metrics:
  - κ_raw, κ_geom, κ_phys  (SVD condition numbers on different parameter scalings)
  - effective rank
  - damped fraction (Im ω < 0) across computed QNMs

This script is designed to be:
  - fast enough for local runs and CI smoke use (with env knobs),
  - honest: it reports spreads and guardrails, not "one magical number".

Why this file exists
--------------------
We want to be *more useful than "one-off ω tables"*:
robust inference needs stability diagnostics and error bars for κ / rank / damping.

Environment overrides
---------------------
GS_MODEL            default: hayward
GS_RESULTS_DIR      default: results
GS_WKB_ORDER        default: 6
GS_ALPHA            default: 0.05
GS_L_MAX            default: 5
GS_DELTA_LIST       default: "1e-3,5e-4,2e-3"
GS_N_R_LIST         default: "2000,2400,2800"
GS_R_MAX_LIST       default: "50.0"   (you can pass "40,50,60")
GS_PROFILE_WIDTH    default: 3.0
GS_ELL_MAX          default: 7   (ells = 2..ELL_MAX)
GS_NS_MAX           default: 1   (ns = 0..NS_MAX)
GS_PS_ELLS          default: "2,3"  (extras)
GS_TIMEOUT_SEC      default: 180
GS_GUARD_KAPPA_GEOM_SPREAD  default: 0.20
GS_GUARD_RANK_SPREAD        default: 0.35
GS_GUARD_DAMPED_FRACTION_MIN default: 1.00

Output
------
Writes:
  - results/stability_monitor_<model>_<timestamp>.csv
  - results/stability_monitor_<model>_<timestamp>.json
"""

from __future__ import annotations

import csv
import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

from geometric_spectroscopy.jacobian import (
    build_jacobian,
    conditioning_raw,
    conditioning_geom,
    conditioning_phys,
    deformation_weights_phys,
)
from geometric_spectroscopy.resonances import compute_data_vector


# -----------------------------
# Small utilities
# -----------------------------

def _parse_int_list(s: str, default: Sequence[int]) -> List[int]:
    try:
        out = []
        for part in s.split(","):
            part = part.strip()
            if not part:
                continue
            out.append(int(part))
        return out if out else list(default)
    except Exception:
        return list(default)


def _parse_float_list(s: str, default: Sequence[float]) -> List[float]:
    try:
        out = []
        for part in s.split(","):
            part = part.strip()
            if not part:
                continue
            out.append(float(part))
        return out if out else list(default)
    except Exception:
        return list(default)


def _parse_int_tuple(s: str, default: Sequence[int]) -> Tuple[int, ...]:
    vals = _parse_int_list(s, default)
    return tuple(vals)


def _percentile_spread(x: Sequence[float]) -> Tuple[float, float, float]:
    """
    Return (median, p10, p90). If empty => (nan, nan, nan).
    """
    if not x:
        return float("nan"), float("nan"), float("nan")
    arr = np.asarray(x, dtype=float)
    med = float(np.median(arr))
    p10 = float(np.percentile(arr, 10))
    p90 = float(np.percentile(arr, 90))
    return med, p10, p90


def _rel_spread(p10: float, p90: float, med: float) -> float:
    """
    Relative spread proxy: (p90 - p10) / median.
    """
    if not np.isfinite(med) or med == 0.0:
        return float("inf")
    return float((p90 - p10) / med)


def effective_rank(Jc: np.ndarray, tol: float = 1e-6) -> float:
    """
    Effective rank proxy: count singular values above tol*smax on the realified matrix.
    Returned as float to allow medians and spreads.
    """
    Jr = np.vstack([Jc.real, Jc.imag])
    s = np.linalg.svd(Jr, compute_uv=False)
    smax = float(s[0]) if s.size else 0.0
    if smax <= 0.0:
        return 0.0
    return float(np.sum(s > tol * smax))


def extract_modes_and_damping(
    model: str,
    alpha: float,
    L_max: int,
    ells: Iterable[int],
    ns: Sequence[int],
    *,
    order: int,
    theta0: np.ndarray,
    grid_kwargs: Dict,
    ps_ells: Sequence[int],
) -> Tuple[List[complex], float, float]:
    """
    Compute data vector and extract only QNM mode block:
      modes = first Nmodes entries in data vector
    Return:
      modes, damped_fraction, max_im_omega
    """
    y = compute_data_vector(
        model=model,
        alpha=float(alpha),
        theta=np.asarray(theta0, dtype=float),
        L_max=int(L_max),
        ells=ells,
        ns=ns,
        order=int(order),
        grid_kwargs=grid_kwargs,
        ps_ells=tuple(ps_ells),
    )

    Nmodes = len(list(ells)) * len(ns)
    modes = [complex(z) for z in np.asarray(y[:Nmodes], dtype=complex)]

    if not modes:
        return modes, 0.0, float("nan")

    im_vals = np.array([z.imag for z in modes], dtype=float)
    damped_fraction = float(np.mean(im_vals < 0.0))
    max_im = float(np.max(im_vals))
    return modes, damped_fraction, max_im


# -----------------------------
# Data model
# -----------------------------

@dataclass(frozen=True)
class OneRun:
    model: str
    order: int
    alpha: float
    L_max: int
    delta: float
    N_r: int
    r_min: float
    r_max: float
    profile_width: float
    ell_max: int
    ns_max: int
    ps_ells: Tuple[int, ...]
    Ndata: int
    Nmodes: int
    kappa_raw: float
    kappa_geom: float
    kappa_phys: float
    rank_eff: float
    damped_fraction: float
    max_im_omega: float


# -----------------------------
# Core sweep
# -----------------------------

def run_one(
    *,
    model: str,
    alpha: float,
    L_max: int,
    order: int,
    delta: float,
    N_r: int,
    r_min: float,
    r_max: float,
    profile_width: float,
    ell_max: int,
    ns_max: int,
    ps_ells: Tuple[int, ...],
) -> OneRun:
    ells = range(2, int(ell_max) + 1)
    ns = tuple(range(0, int(ns_max) + 1))
    theta0 = np.zeros(L_max + 1, dtype=float)

    grid_kwargs = dict(
        r_min=float(r_min),
        r_max=float(r_max),
        N_r=int(N_r),
        profile_width=float(profile_width),
    )

    # Jacobian
    Jc = build_jacobian(
        model=model,
        alpha=float(alpha),
        L_max=int(L_max),
        ells=ells,
        ns=ns,
        order=int(order),
        delta=float(delta),
        theta0=theta0,
        grid_kwargs=grid_kwargs,
        ps_ells=ps_ells,
    )

    Ndata = int(Jc.shape[0])
    Nmodes = len(list(ells)) * len(ns)

    # κ metrics (IMPORTANT: conditioning_* returns (smax, smin, kappa))
    _smax, _smin, k_raw = conditioning_raw(Jc)
    ( _smax_g, _smin_g, k_geom ), _col_norms = conditioning_geom(Jc)

    w_phys = deformation_weights_phys(
        model=model,
        alpha=float(alpha),
        L_max=int(L_max),
        ells=ells,
        grid_kwargs=grid_kwargs,
    )
    _smax_p, _smin_p, k_phys = conditioning_phys(Jc, w_phys)

    r_eff = effective_rank(Jc, tol=1e-6)

    # damping monitor from the *data vector* (not from J)
    _modes, damped_fraction, max_im = extract_modes_and_damping(
        model=model,
        alpha=float(alpha),
        L_max=int(L_max),
        ells=ells,
        ns=ns,
        order=int(order),
        theta0=theta0,
        grid_kwargs=grid_kwargs,
        ps_ells=ps_ells,
    )

    return OneRun(
        model=model,
        order=int(order),
        alpha=float(alpha),
        L_max=int(L_max),
        delta=float(delta),
        N_r=int(N_r),
        r_min=float(r_min),
        r_max=float(r_max),
        profile_width=float(profile_width),
        ell_max=int(ell_max),
        ns_max=int(ns_max),
        ps_ells=tuple(int(x) for x in ps_ells),
        Ndata=Ndata,
        Nmodes=Nmodes,
        kappa_raw=float(k_raw),
        kappa_geom=float(k_geom),
        kappa_phys=float(k_phys),
        rank_eff=float(r_eff),
        damped_fraction=float(damped_fraction),
        max_im_omega=float(max_im),
    )


def summarize(runs: List[OneRun]) -> Dict[str, Dict[str, float]]:
    """
    Summary with median and robust spread.
    """
    def collect(name: str) -> List[float]:
        return [float(getattr(r, name)) for r in runs]

    out: Dict[str, Dict[str, float]] = {}

    for key in ["kappa_raw", "kappa_geom", "kappa_phys", "rank_eff"]:
        med, p10, p90 = _percentile_spread(collect(key))
        out[key] = {
            "median": med,
            "p10": p10,
            "p90": p90,
            "rel_spread": _rel_spread(p10, p90, med),
        }

    # Damping guardrails are “worst-case” oriented:
    damped = collect("damped_fraction")
    max_im = collect("max_im_omega")
    out["damping"] = {
        "damped_fraction_min": float(np.min(damped)) if damped else float("nan"),
        "max_im_omega_max": float(np.max(max_im)) if max_im else float("nan"),
    }

    out["N"] = {"runs": float(len(runs))}
    return out


def guardrail_ok(summary: Dict[str, Dict[str, float]]) -> Tuple[bool, List[str]]:
    """
    Simple guardrail policy (tunable via env).
    """
    k_geom_spread_max = float(os.getenv("GS_GUARD_KAPPA_GEOM_SPREAD", "0.20"))
    rank_spread_max = float(os.getenv("GS_GUARD_RANK_SPREAD", "0.35"))
    damped_min_req = float(os.getenv("GS_GUARD_DAMPED_FRACTION_MIN", "1.00"))

    problems: List[str] = []

    kgeom = summary.get("kappa_geom", {})
    if np.isfinite(kgeom.get("rel_spread", float("inf"))) and kgeom["rel_spread"] > k_geom_spread_max:
        problems.append(f"kappa_geom rel_spread={kgeom['rel_spread']:.3e} > {k_geom_spread_max:.3e}")

    rank = summary.get("rank_eff", {})
    if np.isfinite(rank.get("rel_spread", float("inf"))) and rank["rel_spread"] > rank_spread_max:
        problems.append(f"rank_eff rel_spread={rank['rel_spread']:.3e} > {rank_spread_max:.3e}")

    damp = summary.get("damping", {})
    if np.isfinite(damp.get("damped_fraction_min", float("nan"))) and damp["damped_fraction_min"] < damped_min_req:
        problems.append(f"damped_fraction_min={damp['damped_fraction_min']:.3f} < {damped_min_req:.3f}")

    return (len(problems) == 0), problems


def write_outputs(results_dir: Path, model: str, runs: List[OneRun], summary: Dict) -> Tuple[Path, Path]:
    results_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    csv_path = results_dir / f"stability_monitor_{model}_{ts}.csv"
    json_path = results_dir / f"stability_monitor_{model}_{ts}.json"

    # CSV
    fieldnames = list(asdict(runs[0]).keys()) if runs else []
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in runs:
            w.writerow(asdict(r))

    # JSON
    payload = {
        "summary": summary,
        "runs": [asdict(r) for r in runs],
    }
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    return csv_path, json_path


def main() -> None:
    model = os.getenv("GS_MODEL", "hayward").strip()
    results_dir = Path(os.getenv("GS_RESULTS_DIR", "results")).resolve()

    order = int(os.getenv("GS_WKB_ORDER", "6"))
    alpha = float(os.getenv("GS_ALPHA", "0.05"))
    L_max = int(os.getenv("GS_L_MAX", "5"))

    # Sweep knobs
    delta_list = _parse_float_list(os.getenv("GS_DELTA_LIST", "1e-3,5e-4,2e-3"), default=[1e-3, 5e-4, 2e-3])
    N_r_list = _parse_int_list(os.getenv("GS_N_R_LIST", "2000,2400,2800"), default=[2000, 2400, 2800])
    r_max_list = _parse_float_list(os.getenv("GS_R_MAX_LIST", "50.0"), default=[50.0])

    profile_width = float(os.getenv("GS_PROFILE_WIDTH", "3.0"))
    ell_max = int(os.getenv("GS_ELL_MAX", "7"))
    ns_max = int(os.getenv("GS_NS_MAX", "1"))
    ps_ells = _parse_int_tuple(os.getenv("GS_PS_ELLS", "2,3"), default=[2, 3])

    # r_min: model-dependent defaults (safe values)
    # You can override by exporting GS_R_MIN if you want.
    r_min_env = os.getenv("GS_R_MIN")
    if r_min_env is not None and r_min_env.strip():
        r_min = float(r_min_env)
    else:
        # conservative model-dependent defaults
        if model.lower() == "bardeen":
            r_min = 4.5
        elif model.lower() == "simpson_visser":
            r_min = 3.5
        else:
            r_min = 2.2

    # Build sweep grid
    combos = []
    for delta in delta_list:
        for N_r in N_r_list:
            for r_max in r_max_list:
                combos.append((delta, N_r, r_max))

    runs: List[OneRun] = []
    print(f"Stability scan: N={len(combos)}")

    for (delta, N_r, r_max) in combos:
        r = run_one(
            model=model,
            alpha=alpha,
            L_max=L_max,
            order=order,
            delta=delta,
            N_r=N_r,
            r_min=r_min,
            r_max=r_max,
            profile_width=profile_width,
            ell_max=ell_max,
            ns_max=ns_max,
            ps_ells=ps_ells,
        )
        runs.append(r)

    summ = summarize(runs)

    # Pretty print (matches what ты уже видел)
    print(f"  κ_raw  : median={summ['kappa_raw']['median']:.3e}, rel_spread(p90-p10)/med={summ['kappa_raw']['rel_spread']:.3e}")
    print(f"  κ_geom : median={summ['kappa_geom']['median']:.3e}, rel_spread(p90-p10)/med={summ['kappa_geom']['rel_spread']:.3e}")
    print(f"  κ_phys : median={summ['kappa_phys']['median']:.3e}, rel_spread(p90-p10)/med={summ['kappa_phys']['rel_spread']:.3e}")
    print(f"  rank_eff: median={summ['rank_eff']['median']:.3f}, rel_spread={summ['rank_eff']['rel_spread']:.3e}")
    print(f"  damped_fraction_min={summ['damping']['damped_fraction_min']:.3f}")
    print(f"  max(Im ω)_max={summ['damping']['max_im_omega_max']:.3e}")
    print()

    ok, problems = guardrail_ok(summ)
    if ok:
        print("Stability guardrail: OK")
    else:
        print("Stability guardrail: FAIL")
        for p in problems:
            print("  -", p)

    csv_path, json_path = write_outputs(results_dir, model, runs, summ)
    print(f"\nSaved:\n  {csv_path}\n  {json_path}")


if __name__ == "__main__":
    main()