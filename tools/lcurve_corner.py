import csv
import math
import os
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple

import numpy as np


IN_CSV = os.path.join("artifacts", "order_stability", "tikhonov_scan.csv")
OUT_DIR = os.path.join("artifacts", "order_stability")
OUT_CSV = os.path.join(OUT_DIR, "lcurve_corner.csv")
OUT_POINTS_CSV = os.path.join(OUT_DIR, "lcurve_points_with_curvature.csv")


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def read_csv(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append(row)
    return rows


def write_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        print(f"[warn] no rows to write: {path}")
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"[ok] wrote {path} ({len(rows)} rows)")


def curvature_discrete(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Discrete curvature for a planar curve given by points (x_i, y_i) in order.
    Uses 3-point formula: k = 2*Area / (a*b*c), where a,b,c are triangle side lengths.
    Returns array k with k[0]=k[-1]=nan.
    """
    n = len(x)
    k = np.full(n, np.nan, dtype=float)
    for i in range(1, n - 1):
        x1, y1 = x[i - 1], y[i - 1]
        x2, y2 = x[i], y[i]
        x3, y3 = x[i + 1], y[i + 1]

        a = math.hypot(x2 - x1, y2 - y1)
        b = math.hypot(x3 - x2, y3 - y2)
        c = math.hypot(x3 - x1, y3 - y1)

        # Avoid degenerate triangles
        if a <= 0 or b <= 0 or c <= 0:
            continue

        # Twice signed area of triangle
        area2 = abs((x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1))
        # Curvature (always nonnegative here)
        k[i] = area2 / (a * b * c)
    return k


def main() -> None:
    if not os.path.exists(IN_CSV):
        raise FileNotFoundError(f"Missing input CSV: {IN_CSV}")

    rows = read_csv(IN_CSV)

    # Parse numeric columns
    data: List[Tuple[float, float, float, float]] = []
    # (lambda, mean_error, std_error, noise_gain)
    for row in rows:
        lam = float(row["lambda"])
        mean_err = float(row["mean_error"])
        std_err = float(row["std_error"])
        gain = float(row["noise_gain"])
        # Must be positive for log
        if lam > 0 and mean_err > 0 and gain > 0:
            data.append((lam, mean_err, std_err, gain))

    # Sort by lambda ascending (should already be)
    data.sort(key=lambda t: t[0])

    lam = np.array([t[0] for t in data], dtype=float)
    mean_err = np.array([t[1] for t in data], dtype=float)
    std_err = np.array([t[2] for t in data], dtype=float)
    gain = np.array([t[3] for t in data], dtype=float)

    # L-curve in log-log:
    # x = log10(gain), y = log10(mean_error)
    x = np.log10(gain)
    y = np.log10(mean_err)

    k = curvature_discrete(x, y)

    # Choose corner as max curvature (ignore NaNs at ends)
    valid = np.isfinite(k)
    if not np.any(valid):
        raise RuntimeError("Curvature computation failed (no valid points).")

    i_star = int(np.nanargmax(k))
    lam_star = float(lam[i_star])
    gain_star = float(gain[i_star])
    mean_star = float(mean_err[i_star])
    std_star = float(std_err[i_star])
    k_star = float(k[i_star])

    ensure_dir(OUT_DIR)

    # Write per-point curvature table (useful for debugging/plots)
    points_rows: List[Dict[str, Any]] = []
    for i in range(len(lam)):
        points_rows.append(
            {
                "lambda": float(lam[i]),
                "mean_error": float(mean_err[i]),
                "std_error": float(std_err[i]),
                "noise_gain": float(gain[i]),
                "log10_noise_gain": float(x[i]),
                "log10_mean_error": float(y[i]),
                "curvature": (float(k[i]) if np.isfinite(k[i]) else ""),
                "is_corner": 1 if i == i_star else 0,
            }
        )
    write_csv(OUT_POINTS_CSV, points_rows)

    # Write corner summary CSV
    summary = [
        {
            "lambda_star": lam_star,
            "mean_error": mean_star,
            "std_error": std_star,
            "noise_gain": gain_star,
            "curvature": k_star,
            "index": i_star,
        }
    ]
    write_csv(OUT_CSV, summary)

    print("\n[corner] L-curve corner (max curvature in log-log):")
    print(f"  lambda*     = {lam_star:g}")
    print(f"  mean_error  = {mean_star:.6g}  (std={std_star:.3g})")
    print(f"  noise_gain  = {gain_star:.6g}")
    print(f"  curvature   = {k_star:.6g}")
    print(f"  wrote:\n    - {OUT_CSV}\n    - {OUT_POINTS_CSV}")


if __name__ == "__main__":
    main()