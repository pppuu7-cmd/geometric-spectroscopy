# plot_all_figures.py
# ------------------------------------------------------------
# Generates all figures from CSV tables, but with paths anchored
# to the REPO ROOT (so it works no matter where you run it from).
#
# Output:
#   <repo_root>/artifacts/order_stability/figures/*.png
#
# Input CSV expected in:
#   <repo_root>/artifacts/order_stability/*.csv
# ------------------------------------------------------------

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def find_repo_root(start: Path) -> Path:
    """
    Find repo root by walking upwards until we find a folder
    that contains 'geometric_spectroscopy' package directory.
    Falls back to parent of this script if not found.
    """
    for p in [start] + list(start.parents):
        if (p / "geometric_spectroscopy").is_dir():
            return p
    return start.parent


# ================================
# PATHS (anchored at repo root)
# ================================

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = find_repo_root(SCRIPT_DIR)

DATA_DIR = REPO_ROOT / "artifacts" / "order_stability"
FIG_DIR = DATA_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ================================
# PLOT STYLE
# ================================

plt.rcParams.update({
    "font.size": 12,
    "figure.figsize": (6, 4),
    "axes.grid": True
})


def read_csv(name: str) -> pd.DataFrame:
    path = DATA_DIR / name
    if not path.exists():
        raise FileNotFoundError(
            f"CSV not found: {path}\n"
            f"Expected inputs in: {DATA_DIR}"
        )
    return pd.read_csv(path)


# ================================
# 1. ORDER STABILITY (delta_by_order)
# ================================

df_delta = read_csv("delta_by_order.csv")

plt.figure()
for pade_value, label in [(0, "Raw WKB6"), (1, "Padé WKB6")]:
    subset = df_delta[df_delta["pade"] == pade_value]
    plt.plot(subset["n"], subset["delta_abs"], marker="o", label=label)

plt.yscale("log")
plt.xlabel("Overtone n")
plt.ylabel(r"$\Delta_n = |\omega_n^{(6)} - \omega_n^{(3)}|$")
plt.legend()
plt.tight_layout()
plt.savefig(FIG_DIR / "order_stability_delta.png")
plt.close()

# ================================
# 2. REAL PART FREQUENCIES
# ================================

df_freq = read_csv("freq_by_order.csv")

plt.figure()
for order, pade, label in [
    (3, 0, "WKB3"),
    (6, 0, "WKB6 raw"),
    (6, 1, "WKB6 Padé")
]:
    subset = df_freq[(df_freq["order"] == order) & (df_freq["pade"] == pade)]
    plt.plot(subset["n"], subset["omega_re"], marker="o", label=label)

plt.xlabel("Overtone n")
plt.ylabel(r"Re($\omega_n$)")
plt.legend()
plt.tight_layout()
plt.savefig(FIG_DIR / "frequency_real_parts.png")
plt.close()

# ================================
# 3. CONDITION NUMBER VS L_max
# ================================

df_L = read_csv("jacobian_vs_L.csv")

plt.figure()
plt.plot(df_L["L_max"], np.log10(df_L["cond"]), marker="o")
plt.xlabel(r"$L_{\max}$")
plt.ylabel(r"$\log_{10}(\mathrm{cond})$")
plt.tight_layout()
plt.savefig(FIG_DIR / "condition_vs_L.png")
plt.close()

# ================================
# 4. MONTE CARLO NOISE SCAN
# ================================

df_mc = read_csv("monte_carlo_noise.csv")

plt.figure()
plt.plot(df_mc["noise"], df_mc["mean_error"], marker="o")
plt.xscale("log")
plt.yscale("log")
plt.xlabel("Noise amplitude σ")
plt.ylabel("Mean reconstruction error")
plt.tight_layout()
plt.savefig(FIG_DIR / "monte_carlo_noise.png")
plt.close()

# ================================
# 5. TIKHONOV REGULARIZATION SCAN
# ================================

df_tikh = read_csv("tikhonov_scan.csv")

plt.figure()
plt.plot(df_tikh["lambda"], df_tikh["mean_error"], marker="o")
plt.xscale("log")
plt.yscale("log")
plt.xlabel("Regularization parameter λ")
plt.ylabel("Mean reconstruction error")
plt.tight_layout()
plt.savefig(FIG_DIR / "tikhonov_scan.png")
plt.close()

# ================================
# 6. L-CURVE
# ================================

df_corner = read_csv("lcurve_corner.csv")

plt.figure()
plt.scatter(df_tikh["noise_gain"], df_tikh["mean_error"], label="λ scan")
plt.scatter(
    df_corner["noise_gain"],
    df_corner["mean_error"],
    marker="*",
    s=200,
    label="Corner λ*"
)

plt.xscale("log")
plt.yscale("log")
plt.xlabel("Effective noise gain")
plt.ylabel("Mean reconstruction error")
plt.legend()
plt.tight_layout()
plt.savefig(FIG_DIR / "l_curve.png")
plt.close()

print(f"All figures successfully generated in: {FIG_DIR}")