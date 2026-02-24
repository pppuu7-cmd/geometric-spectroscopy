import os
import csv

from geometric_spectroscopy.detector_forecast import alpha_min_from_snr

RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

DETECTORS = [
    ("LIGO O5", 30),
    ("Einstein Telescope", 400),
    ("LISA", 1000),
    ("ngEHT (1%)", 100),
]


def make_table(C_hayward: float, C_bardeen: float, out_csv: str = "results/detector_forecasts.csv") -> None:
    rows = []
    for name, snr in DETECTORS:
        rows.append(
            [
                name,
                snr,
                alpha_min_from_snr(C_hayward, snr),
                alpha_min_from_snr(C_bardeen, snr),
            ]
        )

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Detector", "SNR", "alpha_min (Hayward)", "alpha_min (Bardeen)"])
        w.writerows(rows)

    print("\nDetector forecasts")
    for r in rows:
        print(f"{r[0]:18s} SNR={r[1]:4d}  alpha_min(H)={r[2]:.4g}  alpha_min(B)={r[3]:.4g}")
    print(f"Saved: {out_csv}")


def _env_float(name: str):
    v = os.getenv(name, "").strip()
    if not v:
        return None
    try:
        return float(v)
    except Exception:
        return None


if __name__ == "__main__":
    # Provide constants used in the paper via environment variables:
    #   set C_HAYWARD=...
    #   set C_BARDEEN=...
    #
    # Example:
    #   C_HAYWARD=1.23 C_BARDEEN=0.98 python examples/detector_table.py
    C_h = _env_float("C_HAYWARD")
    C_b = _env_float("C_BARDEEN")

    if C_h is None or C_b is None:
        raise SystemExit(
            "Missing detector constants.\n"
            "Please set environment variables:\n"
            "  C_HAYWARD=<float>\n"
            "  C_BARDEEN=<float>\n"
            "Then rerun: python examples/detector_table.py\n"
        )

    make_table(C_h, C_b)